"""Training entry point: checkpoint management, loop, final reporting."""
from __future__ import annotations

import csv
import glob
import math
import os
import random
import time

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from .data import build_loader, load_tokens
from .decode import VSARecurrentDecoder
from .evaluate import evaluate_ppl, five_spot_report
from .model import VSALanguageModel

TRAIN_CONFIG = dict(
    batch_size=32, lr=3e-4, weight_decay=0.1, warmup_steps=1000,
    ce_chunk=8, grad_clip=1.0, grad_skip=50.0,
    val_every=1000, log_every=100, print_every=500, ckpt_every=1000,
)


def _find_checkpoint(run: str, working: str):
    """Pick the most-trained `{run}_latest.pth` among all attached inputs."""
    candidates = sorted(glob.glob(f"/kaggle/input/**/{run}_latest.pth", recursive=True))
    local = os.path.join(working, f"{run}_latest.pth")
    if os.path.exists(local):
        candidates.append(local)
    best, best_step = None, -1
    for c in candidates:
        try:
            ck = torch.load(c, map_location="cpu", weights_only=True)
        except Exception:
            try:
                ck = torch.load(c, map_location="cpu")
            except Exception:
                print(f"[ckpt] unreadable, skipping: {c}")
                continue
        s = ck.get("step", 0) if isinstance(ck, dict) else -1
        del ck
        print(f"[ckpt] found: {c}  (step {s})")
        if s > best_step:
            best, best_step = c, s
    print(f"[ckpt] using: {best}  (step {best_step})")
    return best


def main(run: str = "vsa", max_len: int = 512):
    """Train (or resume, or no-op-finish) one arm. Returns (model, tokens)."""
    import bitsandbytes as bnb
    import tiktoken
    import torch.optim as optim
    from tqdm import tqdm

    torch.manual_seed(0)
    random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    cfg = TRAIN_CONFIG
    hybrid_every = 4 if run == "vsa" else 1
    working = "/kaggle/working"

    model = VSALanguageModel(max_T=max_len, chunk_size=32,
                             hybrid_every=hybrid_every).cuda()
    print(f"Model ({run}): {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    tokens = load_tokens()
    val_start = len(tokens) - 1_000_000
    batches, total_steps = build_loader(tokens, cfg["batch_size"], max_len)

    decay_p, no_decay_p = [], []
    for n, p in model.named_parameters():
        if p.ndim < 2 or n.startswith("embed") or "pos_code" in n or "log_gamma" in n:
            no_decay_p.append(p)
        else:
            decay_p.append(p)
    optimizer = bnb.optim.AdamW8bit(
        [{"params": decay_p, "weight_decay": cfg["weight_decay"]},
         {"params": no_decay_p, "weight_decay": 0.0}],
        lr=cfg["lr"], betas=(0.9, 0.95))

    ckpt_path = os.path.join(working, f"{run}_latest.pth")
    start_step = 0
    found = _find_checkpoint(run, working)
    if found:
        ckpt = torch.load(found, map_location="cuda", weights_only=True)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
                print("Optimizer state loaded.")
            except Exception as e:
                print(f"Optimizer state incompatible — Adam fresh. ({e})")
        start_step = ckpt.get("step", 0)
        print(f"Resumed at step {start_step}.")

    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])

    def lr_lambda(step):
        if step < cfg["warmup_steps"]:
            return step / max(1, cfg["warmup_steps"])
        prog = (step - cfg["warmup_steps"]) / max(1, total_steps - cfg["warmup_steps"])
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=start_step - 1)

    log_path = os.path.join(working, f"{run}_log.csv")
    val_log_path = os.path.join(working, f"{run}_val.csv")
    for path, header in [(log_path, ["step", "loss", "grad_norm"]),
                         (val_log_path, ["step", "val_loss", "val_ppl"])]:
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(header)

    def save(path, step):
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "step": step, "config": model.config}, path)

    enc = tiktoken.get_encoding("gpt2")
    decoder = VSARecurrentDecoder(model)
    prompt = torch.tensor([enc.encode_ordinary("The water cycle is a process where")], device="cuda")

    model.train()
    step = start_step
    scaler = torch.amp.GradScaler("cuda")
    t_last = time.perf_counter()
    ema_dt = None

    pbar = tqdm(batches, desc=run, initial=start_step, total=total_steps)
    for x, y in pbar:
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        with torch.autocast("cuda", torch.float16):
            hidden = model(x)
            total_loss = 0.0
            for i in range(0, hidden.shape[0], cfg["ce_chunk"]):
                hc = hidden[i:i + cfg["ce_chunk"]]
                yc = y[i:i + hc.shape[0]]

                def _ce(h, tgt):
                    logits = model.lm_head(h).float()
                    return F.cross_entropy(logits.view(-1, logits.size(-1)),
                                           tgt.reshape(-1), reduction="sum")

                total_loss = total_loss + torch.utils.checkpoint.checkpoint(
                    _ce, hc, yc, use_reentrant=False)
            loss = total_loss / (hidden.shape[0] * hidden.shape[1])

        if not torch.isfinite(loss):
            print(f"\nstep {step}: non-finite loss — skipping batch", flush=True)
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        if not torch.isfinite(total_norm):
            print(f"\nstep {step}: grad norm {total_norm} — skipping update", flush=True)
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
        else:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        now = time.perf_counter()
        dt = now - t_last
        t_last = now
        ema_dt = dt if ema_dt is None else 0.95 * ema_dt + 0.05 * dt
        step += 1
        pbar.set_postfix(loss=f"{loss.item():.3f}", norm=f"{total_norm.item():.2f}",
                         spd=f"{ema_dt:.2f}s/it")

        if step % cfg["log_every"] == 0:
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([step, f"{loss.item():.4f}", f"{total_norm.item():.3f}"])

        if step % cfg["print_every"] == 0:
            print(f"\n--- Step {step} ---  loss {loss.item():.3f}  "
                  f"norm {total_norm.item():.2f}  lr {scheduler.get_last_lr()[0]:.2e}")
            parts = []
            for i, blk in enumerate(model.blocks):
                if hasattr(blk, "vsa"):
                    g = blk.vsa.gamma().detach()
                    parts.append(f"L{i} g [{g.min():.4f},{g.max():.4f}]")
            if parts:
                print("  ".join(parts), flush=True)
            model.eval()
            with torch.no_grad():
                out = decoder.generate(prompt, 60, temperature=0.7, top_k=40)
            model.train()
            print(f"Sample: {enc.decode(out[0].tolist())}\n", flush=True)

        if step % cfg["val_every"] == 0:
            model.eval()
            ppl = evaluate_ppl(model, tokens, val_start, 4096)
            model.train()
            print(f"  >>> step {step}  val_ppl {ppl:.2f}  val_loss {math.log(ppl):.3f}", flush=True)
            with open(val_log_path, "a", newline="") as f:
                csv.writer(f).writerow([step, f"{math.log(ppl):.4f}", f"{ppl:.2f}"])

        if step % cfg["ckpt_every"] == 0:
            save(ckpt_path, step)
        if step % 5000 == 0:
            save(os.path.join(working, f"{run}_backup.pth"), step)

        if step >= total_steps:
            print("\nReached total_steps — training complete.")
            break

    save(ckpt_path, step)
    model.eval()
    final_ppl = evaluate_ppl(model, tokens, val_start, 4096)
    print(f"\nFINAL (spot A): step {step}  val_ppl {final_ppl:.2f}  val_loss {math.log(final_ppl):.3f}")
    print(f"{run.upper()} REPORT:")
    five_spot_report(model, tokens)
    with torch.no_grad():
        out = decoder.generate(prompt, 60, temperature=0.7, top_k=40)
    print(f"Final sample: {enc.decode(out[0].tolist())}")
    return model, tokens
