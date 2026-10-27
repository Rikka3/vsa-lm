"""Training entry point: hardware auto-profiler, checkpoint management, loop, reporting."""
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
    batch_size=32,          # effective batch size (preserved across all GPUs via grad accumulation)
    lr=3e-4,
    vsa_lr_mult=5.0,        # faster learning rate for O(1) VSA parameters (pos_code, log_gamma)
    weight_decay=0.1,
    warmup_steps=1000,
    ce_chunk=4,
    grad_clip=1.0,
    val_every=1000,
    log_every=100,
    print_every=500,
    ckpt_every=1000,
)


def auto_fit_hardware(target_batch_size: int, max_len: int, n_params: int) -> dict:
    """Inspect available GPU VRAM and pick micro_batch_size, grad_accum,
    activation checkpointing, and dtype so training fits on 8GB..96GB GPUs."""
    if not torch.cuda.is_available():
        return dict(
            vram_gb=0.0, gpu_name="CPU", micro_bs=max(1, min(4, target_batch_size)),
            grad_accum=max(1, target_batch_size // 4), grad_ckpt=False,
            amp_dtype=torch.float32, use_scaler=False, use_8bit_optim=False,
        )

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)
    gpu_name = props.name
    bf16_ok = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
    use_scaler = (amp_dtype == torch.float16)

    # Estimate model scale factor (1.0 for ~100M params, ~10.0 for ~1B params)
    scale = max(1.0, n_params / 120_000_000.0) * max(1.0, max_len / 512.0)

    if vram_gb >= 40.0 / scale:
        micro_bs = target_batch_size
        grad_ckpt = False
        use_8bit = False
    elif vram_gb >= 20.0 / scale:
        micro_bs = max(1, target_batch_size // 2)
        grad_ckpt = (scale > 2.0)
        use_8bit = True
    elif vram_gb >= 12.0 / scale:
        micro_bs = max(1, target_batch_size // 4)
        grad_ckpt = True
        use_8bit = True
    else:
        # Ultra-low VRAM (e.g. 8GB laptop GPU or 1B model on 15GB T4)
        micro_bs = max(1, target_batch_size // 8)
        grad_ckpt = True
        use_8bit = True

    # Ensure micro_bs divides target_batch_size cleanly
    while target_batch_size % micro_bs != 0 and micro_bs > 1:
        micro_bs -= 1
    grad_accum = target_batch_size // micro_bs

    return dict(
        vram_gb=vram_gb, gpu_name=gpu_name, micro_bs=micro_bs,
        grad_accum=grad_accum, grad_ckpt=grad_ckpt, amp_dtype=amp_dtype,
        use_scaler=use_scaler, use_8bit_optim=use_8bit,
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


def _build_optimizer(model, lr: float, wd: float, vsa_lr_mult: float, use_8bit: bool):
    """Two-speed AdamW: standard LR on projections/MLP, boosted LR on VSA pos_code/log_gamma."""
    decay_p, no_decay_p, vsa_p = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "pos_code" in n or "log_gamma" in n:
            vsa_p.append(p)
        elif p.ndim < 2 or n.startswith("embed"):
            no_decay_p.append(p)
        else:
            decay_p.append(p)

    groups = [
        {"params": decay_p, "weight_decay": wd, "lr_scale": 1.0},
        {"params": no_decay_p, "weight_decay": 0.0, "lr_scale": 1.0},
        {"params": vsa_p, "weight_decay": 0.0, "lr_scale": vsa_lr_mult},
    ]

    if use_8bit:
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(groups, lr=lr, betas=(0.9, 0.95)), "AdamW8bit"
        except ImportError:
            pass
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95)), "AdamW"


def _forward_model(model, x: torch.Tensor, grad_ckpt: bool) -> torch.Tensor:
    """Forward pass with optional per-block activation checkpointing to save VRAM."""
    h = model.embed(x)
    for block in model.blocks:
        if grad_ckpt and model.training:
            h = torch.utils.checkpoint.checkpoint(block, h, use_reentrant=False)
        else:
            h = block(h)
    return model.norm_f(h)


def main(run: str = "vsa", max_len: int = 512, working: str | None = None):
    """Train (or resume) one arm with automatic hardware fitting. Returns (model, tokens)."""
    import tiktoken
    from tqdm import tqdm

    torch.manual_seed(0)
    random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    cfg = TRAIN_CONFIG
    hybrid_every = 4 if run == "vsa" else 1
    if working is None:
        working = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()

    model = VSALanguageModel(max_T=max_len, chunk_size=32,
                             hybrid_every=hybrid_every).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    hw = auto_fit_hardware(cfg["batch_size"], max_len, n_params)

    print(f"Model ({run}): {n_params/1e6:.2f}M parameters")
    print(f"[auto-hw] GPU: {hw['gpu_name']} ({hw['vram_gb']:.1f} GB VRAM) -> "
          f"micro_bs={hw['micro_bs']} x grad_accum={hw['grad_accum']} "
          f"(effective bs={cfg['batch_size']}), grad_ckpt={hw['grad_ckpt']}, "
          f"dtype={hw['amp_dtype']}")

    tokens = load_tokens()
    val_start = len(tokens) - 1_000_000
    batches, total_steps = build_loader(tokens, cfg["batch_size"], max_len)

    optimizer, opt_name = _build_optimizer(
        model, cfg["lr"], cfg["weight_decay"], cfg["vsa_lr_mult"], hw["use_8bit_optim"]
    )
    print(f"[auto-hw] Optimizer: {opt_name} (backbone lr={cfg['lr']:.1e}, "
          f"VSA lr={cfg['lr']*cfg['vsa_lr_mult']:.1e})")

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

    def lr_mult(step):
        if step < cfg["warmup_steps"]:
            return step / max(1, cfg["warmup_steps"])
        prog = (step - cfg["warmup_steps"]) / max(1, total_steps - cfg["warmup_steps"])
        return 0.05 + 0.475 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    def set_step_lr(step):
        m = lr_mult(step)
        for g in optimizer.param_groups:
            g["lr"] = cfg["lr"] * m * g.get("lr_scale", 1.0)

    set_step_lr(start_step)

    log_path = os.path.join(working, f"{run}_log.csv")
    val_log_path = os.path.join(working, f"{run}_val.csv")
    for path, header in [(log_path, ["step", "loss", "grad_norm"]),
                         (val_log_path, ["step", "val_loss", "val_ppl"])]:
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(header)

    def save(path, step):
        tmp = path + ".tmp"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "step": step, "config": model.config}, tmp)
        os.replace(tmp, path)

    enc = tiktoken.get_encoding("gpt2")
    decoder = VSARecurrentDecoder(model)
    prompt = torch.tensor([enc.encode_ordinary("The water cycle is a process where")], device="cuda")

    model.train()
    step = start_step
    scaler = torch.amp.GradScaler("cuda", enabled=hw["use_scaler"])
    t_last = time.perf_counter()
    ema_dt = None
    micro_bs = hw["micro_bs"]

    pbar = tqdm(batches, desc=run, initial=start_step, total=total_steps)
    for x, y in pbar:
        optimizer.zero_grad(set_to_none=True)
        loss_val = 0.0
        step_ok = True

        for m_idx in range(0, x.shape[0], micro_bs):
            xm = x[m_idx:m_idx + micro_bs].cuda(non_blocking=True)
            ym = y[m_idx:m_idx + micro_bs].cuda(non_blocking=True)
            w = xm.shape[0] / x.shape[0]

            with torch.autocast("cuda", hw["amp_dtype"]):
                hidden = _forward_model(model, xm, hw["grad_ckpt"])
                total_loss = 0.0
                for i in range(0, hidden.shape[0], cfg["ce_chunk"]):
                    hc = hidden[i:i + cfg["ce_chunk"]]
                    yc = ym[i:i + hc.shape[0]]

                    def _ce(h, tgt):
                        logits = model.lm_head(h).float()
                        return F.cross_entropy(logits.view(-1, logits.size(-1)),
                                               tgt.reshape(-1), reduction="sum")

                    total_loss = total_loss + torch.utils.checkpoint.checkpoint(
                        _ce, hc, yc, use_reentrant=False)
                mb_loss = (total_loss / (hidden.shape[0] * hidden.shape[1])) * w

            if not torch.isfinite(mb_loss):
                print(f"\nstep {step}: non-finite loss — skipping batch", flush=True)
                step_ok = False
                break

            scaler.scale(mb_loss).backward()
            loss_val += mb_loss.item()

        if not step_ok:
            optimizer.zero_grad(set_to_none=True)
            step += 1
            set_step_lr(step)
            continue

        scaler.unscale_(optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        if not torch.isfinite(total_norm):
            print(f"\nstep {step}: non-finite grad norm — skipping update", flush=True)
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
        else:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        step += 1
        set_step_lr(step)

        now = time.perf_counter()
        dt = now - t_last
        t_last = now
        ema_dt = dt if ema_dt is None else 0.95 * ema_dt + 0.05 * dt
        pbar.set_postfix(loss=f"{loss_val:.3f}", norm=f"{total_norm.item():.2f}",
                         spd=f"{ema_dt:.2f}s/it")

        if step % cfg["log_every"] == 0:
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([step, f"{loss_val:.4f}", f"{total_norm.item():.3f}"])

        if step % cfg["print_every"] == 0:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"\n--- Step {step} ---  loss {loss_val:.3f}  "
                  f"norm {total_norm.item():.2f}  lr {cur_lr:.2e}")
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
