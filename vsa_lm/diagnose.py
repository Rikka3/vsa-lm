"""Decode-parity and drift-by-position diagnostics (`python -m vsa_lm diagnose`).

1. Decode parity: the token-by-token recurrent decoder must give the same next-token predictions as the
   parallel training forward at every position -- past the 64-token window wrap and past the training T.
2. Drift by position: KL(teacher || student) bucketed by generated-token position, measured on
     - the teacher's own reasoning (teacher-forced: the student reads clean text), and
     - the student's own reasoning (free-run: the student reads what it wrote).
   KL rising with position in BOTH -> long-range memory limit. Flat when teacher-forced but rising in
   free-run -> exposure bias (errors compound on self-generated text), which on-policy KD targets.
"""
from __future__ import annotations

import json
import statistics

import torch
import torch.nn.functional as F

from .benchmark import _load_hf_donor, _load_rows, _load_student, gsm8k_prompt
from .decode import VSARecurrentDecoder
from .server import get_all_eos_ids

PARITY_EDGES = (0, 64, 256, 1024, 2048, 3072)
DRIFT_EDGES = (0, 64, 128, 256, 512, 1024, 2048, 4096)


def vsa_logits_fn(model):
    """Logits for a UniversalVSALM (the student, or the native softmax replica of the teacher)."""
    return lambda x: F.linear(model(x), model.lm_head.weight)


def hf_logits_fn(model):
    return lambda x: model(input_ids=x, use_cache=False).logits


def pad_to_multiple(x: torch.Tensor, m: int, value: int = 0) -> torch.Tensor:
    """Right-pad the last dim to a multiple of m (the VSA chunk kernel needs T % 32 == 0)."""
    pad = (-x.shape[-1]) % m
    return F.pad(x, (0, pad), value=value) if pad else x


def _bucketize(per_seq: list[torch.Tensor], edges=DRIFT_EDGES) -> list[dict]:
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        vals = [v[lo:hi] for v in per_seq if v.numel() > lo]
        n = sum(int(v.numel()) for v in vals)
        if n:
            rows.append({"pos": f"{lo}-{hi}", "mean": round(float(sum(v.sum() for v in vals)) / n, 4), "n": n})
    return rows


@torch.no_grad()
def score_continuation(s_fn, t_fn, seq: torch.Tensor, prompt_len: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """seq: (1, P+n) token ids. Returns per-generated-token KL(teacher||student) and teacher NLL, both (n,)."""
    L = seq.shape[1]
    seq_p = pad_to_multiple(seq, 32)  # VSA chunk kernel needs T % 32 == 0; right-padding is causal-safe
    with torch.autocast("cuda", torch.bfloat16, enabled=(device.type == "cuda")):
        ls = s_fn(seq_p)[0, prompt_len - 1 : L - 1]
        lt = t_fn(seq_p)[0, prompt_len - 1 : L - 1]
    tgt = seq[0, prompt_len:]
    kl, nll = [], []
    for lo in range(0, tgt.numel(), 512):
        a = F.log_softmax(lt[lo : lo + 512].float(), dim=-1)
        b = F.log_softmax(ls[lo : lo + 512].float(), dim=-1)
        kl.append((a.exp() * (a - b)).sum(-1))
        nll.append(-a.gather(-1, tgt[lo : lo + 512, None])[:, 0])
    del ls, lt
    return torch.cat(kl).cpu(), torch.cat(nll).cpu()


@torch.no_grad()
def student_drift(stu, s_fn, t_fn, tok, device, prompts: list[str], max_new: int = 2048,
                  temperature: float = 0.6, top_p: float = 0.95, seed: int = 1000) -> dict:
    """Student generates each prompt (benchmark sampling); KL/teacher-NLL by generated position."""
    stu.eval()
    dec = VSARecurrentDecoder(stu, wrap_positions=False)
    eos_ids = get_all_eos_ids(tok)
    kls, nlls, lens = [], [], []
    for i, p in enumerate(prompts):
        torch.manual_seed(seed + i)
        x = torch.tensor([tok.encode(p, add_special_tokens=False)], dtype=torch.long, device=device)
        toks, n = dec.generate_batch(x, max_new, temperature=temperature, top_p=top_p, eos_ids=eos_ids)
        n = int(n[0])
        lens.append(n)
        kl, nll = score_continuation(s_fn, t_fn, torch.cat([x, toks[:, :n]], dim=1), x.shape[1], device)
        kls.append(kl)
        nlls.append(nll)
    return {"kl": _bucketize(kls), "teacher_nll": _bucketize(nlls),
            "median_len": int(statistics.median(lens)) if lens else 0,
            "capped": f"{sum(int(n >= max_new) for n in lens)}/{len(lens)}",
            "capped_n": sum(int(n >= max_new) for n in lens), "n": len(lens)}


@torch.no_grad()
def teacher_forced_drift(teacher, s_fn, t_fn, tok, device, prompts: list[str], max_new: int = 2048,
                         temperature: float = 0.6, top_p: float = 0.95, seed: int = 1000) -> dict:
    """HF teacher generates; the student reads the teacher's reasoning. KL by position = long-range fidelity."""
    eos_ids = list(get_all_eos_ids(tok))
    kls, nlls, lens = [], [], []
    for i, p in enumerate(prompts):
        torch.manual_seed(seed + i)
        x = torch.tensor([tok.encode(p, add_special_tokens=False)], dtype=torch.long, device=device)
        out = teacher.generate(x, max_new_tokens=max_new, do_sample=True, temperature=temperature, top_p=top_p,
                               eos_token_id=eos_ids, pad_token_id=tok.eos_token_id or 0)
        lens.append(out.shape[1] - x.shape[1])
        kl, nll = score_continuation(s_fn, t_fn, out, x.shape[1], device)
        kls.append(kl)
        nlls.append(nll)
    return {"kl": _bucketize(kls), "teacher_nll": _bucketize(nlls),
            "median_len": int(statistics.median(lens)) if lens else 0}


def _parity_tokens(tok, device, T: int) -> torch.Tensor:
    """One real token sequence (GSM8K prompt + worked solutions), length rounded down to a multiple of 32."""
    rows = _load_rows("openai/gsm8k", "main", "train", 400)
    ids = tok.encode(gsm8k_prompt(tok, rows[0]["question"]), add_special_tokens=False)
    for r in rows:
        if len(ids) >= T:
            break
        ids += tok.encode(f"{r['question'].strip()}\n{r['answer'].strip()}\n\n", add_special_tokens=False)
    T = (min(T, len(ids)) // 32) * 32
    return torch.tensor([ids[:T]], dtype=torch.long, device=device)


@torch.no_grad()
def layer_parity(stu, tok, device, T: int = 1024) -> list[dict]:
    """Per-block parity with no error compounding: every block gets exactly the inputs the parallel forward
    gave it, is stepped token by token through the decoder path, and its contribution (out - in) is compared.
    bf16 noise shows as a small, similar error in every layer; a decode bug shows as one layer type far above it."""
    x = _parity_tokens(tok, device, T)
    T = x.shape[1]
    ins, outs = {}, {}

    def _hook(li):
        def fn(_m, args, out):
            ins[li], outs[li] = args[0].detach(), out.detach()
        return fn

    hooks = [blk.register_forward_hook(_hook(li)) for li, blk in enumerate(stu.blocks)]
    stu.eval()
    try:
        with torch.autocast("cuda", torch.bfloat16, enabled=(device.type == "cuda")):
            stu(x)
    finally:
        for h in hooks:
            h.remove()

    dec = VSARecurrentDecoder(stu, wrap_positions=False)
    states, kv = VSARecurrentDecoder.blank_state(stu, 1, device, max_kv_len=T + 64)
    edges = [e for e in (0, 64, 256, 1024, 2048, 3072) if e < T] + [T]
    rows = []
    for li, blk in enumerate(stu.blocks):
        rec = torch.empty_like(outs[li])
        with torch.autocast("cuda", torch.bfloat16, enabled=(device.type == "cuda")):
            for t in range(T):
                rec[:, t : t + 1] = dec._step_universal_block(li, blk, ins[li][:, t : t + 1], t, states, kv)
        ref_d = (outs[li] - ins[li]).float()
        rec_d = (rec - ins[li]).float()
        err = ((rec_d - ref_d).norm(dim=-1) / ref_d.norm(dim=-1).clamp_min(1e-6))[0]
        row = {"layer": li, "type": "vsa" if hasattr(blk, "vsa") else "softmax"}
        for lo, hi in zip(edges[:-1], edges[1:]):
            row[f"{lo}-{hi}"] = round(float(err[lo:hi].mean()), 4)
        rows.append(row)
        del rec, ref_d, rec_d
    return rows


@torch.no_grad()
def decode_parity(stu, tok, device, T: int = 3072) -> list[dict]:
    """Parallel forward vs recurrent step on one real token sequence of length T."""
    x = _parity_tokens(tok, device, T)
    T = x.shape[1]

    stu.eval()
    with torch.autocast("cuda", torch.bfloat16, enabled=(device.type == "cuda")):
        par = F.linear(stu(x), stu.lm_head.weight)[0]
    dec = VSARecurrentDecoder(stu, wrap_positions=False)
    states, kv = VSARecurrentDecoder.blank_state(stu, 1, device, max_kv_len=T + 64)
    max_diff, agree = torch.zeros(T), torch.zeros(T)
    for t in range(T):
        h, states, kv = dec.step(x[:, t : t + 1], t, states, kv)
        rec = stu.lm_head(h[:, -1])[0].float()
        ref = par[t].float()
        max_diff[t] = (rec - ref).abs().max().item()
        agree[t] = float(rec.argmax() == ref.argmax())
    # Same check for the CUDA-graph step that generate()/generate_batch() actually use.
    g_agree = None
    runner = dec._graph_runner(1, T, device)
    if runner is not None:
        g_agree = torch.zeros(T)
        for t in range(T):
            g_agree[t] = float(runner(x[:, t : t + 1], t)[0].float().argmax() == par[t].float().argmax())
    out = []
    for lo, hi in zip(PARITY_EDGES[:-1], PARITY_EDGES[1:]):
        if lo >= T:
            break
        hi = min(hi, T)
        row = {"pos": f"{lo}-{hi}", "top1_agree_pct": round(100.0 * float(agree[lo:hi].mean()), 2),
               "max_abs_logit_diff": round(float(max_diff[lo:hi].max()), 3)}
        if g_agree is not None:
            row["graph_top1_agree_pct"] = round(100.0 * float(g_agree[lo:hi].mean()), 2)
        out.append(row)
    return out


def _print_rows(title: str, rows: list[dict], key: str = "mean"):
    print(f"  {title}")
    for r in rows:
        print(f"    pos {r['pos']:>10}  {key}={r[key]}" + (f"  (n={r['n']})" if "n" in r else ""))


def run_diagnostics(ckpt_path: str, donor_id: str | None = None, n_prompts: int = 16, max_new: int = 2048,
                    parity_T: int = 3072, out_json: str | None = None, parity_only: bool = False) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stu, cfg, tok, donor_id, donor_dir, step, _ = _load_student(ckpt_path, donor_id, device)
    res = {"checkpoint": ckpt_path, "step": step}

    print("=" * 84)
    print(f"VSA DIAGNOSTICS: {ckpt_path} (step {step})")
    print("=" * 84)
    res["decode_parity"] = decode_parity(stu, tok, device, T=parity_T)
    print("  [1] Decode parity (recurrent step vs parallel forward; pass = >=99% top-1 in every bucket)")
    for r in res["decode_parity"]:
        flag = "ok" if r["top1_agree_pct"] >= 99.0 else "MISMATCH"
        graph = f"  graph top1 {r['graph_top1_agree_pct']:6.2f}%" if "graph_top1_agree_pct" in r else "  graph n/a"
        print(f"    pos {r['pos']:>10}  top1 {r['top1_agree_pct']:6.2f}%  max|dlogit| {r['max_abs_logit_diff']:.3f}{graph}  {flag}")

    res["layer_parity"] = layer_parity(stu, tok, device, T=min(parity_T, 2048))
    print("  [1b] Per-layer parity: relative error of each block's output, same inputs, no compounding")
    for r in res["layer_parity"]:
        cols = "  ".join(f"{k}:{v:.4f}" for k, v in r.items() if k not in ("layer", "type"))
        print(f"    L{r['layer']:02d} {r['type']:>7}  {cols}")

    if parity_only:
        if out_json:
            with open(out_json, "w") as f:
                json.dump(res, f, indent=2)
        return res

    teacher = _load_hf_donor(donor_dir, device)
    s_fn, t_fn = vsa_logits_fn(stu), hf_logits_fn(teacher)
    prompts = [gsm8k_prompt(tok, r["question"]) for r in _load_rows("openai/gsm8k", "main", "test", n_prompts)]

    res["teacher_forced"] = teacher_forced_drift(teacher, s_fn, t_fn, tok, device, prompts, max_new=max_new)
    res["free_run"] = student_drift(stu, s_fn, t_fn, tok, device, prompts, max_new=max_new)

    print(f"  [2] Teacher-forced: student reads the teacher's reasoning (median len {res['teacher_forced']['median_len']})")
    _print_rows("KL(teacher||student) by position", res["teacher_forced"]["kl"])
    print(f"  [3] Free-run: student reads its own reasoning (median len {res['free_run']['median_len']}, capped {res['free_run']['capped']})")
    _print_rows("KL(teacher||student) by position", res["free_run"]["kl"])
    _print_rows("teacher NLL of the student's tokens by position", res["free_run"]["teacher_nll"])

    if out_json:
        with open(out_json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"  Saved {out_json}")
    return res
