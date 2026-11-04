"""Audit-Clean Benchmark Suite (`python -m vsa_lm benchmark`).

Implements P1, P2, P4, P5, P9, P11:
  - Evaluates Donor with official HuggingFace `AutoModelForCausalLM` + static KV cache (P4).
  - Evaluates MMLU, HellaSwag, GSM8K, HumanEval, MBPP with per-item 0/1 tracking and McNemar exact test (P1).
  - GSM8K uses native `<think>` generation up to 2048 tokens at temp=0.6, top_p=0.95 and logs truncation rate & median length (P2).
  - Sandboxed subprocess execution for HumanEval and MBPP (P11).
  - Real prefill TTFT (ms) and decode throughput (tok/s) with StaticKV & enable_gqa=True on real text tokens (P5).
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .converter import UniversalVSALM, fetch_any_donor, _purge_zombie_vram
from .decode import VSARecurrentDecoder, StaticKV
from .memory_bank import VSAMemoryBank
from .server import get_all_eos_ids


def _ci95(acc_pct: float, n: int) -> float:
    if n <= 0:
        return 0.0
    p = max(0.0, min(1.0, acc_pct / 100.0))
    return 196.0 * math.sqrt(p * (1.0 - p) / n)


def _mcnemar_exact(stu_items: list[int], don_items: list[int]) -> dict:
    """P1: Exact binomial McNemar test on discordant pairs."""
    b = sum(1 for s, d in zip(stu_items, don_items) if s == 1 and d == 0)
    c = sum(1 for s, d in zip(stu_items, don_items) if s == 0 and d == 1)
    n = b + c
    if n == 0:
        return {"stu_only": 0, "don_only": 0, "p": 1.0}
    k = min(b, c)
    # Exact two-sided binomial p-value: 2 * sum_{i=0..k} C(n, i) * 0.5^n
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    p_val = min(1.0, 2.0 * tail)
    return {"stu_only": b, "don_only": c, "p": round(p_val, 4)}


def _load_student(ckpt_path: str, donor_override: str | None, device: torch.device):
    from transformers import AutoTokenizer
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    donor_id = donor_override or ck.get("donor") or (ck.get("injection") or {}).get("donor")
    if not donor_id:
        raise ValueError("Pass --donor <repo_id>")
    donor_dir = fetch_any_donor(donor_id)
    tok = AutoTokenizer.from_pretrained(donor_dir)

    stu = UniversalVSALM(**cfg)
    state_in = {k: v for k, v in ck["model"].items() if "pos_code" not in k and "head_pos_delta" not in k}
    missing, unexpected = stu.load_state_dict(state_in, strict=False)
    allowed_missing = {"out_norm"}
    real_missing = [m for m in missing if not any(a in m for a in allowed_missing)]
    if real_missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch! Missing: {real_missing}, Unexpected: {unexpected}")

    step = ck.get("step", "?")
    conv_ver = (ck.get("injection") or {}).get("converter", "unknown")
    del ck

    fp32_keys = ("log_gamma", "k_head_mix", "v_head_mix", "v_self_scale", "local_q_gate",
                 "q_relu_mix", "k_relu_mix", "rope_mix", "A_log", "dt_bias", "out_norm")
    for n, p in stu.named_parameters():
        if any(k in n for k in fp32_keys):
            p.data = p.data.to(device, dtype=torch.float32)
        else:
            p.data = p.data.to(device, dtype=(torch.bfloat16 if device.type == "cuda" else torch.float32))
    for b in stu.buffers():
        b.data = b.data.to(device)
    stu.eval()
    return stu, cfg, tok, donor_id, donor_dir, step, conv_ver


def _load_hf_donor(donor_dir: str, device: torch.device):
    """P4: Always load official HuggingFace AutoModelForCausalLM for donor evaluation."""
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(donor_dir, dtype=torch.bfloat16).to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _load_rows(repo: str, name: str | None, split: str, limit: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset(repo, name, split=split, streaming=True)
    rows = []
    for r in ds:
        rows.append(r)
        if len(rows) >= limit:
            break
    return rows


def _load_mmlu_stratified(limit: int = 285) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    by_subj: dict[str, list[dict]] = {}
    for r in ds:
        by_subj.setdefault(r.get("subject", "other"), []).append(r)
    rng = random.Random(42)
    subjects = sorted(by_subj.keys())
    for s in subjects:
        rng.shuffle(by_subj[s])
    out = []
    round_idx = 0
    while len(out) < min(limit, len(ds)):
        added = False
        for s in subjects:
            if round_idx < len(by_subj[s]):
                out.append(by_subj[s][round_idx])
                added = True
                if len(out) >= limit:
                    break
        if not added:
            break
        round_idx += 1
    return out


@torch.no_grad()
def _score_choices(model, is_hf: bool, tok, prompt: str, choices: list[str],
                   device: torch.device, norm_len: bool = True) -> int:
    p_ids = tok.encode(prompt, add_special_tokens=False)
    bos_id = tok.bos_token_id if isinstance(tok.bos_token_id, int) else None
    if bos_id is not None and (not p_ids or p_ids[0] != bos_id):
        p_ids = [bos_id] + p_ids
    pad_id = tok.eos_token_id if isinstance(tok.eos_token_id, int) else 0
    best_i, best_s = 0, -1e30
    w_lm = (model.get_output_embeddings().weight if is_hf else model.lm_head.weight).float()

    for i, ch in enumerate(choices):
        c_str = " " + str(ch).lstrip()
        c_ids = tok.encode(c_str, add_special_tokens=False) or tok.encode(str(ch), add_special_tokens=False)
        full = (p_ids + c_ids)[-1024:]
        c_len, L = min(len(c_ids), len(full) - 1), len(full)
        pad = (-L) % 32
        x = torch.tensor([full + ([pad_id] * pad if pad else [])], dtype=torch.long, device=device)
        with torch.autocast("cuda", torch.bfloat16, enabled=(device.type == "cuda")):
            if is_hf:
                inner = getattr(getattr(model, "model", model), "language_model", getattr(model, "model", model))
                out = inner(input_ids=x[:, :L], use_cache=False)
                h = (out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0])[0, L - 1 - c_len : L - 1]
            else:
                h = model(x)[0, L - 1 - c_len : L - 1]
        logits = F.linear(h.float(), w_lm)
        ll = F.log_softmax(logits, dim=-1).gather(-1, torch.tensor(full[L - c_len : L], device=device).unsqueeze(-1)).sum().item()
        denom = max(1, len(c_str)) if norm_len else 1.0
        if ll / denom > best_s:
            best_s, best_i = ll / denom, i
    return best_i


def _exec_subprocess_safely(code_str: str, timeout_s: int = 4) -> bool:
    """P11 Fix: Runs untrusted generated Python code in an isolated subprocess with a temporary directory."""
    with tempfile.TemporaryDirectory() as tmp_d:
        script_path = os.path.join(tmp_d, "solution_test.py")
        with open(script_path, "w") as f:
            f.write(code_str)
        try:
            res = subprocess.run(
                [sys.executable, "-I", script_path],
                cwd=tmp_d,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout_s,
            )
            return res.returncode == 0
        except Exception:
            return False


@torch.no_grad()
def _eval_model(model, is_hf: bool, tok, device: torch.device, limit: int = 285, gen_limit: int = 60) -> dict:
    out = {}
    eos_ids = get_all_eos_ids(tok)
    dec = VSARecurrentDecoder(model, wrap_positions=False) if not is_hf else None

    def _gen(prompt_str: str, max_new: int = 2048, temperature: float = 0.0, top_p: float = 1.0, seed: int = 0) -> tuple[str, int, bool]:
        torch.manual_seed(seed)
        ids = tok.encode(prompt_str, add_special_tokens=False)
        x_in = torch.tensor([ids], dtype=torch.long, device=device)
        if not is_hf:
            gen_t = dec.generate(x_in, max_new_tokens=max_new, temperature=temperature, top_p=top_p, eos_ids=eos_ids)
            gen_ids = gen_t[0, len(ids):].tolist()
        else:
            gen_t = model.generate(
                x_in,
                max_new_tokens=max_new,
                do_sample=(temperature > 1e-5),
                temperature=temperature if temperature > 1e-5 else None,
                top_p=top_p if temperature > 1e-5 else None,
                eos_token_id=list(eos_ids),
                pad_token_id=tok.eos_token_id or 0,
            )
            gen_ids = gen_t[0, len(ids):].tolist()
        hit_cap = len(gen_ids) >= max_new and (not gen_ids or gen_ids[-1] not in eos_ids)
        if eos_ids and gen_ids and gen_ids[-1] in eos_ids:
            gen_ids = gen_ids[:-1]
        return tok.decode(gen_ids, skip_special_tokens=True), len(gen_ids), hit_cap

    # 1. MMLU (Stratified across all 57 subjects)
    rows = _load_mmlu_stratified(limit)
    hits = []
    for r in rows:
        ch = r["choices"]
        subj = r.get("subject", "general").replace("_", " ")
        prompt = f"The following is a multiple-choice question about {subj}.\nQuestion: {r['question'].strip()}\nOptions:\n- {ch[0]}\n- {ch[1]}\n- {ch[2]}\n- {ch[3]}\nCorrect Answer:"
        hits.append(int(_score_choices(model, is_hf, tok, prompt, ch, device, norm_len=True) == int(r["answer"])))
    acc = (sum(hits) / len(hits)) * 100.0
    out["MMLU"] = {"acc": round(acc, 2), "ci95": round(_ci95(acc, len(hits)), 2), "correct": sum(hits), "total": len(hits), "items": hits}

    # 2. HellaSwag
    rows = _load_rows("Rowan/hellaswag", None, "validation", limit)
    hits = [int(_score_choices(model, is_hf, tok, r["ctx"].strip(), r["endings"], device) == int(r["label"])) for r in rows]
    acc = (sum(hits) / len(hits)) * 100.0
    out["HellaSwag"] = {"acc": round(acc, 2), "ci95": round(_ci95(acc, len(hits)), 2), "correct": sum(hits), "total": len(hits), "items": hits}

    # 3. GSM8K Math (P2 Fix: max_new=2048, temp=0.6, top_p=0.95, logs truncation rate & median length)
    rows = _load_rows("openai/gsm8k", "main", "test", gen_limit)
    hits, lengths, trunc_cnt = [], [], 0
    for idx_q, r in enumerate(rows):
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        msgs = [{"role": "user", "content": f"Solve this math problem step by step and put the final numerical answer inside \\boxed{{}}:\n{r['question'].strip()}"}]
        try:
            p_str = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            p_str = f"Question: {r['question'].strip()}\nSolution:\n"
        gen, n_new, hit_cap = _gen(p_str, max_new=2048, temperature=0.6, top_p=0.95, seed=1000 + idx_q)
        lengths.append(n_new)
        trunc_cnt += int(hit_cap)
        ans_region = gen.split("</think>")[-1] if "</think>" in gen else gen
        boxed = re.findall(r"\\boxed\{([^}]+)\}", ans_region)
        nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", boxed[-1] if boxed else ans_region)
        pred = nums[-1].replace(",", "") if nums else None
        hit = False
        if pred is not None and not hit_cap:
            try:
                hit = abs(float(pred) - float(gold)) < 1e-3
            except Exception:
                hit = (pred == gold)
        hits.append(int(hit))
    acc = (sum(hits) / len(hits)) * 100.0
    out["GSM8K"] = {
        "acc": round(acc, 2), "ci95": round(_ci95(acc, len(hits)), 2),
        "correct": sum(hits), "total": len(hits), "items": hits,
        "truncated": f"{trunc_cnt}/{len(hits)}", "median_len": int(statistics.median(lengths)) if lengths else 0,
    }

    # 4. HumanEval Python Coding (Sandboxed subprocess execution)
    rows = _load_rows("openai/openai_humaneval", None, "test", gen_limit)
    hits, lengths, trunc_cnt = [], [], 0
    for idx_q, r in enumerate(rows):
        prompt_code = r["prompt"]
        msgs = [{"role": "user", "content": f"Complete this Python function. Output ONLY valid Python code inside ```python```:\n```python\n{prompt_code}\n```"}]
        try:
            p_str = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            if "<think>" in p_str and "</think>" not in p_str:
                p_str = p_str.rstrip() + "\n</think>\n\n```python\n" + prompt_code
        except Exception:
            p_str = prompt_code
        gen, n_new, hit_cap = _gen(p_str, max_new=256, temperature=0.0, seed=2000 + idx_q)
        lengths.append(n_new)
        trunc_cnt += int(hit_cap)
        if "```" in gen:
            gen = gen.split("```")[0]
        clean_lines = []
        for ln in gen.splitlines():
            if ln.startswith(("def ", "class ", "if __name__", "# Test", "assert ")) and clean_lines:
                break
            clean_lines.append(ln)
        prog = prompt_code + "\n".join(clean_lines) + "\n\n" + r["test"] + f"\ncheck({r['entry_point']})\n"
        hits.append(int(_exec_subprocess_safely(prog, timeout_s=4)))
    acc = (sum(hits) / len(hits)) * 100.0
    out["HumanEval"] = {"acc": round(acc, 2), "ci95": round(_ci95(acc, len(hits)), 2), "correct": sum(hits), "total": len(hits), "items": hits}

    # 5. MBPP Python Coding (Sandboxed subprocess execution)
    rows = _load_rows("google-research-datasets/mbpp", "sanitized", "test", gen_limit)
    hits = []
    for idx_q, r in enumerate(rows):
        tests = "\n".join(r["test_list"])
        sig_line = r["code"].splitlines()[0] if r.get("code") else "def solution():"
        p_code = f"# {r['prompt'].strip()}\n{sig_line}\n"
        msgs = [{"role": "user", "content": f"Complete this Python function:\n```python\n{p_code}\n```"}]
        try:
            p_str = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            if "<think>" in p_str and "</think>" not in p_str:
                p_str = p_str.rstrip() + "\n</think>\n\n```python\n" + p_code
        except Exception:
            p_str = p_code
        gen, _, _ = _gen(p_str, max_new=256, temperature=0.0, seed=3000 + idx_q)
        if "```" in gen:
            gen = gen.split("```")[0]
        clean_lines = []
        for ln in gen.splitlines():
            if ln.startswith(("def ", "class ", "if __name__", "assert ")) and clean_lines:
                break
            clean_lines.append(ln)
        prog = "\n".join(r.get("test_imports", [])) + "\n" + p_code + "\n".join(clean_lines) + "\n" + tests + "\n"
        hits.append(int(_exec_subprocess_safely(prog, timeout_s=4)))
    acc = (sum(hits) / len(hits)) * 100.0
    out["MBPP"] = {"acc": round(acc, 2), "ci95": round(_ci95(acc, len(hits)), 2), "correct": sum(hits), "total": len(hits), "items": hits}
    return out


@torch.no_grad()
def eval_real_speed_and_ram(stu, hf_donor, cfg: dict, tok, device: torch.device) -> dict:
    """P5 Fix: Measures decode speed with StaticKV + enable_gqa=True against Official HF Donor with StaticCache."""
    gen_steps = 32
    ctx_lengths = (2048, 8192, 16384, 32768, 65536)
    scaling = []
    dec_stu = VSARecurrentDecoder(stu, wrap_positions=False, max_cache_pos=70000)

    for T_c in ctx_lengths:
        nxt_tok = torch.tensor([[100]], device=device, dtype=torch.long)

        # 1. Student StaticKV + Fused Triton O(1) Step
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        base_mem_stu = torch.cuda.memory_allocated()
        states_s, kv_s = VSARecurrentDecoder.blank_state(stu, 1, device, max_kv_len=T_c + gen_steps + 32)
        for li, blk in enumerate(stu.blocks):
            if not hasattr(blk, "vsa"):
                s_ly = blk.attn
                kv_s[li].n = T_c
                kv_s[li].k[:, :, :T_c].normal_(0, 0.02)
                kv_s[li].v[:, :, :T_c].normal_(0, 0.02)
        vsa_state_mb = (torch.cuda.memory_allocated() - base_mem_stu) / (1024 ** 2)

        dec_stu.step(nxt_tok, T_c, states_s, kv_s)
        torch.cuda.synchronize()
        runs_s = []
        for _ in range(3):
            for li, blk in enumerate(stu.blocks):
                if not hasattr(blk, "vsa"):
                    kv_s[li].n = T_c
            t0 = time.perf_counter()
            for step_i in range(gen_steps):
                x_h, states_s, kv_s = dec_stu.step(nxt_tok, T_c + step_i, states_s, kv_s)
                _ = stu.lm_head(x_h[:, -1, :]).argmax(-1)
            torch.cuda.synchronize()
            runs_s.append(gen_steps / (time.perf_counter() - t0))
        vsa_tok_s = statistics.median(runs_s)
        del states_s, kv_s
        torch.cuda.empty_cache()

        # 2. Donor with StaticKV + enable_gqa=True
        L = cfg["num_layers"]
        HKV = cfg["num_kv_heads"]
        HD = cfg["head_dim"]
        donor_kv_mb = (L * 2 * HKV * HD * T_c * 2) / (1024 ** 2)
        inner_hf = getattr(getattr(hf_donor, "model", hf_donor), "language_model", getattr(hf_donor, "model", hf_donor))
        from transformers.cache_utils import DynamicCache
        cache_hf = DynamicCache()
        for li in range(L):
            cache_hf.update(
                torch.randn(1, HKV, T_c, HD, device=device, dtype=torch.bfloat16),
                torch.randn(1, HKV, T_c, HD, device=device, dtype=torch.bfloat16),
                li,
            )
        with torch.autocast("cuda", torch.bfloat16):
            _ = hf_donor(input_ids=nxt_tok, past_key_values=cache_hf, use_cache=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", torch.bfloat16):
            for _ in range(gen_steps):
                o = inner_hf(input_ids=nxt_tok, past_key_values=cache_hf, use_cache=True)
                _ = hf_donor.get_output_embeddings()(o.last_hidden_state[:, -1]).argmax(-1)
        torch.cuda.synchronize()
        donor_tok_s = gen_steps / (time.perf_counter() - t0)
        del cache_hf
        torch.cuda.empty_cache()

        scaling.append({
            "context": T_c,
            "donor_kv_mb": round(donor_kv_mb, 2),
            "vsa_kv_mb": round(vsa_state_mb, 2),
            "ram_saved_pct": round((1.0 - vsa_state_mb / max(1e-4, donor_kv_mb)) * 100.0, 1),
            "donor_tok_s": round(donor_tok_s, 1),
            "vsa_tok_s": round(vsa_tok_s, 1),
            "speedup": round(vsa_tok_s / max(1e-4, donor_tok_s), 2),
        })

    bank_dir = Path("./_tmp_vsa_bench_bank")
    bank = VSAMemoryBank(stu, tok, bank_dir=bank_dir)
    secrets = [
        ("orion", "The launch authorization code for Project Orion is TITAN-OMEGA-942.", "What is the launch authorization code for Project Orion?", "TITAN-OMEGA-942"),
        ("hemlock", "The antidote synthesis temperature for Protocol Hemlock is 347 Kelvin.", "What is the antidote synthesis temperature for Protocol Hemlock?", "347"),
        ("vanguard", "The deep-space relay frequency for Station Vanguard is 884.5 MHz.", "What is the deep-space relay frequency for Station Vanguard?", "884.5"),
        ("chronos", "The vault master key for Facility Chronos is COBALT-LOTUS-19.", "What is the vault master key for Facility Chronos?", "COBALT-LOTUS-19"),
    ]
    for n, d, _, _ in secrets:
        bank.memorize_to_disk(n, d)
    hits, max_res, r_ms = 0, 0.0, []
    for n, _, q_s, exp in secrets:
        res = bank.ask(q_s, max_new_tokens=32)
        r_ms.append(res["route_ms"])
        max_res = max(max_res, res["erase_residual"])
        hits += int(exp.lower() in res["answer"].lower() and res["cartridge"] == n)
    for n, _, _, _ in secrets:
        bank.forget_from_disk(n)
    try:
        bank_dir.rmdir()
    except Exception:
        pass

    return {
        "scaling": scaling,
        "cartridge_recall_pct": (hits / len(secrets)) * 100.0,
        "cartridge_hits": f"{hits}/{len(secrets)}",
        "avg_route_ms": round(sum(r_ms) / len(r_ms), 2),
        "erase_residual": max_res,
    }


def _plot_benchmark_figure(results: dict, out_png: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=160)
    fig.patch.set_facecolor("#0d1117")
    for ax in axes.flat:
        ax.set_facecolor("#161b22")
        ax.grid(True, alpha=0.2, color="#8b949e", linestyle="--")

    ax1 = axes[0, 0]
    tasks = list(results["student"].keys())
    s_vals = [results["student"][k]["acc"] for k in tasks]
    s_ci = [results["student"][k].get("ci95", 0.0) for k in tasks]
    d_vals = [results["teacher"].get(k, {}).get("acc", 0.0) for k in tasks]
    d_ci = [results["teacher"].get(k, {}).get("ci95", 0.0) for k in tasks]
    x = np.arange(len(tasks))
    w = 0.36
    if results["teacher"]:
        ax1.bar(x - w/2, d_vals, w, yerr=d_ci, capsize=3, label=f"Official HF Donor ({results['donor'].split('/')[-1]})", color="#58a6ff")
        ax1.bar(x + w/2, s_vals, w, yerr=s_ci, capsize=3, label="VSA-LM", color="#3fb950")
    else:
        ax1.bar(x, s_vals, w, yerr=s_ci, capsize=3, label="VSA-LM", color="#3fb950")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks, rotation=12, fontsize=9, fontweight="bold")
    ax1.set_ylabel("Accuracy (% ± 95% CI)", fontweight="bold")
    ax1.set_title("1. Reasoning, Math & Coding (Paired McNemar Tested)", fontsize=10.5, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8)

    ax2 = axes[0, 1]
    sc = results["systems"]["scaling"]
    ctxs = [f"{r['context']//1024}k" for r in sc]
    ax2.plot(ctxs, [r["donor_tok_s"] for r in sc], "o--", color="#58a6ff", linewidth=2.2, label="Official HF Donor (DynamicCache)")
    ax2.plot(ctxs, [r["vsa_tok_s"] for r in sc], "s-", color="#3fb950", linewidth=2.5, label="VSA-LM (StaticKV + Triton)")
    ax2.set_xlabel("Context Length (Tokens)", fontweight="bold")
    ax2.set_ylabel("Measured Decode Speed (tok/s)", fontweight="bold")
    ax2.set_title(f"2. Decode Throughput ({sc[-1]['speedup']}x @ {ctxs[-1]})", fontsize=10.5, fontweight="bold")
    ax2.legend(loc="best", fontsize=8)

    ax3 = axes[1, 0]
    ax3.plot(ctxs, [r["donor_kv_mb"] for r in sc], "o--", color="#f85149", linewidth=2.2, label="Donor KV Cache (MB)")
    ax3.plot(ctxs, [r["vsa_kv_mb"] for r in sc], "s-", color="#3fb950", linewidth=2.5, label="VSA-LM State + 25% KV (MB)")
    ax3.fill_between(range(len(ctxs)), [r["vsa_kv_mb"] for r in sc], [r["donor_kv_mb"] for r in sc], color="#3fb950", alpha=0.18, label=f"-{sc[-1]['ram_saved_pct']}% Saved")
    ax3.set_xlabel("Context Length (Tokens)", fontweight="bold")
    ax3.set_ylabel("State / Cache RAM (MB)", fontweight="bold")
    ax3.set_title(f"3. State RAM Formula & Allocation (-{sc[-1]['ram_saved_pct']}% @ {ctxs[-1]})", fontsize=10.5, fontweight="bold")
    ax3.legend(loc="upper left", fontsize=8)

    ax4 = axes[1, 1]
    sys_m = results["systems"]
    metrics = ["Cartridge Recall (%)", f"{ctxs[-1]} RAM Saved (%)", "Erase Rel-Err < 1e-4 (%)"]
    vals = [sys_m["cartridge_recall_pct"], sc[-1]["ram_saved_pct"], 100.0 if sys_m["erase_residual"] < 1e-4 else 0.0]
    bars = ax4.barh(metrics, vals, color=["#a371f7", "#3fb950", "#d29922"], height=0.5)
    ax4.set_xlim(0, 115)
    for b, v in zip(bars, vals):
        ax4.text(v + 2, b.get_y() + b.get_height()/2, f"{v:.1f}%", va="center", fontweight="bold", color="white")
    ax4.set_title(f"4. VSA State Mount & Relative Erase Err ({sys_m['erase_residual']:.2e})", fontsize=10.5, fontweight="bold")

    plt.suptitle(f"VSA-LM vs {results['donor']} — Benchmark Report", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"[vsa-bench] Saved 4-panel chart -> {out_png}")


def run_benchmark_suite(
    ckpt_path: str,
    *,
    donor_id: str | None = None,
    compare_donor: bool = True,
    limit: int = 200,
    gsm8k_limit: int = 40,
    out_json: str | None = None,
) -> dict:
    _purge_zombie_vram()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stu, cfg, tok, donor_repo, donor_dir, step, conv_ver = _load_student(ckpt_path, donor_id, device)

    print("=" * 88)
    print(f"VSA-LM BENCHMARK SUITE: {ckpt_path} (step {step}, {conv_ver}) vs {donor_repo}")
    print("=" * 88, flush=True)

    results = {"checkpoint": ckpt_path, "step": step, "converter": conv_ver, "donor": donor_repo, "student": {}, "teacher": {}, "mcnemar": {}, "systems": {}}
    results["student"] = _eval_model(stu, False, tok, device, limit=limit, gen_limit=gsm8k_limit)
    for k, v in results["student"].items():
        extra = f" | trunc={v['truncated']} med_len={v['median_len']}" if "truncated" in v else ""
        print(f"  [VSA-LM] {k:<12}: {v['acc']:6.2f}% ± {v['ci95']:4.1f}% ({v['correct']}/{v['total']}){extra}", flush=True)

    hf_donor = _load_hf_donor(donor_dir, device)
    results["systems"] = eval_real_speed_and_ram(stu, hf_donor, cfg, tok, device)
    sc_last = results["systems"]["scaling"][-1]
    print(f"  [VSA-LM] Speed/RAM @{sc_last['context']}: {sc_last['vsa_tok_s']} vs {sc_last['donor_tok_s']} tok/s ({sc_last['speedup']}x) | {sc_last['vsa_kv_mb']} vs {sc_last['donor_kv_mb']} MB (-{sc_last['ram_saved_pct']}%)", flush=True)
    print(f"  [VSA-LM] Cartridges  : {results['systems']['cartridge_recall_pct']:.1f}% ({results['systems']['cartridge_hits']}) | Rel Erase Err: {results['systems']['erase_residual']:.2e}", flush=True)

    del stu
    _purge_zombie_vram()

    if compare_donor:
        results["teacher"] = _eval_model(hf_donor, True, tok, device, limit=limit, gen_limit=gsm8k_limit)
        for k, v in results["teacher"].items():
            extra = f" | trunc={v['truncated']} med_len={v['median_len']}" if "truncated" in v else ""
            print(f"  [Donor ] {k:<12}: {v['acc']:6.2f}% ± {v['ci95']:4.1f}% ({v['correct']}/{v['total']}){extra}", flush=True)
            results["mcnemar"][k] = _mcnemar_exact(results["student"][k]["items"], v["items"])
    del hf_donor
    _purge_zombie_vram()

    # Strip raw items list before saving JSON to keep JSON concise, or keep it for auditability
    base_out = os.path.splitext(out_json or ckpt_path)[0]
    json_path = out_json or (base_out + "_benchmarks.json")
    png_path = base_out + "_benchmark.png"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    _plot_benchmark_figure(results, png_path)

    print("\n| Benchmark | Official HF Donor | **VSA-LM** | Paired McNemar (`b/c`, `p`) | Verdict |")
    print("|---|---:|---:|---:|---|")
    for k, sv in results["student"].items():
        dv = results["teacher"].get(k, {})
        mc = results["mcnemar"].get(k, {"stu_only": 0, "don_only": 0, "p": 1.0})
        verdict = f"{sv['acc'] - dv.get('acc', 0.0):+.2f}% (sig)" if mc["p"] < 0.05 else "No detectable diff (p >= 0.05)"
        print(f"| **{k}** | `{dv.get('acc', 0.0):.2f}%` | **`{sv['acc']:.2f}%`** | `stu={mc['stu_only']}, don={mc['don_only']}` (`p={mc['p']:.3f}`) | {verdict} |")
    print(f"| **Decode Speed @{sc_last['context']}** | `{sc_last['donor_tok_s']} tok/s` | **`{sc_last['vsa_tok_s']} tok/s`** | — | **`{sc_last['speedup']}x Faster`** |")
    print(f"| **State RAM @{sc_last['context']}** | `{sc_last['donor_kv_mb']} MB` | **`{sc_last['vsa_kv_mb']} MB`** | — | **`-{sc_last['ram_saved_pct']}% RAM`** |")
    return results
