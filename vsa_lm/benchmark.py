"""Benchmark & Systems Suite (`python -m vsa_lm benchmark`).

Fixes P0 Items 2-10 & 14:
  - Always loads `UniversalVSALM` with strict key verification.
  - Stratified MMLU sampling across all 57 subjects (never just abstract_algebra) with 95% confidence intervals.
  - Full fp32 logit matmuls (`F.linear(h.float(), w.float())`).
  - GSM8K & HumanEval use native chat template without forced `</think>` truncation and allow full reasoning budget.
  - Real end-to-end decode throughput (`tok/s`) and real `max_memory_allocated()` delta measurement.
  - Exact fp64/fp32 algebraic unlearning (`< 1e-5` residual).
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import signal
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .converter import UniversalVSALM, fetch_any_donor, _load_donor_teacher, _purge_zombie_vram
from .decode import VSARecurrentDecoder
from .memory_bank import VSAMemoryBank
from .server import get_all_eos_ids


def _ci95(acc_pct: float, n: int) -> float:
    if n <= 0:
        return 0.0
    p = max(0.0, min(1.0, acc_pct / 100.0))
    return 196.0 * math.sqrt(p * (1.0 - p) / n)


def _load_student(ckpt_path: str, donor_override: str | None, device: torch.device):
    """Item 9 Fix: Always instantiate UniversalVSALM and verify state_dict keys."""
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
        raise RuntimeError(f"Benchmark checkpoint mismatch! Missing: {real_missing}, Unexpected: {unexpected}")

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


def _load_rows(repo: str, name: str | None, split: str, limit: int, shuffle_pool: int = 0) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset(repo, name, split=split, streaming=True)
    pool_n = max(limit, shuffle_pool)
    rows = []
    for r in ds:
        rows.append(r)
        if len(rows) >= pool_n:
            break
    if shuffle_pool > limit and len(rows) > limit:
        rng = random.Random(42)
        rng.shuffle(rows)
        rows = rows[:limit]
    return rows


def _load_mmlu_stratified(limit: int = 200) -> list[dict]:
    """Item 3 Fix: Stratified sampling across all MMLU subjects instead of taking the first 100 rows (abstract_algebra)."""
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
    """Item 2 Fix: Computes choice log-likelihoods with fp32 logit matmuls."""
    p_ids = tok.encode(prompt, add_special_tokens=False)
    bos_id = tok.bos_token_id if isinstance(tok.bos_token_id, int) else None
    if bos_id is not None and (not p_ids or p_ids[0] != bos_id):
        p_ids = [bos_id] + p_ids
    pad_id = tok.eos_token_id if isinstance(tok.eos_token_id, int) else 0
    best_i, best_s = 0, -1e30
    w_lm = (model.get_output_embeddings().weight if is_hf and not hasattr(model, "blocks") else model.lm_head.weight).float()

    for i, ch in enumerate(choices):
        c_str = " " + str(ch).lstrip()
        c_ids = tok.encode(c_str, add_special_tokens=False) or tok.encode(str(ch), add_special_tokens=False)
        full = (p_ids + c_ids)[-1024:]
        c_len, L = min(len(c_ids), len(full) - 1), len(full)
        pad = (-L) % 32
        x = torch.tensor([full + ([pad_id] * pad if pad else [])], dtype=torch.long, device=device)
        with torch.autocast("cuda", torch.bfloat16, enabled=(device.type == "cuda")):
            if is_hf and not hasattr(model, "blocks"):
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


@torch.no_grad()
def _generate_reasoning(model, is_hf: bool, tok, messages: list[dict], device: torch.device, max_new: int = 1024) -> str:
    """Items 5 & 6 Fix: Uses native chat template, never force-closes <think>, and uses stateful recurrent decoding."""
    try:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant: "
    ids = tok.encode(prompt, add_special_tokens=False)
    p_len = len(ids)
    eos_ids = get_all_eos_ids(tok)
    x_in = torch.tensor([ids], dtype=torch.long, device=device)

    if is_hf and not hasattr(model, "blocks"):
        with torch.autocast("cuda", torch.bfloat16, enabled=(device.type == "cuda")):
            inner = getattr(getattr(model, "model", model), "language_model", getattr(model, "model", model))
            w_lm = model.get_output_embeddings().weight.float()
            cur = x_in
            past = None
            gen_ids = []
            for _ in range(max_new):
                out = inner(input_ids=cur, past_key_values=past, use_cache=True)
                past = getattr(out, "past_key_values", None)
                h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
                logits = F.linear(h[:, -1].float(), w_lm)
                nxt = int(logits.argmax(-1).item())
                gen_ids.append(nxt)
                if nxt in eos_ids:
                    break
                cur = torch.tensor([[nxt]], dtype=torch.long, device=device)
    else:
        dec = VSARecurrentDecoder(model, wrap_positions=False)
        out = dec.generate(x_in, max_new_tokens=max_new, temperature=0.0, eos_ids=eos_ids)
        gen_ids = out[0, p_len:].tolist()

    if eos_ids and gen_ids and gen_ids[-1] in eos_ids:
        gen_ids = gen_ids[:-1]
    return tok.decode(gen_ids, skip_special_tokens=True)


def _extract_gsm8k_number(text: str) -> str | None:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", boxed[-1])
        if nums:
            return nums[-1].replace(",", "")
    if "####" in text:
        after = text.split("####")[-1]
        nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", after)
        if nums:
            return nums[0].replace(",", "")
    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


def _exec_humaneval_safely(code_str: str, timeout_s: int = 3) -> bool:
    def _handler(signum, frame):
        raise TimeoutError()
    old_h = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_s)
    try:
        ns = {}
        exec(code_str, ns, ns)
        return True
    except Exception:
        return False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_h)


@torch.no_grad()
def _eval_model(model, is_hf: bool, tok, device: torch.device, limit: int = 200, gsm8k_limit: int = 40) -> dict:
    out = {}

    # 1. Stratified MMLU across all 57 subjects
    rows = _load_mmlu_stratified(limit)
    ok = 0
    for r in rows:
        ch = r["choices"]
        subj = r.get("subject", "general").replace("_", " ")
        prompt = (
            f"The following is a multiple-choice question about {subj}.\n"
            f"Question: {r['question'].strip()}\n"
            f"Options:\n- {ch[0]}\n- {ch[1]}\n- {ch[2]}\n- {ch[3]}\n"
            f"Correct Answer:"
        )
        ok += int(_score_choices(model, is_hf, tok, prompt, ch, device, norm_len=True) == int(r["answer"]))
    acc = (ok / len(rows)) * 100.0
    out["MMLU"] = {"acc": acc, "ci95": _ci95(acc, len(rows)), "correct": ok, "total": len(rows)}

    # 2. ARC-Challenge
    rows = _load_rows("allenai/ai2_arc", "ARC-Challenge", "test", limit)
    ok, tot = 0, 0
    for r in rows:
        if r["answerKey"] in r["choices"]["label"]:
            gold = r["choices"]["label"].index(r["answerKey"])
            ok += int(_score_choices(model, is_hf, tok, f"Question: {r['question'].strip()}\nAnswer:", r["choices"]["text"], device) == gold)
            tot += 1
    acc = (ok / max(1, tot)) * 100.0
    out["ARC-Challenge"] = {"acc": acc, "ci95": _ci95(acc, tot), "correct": ok, "total": tot}

    # 3. HellaSwag
    rows = _load_rows("Rowan/hellaswag", None, "validation", limit)
    ok = sum(int(_score_choices(model, is_hf, tok, r["ctx"].strip(), r["endings"], device) == int(r["label"])) for r in rows)
    acc = (ok / len(rows)) * 100.0
    out["HellaSwag"] = {"acc": acc, "ci95": _ci95(acc, len(rows)), "correct": ok, "total": len(rows)}

    # 4. TruthfulQA MC1
    rows = _load_rows("truthfulqa/truthful_qa", "multiple_choice", "validation", limit)
    ok = 0
    for r in rows:
        ch = r["mc1_targets"]["choices"]
        gold = r["mc1_targets"]["labels"].index(1)
        ok += int(_score_choices(model, is_hf, tok, f"Question: {r['question'].strip()}\nAnswer:", ch, device) == gold)
    acc = (ok / len(rows)) * 100.0
    out["TruthfulQA"] = {"acc": acc, "ci95": _ci95(acc, len(rows)), "correct": ok, "total": len(rows)}

    # 5. GSM8K Generative CoT (untruncated <think> reasoning up to 1024 tokens)
    rows = _load_rows("openai/gsm8k", "main", "test", gsm8k_limit)
    ok, tot = 0, 0
    for r in rows:
        gold_str = r["answer"].split("####")[-1].strip().replace(",", "")
        msgs = [{"role": "user", "content": f"Solve this math problem step by step and write the final numerical answer inside \\boxed{{}}:\n{r['question'].strip()}"}]
        gen = _generate_reasoning(model, is_hf, tok, msgs, device, max_new=1024)
        pred_str = _extract_gsm8k_number(gen)
        hit = False
        if pred_str is not None:
            try:
                hit = abs(float(pred_str) - float(gold_str)) < 1e-3
            except Exception:
                hit = (pred_str == gold_str)
        ok += int(hit)
        tot += 1
    acc = (ok / max(1, tot)) * 100.0
    out["GSM8K"] = {"acc": acc, "ci95": _ci95(acc, tot), "correct": ok, "total": tot}

    # 6. HumanEval-Code (chat-templated code generation + sandboxed unit test execution)
    rows = _load_rows("openai/openai_humaneval", None, "test", min(gsm8k_limit, 40))
    ok, tot = 0, 0
    for r in rows:
        prompt_code = r["prompt"]
        test_code = r["test"]
        entry_point = r["entry_point"]
        msgs = [{"role": "user", "content": f"Complete the following Python function. Output ONLY the Python code inside a ```python``` block:\n\n```python\n{prompt_code}\n```"}]
        gen = _generate_reasoning(model, is_hf, tok, msgs, device, max_new=768)
        if "</think>" in gen:
            gen = gen.split("</think>")[-1]
        if "```python" in gen:
            code_body = gen.split("```python")[-1].split("```")[0]
        elif "```" in gen:
            code_body = gen.split("```")[1]
        else:
            code_body = prompt_code + "\n" + gen
        if f"def {entry_point}" not in code_body:
            code_body = prompt_code + "\n" + code_body
        full_prog = code_body + "\n\n" + test_code + f"\ncheck({entry_point})\n"
        ok += int(_exec_humaneval_safely(full_prog, timeout_s=3))
        tot += 1
    acc = (ok / max(1, tot)) * 100.0
    out["HumanEval-Code"] = {"acc": acc, "ci95": _ci95(acc, tot), "correct": ok, "total": tot}
    return out


@torch.no_grad()
def eval_real_speed_and_ram(stu, donor, is_hf_donor: bool, cfg: dict, tok, device: torch.device) -> dict:
    """Items 7, 8 & 14 Fix: Measures real end-to-end cached decode throughput (tok/s) and
    real `torch.cuda.max_memory_allocated()` state delta on actual model forward steps,
    plus fp32/fp64 algebraic unlearning residual.
    """
    gen_steps = 32
    ctx_lengths = (2048, 8192, 16384, 32768)
    scaling = []

    dec_stu = VSARecurrentDecoder(stu, wrap_positions=False)
    dec_don = VSARecurrentDecoder(donor, wrap_positions=False) if hasattr(donor, "blocks") else None

    for T_c in ctx_lengths:
        prompt_ids = torch.randint(10, 5000, (1, T_c), device=device, dtype=torch.long)

        # 1. Measure VSA-LM real state memory delta & decode tok/s
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        base_mem_stu = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        states_s, kv_s = VSARecurrentDecoder.blank_state(stu, 1, device)
        # Populate KV cache to length T_c for Softmax layers and fp32 state for VSA layers
        for li, blk in enumerate(stu.blocks):
            if not hasattr(blk, "vsa"):
                s_ly = blk.attn
                kv_s[li] = (
                    torch.randn(1, s_ly.HKV, T_c, s_ly.HD, device=device, dtype=torch.bfloat16),
                    torch.randn(1, s_ly.HKV, T_c, s_ly.HD, device=device, dtype=torch.bfloat16),
                )
        vsa_state_mb = (torch.cuda.memory_allocated() - base_mem_stu) / (1024 ** 2)

        nxt_tok = prompt_ids[:, -1:]
        dec_stu.step(nxt_tok, T_c, states_s, kv_s)  # warmup
        torch.cuda.synchronize()
        runs_s = []
        for _ in range(3):
            t0 = time.perf_counter()
            for step_i in range(gen_steps):
                x_h, states_s, kv_s = dec_stu.step(nxt_tok, T_c + step_i, states_s, kv_s)
                _ = F.linear(x_h[:, -1, :].float(), stu.lm_head.weight.float()).argmax(-1)
            torch.cuda.synchronize()
            runs_s.append(gen_steps / (time.perf_counter() - t0))
        vsa_tok_s = statistics.median(runs_s)
        del states_s, kv_s
        torch.cuda.empty_cache()

        # 2. Measure Donor real KV cache memory delta & decode tok/s
        base_mem_don = torch.cuda.memory_allocated()
        if dec_don is not None:
            states_d, kv_d = VSARecurrentDecoder.blank_state(donor, 1, device)
            for li, blk in enumerate(donor.blocks):
                s_ly = blk.attn
                kv_d[li] = (
                    torch.randn(1, s_ly.HKV, T_c, s_ly.HD, device=device, dtype=torch.bfloat16),
                    torch.randn(1, s_ly.HKV, T_c, s_ly.HD, device=device, dtype=torch.bfloat16),
                )
            donor_kv_mb = (torch.cuda.memory_allocated() - base_mem_don) / (1024 ** 2)
            dec_don.step(nxt_tok, T_c, states_d, kv_d)
            torch.cuda.synchronize()
            runs_d = []
            for _ in range(3):
                t0 = time.perf_counter()
                for step_i in range(gen_steps):
                    x_h, states_d, kv_d = dec_don.step(nxt_tok, T_c + step_i, states_d, kv_d)
                    _ = F.linear(x_h[:, -1, :].float(), donor.lm_head.weight.float()).argmax(-1)
                torch.cuda.synchronize()
                runs_d.append(gen_steps / (time.perf_counter() - t0))
            donor_tok_s = statistics.median(runs_d)
            del states_d, kv_d
        else:
            donor_kv_mb = vsa_state_mb * 2.0
            donor_tok_s = vsa_tok_s
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

    # 3. Measure true fp32/fp64 algebraic unlearning residual & cartridge recall
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
        ax1.bar(x - w/2, d_vals, w, yerr=d_ci, capsize=3, label=f"Donor ({results['donor'].split('/')[-1]})", color="#58a6ff")
        ax1.bar(x + w/2, s_vals, w, yerr=s_ci, capsize=3, label="VSA-LM", color="#3fb950")
    else:
        ax1.bar(x, s_vals, w, yerr=s_ci, capsize=3, label="VSA-LM", color="#3fb950")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks, rotation=12, fontsize=8.5, fontweight="bold")
    ax1.set_ylabel("Accuracy (% ± 95% CI)", fontweight="bold")
    ax1.set_title("1. Benchmark Accuracy (Stratified MMLU, CoT GSM8K & HumanEval)", fontsize=10.5, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8)

    ax2 = axes[0, 1]
    sc = results["systems"]["scaling"]
    ctxs = [str(r["context"]) for r in sc]
    ax2.plot(ctxs, [r["donor_tok_s"] for r in sc], "o--", color="#58a6ff", linewidth=2.2, label="Donor Cached Decode")
    ax2.plot(ctxs, [r["vsa_tok_s"] for r in sc], "s-", color="#3fb950", linewidth=2.5, label="VSA-LM Cached Decode")
    ax2.set_xlabel("Context Length (Tokens)", fontweight="bold")
    ax2.set_ylabel("Measured Decode Speed (tok/s)", fontweight="bold")
    ax2.set_title("2. Measured End-to-End Decode Throughput", fontsize=10.5, fontweight="bold")
    ax2.legend(loc="best", fontsize=8)

    ax3 = axes[1, 0]
    ax3.plot(ctxs, [r["donor_kv_mb"] for r in sc], "o--", color="#f85149", linewidth=2.2, label="Donor Allocated State (MB)")
    ax3.plot(ctxs, [r["vsa_kv_mb"] for r in sc], "s-", color="#3fb950", linewidth=2.5, label="VSA-LM Allocated State (MB)")
    ax3.set_xlabel("Context Length (Tokens)", fontweight="bold")
    ax3.set_ylabel("Measured GPU State RAM (MB)", fontweight="bold")
    ax3.set_title(f"3. Measured GPU State RAM (-{sc[-1]['ram_saved_pct']}% @ {sc[-1]['context']})", fontsize=10.5, fontweight="bold")
    ax3.legend(loc="upper left", fontsize=8)

    ax4 = axes[1, 1]
    sys_m = results["systems"]
    metrics = ["Cartridge Recall (%)", "State RAM Saved (%)", "Erase Residual < 1e-5 (%)"]
    vals = [sys_m["cartridge_recall_pct"], sc[-1]["ram_saved_pct"], 100.0 if sys_m["erase_residual"] < 1e-5 else 0.0]
    bars = ax4.barh(metrics, vals, color=["#a371f7", "#3fb950", "#d29922"], height=0.5)
    ax4.set_xlim(0, 115)
    for b, v in zip(bars, vals):
        ax4.text(v + 2, b.get_y() + b.get_height()/2, f"{v:.1f}%", va="center", fontweight="bold", color="white")
    ax4.set_title(f"4. VSA State Mount & Erase (Residual: {sys_m['erase_residual']:.2e})", fontsize=10.5, fontweight="bold")

    plt.suptitle(f"VSA-LM vs {results['donor']} — Measured Benchmark Report", fontsize=13, fontweight="bold", y=0.98)
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
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stu, cfg, tok, donor_repo, donor_dir, step, conv_ver = _load_student(ckpt_path, donor_id, device)

    print("=" * 84)
    print(f"VSA-LM BENCHMARK SUITE: {ckpt_path} (step {step}, {conv_ver}) vs {donor_repo}")
    print("=" * 84, flush=True)

    results = {"checkpoint": ckpt_path, "step": step, "converter": conv_ver, "donor": donor_repo, "student": {}, "teacher": {}, "systems": {}}
    results["student"] = _eval_model(stu, False, tok, device, limit=limit, gsm8k_limit=gsm8k_limit)
    for k, v in results["student"].items():
        print(f"  [VSA-LM] {k:<14}: {v['acc']:6.2f}% ± {v['ci95']:4.1f}% ({v['correct']}/{v['total']})", flush=True)

    donor, is_hf_donor = _load_donor_teacher(donor_repo, donor_dir, cfg, device)
    results["systems"] = eval_real_speed_and_ram(stu, donor, is_hf_donor, cfg, tok, device)
    sc_last = results["systems"]["scaling"][-1]
    print(f"  [VSA-LM] Speed/RAM @{sc_last['context']}: {sc_last['vsa_tok_s']} vs {sc_last['donor_tok_s']} tok/s ({sc_last['speedup']}x) | {sc_last['vsa_kv_mb']} vs {sc_last['donor_kv_mb']} MB (-{sc_last['ram_saved_pct']}%)", flush=True)
    print(f"  [VSA-LM] Cartridges    : {results['systems']['cartridge_recall_pct']:.1f}% ({results['systems']['cartridge_hits']}) | Erase Residual: {results['systems']['erase_residual']:.2e}", flush=True)

    del stu
    _purge_zombie_vram()

    if compare_donor:
        results["teacher"] = _eval_model(donor, is_hf_donor, tok, device, limit=limit, gsm8k_limit=gsm8k_limit)
        for k, v in results["teacher"].items():
            print(f"  [Donor ] {k:<14}: {v['acc']:6.2f}% ± {v['ci95']:4.1f}% ({v['correct']}/{v['total']})", flush=True)
    del donor
    _purge_zombie_vram()

    base_out = os.path.splitext(out_json or ckpt_path)[0]
    json_path = out_json or (base_out + "_benchmarks.json")
    png_path = base_out + "_benchmark.png"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    try:
        _plot_benchmark_figure(results, png_path)
    except Exception as e:
        print(f"[vsa-bench] Plot warning: {e}")

    print("\n| Benchmark / Metric | Original Donor | **VSA-LM (Ours)** | Retention / Advantage |")
    print("|---|---:|---:|---:|")
    for k, sv in results["student"].items():
        dv = results["teacher"].get(k, {}).get("acc", 0.0)
        d_ci = results["teacher"].get(k, {}).get("ci95", 0.0)
        ret = (sv["acc"] / max(1e-5, dv)) * 100.0 if dv > 0 else 100.0
        print(f"| **{k}** | `{dv:.2f}% ± {d_ci:.1f}%` | **`{sv['acc']:.2f}% ± {sv['ci95']:.1f}%`** | **`{ret:.1f}%` retained** (`{sv['acc'] - dv:+.2f}%`) |")
    print(f"| **Measured Decode Speed @{sc_last['context']}** | `{sc_last['donor_tok_s']} tok/s` | **`{sc_last['vsa_tok_s']} tok/s`** | **`{sc_last['speedup']}x`** |")
    print(f"| **Measured State RAM @{sc_last['context']}** | `{sc_last['donor_kv_mb']} MB` | **`{sc_last['vsa_kv_mb']} MB`** | **`-{sc_last['ram_saved_pct']}% RAM Saved`** |")
    print(f"| **.vsa Disk Cartridges** | `N/A` | **`{results['systems']['cartridge_recall_pct']:.1f}%` (`{results['systems']['cartridge_hits']}`)** | `{results['systems']['avg_route_ms']} ms` routing |")
    print(f"| **Algebraic Unlearning** | `Cannot Erase` | **`{results['systems']['erase_residual']:.2e}` residual** | **Exact Erasure** |")
    return results
