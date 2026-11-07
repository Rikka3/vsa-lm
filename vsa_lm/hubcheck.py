"""Hub readiness check (`python -m vsa_lm hubcheck`): everything a Hugging Face user relies on, as pass/fail.

  1 load            checkpoint, tokenizer, chat template, stop tokens, donor profile, state size
  2 chat            chat replies through the profile's template and sampling: stop on EOS, no loops, correct
  3 decode_parity   recurrent decoder (eager + CUDA graph) vs the parallel forward
  4 batch           batched greedy generation vs single-sequence generation
  5 stream_parity   infinite-context streaming vs exact decoding: identical until the window fills
  6 passkey         needle recall by context length and depth, exact vs streaming (beyond-window recall)
  7 stream_ppl      perplexity over a long stream in constant RAM; flat memory; cost of eviction vs exact
  8 memory_bank     .vsa cartridges: memorize -> route -> answer -> algebraic erase
  9 server          OpenAI-style completion function, in streaming mode

Checks that measure a capability rather than a contract (passkey beyond the window, eviction cost) are
reported as numbers and never fail the run.
"""
from __future__ import annotations

import json
import math
import random
import re
import tempfile
import time

import torch
import torch.nn.functional as F

from .benchmark import _load_rows, _load_student
from .config.profiles import format_user_prompt, resolve_profile
from .decode import VSARecurrentDecoder
from .diagnose import _parity_tokens, decode_parity
from .server import generate_completion, get_all_eos_ids

FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "


def _feeder(dec: VSARecurrentDecoder, B: int, need_len: int, device, graph: bool):
    """feed(tok [B,1], pos) -> logits [B,V] from an empty state: the CUDA graph when asked and possible,
    else eager steps. A graph's logits live in its output buffer: use them before the next call."""
    if graph:
        runner = dec._graph_runner(B, need_len, device)
        if runner is not None:
            return runner
    states, kv = VSARecurrentDecoder.blank_state(dec.model, B, device, max_kv_len=need_len + 64, stream=dec.stream)

    def feed(tok, pos):
        h, _, _ = dec.step(tok, pos, states, kv)
        return dec.model.lm_head(h[:, -1, :])

    return feed


def _repeat_frac(ids: list[int], n: int = 8) -> float:
    """Fraction of n-grams that already occurred earlier in the text (loops push this toward 1)."""
    grams = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)


def _ppl(mean_nll) -> float:
    return math.exp(min(50.0, float(mean_nll)))  # a broken model must still produce a report


def _final_answer(text: str) -> str:
    return text.split("</think>")[-1]


# ----------------------------------------------------------------------------------------------- checks

def check_load(stu, tok, prof, eos_ids, step, conv_ver, stream) -> dict:
    fixed, per_tok = VSARecurrentDecoder.state_bytes(stu)
    has_template = bool(getattr(tok, "chat_template", None))
    res = {
        "step": step, "converter": conv_ver, "profile": prof.name,
        "sampling": {"temperature": prof.temperature, "top_p": prof.top_p, "max_new": prof.max_new},
        "params_b": round(sum(p.numel() for p in stu.parameters()) / 1e9, 3),
        "chat_template": has_template, "eos_ids": sorted(eos_ids),
        "state_mb_fixed": round(fixed / 2**20, 2), "kv_kb_per_token": round(per_tok / 2**10, 2),
        "exact_state_mb_at_65k": round((fixed + 65536 * per_tok) / 2**20, 1),
        "stream_state_mb_any_length": round((fixed + sum(stream) * per_tok) / 2**20, 1),
    }
    res["pass"] = bool(eos_ids) and (has_template or not prof.use_chat_template)
    return res


SMOKE = (
    ("What is 17 * 23? Give the final answer.", lambda s: "391" in s),
    ("What is the capital of France? Answer in one word.", lambda s: "paris" in s.lower()),
    ("Write a Python function is_prime(n) that returns True when n is prime.", lambda s: "def is_prime" in s),
)


def check_chat(dec, tok, prof, eos_ids, device) -> dict:
    rows = []
    for i, (q, ok) in enumerate(SMOKE):
        torch.manual_seed(100 + i)
        ids = tok.encode(format_user_prompt(tok, prof, q), add_special_tokens=False)
        out = dec.generate(torch.tensor([ids], device=device), prof.max_new, temperature=prof.temperature,
                           top_p=prof.top_p, eos_ids=eos_ids)[0, len(ids):].tolist()
        stopped = bool(out) and out[-1] in eos_ids
        text = tok.decode(out[:-1] if stopped else out, skip_special_tokens=True)
        rows.append({"prompt": q, "tokens": len(out), "stopped": stopped, "repeat_frac": round(_repeat_frac(out), 3),
                     "correct": ok(_final_answer(text)), "answer_tail": _final_answer(text).strip()[-160:]})
    n_ok = sum(r["correct"] for r in rows)
    return {"rows": rows, "pass": all(r["stopped"] and r["repeat_frac"] < 0.5 for r in rows) and n_ok >= len(rows) - 1}


def check_decode_parity(stu, tok, device, T: int) -> dict:
    rows = decode_parity(stu, tok, device, T=T)
    ok = all(r["top1_agree_pct"] >= 99.0 and r.get("graph_top1_agree_pct", 100.0) >= 99.0 for r in rows)
    return {"rows": rows, "pass": ok}


def check_batch(dec, tok, prof, device, n_new: int = 64, B: int = 4) -> dict:
    ids = tok.encode(format_user_prompt(tok, prof, "Explain in two sentences why the sky is blue."), add_special_tokens=False)
    x = torch.tensor([ids], device=device)
    single = dec.generate(x, n_new, temperature=0.0)[0, len(ids):].tolist()
    batch, _ = dec.generate_batch(x.repeat(B, 1), n_new, temperature=0.0)
    rows = batch.tolist()
    prefix = [next((i for i, (a, b) in enumerate(zip(r, single)) if a != b), min(len(r), len(single))) for r in rows]
    same_rows = all(r == rows[0] for r in rows)
    # bf16 kernels at a different batch size round differently, so a near-tie may flip late; the first
    # half must match exactly.
    return {"match_prefix_tokens": prefix, "rows_identical": same_rows, "pass": same_rows and min(prefix) >= n_new // 2}


def check_stream_parity(stu, tok, device, T: int, W: int, S: int) -> dict:
    """Exact vs streaming (small window, so eviction happens inside T) on one real sequence, eager and graph."""
    x = _parity_tokens(tok, device, T)
    T = x.shape[1]
    ex = VSARecurrentDecoder(stu)
    sd = VSARecurrentDecoder(stu, stream_window=W, stream_sinks=S)
    f_ex = _feeder(ex, 1, T, device, graph=False)
    f_st = _feeder(sd, 1, T, device, graph=False)
    agree, diff = torch.zeros(T), torch.zeros(T)
    for t in range(T):
        le = f_ex(x[:, t : t + 1], t)[0].float()
        ls = f_st(x[:, t : t + 1], t)[0].float()
        agree[t] = float(le.argmax() == ls.argmax())
        diff[t] = (le - ls).abs().max()
    g_agree = None
    g_st = sd._graph_runner(1, T, device)
    if g_st is not None:
        f_st = _feeder(sd, 1, T, device, graph=False)
        g_agree = torch.zeros(T)
        for t in range(T):
            ls = f_st(x[:, t : t + 1], t)[0].float()
            g_agree[t] = float(g_st(x[:, t : t + 1], t)[0].float().argmax() == ls.argmax())
    full = min(T, S + W)
    res = {
        "window": W, "sinks": S, "T": T,
        "before_eviction_top1_pct": round(100 * float(agree[:full].mean()), 2),
        "before_eviction_max_dlogit": round(float(diff[:full].max()), 4),
        "after_eviction_top1_pct": round(100 * float(agree[full:].mean()), 2) if full < T else None,
        "graph_vs_eager_top1_pct": round(100 * float(g_agree.mean()), 2) if g_agree is not None else None,
    }
    res["pass"] = res["before_eviction_top1_pct"] >= 99.0 and (g_agree is None or res["graph_vs_eager_top1_pct"] >= 99.0)
    return res


def _passkey_batch(tok, prof, L: int, depths, reps: int) -> tuple[torch.Tensor, list[int], list[float]]:
    """B prompts of exactly L tokens: template + filler with a 5-digit pass key at each depth + question."""
    pre_t, post_t = format_user_prompt(tok, prof, "@@HAY@@").split("@@HAY@@")
    if prof.name == "reasoning":  # skip the reasoning chain: this measures recall, not reasoning
        answer = ("" if post_t.rstrip().endswith("<think>") else "<think>\n") + "\n</think>\n\nThe pass key is"
    else:
        answer = " The pass key is"
    pre = tok.encode(pre_t + "There is a pass key hidden inside a lot of irrelevant text. Find it and remember it.\n\n",
                     add_special_tokens=False)
    if not prof.use_chat_template and isinstance(tok.bos_token_id, int):
        pre = [tok.bos_token_id] + pre
    post = tok.encode("\n\nWhat is the pass key? Answer with just the number." + post_t + answer, add_special_tokens=False)
    unit = tok.encode(FILLER * 64, add_special_tokens=False)
    rng = random.Random(1234 + L)
    rows, keys, ds = [], [], []
    for d in depths:
        for _ in range(reps):
            key = rng.randint(10000, 99999)
            needle = tok.encode(f" The pass key is {key}. Remember it. {key} is the pass key. ", add_special_tokens=False)
            n_hay = L - len(pre) - len(post) - len(needle)
            if n_hay < 16:
                raise ValueError(f"passkey length {L} is too short for the prompt template")
            hay = (unit * (n_hay // len(unit) + 1))[:n_hay]
            a = int(n_hay * d)
            rows.append(pre + hay[:a] + needle + hay[a:] + post)
            keys.append(key)
            ds.append(d)
    return torch.tensor(rows), keys, ds


def check_passkey(stu, tok, prof, eos_ids, device, lengths, depths, reps: int, W: int, S: int) -> dict:
    out = {"window": W, "runs": []}
    for L in lengths:
        x, keys, ds = _passkey_batch(tok, prof, L, depths, reps)
        x = x.to(device)
        for mode in ("exact", "stream"):
            dec = VSARecurrentDecoder(stu, max_cache_pos=max(65536, L + 64),
                                      stream_window=(W if mode == "stream" else None), stream_sinks=S)
            torch.cuda.empty_cache()
            base = torch.cuda.memory_allocated() if device.type == "cuda" else 0
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            toks, n = dec.generate_batch(x, 8, temperature=0.0, eos_ids=eos_ids)
            secs = time.perf_counter() - t0
            peak = (torch.cuda.max_memory_allocated() - base) / 2**20 if device.type == "cuda" else 0.0
            hits = []
            for r, key in enumerate(keys):
                m = re.search(r"\d+", tok.decode(toks[r, : int(n[r])].tolist(), skip_special_tokens=True))
                hits.append(bool(m) and m.group(0) == str(key))
            by_depth = {f"{d:.2f}": round(100 * sum(h for h, dd in zip(hits, ds) if dd == d) / max(1, ds.count(d)), 1)
                        for d in sorted(set(ds))}
            run = {"mode": mode, "length": L, "recall_pct": round(100 * sum(hits) / len(hits), 1), "by_depth": by_depth,
                   "beyond_window": mode == "stream" and L > W + S, "peak_mb": round(peak, 1), "secs": round(secs, 1)}
            out["runs"].append(run)
            print(f"    passkey {mode:>6} L={L:>6}: recall {run['recall_pct']:5.1f}%  by depth {by_depth}  "
                  f"peak +{run['peak_mb']:.0f} MB  {secs:.0f}s", flush=True)
            del dec
    # Contract: streaming inside its window recalls as well as exact decoding (beyond the window is reported only).
    inside = [(r["recall_pct"], e["recall_pct"]) for r in out["runs"] if r["mode"] == "stream" and not r["beyond_window"]
              for e in out["runs"] if e["mode"] == "exact" and e["length"] == r["length"]]
    out["pass"] = all(s >= e - 20.0 for s, e in inside)
    return out


def _long_text_ids(tok, n_tokens: int) -> list[int]:
    rows = _load_rows("Salesforce/wikitext", "wikitext-103-raw-v1", "test", 1_000_000)
    ids, buf = [], []
    for r in rows:
        buf.append(r["text"])
        if len(buf) >= 2000:
            ids += tok.encode("".join(buf), add_special_tokens=False)
            buf = []
            if len(ids) >= n_tokens:
                break
    if buf and len(ids) < n_tokens:
        ids += tok.encode("".join(buf), add_special_tokens=False)
    if len(ids) < n_tokens:
        raise RuntimeError(f"wikitext-103 test has only {len(ids)} tokens (< {n_tokens})")
    return ids[:n_tokens]


@torch.no_grad()
def check_stream_ppl(stu, tok, device, n_tokens: int, W: int, S: int, ref_len: int, seg: int = 1024) -> dict:
    """Stream n_tokens through the streaming decoder (constant RAM), NLL per position; compare with the exact
    parallel forward on the first ref_len tokens, where both see the same text."""
    ids = torch.tensor([_long_text_ids(tok, n_tokens)], device=device)
    dec = VSARecurrentDecoder(stu, stream_window=W, stream_sinks=S)
    feed = _feeder(dec, 1, n_tokens, device, graph=True)
    nll = torch.zeros(n_tokens - 1, device=device)
    mem = {}
    t0 = time.perf_counter()
    for t in range(n_tokens - 1):
        logits = feed(ids[:, t : t + 1], t)
        nll[t] = -F.log_softmax(logits[0].float(), dim=-1)[ids[0, t + 1]]
        if device.type == "cuda" and t in (S + W, n_tokens - 2):
            mem[t] = torch.cuda.memory_allocated()
    if device.type == "cuda":
        torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    nll = nll.cpu()
    segs = [round(_ppl(nll[i : i + seg].mean()), 2) for i in range(0, len(nll), seg)]

    def ppl(a, b):
        a, b = max(0, a), min(len(nll), b)
        return round(_ppl(nll[a:b].mean()), 3) if b > a else None

    res = {"tokens": n_tokens, "window": W, "sinks": S, "tok_s": round((n_tokens - 1) / secs, 1),
           "segment_ppl": segs, "ppl_window_filling": ppl(W // 2, W), "ppl_last_quarter": ppl(3 * n_tokens // 4, n_tokens)}
    if len(mem) == 2:
        a, b = sorted(mem)
        res["mem_growth_mb_after_window_full"] = round((mem[b] - mem[a]) / 2**20, 2)

    # Exact reference: the parallel forward (full attention) on the first ref_len tokens.
    ref_len = (min(ref_len, n_tokens) // 32) * 32
    if ref_len > W + S:
        try:
            with torch.autocast("cuda", torch.bfloat16, enabled=(device.type == "cuda")):
                h = stu(ids[:, :ref_len])[0]
            w_lm = stu.lm_head.weight.float()
            ref = torch.cat([-F.log_softmax(F.linear(h[i : i + 1024].float(), w_lm), dim=-1)
                            .gather(-1, ids[0, i + 1 : i + 1025].unsqueeze(-1)).squeeze(-1)
                             for i in range(0, ref_len - 1, 1024)])[: ref_len - 1].cpu()
            del h, w_lm
            res["exact_vs_stream_ppl_after_eviction"] = {
                "range": f"{W + S}-{ref_len}",
                "exact": round(_ppl(ref[W + S :].mean()), 3),
                "stream": round(_ppl(nll[W + S : ref_len - 1].mean()), 3),
            }
        except torch.cuda.OutOfMemoryError:
            res["exact_vs_stream_ppl_after_eviction"] = "skipped (OOM)"
            torch.cuda.empty_cache()
    late, filling = res["ppl_last_quarter"], res["ppl_window_filling"]
    flat = res.get("mem_growth_mb_after_window_full", 0.0) < 16.0
    res["pass"] = flat and all(math.isfinite(s) for s in segs) and (late is None or filling is None or late <= 1.5 * filling)
    return res


def check_memory_bank(stu, tok) -> dict:
    from .memory_bank import VSAMemoryBank
    facts = {
        "orion": ("The launch authorization code for Project Orion is TITAN-942.", "What is the launch authorization code for Project Orion?", "TITAN-942"),
        "hemlock": ("The antidote synthesis temperature for Protocol Hemlock is 347 Kelvin.", "What is the antidote synthesis temperature for Protocol Hemlock?", "347"),
        "lyra": ("The vault combination for the Lyra archive is 58-12-77.", "What is the vault combination for the Lyra archive?", "58-12-77"),
    }
    rows = []
    with tempfile.TemporaryDirectory() as d:
        bank = VSAMemoryBank(stu, tok, bank_dir=d)
        for name, (doc, _, _) in facts.items():
            bank.memorize_to_disk(name, doc)
        for name, (_, q, secret) in facts.items():
            r = bank.ask(q)
            rows.append({"doc": name, "routed": r["cartridge"], "answer": r["answer"][:80],
                         "recalled": secret in r["answer"], "erase_residual": r["erase_residual"]})
    routed = all(r["routed"] == r["doc"] for r in rows)
    erased = all(r["erase_residual"] < 1e-3 for r in rows)
    recalled = sum(r["recalled"] for r in rows)
    return {"rows": rows, "pass": routed and erased and recalled >= len(rows) - 1}


def check_server(stu, tok, device, prof, W: int, S: int) -> dict:
    dec = VSARecurrentDecoder(stu, stream_window=W, stream_sinks=S)
    reply = generate_completion(stu, tok, device, [{"role": "user", "content": "Say hello in one word."}],
                                max_tokens=prof.max_new, temperature=0.0, decoder=dec)
    return {"reply_tail": _final_answer(reply).strip()[-120:], "pass": bool(reply.strip())}


# ----------------------------------------------------------------------------------------------- runner

def run_hubcheck(ckpt_path: str, donor_id: str | None = None, profile: str | None = None,
                 stream_window: int = 8192, stream_sinks: int = 4,
                 passkey_lengths=(4096, 16384, 32768), passkey_depths=(0.1, 0.5, 0.9), passkey_reps: int = 2,
                 ppl_tokens: int = 32768, ppl_ref_len: int = 16384, parity_T: int = 1024,
                 skip: tuple[str, ...] = (), out_json: str | None = None) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    stu, cfg, tok, donor_id, donor_dir, step, conv_ver = _load_student(ckpt_path, donor_id, device)
    prof = resolve_profile(tok, profile, donor_dir)
    eos_ids = get_all_eos_ids(tok)
    W, S = int(stream_window), int(stream_sinks)
    dec = VSARecurrentDecoder(stu)

    print("=" * 84)
    print(f"VSA-LM HUB CHECK: {ckpt_path} (step {step}, {conv_ver}) | donor {donor_id} | profile {prof.name} | "
          f"stream window {W} + {S} sinks")
    print("=" * 84)
    checks = [
        ("load", lambda: check_load(stu, tok, prof, eos_ids, step, conv_ver, (S, W))),
        ("chat", lambda: check_chat(dec, tok, prof, eos_ids, device)),
        ("decode_parity", lambda: check_decode_parity(stu, tok, device, parity_T)),
        ("batch", lambda: check_batch(dec, tok, prof, device)),
        # Small window so the sequence crosses it: tests eviction itself, not just the no-op case.
        ("stream_parity", lambda: check_stream_parity(stu, tok, device, max(parity_T, 1024), 256, S)),
        ("passkey", lambda: check_passkey(stu, tok, prof, eos_ids, device, passkey_lengths, passkey_depths, passkey_reps, W, S)),
        ("stream_ppl", lambda: check_stream_ppl(stu, tok, device, ppl_tokens, W, S, ppl_ref_len)),
        ("memory_bank", lambda: check_memory_bank(stu, tok)),
        ("server", lambda: check_server(stu, tok, device, prof, W, S)),
    ]
    res = {"checkpoint": ckpt_path, "donor": donor_id, "profile": prof.name, "stream": {"window": W, "sinks": S}, "checks": {}}
    for name, fn in checks:
        if name in skip:
            continue
        print(f"  [{name}] ...", flush=True)
        t0 = time.perf_counter()
        try:
            r = fn()
        except Exception as e:  # one broken feature must not hide the others
            r = {"pass": False, "error": f"{type(e).__name__}: {e}"}
        r["secs"] = round(time.perf_counter() - t0, 1)
        res["checks"][name] = r
        detail = {k: v for k, v in r.items() if k not in ("pass", "rows", "runs", "segment_ppl", "eos_ids")}
        print(f"  [{name}] {'PASS' if r['pass'] else 'FAIL'}  {json.dumps(detail, default=str)[:600]}", flush=True)
        for row in r.get("rows", []):
            print(f"      {json.dumps(row, default=str)[:300]}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    res["ready"] = all(c["pass"] for c in res["checks"].values())
    print("-" * 84)
    print(f"  {'check':<15} result")
    for name, c in res["checks"].items():
        print(f"  {name:<15} {'PASS' if c['pass'] else 'FAIL'}" + (f"  ({c['error']})" if "error" in c else ""))
    print(f"  READY FOR HUB USERS: {'YES' if res['ready'] else 'NO'}")
    if out_json:
        with open(out_json, "w") as f:
            json.dump(res, f, indent=2, default=str)
        print(f"  Saved {out_json}")
    return res
