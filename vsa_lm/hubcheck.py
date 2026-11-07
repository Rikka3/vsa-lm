from __future__ import annotations

import json
import math
import random
import re
import tempfile
import time

import torch
import torch.nn.functional as F

from .benchmark import _load_hf_donor, _load_rows, _load_student
from .config.profiles import format_user_prompt, resolve_profile
from .decode import VSARecurrentDecoder
from .diagnose import _parity_tokens, decode_parity
from .server import generate_completion, get_all_eos_ids

FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "


def _feeder(dec: VSARecurrentDecoder, B: int, need_len: int, device, graph: bool):
    if graph:
        runner = dec._graph_runner(B, need_len, device)
        if runner is not None:
            return runner
    states, kv = VSARecurrentDecoder.blank_state(dec.model, B, device, max_kv_len=need_len + 64, stream=dec.stream)

    @torch.no_grad()
    def feed(tok, pos):
        h, _, _ = dec.step(tok, pos, states, kv)
        return dec.model.lm_head(h[:, -1, :])

    return feed


def _donor_generate(donor, tok, x: torch.Tensor, max_new: int, eos_ids, temperature: float = 0.0, top_p: float = 1.0):
    sample = temperature > 1e-5
    out = donor.generate(x, max_new_tokens=max_new, do_sample=sample, temperature=temperature if sample else None,
                         top_p=top_p if sample else None, eos_token_id=sorted(eos_ids) or None,
                         pad_token_id=tok.eos_token_id if isinstance(tok.eos_token_id, int) else 0)
    toks = out[:, x.shape[1]:]
    hit = torch.isin(toks, torch.tensor(sorted(eos_ids), dtype=torch.long, device=toks.device))
    n = torch.where(hit.any(1), hit.float().argmax(1) + 1, torch.full_like(hit[:, 0], toks.shape[1], dtype=torch.long))
    return toks, n


def _nll_from_hidden(h: torch.Tensor, head, ids: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    out = []
    for i in range(0, h.shape[0] - 1, chunk):
        j = min(i + chunk, h.shape[0] - 1)
        logp = F.log_softmax(head(h[i:j]).float(), dim=-1)
        out.append(-logp.gather(-1, ids[i + 1 : j + 1].unsqueeze(-1)).squeeze(-1))
    return torch.cat(out).cpu()


def _repeat_frac(ids: list[int], n: int = 8) -> float:
    grams = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)


def _ppl(mean_nll) -> float:
    return math.exp(min(50.0, float(mean_nll)))


def _seg_ppl(nll: torch.Tensor, seg: int = 1024) -> list[float]:
    return [round(_ppl(nll[i : i + seg].mean()), 2) for i in range(0, len(nll), seg)]


def _final_answer(text: str) -> str:
    return text.split("</think>")[-1]


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
    ("Write a Python function is_prime(n) that returns True when n is prime. Output only code.", lambda s: "def is_prime" in s),
)


@torch.no_grad()
def check_chat(dec, donor, tok, prof, eos_ids, device) -> dict:
    rows = []
    for i, (q, ok) in enumerate(SMOKE):
        ids = tok.encode(format_user_prompt(tok, prof, q), add_special_tokens=False)
        x = torch.tensor([ids], device=device)
        torch.manual_seed(100 + i)
        out = dec.generate(x, prof.max_new, temperature=prof.temperature, top_p=prof.top_p, eos_ids=eos_ids)[0, len(ids):].tolist()
        stopped = bool(out) and out[-1] in eos_ids
        text = tok.decode(out[:-1] if stopped else out, skip_special_tokens=True)
        if not (stopped and ok(_final_answer(text))):
            out = dec.generate(x, prof.max_new, temperature=0.0, eos_ids=eos_ids)[0, len(ids):].tolist()
            stopped = bool(out) and out[-1] in eos_ids
            text = tok.decode(out[:-1] if stopped else out, skip_special_tokens=True)
        row = {"prompt": q, "tokens": len(out), "stopped": stopped, "repeat_frac": round(_repeat_frac(out), 3),
               "correct": ok(_final_answer(text)), "answer_tail": _final_answer(text).strip()[-160:]}
        if donor is not None:
            torch.manual_seed(100 + i)
            d_toks, d_n = _donor_generate(donor, tok, x, prof.max_new, eos_ids, prof.temperature, prof.top_p)
            d_n = int(d_n[0])
            d_stopped = d_n < d_toks.shape[1] or (d_n > 0 and int(d_toks[0, d_n - 1]) in eos_ids)
            d_text = tok.decode(d_toks[0, :d_n].tolist(), skip_special_tokens=True)
            row.update(donor_tokens=d_n, donor_stopped=d_stopped, donor_correct=ok(_final_answer(d_text)))
        row["ok"] = (row["repeat_frac"] < 0.35 and (row["correct"] or not row.get("donor_correct", True))
                     and (row["stopped"] or row["correct"] or not row.get("donor_stopped", True)))
        rows.append(row)
    return {"rows": rows, "pass": all(r["ok"] for r in rows)}


@torch.no_grad()
def _donor_cached_parity(donor, x: torch.Tensor) -> float:
    full = donor(x, use_cache=False).logits[0].argmax(-1)
    pkv, agree = None, 0
    for t in range(x.shape[1]):
        out = donor(x[:, t : t + 1], past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        agree += int(out.logits[0, -1].argmax() == full[t])
    return 100.0 * agree / x.shape[1]


def check_decode_parity(stu, donor, tok, device, T: int, ctx: dict) -> dict:
    rows = decode_parity(stu, tok, device, T=T)
    sizes = [int(r["pos"].split("-")[1]) - int(r["pos"].split("-")[0]) for r in rows]
    res = {"rows": rows}
    for key in ("top1_agree_pct", "graph_top1_agree_pct"):
        if all(key in r for r in rows):
            res[key] = round(sum(r[key] * n for r, n in zip(rows, sizes)) / sum(sizes), 2)
    floor = 99.0
    if donor is not None:
        res["donor_top1_agree_pct"] = round(_donor_cached_parity(donor, _parity_tokens(tok, device, T)), 2)
        floor = res["donor_top1_agree_pct"] - 0.5
    ctx["parity_floor"] = floor
    res["pass"] = all(res[k] >= floor for k in ("top1_agree_pct", "graph_top1_agree_pct") if k in res)
    return res


@torch.no_grad()
def check_batch(dec, donor, tok, prof, device, n_new: int = 64, B: int = 4) -> dict:
    ids = tok.encode(format_user_prompt(tok, prof, "Explain in two sentences why the sky is blue."), add_special_tokens=False)
    x = torch.tensor([ids], device=device)

    def prefix(rows, single):
        return [next((i for i, (a, b) in enumerate(zip(r, single)) if a != b), min(len(r), len(single))) for r in rows]

    single = dec.generate(x, n_new, temperature=0.0)[0, len(ids):].tolist()
    rows = dec.generate_batch(x.repeat(B, 1), n_new, temperature=0.0)[0].tolist()
    res = {"match_prefix_tokens": prefix(rows, single), "rows_identical": all(r == rows[0] for r in rows)}
    bar = n_new // 4
    if donor is not None:
        d_single = _donor_generate(donor, tok, x, n_new, set())[0][0].tolist()
        d_rows = _donor_generate(donor, tok, x.repeat(B, 1), n_new, set())[0].tolist()
        res["donor_match_prefix_tokens"] = prefix(d_rows, d_single)
        bar = min(bar, min(res["donor_match_prefix_tokens"]))
    res["pass"] = res["rows_identical"] and min(res["match_prefix_tokens"]) >= bar
    return res


@torch.no_grad()
def check_stream_parity(stu, tok, device, T: int, W: int, S: int, ctx: dict) -> dict:
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
        "before_eviction_max_dlogit": round(float(diff[:full].detach().max()), 4),
        "after_eviction_top1_pct": round(100 * float(agree[full:].mean()), 2) if full < T else None,
        "graph_vs_eager_top1_pct": round(100 * float(g_agree.mean()), 2) if g_agree is not None else None,
    }
    floor = ctx.get("parity_floor", 99.0)
    res["pass"] = res["before_eviction_top1_pct"] >= floor and (g_agree is None or res["graph_vs_eager_top1_pct"] >= floor)
    return res


def _passkey_batch(tok, prof, L: int, depths, reps: int) -> tuple[torch.Tensor, list[int], list[float]]:
    pre_t, post_t = format_user_prompt(tok, prof, "@@HAY@@").split("@@HAY@@")
    if prof.name == "reasoning":
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
                raise ValueError(f"passkey length {L} is too short for prompt template")
            hay = (unit * (n_hay // len(unit) + 1))[:n_hay]
            a = int(n_hay * d)
            rows.append(pre + hay[:a] + needle + hay[a:] + post)
            keys.append(key)
            ds.append(d)
    return torch.tensor(rows), keys, ds


@torch.no_grad()
def check_passkey(stu, donor, tok, prof, eos_ids, device, lengths, depths, reps: int, W: int, S: int) -> dict:
    out = {"window": W, "runs": []}
    for L in lengths:
        x, keys, ds = _passkey_batch(tok, prof, L, depths, reps)
        x = x.to(device)
        for mode in ("donor", "exact", "stream"):
            if mode == "donor" and donor is None:
                continue
            dec = None if mode == "donor" else VSARecurrentDecoder(
                stu, max_cache_pos=max(65536, L + 64), stream_window=(W if mode == "stream" else None), stream_sinks=S)
            torch.cuda.empty_cache()
            base = torch.cuda.memory_allocated() if device.type == "cuda" else 0
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            if mode == "donor":
                toks, n = _donor_generate(donor, tok, x, 8, eos_ids)
            else:
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
    max_T = int(stu.config.get("max_T", 1024))
    slack = 100.0 / (len(depths) * reps) + 0.1
    ref = {r["length"]: r["recall_pct"] for r in out["runs"] if r["mode"] == "donor"}
    judged = [r for r in out["runs"] if r["length"] <= max_T and (r["mode"] == "exact" or (r["mode"] == "stream" and not r["beyond_window"]))]
    out["pass"] = all(r["recall_pct"] >= ref.get(r["length"], 100.0) - slack for r in judged)
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
def check_stream_ppl(stu, donor, tok, device, n_tokens: int, W: int, S: int, ref_len: int, seg: int = 1024) -> dict:
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

    def ppl(x, a, b):
        a, b = max(0, a), min(len(x), b)
        return round(_ppl(x[a:b].mean()), 3) if b > a else None

    res = {"tokens": n_tokens, "window": W, "sinks": S, "tok_s": round((n_tokens - 1) / secs, 1),
           "stream_segment_ppl": _seg_ppl(nll, seg),
           "stream_ppl_window_filling": ppl(nll, W // 2, W), "stream_ppl_last_quarter": ppl(nll, 3 * n_tokens // 4, n_tokens)}
    if len(mem) == 2:
        a, b = sorted(mem)
        res["mem_growth_mb_after_window_full"] = round((mem[b] - mem[a]) / 2**20, 2)

    max_T = int(stu.config.get("max_T", 1024))
    ref_len = (min(ref_len, n_tokens, max_T) // 32) * 32
    x = ids[:, :ref_len]
    ref = {"stream": nll[: ref_len - 1]}
    try:
        with torch.autocast("cuda", torch.bfloat16, enabled=(device.type == "cuda")):
            h = stu(x)[0]
            w_lm = stu.lm_head.weight
            ref["exact"] = _nll_from_hidden(h, lambda z: F.linear(z, w_lm), x[0])
            del h
            if donor is not None:
                h = donor.model(input_ids=x, use_cache=False).last_hidden_state[0]
                ref["donor"] = _nll_from_hidden(h, donor.lm_head, x[0])
                del h
    except torch.cuda.OutOfMemoryError:
        res["reference"] = "skipped (OOM)"
        torch.cuda.empty_cache()
    res["reference_range"] = f"0-{ref_len}"
    for name, v in ref.items():
        res[f"{name}_ref_segment_ppl"] = _seg_ppl(v, seg)
        res[f"{name}_ref_ppl"] = ppl(v, 0, ref_len)
    flat = res.get("mem_growth_mb_after_window_full", 0.0) < 16.0
    stable = all(math.isfinite(s) for s in res["stream_segment_ppl"])
    res["pass"] = flat and stable
    if "donor" in ref and "exact" in ref:
        res["exact_over_donor"] = round(res["exact_ref_ppl"] / res["donor_ref_ppl"], 3)
        res["stream_over_donor_in_window"] = round(ppl(ref["stream"], 0, ref_len) / ppl(ref["donor"], 0, ref_len), 3)
        res["pass"] = res["pass"] and res["exact_over_donor"] <= 1.25 and res["stream_over_donor_in_window"] <= 1.25
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


def run_hubcheck(ckpt_path: str, donor_id: str | None = None, profile: str | None = None,
                 stream_window: int = 8192, stream_sinks: int = 4,
                 passkey_lengths=(4096, 16384, 32768), passkey_depths=(0.1, 0.5, 0.9), passkey_reps: int = 2,
                 ppl_tokens: int = 32768, ppl_ref_len: int = 16384, parity_T: int = 1024,
                 skip: tuple[str, ...] = (), compare_donor: bool = True, out_json: str | None = None) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    stu, cfg, tok, donor_id, donor_dir, step, conv_ver = _load_student(ckpt_path, donor_id, device)
    prof = resolve_profile(tok, profile, donor_dir)
    eos_ids = get_all_eos_ids(tok)
    W, S = int(stream_window), int(stream_sinks)
    dec = VSARecurrentDecoder(stu)
    donor = _load_hf_donor(donor_dir, device) if compare_donor else None
    ctx: dict = {}

    print("=" * 84)
    print(f"VSA-LM HUB CHECK: {ckpt_path} (step {step}, {conv_ver}) | donor {donor_id} "
          f"({'compared' if donor is not None else 'not compared'}) | profile {prof.name} | stream window {W} + {S} sinks")
    print("=" * 84)
    checks = [
        ("load", lambda: check_load(stu, tok, prof, eos_ids, step, conv_ver, (S, W))),
        ("chat", lambda: check_chat(dec, donor, tok, prof, eos_ids, device)),
        ("decode_parity", lambda: check_decode_parity(stu, donor, tok, device, parity_T, ctx)),
        ("batch", lambda: check_batch(dec, donor, tok, prof, device)),
        ("stream_parity", lambda: check_stream_parity(stu, tok, device, max(parity_T, 1024), 256, S, ctx)),
        ("passkey", lambda: check_passkey(stu, donor, tok, prof, eos_ids, device, passkey_lengths, passkey_depths,
                                          passkey_reps, W, S)),
        ("stream_ppl", lambda: check_stream_ppl(stu, donor, tok, device, ppl_tokens, W, S, ppl_ref_len)),
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
        except Exception as e:
            r = {"pass": False, "error": f"{type(e).__name__}: {e}"}
        r["secs"] = round(time.perf_counter() - t0, 1)
        res["checks"][name] = r
        detail = {k: v for k, v in r.items() if k not in ("pass", "rows", "runs", "eos_ids")}
        print(f"  [{name}] {'PASS' if r['pass'] else 'FAIL'}  {json.dumps(detail, default=str)[:1500]}", flush=True)
        for row in r.get("rows", []):
            print(f"      {json.dumps(row, default=str)[:300]}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    res["ready"] = all(c["pass"] for c in res["checks"].values())
    print("-" * 84)
    print(f"  {'check':<15} result")
    for name, c in res["checks"].items():
        print(f"  {name:<15} {'PASS' if c['pass'] else 'FAIL'}" + (f"  ({c['error']})" if "error" in c else ""))
    print(f"  DROP-IN REPLACEMENT FOR THE DONOR: {'YES' if res['ready'] else 'NO'}")
    if out_json:
        with open(out_json, "w") as f:
            json.dump(res, f, indent=2, default=str)
        print(f"  Saved {out_json}")
    return res
