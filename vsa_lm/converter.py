"""Universal Donor-to-VSA Converter & 2-Stage Attention-Transfer + KD Healer (vsa_lm.converter)."""
from __future__ import annotations

import gc
import math
import os
import random
import sys
import threading
import time
import traceback
from array import array

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from .config import (
    fetch_any_donor,
    load_all_safetensors,
    inspect_donor_config,
    map_any_hf_to_student,
    Qwen35ZeroCenteredRMSNorm,
    Qwen35RMSNormGated,
    Qwen35PartialRoPE,
    Qwen35VSAAttn,
    preserve_visual_tower,
    reattach_visual_tower,
    UniversalSoftmaxAttn,
    UniversalVSAAttn,
    UniversalBlock,
    UniversalVSALM,
)


def _purge_zombie_vram():
    for attr in ("last_traceback", "last_value", "last_type", "last_exc"):
        if hasattr(sys, attr):
            try:
                tb = getattr(sys, attr)
                if attr == "last_traceback" and tb is not None:
                    traceback.clear_frames(tb)
                elif attr in ("last_value", "last_exc") and hasattr(tb, "__traceback__") and tb.__traceback__ is not None:
                    traceback.clear_frames(tb.__traceback__)
                setattr(sys, attr, None)
            except Exception:
                pass
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def convert_donor_model(
    donor_id_or_dir: str,
    out_ckpt: str,
    *,
    hybrid_every: int = 4,
    max_T: int = 2048,
    chunk_size: int = 32,
    gamma_cap: float = 0.9995,
    hf_token: str | None = None,
    visual_sidecar_out: str | None = None,
    swa_window: int = 64,
) -> str:
    t0 = time.perf_counter()
    donor_dir = fetch_any_donor(donor_id_or_dir, hf_token=hf_token)
    safe_tag = donor_id_or_dir.split("/")[-1].lower().replace("-", "_").replace(".", "_")
    vis_path = visual_sidecar_out or os.path.join(os.path.dirname(os.path.abspath(out_ckpt)), f"{safe_tag}_visual_tower.safetensors")
    n_vis, vis_mb = preserve_visual_tower(donor_dir, vis_path)

    hf_sd = load_all_safetensors(donor_dir)
    cfg = inspect_donor_config(donor_dir, hf_sd, max_T=max_T, chunk_size=chunk_size, hybrid_every=hybrid_every)
    cfg["vsa_swa_window"] = int(swa_window)
    if n_vis > 0:
        cfg["visual_sidecar_path"] = vis_path
        cfg["visual_tensors_count"] = n_vis

    L = cfg["num_layers"]
    layer_types = cfg["layer_types"]
    n_soft = sum(1 for t in layer_types if t == "softmax")
    n_vsa = sum(1 for t in layer_types if t == "vsa")
    n_gdn = sum(1 for t in layer_types if t == "gated_deltanet")

    print("=" * 80)
    print(f"UNIVERSAL VSA-LM CONVERTER: {donor_id_or_dir}")
    print("=" * 80)
    print(f"  Detected family     : {cfg['donor_arch']}  (tie_emb={cfg['tie_word_embeddings']}, qkv_bias={cfg['qkv_bias']}, max_T={max_T})")
    if n_vis > 0:
        print(f"  Vision Tower        : Preserved {n_vis} tensors ({vis_mb:.1f} MB) -> {vis_path}")
    print(f"  Target layout       : {n_vsa} VSA + {n_soft} Softmax + {n_gdn} Native GatedDeltaNet layers")
    print(f"  VSA sliding window  : {swa_window} tokens + sink (0 = off)")

    stu = UniversalVSALM(**cfg)
    sd = map_any_hf_to_student(hf_sd, cfg, hybrid_every)
    del hf_sd

    H, HD, D = cfg["vsa_num_heads"], cfg["vsa_head_dim"], cfg["dim"]
    dtype = sd["embed_tokens.weight"].dtype
    max_h = min(float(max(256, max_T)), 1.0 / max(1e-4, 1.0 - gamma_cap))
    horizons = torch.logspace(math.log10(4.0), math.log10(max_h), H)
    g_init = (1.0 - 1.0 / horizons).clamp(0.50, gamma_cap)
    sig = (g_init / 0.9999).clamp(1e-6, 1.0 - 1e-6)
    lg_init = torch.log(sig / (1.0 - sig)).float()

    for li, ltype in enumerate(layer_types):
        if ltype != "vsa":
            continue
        p = f"blocks.{li}.vsa"
        sd[f"{p}.log_gamma"] = lg_init.clone()
        sd[f"{p}.out_norm.weight"] = torch.ones(HD, dtype=torch.float32)
        eye = torch.eye(HD, dtype=torch.float32).unsqueeze(0).repeat(H, 1, 1)
        sd[f"{p}.k_head_mix"] = eye.clone()
        sd[f"{p}.v_head_mix"] = eye.clone()
        sd[f"{p}.v_self_scale"] = torch.zeros(H, dtype=torch.float32)
        sd.setdefault(f"{p}.g_proj.weight", torch.zeros(H * HD, D, dtype=dtype))
        sd[f"{p}.v_conv.weight"] = torch.zeros(H * HD, 1, 8, dtype=dtype)
        sd[f"{p}.k_conv.weight"] = torch.zeros(H * HD, 1, 4, dtype=dtype)
        sd[f"{p}.local_q_gate"] = torch.zeros(H, HD, dtype=torch.float32)
        sd[f"{p}.q_relu_mix"] = torch.zeros(H, 1, 1, dtype=torch.float32)
        sd[f"{p}.k_relu_mix"] = torch.zeros(H, 1, 1, dtype=torch.float32)
        if swa_window > 0:
            sd[f"{p}.swa_mix_logit"] = torch.full((H,), 2.0, dtype=torch.float32)

    stu.load_state_dict(sd, strict=True)
    n_params = sum(p.numel() for p in stu.parameters())
    tmp = out_ckpt + ".tmp"
    torch.save({
        "model": stu.state_dict(),
        "step": 0,
        "config": stu.config,
        "injection": {
            "donor": donor_id_or_dir,
            "converter": "vsa_lm.converter v12-code-math",
            "visual_sidecar_path": vis_path if n_vis > 0 else None,
        },
    }, tmp)
    os.replace(tmp, out_ckpt)
    print(f"  Strict weight load  : PASS ({n_params/1e9:.3f}B text-model params)")
    print(f"  Saved VSA student   : {out_ckpt}  ({time.perf_counter() - t0:.1f}s)")
    print("=" * 80)
    return out_ckpt


def _format_conversation_donor_native(tok, msgs: list[dict]) -> str:
    norm_m = []
    for m in msgs:
        raw_role = str(m.get("from", m.get("role", "user"))).lower()
        r = "system" if raw_role in ("system", "sys") else ("user" if raw_role in ("human", "user") else "assistant")
        c = str(m.get("value", m.get("content", "")))
        c = c.replace("<|begin_of_thought|>", "<think>\n").replace("<|end_of_thought|>", "\n</think>\n").replace("<|begin_of_solution|>", "").replace("<|end_of_solution|>", "")
        norm_m.append({"role": r, "content": c})

    has_cot = any("<think>" in m["content"] and m["role"] == "assistant" for m in norm_m)
    if not has_cot:
        try:
            return tok.apply_chat_template(norm_m, tokenize=False)
        except Exception:
            return "\n".join(f"{x['role']}: {x['content']}" for x in norm_m)

    eos = tok.eos_token or ""
    parts, history = [], []
    for m in norm_m:
        if m["role"] != "assistant":
            history.append(m)
        else:
            try:
                prefix = tok.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
            except Exception:
                prefix = "\n".join(f"{x['role']}: {x['content']}" for x in history) + "\nassistant: "
            content = m["content"]
            if prefix.rstrip().endswith("<think>") and content.lstrip().startswith("<think>"):
                content = content.lstrip()[len("<think>"):].lstrip("\n")
            parts.append(prefix + content + eos)
            history = []
    return "\n".join(parts) if parts else "\n".join(f"{x['role']}: {x['content']}" for x in norm_m)


def _build_reasoning_and_eval_caches(work: str, donor_dir: str, donor_repo: str, target_tokens: int, max_T: int = 2048):
    """P3 Fix: Mixes 40% Math/CoT (Bespoke-Stratos), 30% Code (Evol-Instruct-Code / Magicoder),
    and 30% Chat/Web (UltraChat + FineWeb-Edu) so VSA layers heal on Python code & long math chains!
    Strictly disjoint from evaluation benchmarks (GSM8K test, HumanEval, MBPP test, MMLU, HellaSwag).
    """
    import pyarrow.parquet as pq
    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer

    safe_tag = donor_repo.split("/")[-1].lower().replace("-", "_").replace(".", "_")
    train_cache = os.path.join(work, f"{safe_tag}_v12_code_math_train.pt")
    eval_cache = os.path.join(work, f"{safe_tag}_v12_code_math_eval.pt")

    if os.path.isfile(train_cache) and os.path.isfile(eval_cache):
        tr_t = torch.load(train_cache, weights_only=True)
        ev_t = torch.load(eval_cache, weights_only=True)
        if len(tr_t) >= int(target_tokens * 0.8) and len(ev_t) >= 500_000:
            print(f"[vsa-data] Loaded v12 Code+Math+Chat cache ({len(tr_t)/1e6:.1f}M) & eval cache ({len(ev_t)/1e6:.2f}M)")
            return tr_t, ev_t

    tok = AutoTokenizer.from_pretrained(donor_dir)
    eos_id = tok.eos_token_id if isinstance(tok.eos_token_id, int) else (tok.eos_token_id[0] if tok.eos_token_id else 0)
    bos_id = tok.bos_token_id if isinstance(tok.bos_token_id, int) else eos_id
    api = HfApi()
    t0 = time.perf_counter()

    def _tokenize_docs(docs: list[str], budget: int) -> tuple[array, int]:
        buf = array("i")
        for b_i in range(0, len(docs), 128):
            for ids in tok(docs[b_i : b_i + 128], add_special_tokens=False)["input_ids"]:
                if bos_id is not None:
                    buf.append(int(bos_id))
                buf.extend(ids)
                buf.append(int(eos_id))
            if len(buf) >= budget:
                break
        return buf, len(buf)

    cot_docs, chat_docs, code_docs = [], [], []
    for ds_id, target_list, max_f in (("bespokelabs/Bespoke-Stratos-17k", cot_docs, 4), ("HuggingFaceH4/ultrachat_200k", chat_docs, 4)):
        for rev in ("main", "refs/convert/parquet"):
            try:
                files = [f for f in api.list_repo_files(ds_id, revision=rev, repo_type="dataset") if f.endswith(".parquet") and "train" in f]
                if not files:
                    files = [f for f in api.list_repo_files(ds_id, revision=rev, repo_type="dataset") if f.endswith(".parquet")]
                for f in files[:max_f]:
                    lp = hf_hub_download(repo_id=ds_id, filename=f, revision=rev, repo_type="dataset")
                    for row in pq.read_table(lp).to_pylist():
                        msgs = row.get("messages") or row.get("conversations")
                        if isinstance(msgs, list):
                            target_list.append(_format_conversation_donor_native(tok, msgs))
                if target_list:
                    break
            except Exception:
                continue

    # Load dedicated Python/Code instruction datasets (strictly excluding HumanEval / MBPP test sets)
    for code_ds_id in ("ise-uiuc/Magicoder-OSS-Instruct-75K", "nickrosh/Evol-Instruct-Code-80k-v1", "iamtarun/python_code_instructions_18k_alpaca"):
        try:
            cds = load_dataset(code_ds_id, split="train", streaming=True)
            for ex in cds:
                q_c = str(ex.get("problem") or ex.get("instruction") or ex.get("prompt") or "")
                a_c = str(ex.get("solution") or ex.get("output") or ex.get("response") or "")
                if q_c and a_c:
                    code_docs.append(_format_conversation_donor_native(tok, [{"role": "user", "content": q_c}, {"role": "assistant", "content": a_c}]))
                if len(code_docs) >= 25_000:
                    break
            if len(code_docs) >= 10_000:
                break
        except Exception:
            continue

    rng = random.Random(7777)
    rng.shuffle(cot_docs)
    rng.shuffle(code_docs)
    rng.shuffle(chat_docs)

    eval_cot_cut = max(50, len(cot_docs) // 20)
    eval_code_cut = max(100, len(code_docs) // 20)
    eval_chat_cut = max(150, len(chat_docs) // 20)

    ev_cot_buf, _ = _tokenize_docs(cot_docs[:eval_cot_cut], 350_000)
    ev_code_buf, _ = _tokenize_docs(code_docs[:eval_code_cut], 350_000)
    ev_chat_buf, _ = _tokenize_docs(chat_docs[:eval_chat_cut], 350_000)
    eval_t = torch.cat([
        torch.frombuffer(ev_cot_buf, dtype=torch.int32).clone(),
        torch.frombuffer(ev_code_buf, dtype=torch.int32).clone(),
        torch.frombuffer(ev_chat_buf, dtype=torch.int32).clone(),
    ])

    cot_budget = int(target_tokens * 0.40)
    code_budget = int(target_tokens * 0.35)
    chat_budget = max(0, target_tokens - cot_budget - code_budget)

    tr_cot_buf, n_tr_cot = _tokenize_docs(cot_docs[eval_cot_cut:], cot_budget)
    tr_code_buf, n_tr_code = _tokenize_docs(code_docs[eval_code_cut:], code_budget)
    tr_chat_buf, n_tr_chat = _tokenize_docs(chat_docs[eval_chat_cut:], chat_budget)

    T_chunk = max_T + 1
    chunks = []
    for buf in (tr_cot_buf, tr_code_buf, tr_chat_buf):
        t_arr = torch.frombuffer(buf, dtype=torch.int32).clone()
        n_c = len(t_arr) // T_chunk
        if n_c > 0:
            chunks.append(t_arr[: n_c * T_chunk].view(n_c, T_chunk))
    all_chunks = torch.cat(chunks, dim=0)
    perm = torch.randperm(all_chunks.size(0), generator=torch.Generator().manual_seed(7777))
    train_t = all_chunks[perm].reshape(-1)[:target_tokens].contiguous()

    torch.save(train_t, train_cache)
    torch.save(eval_t, eval_cache)
    tot_tr = max(1, n_tr_cot + n_tr_code + n_tr_chat)
    print(
        f"[vsa-data] Token mix -> Train: {len(train_t)/1e6:.1f}M "
        f"(CoT/Math: {n_tr_cot/1e6:.1f}M [{n_tr_cot/tot_tr*100:.0f}%], "
        f"Code: {n_tr_code/1e6:.1f}M [{n_tr_code/tot_tr*100:.0f}%], "
        f"Chat: {n_tr_chat/1e6:.1f}M [{n_tr_chat/tot_tr*100:.0f}%]) | "
        f"Held-out Eval: {len(eval_t)/1e6:.2f}M in {time.perf_counter()-t0:.1f}s!",
        flush=True,
    )
    return train_t, eval_t


def _load_donor_teacher(donor_repo: str, donor_dir: str, stu_cfg: dict, device: torch.device):
    if stu_cfg.get("has_qwen35_lin", False):
        import transformers
        last_err = None
        for cls_name in ("Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration", "AutoModelForCausalLM", "AutoModelForImageTextToText"):
            cls = getattr(transformers, cls_name, None)
            if cls is not None:
                try:
                    m = cls.from_pretrained(donor_dir, dtype=torch.bfloat16)
                    if hasattr(m, "model") and hasattr(m.model, "visual"):
                        del m.model.visual
                    elif hasattr(m, "visual"):
                        del m.visual
                    m = m.to(device, dtype=torch.bfloat16).eval()
                    for p in m.parameters():
                        p.requires_grad_(False)
                    return m, True
                except Exception as e:
                    last_err = e
                    continue
        raise RuntimeError(f"Failed to load native Qwen3.5 teacher from {donor_dir}: {last_err}")

    hf_sd = load_all_safetensors(donor_dir)
    rep_cfg = dict(stu_cfg, hybrid_every=1, layer_types=["softmax"] * stu_cfg["num_layers"], vsa_untie_gqa=False, vsa_local_conv=False)
    donor = UniversalVSALM(**rep_cfg)
    donor.load_state_dict(map_any_hf_to_student(hf_sd, rep_cfg, 1), strict=True)
    del hf_sd
    donor = donor.to(device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32).eval()
    for p in donor.parameters():
        p.requires_grad_(False)
    return donor, False


def _forward_hf_teacher_body(donor, xm: torch.Tensor, num_layers: int, hybrid_every: int):
    inner = getattr(donor, "model", donor)
    if hasattr(inner, "language_model"):
        inner = inner.language_model
    out = inner(input_ids=xm, output_hidden_states=True, use_cache=False)
    h_teacher = (out.last_hidden_state if hasattr(out, "last_hidden_state") else out.hidden_states[-1]).detach()
    t_mids = [
        out.hidden_states[li + 1].detach()
        for li in range(num_layers - 1)
        if (li + 1) % hybrid_every == 0
    ]
    del out
    return h_teacher, t_mids


@torch.no_grad()
def _eval_reasoning_metrics(stu, donor, is_hf_donor: bool, eval_tokens: torch.Tensor,
                            device: torch.device, n_tokens: int = 131_072, bs: int = 2,
                            bos_id: int | None = None) -> tuple[float, float, float, float]:
    stu.eval()
    T, V = stu.config["max_T"], stu.config["vocab_size"]
    avail = len(eval_tokens) - 1
    n_b = max(1, min(n_tokens // (bs * T), avail // (bs * T)))
    ar = torch.arange(T)
    nll_s, nll_d, kl_sum, top1_ok, cnt = 0.0, 0.0, 0.0, 0, 0
    w_lm_s = stu.lm_head.weight
    w_lm_t = donor.get_output_embeddings().weight if is_hf_donor else donor.lm_head.weight

    for i in range(n_b):
        s = i * bs * T
        base = s + torch.arange(bs)[:, None] * T + ar[None, :]
        x, y = eval_tokens[base].long().to(device), eval_tokens[base + 1].long().to(device)
        if bos_id is not None:
            x[:, 0] = int(bos_id)
        with torch.autocast("cuda", torch.bfloat16, enabled=(device.type == "cuda")):
            hs = stu(x)
            hd, _ = _forward_hf_teacher_body(donor, x, stu.config["num_layers"], stu.config["hybrid_every"]) if is_hf_donor else (donor(x), None)
        for c in range(bs):
            slog = F.linear(hs[c : c + 1].float(), w_lm_s.float())
            dlog = F.linear(hd[c : c + 1].float(), w_lm_t.float())
            yv = y[c : c + 1].reshape(-1)
            nll_s += F.cross_entropy(slog.reshape(-1, V), yv, reduction="sum").item()
            nll_d += F.cross_entropy(dlog.reshape(-1, V), yv, reduction="sum").item()
            lsm_s, lsm_d = F.log_softmax(slog, dim=-1), F.log_softmax(dlog, dim=-1)
            p_d = lsm_d.exp()
            kl_sum += float((p_d * (lsm_d - lsm_s)).sum().item())
            top1_ok += int((slog.argmax(dim=-1) == dlog.argmax(dim=-1)).sum().item())
            cnt += yv.numel()
            del slog, dlog, lsm_s, lsm_d, p_d
        del hs, hd, x, y

    return math.exp(nll_s / max(1, cnt)), math.exp(nll_d / max(1, cnt)), kl_sum / max(1, cnt), (top1_ok / max(1, cnt)) * 100.0


def _attn_transfer_loss(a_hat: torch.Tensor, a_tgt: torch.Tensor) -> torch.Tensor:
    a_hat, a_tgt = a_hat.float(), a_tgt.float()
    rel_mse = F.mse_loss(a_hat, a_tgt) / a_tgt.pow(2).mean().clamp_min(1e-5)
    return rel_mse + (1.0 - F.cosine_similarity(a_hat, a_tgt, dim=-1).mean())


def _stage1_layerwise_attention_transfer(stu, donor, is_hf_donor: bool, train_tokens: torch.Tensor,
                                         device: torch.device, stage1_tokens: int = 16_000_000, bs: int = 4,
                                         on_policy: bool = False):
    """P3 Step 1 Fix: True Per-Layer Attention-Output Transfer!
    Feeds the exact same normed input `h_norm = donor.blocks[li].norm1(x)` into both `donor.blocks[li].attn`
    and `stu.blocks[li].vsa` and minimizes normalized MSE + Cosine distance on the attention output itself!
    """
    if stage1_tokens <= 0:
        return
    T = min(1024, int(stu.config["max_T"]))
    steps = max(1, stage1_tokens // (bs * T))
    L = int(stu.config["num_layers"])
    layer_types = stu.config.get("layer_types") or []
    vsa_indices = [li for li in range(L) if layer_types[li] == "vsa" or (not layer_types and hasattr(stu.blocks[li], "vsa"))]
    if not vsa_indices:
        return
    vsa_set = set(vsa_indices)
    # On-policy needs the teacher's attention modules callable on arbitrary inputs (native replica donor only).
    # The first quarter stays teacher-forced so early VSA layers are roughly fitted before their outputs drive later ones.
    if on_policy and is_hf_donor:
        print("  [Stage-1] on-policy mode needs a native donor replica; HF donor -> teacher-forced only.", flush=True)
    on_policy_from = steps // 4 if (on_policy and not is_hf_donor) else None

    for p in stu.parameters():
        p.requires_grad_(False)
    vsa_params = []
    for li in vsa_indices:
        for p in stu.blocks[li].vsa.parameters():
            p.data = p.data.to(device, dtype=torch.float32)
            p.requires_grad_(True)
            vsa_params.append(p)

    opt = torch.optim.AdamW(vsa_params, lr=1e-3, weight_decay=0.0, fused=(device.type == "cuda"))
    n_win = max(1, (len(train_tokens) - 1) // T)
    gen = torch.Generator().manual_seed(101)
    ar = torch.arange(T)
    t0 = time.perf_counter()
    print(f"  [Stage-1 Direct Attention Transfer] {stage1_tokens/1e6:.1f}M tokens ({steps} steps) across {len(vsa_indices)} VSA layers...", flush=True)

    stu.train()
    for s in range(steps):
        idx = torch.randint(0, n_win, (bs,), generator=gen)
        x = train_tokens[idx[:, None] * T + ar[None, :]].long().to(device)
        opt.zero_grad(set_to_none=True)

        if on_policy_from is not None and s == on_policy_from:
            print(f"    [Stage-1] step {s+1}: switching to on-policy inputs (student's own hidden states)", flush=True)
        total_loss = torch.zeros((), device=device, dtype=torch.float32)

        if on_policy_from is not None and s >= on_policy_from:
            # On-policy: walk the STUDENT's residual stream. At each VSA layer the target is the teacher's
            # attention applied to the student's own normed input, so the layer learns to imitate the teacher
            # on the drifted inputs it will actually see, instead of on teacher inputs it never sees after
            # the earlier VSA layers have made small errors. The stream itself carries no gradient.
            with torch.autocast("cuda", torch.bfloat16, enabled=(device.type == "cuda")):
                with torch.no_grad():
                    xs = stu.embed_tokens(x)
                for li, blk in enumerate(stu.blocks):
                    with torch.no_grad():
                        h_n = blk.norm1(xs)
                    if li in vsa_set:
                        with torch.no_grad():
                            a_tgt = donor.blocks[li].attn(h_n)
                        a_hat = blk.vsa(h_n)
                        total_loss = total_loss + _attn_transfer_loss(a_hat, a_tgt) / len(vsa_indices)
                        a_out = a_hat.detach()
                    else:
                        with torch.no_grad():
                            a_out = blk.attn(h_n)
                    with torch.no_grad():
                        xs = xs + a_out
                        xs = xs + blk.mlp(blk.norm2(xs))
            del xs
        else:
            # Teacher-forced: capture normed inputs and exact attention outputs from the teacher's stream.
            norm_inputs, attn_targets = {}, {}
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=(device.type == "cuda")):
                if not is_hf_donor:
                    xt = donor.embed_tokens(x)
                    for li, blk in enumerate(donor.blocks):
                        h_n = blk.norm1(xt)
                        a_out = blk.attn(h_n) if hasattr(blk, "attn") else blk.vsa(h_n)
                        if li in vsa_set:
                            norm_inputs[li] = h_n.detach()
                            attn_targets[li] = a_out.detach()
                        xt = xt + a_out
                        xt = xt + blk.mlp(blk.norm2(xt))
                else:
                    inner = getattr(getattr(donor, "model", donor), "language_model", getattr(donor, "model", donor))
                    out = inner(input_ids=x, output_hidden_states=True, use_cache=False)
                    for li in vsa_indices:
                        if li < L - 1:
                            norm_inputs[li] = stu.blocks[li].norm1(out.hidden_states[li].detach())
                            attn_targets[li] = (out.hidden_states[li + 1] - out.hidden_states[li]).detach()
                    del out

            with torch.autocast("cuda", torch.bfloat16, enabled=(device.type == "cuda")):
                for li, h_n in norm_inputs.items():
                    a_hat = stu.blocks[li].vsa(h_n)
                    total_loss = total_loss + _attn_transfer_loss(a_hat, attn_targets[li]) / max(1, len(norm_inputs))
            del norm_inputs, attn_targets

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(vsa_params, 1.0)
        opt.step()
        if (s + 1) in (1, max(1, steps // 4), max(1, steps // 2), steps):
            print(f"    [Stage-1] step {s+1:4d}/{steps} | attn_rel_mse+cos = {float(total_loss.item()):.4f} ({time.perf_counter()-t0:.1f}s)", flush=True)


def heal_universal_model(
    ckpt_path: str,
    *,
    donor_id: str | None = None,
    work: str = ".",
    out_prefix: str = "vsa_healed",
    tokens_budget: int = 80_000_000,
    data_tokens: int = 100_000_000,
    stage1_tokens: int = 16_000_000,
    bs: int = 8,
    lr: float = 1.5e-4,
    warmup: int = 150,
    kd: float = 0.85,
    eval_every: int = 250,
    save_every: int = 250,
    seed: int = 7777,
    hf_repo: str | None = None,
    highspeed: bool = False,
    resume: bool = False,
    swa_window: int | None = None,
    freeze_mlp: bool = False,
    stage1_onpolicy: bool = False,
):
    _purge_zombie_vram()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # The KD head runs three [tokens x 1536 x 151936] matmuls per step in fp32; without TF32 they cost
    # roughly as much as the whole bf16 body. train.py already enables this; healing never did.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out_ckpt = os.path.join(work, f"{out_prefix}_latest.pth")
    opt_ckpt = os.path.join(work, f"{out_prefix}_optim.pth")
    active_ckpt = out_ckpt if (resume and os.path.isfile(out_ckpt)) else ckpt_path

    ck = torch.load(active_ckpt, map_location="cpu", weights_only=False)
    stu_cfg = dict(ck["config"])
    # Sliding-window branch: explicit value wins; checkpoints converted before it existed get the default 64.
    if swa_window is not None:
        stu_cfg["vsa_swa_window"] = int(swa_window)
    else:
        stu_cfg.setdefault("vsa_swa_window", 64)
    start_step = int(ck.get("step", 0)) if resume else 0
    donor_repo = donor_id or (ck.get("injection") or {}).get("donor")
    if not donor_repo:
        raise ValueError("Pass --donor <repo_id> or use a checkpoint created by `python -m vsa_lm convert`")

    stu = UniversalVSALM(**stu_cfg)
    state_in = {k: v for k, v in ck["model"].items() if "pos_code" not in k and "head_pos_delta" not in k}
    missing, unexpected = stu.load_state_dict(state_in, strict=False)
    allowed_missing = {"rope_mix", "v_conv", "k_conv", "local_q_gate", "q_relu_mix", "k_relu_mix", "out_norm", "swa_mix"}
    real_missing = [m for m in missing if not any(a in m for a in allowed_missing)]
    if real_missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch! Missing: {real_missing[:10]}, Unexpected: {unexpected[:10]}")

    vsa_conv, vsa_core, vsa_proj, mlp_and_norm = [], [], [], []
    layer_types = stu_cfg.get("layer_types") or []
    for n_, p_ in stu.named_parameters():
        li = int(n_.split(".")[1]) if n_.startswith("blocks.") else -1
        is_vsa_blk = (li >= 0 and (not layer_types or layer_types[li] == "vsa"))
        if any(k in n_ for k in ("v_conv", "k_conv", "local_q_gate", "q_relu_mix", "k_relu_mix", "rope_mix")):
            p_.requires_grad_(True)
            vsa_conv.append(p_)
        elif any(k in n_ for k in ("log_gamma", "k_head_mix", "v_head_mix", "v_self_scale", "out_norm", "swa_mix")):
            p_.requires_grad_(True)
            vsa_core.append(p_)
        elif ".vsa." in n_ and is_vsa_blk:
            p_.requires_grad_(True)
            vsa_proj.append(p_)
        elif is_vsa_blk and not freeze_mlp and (".mlp." in n_ or ".norm1." in n_ or ".norm2." in n_):
            p_.requires_grad_(True)
            mlp_and_norm.append(p_)
        else:
            p_.requires_grad_(False)

    for n_, p_ in stu.named_parameters():
        p_.data = p_.data.to(device, dtype=(torch.float32 if p_.requires_grad else (torch.bfloat16 if device.type == "cuda" else torch.float32)))
    for b in stu.buffers():
        b.data = b.data.to(device)

    n_params = sum(p.numel() for p in stu.parameters())
    T = int(stu_cfg["max_T"])
    vram_gb = (torch.cuda.get_device_properties(0).total_memory / 1024**3) if device.type == "cuda" else 16.0

    # Frozen MLPs drop ~14 GB of fp32 master weights + Adam moments, so no-checkpoint mode fits on smaller GPUs.
    if highspeed and vram_gb >= (32.0 if freeze_mlp else 45.0) and n_params < 3_000_000_000:
        micro_bs = 4 if T <= 1024 else 2
        use_ckpt = False
        mode_str = f"HIGHSPEED (use_ckpt=False, micro_bs={micro_bs}, T={T})"
    else:
        micro_bs = 2
        use_ckpt = True
        mode_str = f"LOW-VRAM (use_ckpt=True, micro_bs={micro_bs}, T={T})"

    donor_dir = fetch_any_donor(donor_repo)
    donor, is_hf_donor = _load_donor_teacher(donor_repo, donor_dir, stu_cfg, device)

    if is_hf_donor:
        t_in_emb, t_out_emb = donor.get_input_embeddings(), donor.get_output_embeddings()
        if t_in_emb is not None and t_in_emb.weight.shape == stu.embed_tokens.weight.shape and torch.equal(t_in_emb.weight, stu.embed_tokens.weight):
            t_in_emb.weight = stu.embed_tokens.weight
        if t_out_emb is not None and t_out_emb.weight.shape == stu.lm_head.weight.shape and torch.equal(t_out_emb.weight, stu.lm_head.weight):
            t_out_emb.weight = stu.lm_head.weight
    else:
        if torch.equal(donor.embed_tokens.weight, stu.embed_tokens.weight):
            donor.embed_tokens.weight = stu.embed_tokens.weight
        if torch.equal(donor.lm_head.weight, stu.lm_head.weight):
            donor.lm_head.weight = stu.lm_head.weight
    _purge_zombie_vram()

    w_lm_t = donor.get_output_embeddings().weight if is_hf_donor else donor.lm_head.weight
    train_tokens, eval_tokens = _build_reasoning_and_eval_caches(
        work, donor_dir, donor_repo, max(data_tokens, int(tokens_budget * 1.05)), max_T=T
    )

    if start_step == 0 and stage1_tokens > 0:
        _stage1_layerwise_attention_transfer(stu, donor, is_hf_donor, train_tokens, device, stage1_tokens=stage1_tokens, bs=max(2, micro_bs), on_policy=stage1_onpolicy)
        for p_ in vsa_conv + vsa_core + vsa_proj + mlp_and_norm:
            p_.data = p_.data.to(device, dtype=torch.float32)
            p_.requires_grad_(True)
        _purge_zombie_vram()

    start_ppl, donor_ppl, start_kl, start_top1 = _eval_reasoning_metrics(stu, donor, is_hf_donor, eval_tokens, device, bs=2)
    total_steps = max(start_step + 1, math.ceil(tokens_budget / (bs * T)))

    print("=" * 84)
    print(f"VSA-FASTDISTILL 2-STAGE HEALER: {donor_repo} (kd={kd}, ce={1.0-kd:.2f})")
    print("=" * 84)
    print(f"  Memory Mode    : {mode_str} ({n_params/1e9:.2f}B Student, FP32 Master Weights)")
    print(f"  Held-Out Eval  : KL={start_kl:.4f} nats | Top1={start_top1:.2f}% | Student PPL={start_ppl:.2f} / Donor PPL={donor_ppl:.2f} ({start_ppl/donor_ppl:.3f}x)")
    print(f"  Stage-2 Budget : {tokens_budget/1e6:.1f}M Code+Math+Chat tokens ({total_steps} steps, resume={start_step})")

    opt_groups = [
        {"params": vsa_conv,     "weight_decay": 0.0, "lr_scale": 5.0},
        {"params": vsa_core,     "weight_decay": 0.0, "lr_scale": 5.0},
        {"params": vsa_proj,     "weight_decay": 0.0, "lr_scale": 1.0},
        {"params": mlp_and_norm, "weight_decay": 0.0, "lr_scale": 0.08},
    ]
    opt_groups = [g for g in opt_groups if g["params"]]
    trainable = vsa_conv + vsa_core + vsa_proj + mlp_and_norm
    print(f"  Trainable      : {sum(p.numel() for p in trainable)/1e6:.1f}M params (MLPs {'frozen' if freeze_mlp else 'trained @0.08x lr'})")
    opt = torch.optim.AdamW(opt_groups, lr=lr, betas=(0.9, 0.95), fused=(device.type == "cuda"))
    if resume and os.path.isfile(opt_ckpt):
        try:
            opt.load_state_dict(torch.load(opt_ckpt, map_location=device, weights_only=False))
            print("  [Resume] Restored optimizer moments from sidecar.")
        except Exception as e:
            print(f"  [Resume] Optimizer sidecar skipped: {e}")

    stable_end = int(total_steps * 0.65)
    def set_step_lr(s: int):
        if s < warmup:
            m = (s + 1) / max(1, warmup)
        elif s < stable_end:
            m = 1.0
        else:
            prog = (s - stable_end) / max(1, total_steps - stable_end)
            m = 0.05 + 0.475 * (1.0 + math.cos(math.pi * min(1.0, prog)))
        for g in opt.param_groups:
            g["lr"] = lr * m * g.get("lr_scale", 1.0)

    n_win = max(1, (len(train_tokens) - 1) // T)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(start_step):
        torch.randint(0, n_win, (bs,), generator=gen)
    ar = torch.arange(T)

    _purge_zombie_vram()
    stu.train()
    set_step_lr(start_step)
    t_window = time.perf_counter()
    step_window = start_step
    tok_slice = 512
    L_num = int(stu_cfg["num_layers"])

    for step in range(start_step, total_steps):
        idx = torch.randint(0, n_win, (bs,), generator=gen)
        base = idx[:, None] * T + ar[None, :]
        x, y = train_tokens[base].long(), train_tokens[base + 1].long()
        opt.zero_grad(set_to_none=True)
        loss_gpu = torch.zeros((), device=device, dtype=torch.float32)

        for lo in range(0, bs, micro_bs):
            hi = min(lo + micro_bs, bs)
            xm, ym = x[lo:hi].to(device, non_blocking=True), y[lo:hi].to(device, non_blocking=True)
            w = (hi - lo) / bs

            with torch.autocast("cuda", torch.bfloat16, enabled=(device.type == "cuda")):
                with torch.no_grad():
                    if is_hf_donor:
                        h_teacher, t_mids = _forward_hf_teacher_body(
                            donor, xm, L_num, stu_cfg["hybrid_every"]
                        )
                    else:
                        xt = donor.embed_tokens(xm)
                        t_mids = []
                        for li, blk in enumerate(donor.blocks):
                            xt = blk(xt)
                            if li < L_num - 1 and (li + 1) % stu_cfg["hybrid_every"] == 0:
                                t_mids.append(xt)
                        h_teacher = donor.norm(xt)

                xs = stu.embed_tokens(xm)
                s_mids = []
                for li, blk in enumerate(stu.blocks):
                    xs = torch.utils.checkpoint.checkpoint(blk, xs, use_reentrant=False) if use_ckpt else blk(xs)
                    if li < L_num - 1 and (li + 1) % stu_cfg["hybrid_every"] == 0:
                        s_mids.append(xs)
                h_student = stu.norm(xs)

                mid_loss = sum(
                    1.0 - F.cosine_similarity(sm.float(), tm.float(), dim=-1).mean()
                    for sm, tm in zip(s_mids, t_mids)
                ) / max(1, len(s_mids)) if s_mids else torch.zeros((), device=device)
                final_h_loss = 1.0 - F.cosine_similarity(h_student.float(), h_teacher.float(), dim=-1).mean()
                del s_mids, t_mids

            B_m, T_m, D_m = h_student.shape
            N_m = B_m * T_m
            hs_2d, ht_2d, ym_1d = h_student.reshape(N_m, D_m), h_teacher.reshape(N_m, D_m), ym.reshape(N_m)
            dh_2d = torch.empty_like(hs_2d)
            w_lm_s_f32 = stu.lm_head.weight.float()
            w_lm_t_f32 = w_lm_t.float()

            with torch.no_grad():
                for t_lo in range(0, N_m, tok_slice):
                    t_hi = min(t_lo + tok_slice, N_m)
                    hs_c = hs_2d[t_lo:t_hi].float()
                    ht_c = ht_2d[t_lo:t_hi].float()
                    ym_c = ym_1d[t_lo:t_hi]

                    slog = F.linear(hs_c, w_lm_s_f32)
                    lsm_s = F.log_softmax(slog, dim=-1)
                    prob_s = lsm_s.exp()

                    rlog = F.linear(ht_c, w_lm_t_f32)
                    rlsm = F.log_softmax(rlog, dim=-1)
                    prob_t = rlsm.exp()
                    kd_i = (prob_t * (rlsm - lsm_s)).sum()
                    del rlog, rlsm

                    if kd >= 0.999:
                        loss_gpu.add_(kd_i * (w / N_m))
                        g_logits = (prob_s - prob_t) * (w / N_m)
                    else:
                        ce_i = -lsm_s.gather(-1, ym_c.unsqueeze(-1)).sum()
                        loss_gpu.add_(((1.0 - kd) * ce_i + kd * kd_i) * (w / N_m))
                        g_logits = prob_s
                        g_logits.sub_(prob_t, alpha=kd)
                        g_logits.scatter_add_(-1, ym_c.unsqueeze(-1), torch.full_like(ym_c.unsqueeze(-1), -(1.0 - kd), dtype=torch.float32))
                        g_logits.mul_(w / N_m)

                    dh_2d[t_lo:t_hi] = (g_logits @ w_lm_s_f32).to(h_student.dtype)
                    del slog, lsm_s, prob_s, prob_t, g_logits

            del h_teacher, ht_2d, w_lm_s_f32, w_lm_t_f32
            torch.autograd.backward(
                [h_student, (mid_loss + final_h_loss) * (0.20 * w)],
                [dh_2d.view(B_m, T_m, D_m), torch.ones_like(final_h_loss)],
            )
            del h_student, hs_2d, dh_2d, mid_loss, final_h_loss

        gn_tensor = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if torch.isfinite(gn_tensor):
            opt.step()
        step_done = step + 1
        set_step_lr(step_done)

        if step_done in (start_step + 1, start_step + 5) or (step_done % 25 == 0 and step_done % eval_every != 0):
            loss_val, gn = float(loss_gpu.item()), float(gn_tensor.item())
            now = time.perf_counter()
            dt = (now - t_window) / max(1, step_done - step_window)
            t_window, step_window = now, step_done
            peak_gb = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0
            print(f"  [{out_prefix}] step {step_done:5d}/{total_steps}  kd_ce {loss_val:.4f}  gnorm {gn:.2f}  toks {step_done*bs*T/1e6:.2f}M  {dt:.2f}s/it  VRAM {peak_gb:.2f} GiB", flush=True)

        if step_done % eval_every == 0 or step_done == total_steps:
            s_ppl, d_ppl, kl_v, top1_v = _eval_reasoning_metrics(stu, donor, is_hf_donor, eval_tokens, device, bs=2)
            stu.train()
            now = time.perf_counter()
            dt = (now - t_window) / max(1, step_done - step_window)
            t_window, step_window = now, step_done
            peak_gb = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0
            print(f"  >>> [{out_prefix} EVAL] step {step_done:5d}/{total_steps}  KL {kl_v:.4f} nats  Top1 {top1_v:.2f}%  "
                  f"val_ppl {s_ppl:.2f}/{d_ppl:.2f} ({s_ppl/d_ppl:.3f}x donor)  {dt:.2f}s/it  VRAM {peak_gb:.2f} GiB", flush=True)

        if step_done % save_every == 0 or step_done == total_steps:
            tmp = out_ckpt + ".tmp"
            save_sd = {k: (v.cpu().float() if any(f in k for f in ("log_gamma", "k_head_mix", "v_head_mix", "v_self_scale", "local_q_gate", "q_relu_mix", "k_relu_mix", "out_norm", "swa_mix")) else v.cpu().to(torch.bfloat16)) for k, v in stu.state_dict().items()}
            torch.save({"model": save_sd, "step": step_done,
                        "config": stu_cfg, "injection": ck.get("injection", {})}, tmp)
            os.replace(tmp, out_ckpt)
            if step_done < total_steps:
                torch.save(opt.state_dict(), opt_ckpt)
            elif os.path.isfile(opt_ckpt):
                os.remove(opt_ckpt)
            if hf_repo and (step_done % 500 == 0 or step_done == total_steps):
                def _up(p, r):
                    try:
                        from huggingface_hub import HfApi
                        HfApi().upload_file(path_or_fileobj=p, path_in_repo=f"results/{out_prefix}/{os.path.basename(p)}", repo_id=r, repo_type="model")
                        print(f"[vsa-hf] Uploaded {os.path.basename(p)} to {r}", flush=True)
                    except Exception as e:
                        print(f"[vsa-hf] Upload skipped: {e}", flush=True)
                threading.Thread(target=_up, args=(out_ckpt, hf_repo), daemon=True).start()

    vis_sidecar = (ck.get("injection") or {}).get("visual_sidecar_path") or stu_cfg.get("visual_sidecar_path")
    if vis_sidecar and os.path.isfile(vis_sidecar):
        reattach_visual_tower(out_ckpt, vis_sidecar)
