"""Universal Donor-to-VSA Converter & Pure-KD FastDistill Healer (vsa_lm.converter)."""
from __future__ import annotations

import gc
import math
import os
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
    max_T: int = 1024,
    chunk_size: int = 32,
    gamma_cap: float = 0.992,
    hf_token: str | None = None,
    visual_sidecar_out: str | None = None,
) -> str:
    t0 = time.perf_counter()
    donor_dir = fetch_any_donor(donor_id_or_dir, hf_token=hf_token)
    safe_tag = donor_id_or_dir.split("/")[-1].lower().replace("-", "_").replace(".", "_")
    vis_path = visual_sidecar_out or os.path.join(os.path.dirname(os.path.abspath(out_ckpt)), f"{safe_tag}_visual_tower.safetensors")
    n_vis, vis_mb = preserve_visual_tower(donor_dir, vis_path)

    hf_sd = load_all_safetensors(donor_dir)
    cfg = inspect_donor_config(donor_dir, hf_sd, max_T=max_T, chunk_size=chunk_size, hybrid_every=hybrid_every)
    if n_vis > 0:
        cfg["visual_sidecar_path"] = vis_path
        cfg["visual_tensors_count"] = n_vis

    L = cfg["num_layers"]
    n_soft = (L // hybrid_every) if hybrid_every > 0 else 0
    n_vsa = L - n_soft
    print("=" * 80)
    print(f"UNIVERSAL VSA-LM CONVERTER: {donor_id_or_dir}")
    print("=" * 80)
    print(f"  Detected family     : {cfg['donor_arch']}  (qkv_bias={cfg['qkv_bias']}, qk_norm={cfg['qk_norm']}, qwen35_lin={cfg['has_qwen35_lin']}, packed_q_gate={cfg['packed_q_gate']})")
    if n_vis > 0:
        print(f"  Vision Tower        : Preserved {n_vis} tensors ({vis_mb:.1f} MB) -> {vis_path}")
    print(f"  Dimensions          : dim={cfg['dim']}, L={L}, Softmax H={cfg['num_heads']}/HKV={cfg['num_kv_heads']} (HD={cfg['head_dim']}), VSA H={cfg['vsa_num_heads']}/HKV={cfg['vsa_num_kv_heads']} (HD={cfg['vsa_head_dim']})")
    print(f"  Target layout       : {n_vsa} VSA layers + {n_soft} Softmax layers (hybrid_every={hybrid_every})")

    stu = UniversalVSALM(**cfg)
    sd = map_any_hf_to_student(hf_sd, cfg, hybrid_every)
    del hf_sd

    H, HD, D = cfg["vsa_num_heads"], cfg["vsa_head_dim"], cfg["dim"]
    dtype = sd["embed_tokens.weight"].dtype
    horizons = torch.logspace(math.log10(2.0), math.log10(1.0 / max(1e-4, 1.0 - gamma_cap)), H)
    g_init = (1.0 - 1.0 / horizons).clamp(0.50, gamma_cap)
    sig = (g_init / 0.9999).clamp(1e-6, 1.0 - 1e-6)
    lg_init = torch.log(sig / (1.0 - sig)).float()

    for li in range(L):
        if hybrid_every > 0 and (li + 1) % hybrid_every == 0:
            continue
        p = f"blocks.{li}.vsa"
        sd[f"{p}.pos_code"] = torch.ones(max_T, HD, dtype=torch.float32)
        if cfg.get("has_qwen35_lin", False) and f"{p}.A_log" in sd and f"{p}.dt_bias" in sd:
            A_log, dt_bias = sd[f"{p}.A_log"].float(), sd[f"{p}.dt_bias"].float()
            g_static = torch.exp(-A_log.exp() * F.softplus(dt_bias)).clamp(0.20, gamma_cap)
            sd[f"{p}.log_gamma"] = torch.logit(g_static / 0.9999).float()
        else:
            sd[f"{p}.log_gamma"] = lg_init.clone()

        eye = torch.eye(HD, dtype=torch.float32).unsqueeze(0).repeat(H, 1, 1)
        sd[f"{p}.k_head_mix"] = eye.clone()
        sd[f"{p}.v_head_mix"] = eye.clone()
        sd[f"{p}.head_pos_delta"] = torch.zeros(H, max_T, HD, dtype=torch.float32)
        sd[f"{p}.v_self_scale"] = torch.zeros(H, dtype=torch.float32)

        sd[f"{p}.v_conv.weight"] = torch.zeros(H * HD, 1, 8, dtype=dtype)
        sd[f"{p}.local_q_gate"] = torch.zeros(H, HD, dtype=torch.float32)
        sd[f"{p}.q_relu_mix"] = torch.zeros(H, 1, 1, dtype=torch.float32)
        sd[f"{p}.k_relu_mix"] = torch.zeros(H, 1, 1, dtype=torch.float32)
        if cfg.get("has_qwen35_lin", False):
            sd[f"{p}.rope_mix"] = torch.zeros(H, 1, 1, dtype=torch.float32)
        else:
            sd.setdefault(f"{p}.g_proj.weight", torch.zeros(H * HD, D, dtype=dtype))
            sd[f"{p}.k_conv.weight"] = torch.zeros(H * HD, 1, 4, dtype=dtype)

    stu.load_state_dict(sd, strict=True)
    n_params = sum(p.numel() for p in stu.parameters())
    tmp = out_ckpt + ".tmp"
    torch.save({
        "model": stu.state_dict(),
        "step": 0,
        "config": stu.config,
        "injection": {
            "donor": donor_id_or_dir,
            "converter": "vsa_lm.converter v9-rope-conv",
            "visual_sidecar_path": vis_path if n_vis > 0 else None,
        },
    }, tmp)
    os.replace(tmp, out_ckpt)
    print(f"  Strict weight load  : PASS ({n_params/1e9:.3f}B text-model params)")
    print(f"  Saved VSA student   : {out_ckpt}  ({time.perf_counter() - t0:.1f}s)")
    print("=" * 80)
    return out_ckpt


def _build_reasoning_and_eval_caches(work: str, donor_dir: str, donor_repo: str, target_tokens: int):
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer

    safe_tag = donor_repo.split("/")[-1].lower().replace("-", "_").replace(".", "_")
    train_cache = os.path.join(work, f"{safe_tag}_disjoint_train.pt")
    eval_cache = os.path.join(work, f"{safe_tag}_disjoint_eval.pt")

    if os.path.isfile(train_cache) and os.path.isfile(eval_cache):
        tr_t = torch.load(train_cache, weights_only=True)
        ev_t = torch.load(eval_cache, weights_only=True)
        if len(tr_t) >= int(target_tokens * 0.8):
            print(f"[vsa-data] Loaded disjoint train cache ({len(tr_t)/1e6:.1f}M) & eval cache ({len(ev_t)/1e6:.2f}M)")
            return tr_t, ev_t

    tok = AutoTokenizer.from_pretrained(donor_dir)
    eos_id = tok.eos_token_id or tok.pad_token_id or 0
    api = HfApi()
    print(f"[vsa-data] Building strictly disjoint distillation cache ({target_tokens/1e6:.0f}M tokens, 0% eval overlap)...", flush=True)
    t0 = time.perf_counter()
    all_texts = []

    for ds_id, max_f in (("HuggingFaceH4/ultrachat_200k", 4), ("bespokelabs/Bespoke-Stratos-17k", 2)):
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
                            norm_m = []
                            for m in msgs:
                                r = "user" if m.get("from", m.get("role")) in ("human", "user") else "assistant"
                                norm_m.append({"role": r, "content": str(m.get("value", m.get("content", "")))})
                            try:
                                all_texts.append(tok.apply_chat_template(norm_m, tokenize=False))
                            except Exception:
                                all_texts.append("\n".join(x["content"] for x in norm_m))
                if all_texts:
                    break
            except Exception:
                continue

    n_eval_docs = min(400, max(50, len(all_texts) // 20))
    eval_docs, train_docs = all_texts[:n_eval_docs], all_texts[n_eval_docs:]

    eval_buf = array("i")
    for ids in tok(eval_docs, add_special_tokens=False)["input_ids"]:
        eval_buf.extend(ids)
        eval_buf.append(int(eos_id))
        if len(eval_buf) >= 250_000:
            break
    eval_t = torch.frombuffer(eval_buf, dtype=torch.int32).clone()

    train_buf = array("i")
    for b_idx in range(0, len(train_docs), 128):
        for ids in tok(train_docs[b_idx : b_idx + 128], add_special_tokens=False)["input_ids"]:
            train_buf.extend(ids)
            train_buf.append(int(eos_id))
        if len(train_buf) >= target_tokens:
            break

    if len(train_buf) < target_tokens and len(train_buf) > 0:
        reps = (target_tokens // len(train_buf)) + 1
        train_t = torch.frombuffer(train_buf, dtype=torch.int32).clone().repeat(reps)[:target_tokens]
    else:
        train_t = torch.frombuffer(train_buf, dtype=torch.int32).clone()[:target_tokens]

    torch.save(train_t, train_cache)
    torch.save(eval_t, eval_cache)
    print(f"[vsa-data] Built disjoint train ({len(train_t)/1e6:.1f}M) & held-out eval ({len(eval_t)/1e6:.2f}M) in {time.perf_counter()-t0:.1f}s!", flush=True)
    return train_t, eval_t


def _load_donor_teacher(donor_repo: str, donor_dir: str, stu_cfg: dict, device: torch.device):
    if stu_cfg.get("has_qwen35_lin", False):
        import transformers
        print(f"[vsa-teacher] Loading native Qwen3.5 teacher from {donor_dir} (vision-stripped, bfloat16)...", flush=True)
        for cls_name in ("Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration", "AutoModelForCausalLM", "AutoModelForImageTextToText"):
            cls = getattr(transformers, cls_name, None)
            if cls is not None:
                try:
                    m = cls.from_pretrained(donor_dir, torch_dtype=torch.bfloat16)
                    if hasattr(m, "model") and hasattr(m.model, "visual"):
                        del m.model.visual
                    elif hasattr(m, "visual"):
                        del m.visual
                    m = m.to(device, dtype=torch.bfloat16).eval()
                    for p in m.parameters():
                        p.requires_grad_(False)
                    return m, True
                except Exception:
                    continue
    hf_sd = load_all_safetensors(donor_dir)
    rep_cfg = dict(stu_cfg, hybrid_every=1, vsa_untie_gqa=False, vsa_local_conv=False)
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
    t_mids = [out.hidden_states[li + 1].detach() for li in range(num_layers) if (li + 1) % hybrid_every == 0]
    del out
    return h_teacher, t_mids


@torch.no_grad()
def _eval_reasoning_metrics(stu, donor, is_hf_donor: bool, eval_tokens: torch.Tensor,
                            device: torch.device, n_tokens: int = 32768, bs: int = 2) -> tuple[float, float, float, float]:
    stu.eval()
    T, V = stu.config["max_T"], stu.config["vocab_size"]
    avail = len(eval_tokens) - 1
    n_b = max(1, min(n_tokens // (bs * T), avail // (bs * T)))
    ar = torch.arange(T)
    nll_s, nll_d, kl_sum, top1_ok, cnt = 0.0, 0.0, 0.0, 0, 0
    d_lm_head = donor.get_output_embeddings() if is_hf_donor else donor.lm_head

    for i in range(n_b):
        s = i * bs * T
        base = s + torch.arange(bs)[:, None] * T + ar[None, :]
        x, y = eval_tokens[base].long().to(device), eval_tokens[base + 1].long().to(device)
        with torch.autocast("cuda", torch.bfloat16, enabled=(device.type == "cuda")):
            hs = stu(x)
            hd, _ = _forward_hf_teacher_body(donor, x, stu.config["num_layers"], stu.config["hybrid_every"]) if is_hf_donor else (donor(x), None)
        for c in range(bs):
            slog = stu.lm_head(hs[c:c+1]).float()
            dlog = d_lm_head(hd[c:c+1]).float()
            yv = y[c:c+1].reshape(-1)
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


def heal_universal_model(
    ckpt_path: str,
    *,
    donor_id: str | None = None,
    work: str = ".",
    out_prefix: str = "vsa_healed",
    tokens_budget: int = 80_000_000,
    data_tokens: int = 100_000_000,
    bs: int = 16,
    lr: float = 1.5e-4,
    warmup: int = 150,
    kd: float = 1.0,
    eval_every: int = 250,
    save_every: int = 250,
    seed: int = 7777,
    hf_repo: str | None = None,
    highspeed: bool = False,
):
    _purge_zombie_vram()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_ckpt = os.path.join(work, f"{out_prefix}_latest.pth")
    active_ckpt = out_ckpt if os.path.isfile(out_ckpt) else ckpt_path

    ck = torch.load(active_ckpt, map_location="cpu", weights_only=False)
    stu_cfg = dict(ck["config"])
    start_step = int(ck.get("step", 0))
    donor_repo = donor_id or (ck.get("injection") or {}).get("donor")
    if not donor_repo:
        raise ValueError("Pass --donor <repo_id> or use a checkpoint created by `python -m vsa_lm convert`")

    stu = UniversalVSALM(**stu_cfg)
    # Use strict=False so upgraded VSA layers (e.g. rope_mix, v_conv) initialize cleanly on existing checkpoints!
    stu.load_state_dict(ck["model"], strict=False)
    fp32_keys = ("pos_code", "log_gamma", "k_head_mix", "v_head_mix", "head_pos_delta",
                 "v_self_scale", "local_q_gate", "q_relu_mix", "k_relu_mix", "rope_mix", "A_log", "dt_bias")
    for name, p in stu.named_parameters():
        if any(k in name for k in fp32_keys):
            p.data = p.data.to(device, dtype=torch.float32)
        else:
            p.data = p.data.to(device, dtype=(torch.bfloat16 if device.type == "cuda" else torch.float32))
    for b in stu.buffers():
        b.data = b.data.to(device)

    n_params = sum(p.numel() for p in stu.parameters())
    T = int(stu_cfg["max_T"])
    vram_gb = (torch.cuda.get_device_properties(0).total_memory / 1024**3) if device.type == "cuda" else 16.0

    if highspeed and vram_gb >= 45.0 and n_params < 4_600_000_000:
        micro_bs = 4 if n_params >= 3_000_000_000 else 8
        use_ckpt = False
        mode_str = f"HIGHSPEED (use_ckpt=False, micro_bs={micro_bs})"
    else:
        micro_bs = 2 if n_params >= 3_000_000_000 else (4 if vram_gb >= 20.0 else 2)
        use_ckpt = True
        mode_str = f"LOW-VRAM (use_ckpt=True, micro_bs={micro_bs})"

    donor_dir = fetch_any_donor(donor_repo)
    donor, is_hf_donor = _load_donor_teacher(donor_repo, donor_dir, stu_cfg, device)

    if is_hf_donor:
        t_in_emb, t_out_emb = donor.get_input_embeddings(), donor.get_output_embeddings()
        if t_in_emb is not None and t_in_emb.weight.shape == stu.embed_tokens.weight.shape:
            t_in_emb.weight = stu.embed_tokens.weight
        if t_out_emb is not None and t_out_emb.weight.shape == stu.lm_head.weight.shape:
            t_out_emb.weight = stu.lm_head.weight
    else:
        donor.embed_tokens.weight, donor.lm_head.weight = stu.embed_tokens.weight, stu.lm_head.weight
    _purge_zombie_vram()

    w_lm_t = donor.get_output_embeddings().weight if is_hf_donor else donor.lm_head.weight
    train_tokens, eval_tokens = _build_reasoning_and_eval_caches(
        work, donor_dir, donor_repo, max(data_tokens, int(tokens_budget * 1.05))
    )
    start_ppl, donor_ppl, start_kl, start_top1 = _eval_reasoning_metrics(stu, donor, is_hf_donor, eval_tokens, device, bs=2)
    total_steps = max(start_step + 1, math.ceil(tokens_budget / (bs * T)))

    print("=" * 84)
    print(f"VSA-FASTDISTILL PURE-TEACHER HEALER: {donor_repo} (kd={kd})")
    print("=" * 84)
    print(f"  Memory Mode    : {mode_str} ({n_params/1e9:.2f}B Student, RoPE+Conv+FrozenBackbone)")
    print(f"  Donor Eval PPL : {donor_ppl:.3f}  |  Student Start PPL: {start_ppl:.3f} ({start_ppl/donor_ppl:.3f}x) | KL: {start_kl:.3f} | Top1: {start_top1:.1f}%")
    print(f"  Budget         : {tokens_budget/1e6:.1f}M Reasoning/Chat tokens ({total_steps} steps, resume={start_step})")

    vsa_conv, vsa_core, vsa_proj = [], [], []
    for n_, p_ in stu.named_parameters():
        if any(k in n_ for k in ("v_conv", "k_conv", "conv1d", "local_q_gate", "q_relu_mix", "k_relu_mix", "rope_mix")):
            p_.requires_grad_(True)
            vsa_conv.append(p_)
        elif any(k in n_ for k in ("pos_code", "log_gamma", "k_head_mix", "v_head_mix", "head_pos_delta", "v_self_scale", "A_log", "dt_bias")):
            p_.requires_grad_(True)
            vsa_core.append(p_)
        elif ".vsa." in n_:
            p_.requires_grad_(True)
            vsa_proj.append(p_)
        else:
            p_.requires_grad_(False)

    opt_groups = [
        {"params": vsa_conv, "weight_decay": 0.0, "lr_scale": 6.0},
        {"params": vsa_core, "weight_decay": 0.0, "lr_scale": 5.0},
        {"params": vsa_proj, "weight_decay": 0.0, "lr_scale": 1.0},
    ]
    trainable = vsa_conv + vsa_core + vsa_proj
    opt = torch.optim.AdamW(opt_groups, lr=lr, betas=(0.9, 0.95), fused=(device.type == "cuda"))

    stable_end = int(total_steps * 0.65)
    def set_step_lr(s: int):
        if s < warmup:
            m = (s + 1) / max(1, warmup)
        elif s < stable_end:
            m = 1.0
        else:
            prog = (s - stable_end) / max(1, total_steps - stable_end)
            m = 0.05 + 0.475 * (1.0 + math.cos(math.pi * min(1.0, prog)))
        if start_step > 0 and s >= start_step:
            m = max(m, 0.5) * min(1.0, (s - start_step + 1) / 50.0)
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
                            donor, xm, stu_cfg["num_layers"], stu_cfg["hybrid_every"]
                        )
                    else:
                        xt = donor.embed_tokens(xm)
                        t_mids = []
                        for li, blk in enumerate(donor.blocks):
                            xt = blk(xt)
                            if (li + 1) % stu_cfg["hybrid_every"] == 0:
                                t_mids.append(xt)
                        h_teacher = donor.norm(xt)

                xs = stu.embed_tokens(xm)
                s_mids = []
                for li, blk in enumerate(stu.blocks):
                    xs = torch.utils.checkpoint.checkpoint(blk, xs, use_reentrant=False) if use_ckpt else blk(xs)
                    if (li + 1) % stu_cfg["hybrid_every"] == 0:
                        s_mids.append(xs)
                h_student = stu.norm(xs)

                mid_loss = sum(
                    1.0 - F.cosine_similarity(sm.float(), tm.float(), dim=-1).mean()
                    for sm, tm in zip(s_mids, t_mids)
                ) / max(1, len(s_mids))
                final_h_loss = 1.0 - F.cosine_similarity(h_student.float(), h_teacher.float(), dim=-1).mean()
                del s_mids, t_mids

            B_m, T_m, D_m = h_student.shape
            N_m = B_m * T_m
            hs_2d, ht_2d, ym_1d = h_student.reshape(N_m, D_m), h_teacher.reshape(N_m, D_m), ym.reshape(N_m)
            dh_2d = torch.empty_like(hs_2d)
            w_lm_s = stu.lm_head.weight

            with torch.no_grad():
                for t_lo in range(0, N_m, tok_slice):
                    t_hi = min(t_lo + tok_slice, N_m)
                    hs_c, ht_c, ym_c = hs_2d[t_lo:t_hi], ht_2d[t_lo:t_hi], ym_1d[t_lo:t_hi]

                    slog = F.linear(hs_c, w_lm_s).float()
                    lsm_s = F.log_softmax(slog, dim=-1)
                    prob_s = lsm_s.exp()

                    rlog = F.linear(ht_c, w_lm_t).float()
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

                    dh_2d[t_lo:t_hi] = (g_logits.to(h_student.dtype) @ w_lm_s)
                    del slog, lsm_s, prob_s, prob_t, g_logits

            del h_teacher, ht_2d
            torch.autograd.backward(
                [h_student, (mid_loss + final_h_loss) * (0.20 * w)],
                [dh_2d.view(B_m, T_m, D_m), torch.ones_like(mid_loss)],
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
            print(f"  [{out_prefix}] step {step_done:5d}/{total_steps}  kd_kl {loss_val:.4f}  gnorm {gn:.2f}  toks {step_done*bs*T/1e6:.2f}M  {dt:.2f}s/it  VRAM {peak_gb:.2f} GiB", flush=True)

        if step_done % eval_every == 0 or step_done == total_steps:
            s_ppl, d_ppl, kl_v, top1_v = _eval_reasoning_metrics(stu, donor, is_hf_donor, eval_tokens, device, bs=2)
            stu.train()
            now = time.perf_counter()
            dt = (now - t_window) / max(1, step_done - step_window)
            t_window, step_window = now, step_done
            peak_gb = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0
            print(f"  >>> [{out_prefix} EVAL] step {step_done:5d}/{total_steps}  val_ppl {s_ppl:.2f}/{d_ppl:.2f} ({s_ppl/d_ppl:.3f}x donor)  "
                  f"KL {kl_v:.3f} nats  Top1 {top1_v:.1f}%  {dt:.2f}s/it  VRAM {peak_gb:.2f} GiB", flush=True)

        if step_done % save_every == 0 or step_done == total_steps:
            tmp = out_ckpt + ".tmp"
            torch.save({"model": stu.state_dict(), "step": step_done,
                        "config": stu_cfg, "injection": ck.get("injection", {})}, tmp)
            os.replace(tmp, out_ckpt)
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
