"""Donor snapshot fetching, safetensor loading, config inspection, and weight mapping."""
from __future__ import annotations

import glob
import json
import os
import torch


def fetch_any_donor(repo_or_dir: str, hf_token: str | None = None) -> str:
    if os.path.isdir(repo_or_dir):
        return repo_or_dir
    from huggingface_hub import snapshot_download
    tok = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    pats = ["config.json", "*.safetensors", "model.safetensors.index.json",
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"]
    path = snapshot_download(repo_or_dir, allow_patterns=pats, token=tok)
    print(f"[vsa-convert] Donor snapshot ready: {path}")
    return path


def load_all_safetensors(donor_dir: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file
    shards = sorted(glob.glob(os.path.join(donor_dir, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"No .safetensors files found in {donor_dir}")
    sd = {}
    for sh in shards:
        for k, v in load_file(sh).items():
            if k.startswith("visual.") or k.startswith("model.visual.") or ".visual." in k:
                continue
            if k.startswith("model.language_model."):
                k = "model." + k[len("model.language_model."):]
            elif k.startswith("language_model.model."):
                k = "model." + k[len("language_model.model."):]
            elif k.startswith("language_model."):
                k = k[len("language_model."):]
            sd[k] = v
    return sd


def inspect_donor_config(donor_dir: str, hf_sd: dict[str, torch.Tensor],
                         max_T: int = 1024, chunk_size: int = 32,
                         hybrid_every: int = 4) -> dict:
    with open(os.path.join(donor_dir, "config.json")) as f:
        raw_cfg = json.load(f)
    hc = raw_cfg.get("text_config", raw_cfg)
    dim = int(hc["hidden_size"])
    num_layers = int(hc["num_hidden_layers"])
    num_heads = int(hc["num_attention_heads"])
    num_kv_heads = int(hc.get("num_key_value_heads", num_heads))
    head_dim = int(hc.get("head_dim") or (dim // num_heads))
    intermediate = int(hc["intermediate_size"])
    vocab_size = int(hc["vocab_size"])
    eps = float(hc.get("rms_norm_eps", 1e-6))
    rope_params = hc.get("rope_parameters") or {}
    rope_theta = float(hc.get("rope_theta", rope_params.get("rope_theta", 10000.0)))
    partial_rotary_factor = float(hc.get("partial_rotary_factor", rope_params.get("partial_rotary_factor", 1.0)))
    rope_scaling = hc.get("rope_scaling")
    tie_emb = bool(hc.get("tie_word_embeddings", raw_cfg.get("tie_word_embeddings", "lm_head.weight" not in hf_sd)))

    has_qkv_bias = any(k.endswith("self_attn.q_proj.bias") for k in hf_sd)
    has_qk_norm = any(k.endswith("self_attn.q_norm.weight") for k in hf_sd)
    has_qwen35_lin = any(".linear_attn.in_proj_qkv.weight" in k for k in hf_sd)
    packed_q_gate = any(
        k.endswith("self_attn.q_proj.weight") and t.shape[0] == 2 * num_heads * head_dim
        for k, t in hf_sd.items()
    )

    vsa_num_heads = int(hc.get("linear_num_value_heads", num_heads)) if has_qwen35_lin else num_heads
    vsa_num_kv_heads = int(hc.get("linear_num_key_heads", num_kv_heads)) if has_qwen35_lin else num_kv_heads
    vsa_head_dim = int(hc.get("linear_value_head_dim", min(128, head_dim))) if has_qwen35_lin else min(128, head_dim)
    if not has_qwen35_lin and head_dim > 128:
        scale_f = head_dim // 128
        vsa_num_heads, vsa_num_kv_heads, vsa_head_dim = num_heads * scale_f, num_kv_heads * scale_f, 128

    return dict(
        donor_arch=raw_cfg.get("model_type", hc.get("model_type", "unknown")),
        vocab_size=vocab_size, dim=dim, num_layers=num_layers,
        num_heads=num_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, intermediate=intermediate,
        max_T=max_T, chunk_size=chunk_size, hybrid_every=hybrid_every,
        rope_theta=rope_theta, partial_rotary_factor=partial_rotary_factor,
        rope_scaling=rope_scaling, eps=eps,
        qkv_bias=has_qkv_bias, vsa_qkv_bias=(has_qkv_bias and not has_qwen35_lin),
        qk_norm=has_qk_norm, has_qwen35_lin=has_qwen35_lin,
        packed_q_gate=packed_q_gate,
        linear_conv_kernel_dim=int(hc.get("linear_conv_kernel_dim", 4)),
        vsa_num_heads=vsa_num_heads, vsa_num_kv_heads=vsa_num_kv_heads,
        vsa_head_dim=vsa_head_dim,
        tie_word_embeddings=tie_emb,
        vsa_untie_gqa=True, vsa_local_conv=(not has_qwen35_lin),
    )


def map_any_hf_to_student(hf_sd: dict[str, torch.Tensor], cfg: dict, hybrid_every: int) -> dict[str, torch.Tensor]:
    sd = {}
    has_qwen35_lin = bool(cfg.get("has_qwen35_lin", False))
    for k, t in hf_sd.items():
        if k == "model.embed_tokens.weight":
            sd["embed_tokens.weight"] = t.detach().cpu().clone()
        elif k == "model.norm.weight":
            sd["norm.weight"] = t.detach().cpu().clone()
        elif k == "lm_head.weight" and not cfg["tie_word_embeddings"]:
            sd["lm_head.weight"] = t.detach().cpu().clone()
        elif k.startswith("model.layers."):
            parts = k.split(".")
            li, sub = int(parts[2]), ".".join(parts[3:])
            stays_soft = hybrid_every > 0 and (li + 1) % hybrid_every == 0
            core = "attn" if stays_soft else "vsa"
            if sub.startswith("self_attn."):
                proj = sub.split(".", 1)[1]
                if proj in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                            "q_proj.bias", "k_proj.bias", "v_proj.bias", "q_norm.weight", "k_norm.weight"):
                    if not stays_soft and proj in ("q_norm.weight", "k_norm.weight"):
                        continue
                    sd[f"blocks.{li}.{core}.{proj}"] = t.detach().cpu().clone()
            elif sub.startswith("linear_attn.") and not stays_soft and has_qwen35_lin:
                lproj = sub.split(".", 1)[1]
                if lproj in ("in_proj_qkv.weight", "conv1d.weight", "in_proj_z.weight",
                             "in_proj_b.weight", "in_proj_a.weight", "dt_bias", "A_log",
                             "norm.weight", "out_proj.weight"):
                    sd[f"blocks.{li}.vsa.{lproj}"] = t.detach().cpu().clone()
            elif sub == "input_layernorm.weight":
                sd[f"blocks.{li}.norm1.weight"] = t.detach().cpu().clone()
            elif sub == "post_attention_layernorm.weight":
                sd[f"blocks.{li}.norm2.weight"] = t.detach().cpu().clone()
            elif sub in ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"):
                sd[f"blocks.{li}.{sub}"] = t.detach().cpu().clone()
    if cfg["tie_word_embeddings"]:
        sd["lm_head.weight"] = sd["embed_tokens.weight"]
    return sd
