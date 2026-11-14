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


def compute_layer_types(num_layers: int, hybrid_every: int, donor_layer_types: list[str] | None = None,
                        pin_endpoints: bool = False) -> list[str]:
    """Preserve native GatedDeltaNet layers and optionally pin L0 + final layer as Softmax."""
    if donor_layer_types and "linear_attention" in donor_layer_types:
        out = []
        full_seen = 0
        for lt in donor_layer_types:
            if lt == "linear_attention":
                out.append("gated_deltanet")
            else:
                full_seen += 1
                out.append("softmax" if (hybrid_every > 0 and full_seen % 2 == 0) else "vsa")
        return out
    return [
        "softmax" if ((pin_endpoints and (li == 0 or li == num_layers - 1)) or (hybrid_every > 0 and (li + 1) % hybrid_every == 0)) else "vsa"
        for li in range(num_layers)
    ]


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

    # B2 Fix: Only tie word embeddings if lm_head.weight is absent or tie_word_embeddings is explicitly True
    if "lm_head.weight" in hf_sd and not bool(hc.get("tie_word_embeddings", raw_cfg.get("tie_word_embeddings", False))):
        tie_emb = False
    else:
        tie_emb = "lm_head.weight" not in hf_sd or bool(hc.get("tie_word_embeddings", raw_cfg.get("tie_word_embeddings", True)))

    has_qkv_bias = any(k.endswith("self_attn.q_proj.bias") for k in hf_sd)
    has_qk_norm = any(k.endswith("self_attn.q_norm.weight") for k in hf_sd)
    has_qwen35_lin = any(".linear_attn.in_proj_qkv.weight" in k for k in hf_sd)
    packed_q_gate = any(
        k.endswith("self_attn.q_proj.weight") and t.shape[0] == 2 * num_heads * head_dim
        for k, t in hf_sd.items()
    )

    donor_layer_types = hc.get("layer_types")
    if not donor_layer_types and has_qwen35_lin:
        donor_layer_types = [
            "linear_attention" if f"model.layers.{li}.linear_attn.in_proj_qkv.weight" in hf_sd else "full_attention"
            for li in range(num_layers)
        ]
    layer_types = compute_layer_types(num_layers, hybrid_every, donor_layer_types)

    # VSA layers always replace full_attention (self_attn) layers! Cap VSA head_dim <= 128 for SRAM.
    vsa_num_heads, vsa_num_kv_heads, vsa_head_dim = num_heads, num_kv_heads, head_dim
    if head_dim > 128:
        scale_f = head_dim // 128
        vsa_num_heads = num_heads * scale_f
        vsa_num_kv_heads = num_kv_heads * scale_f
        vsa_head_dim = 128

    return dict(
        donor_arch=raw_cfg.get("model_type", hc.get("model_type", "unknown")),
        vocab_size=vocab_size, dim=dim, num_layers=num_layers,
        num_heads=num_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, intermediate=intermediate,
        max_T=max_T, chunk_size=chunk_size, hybrid_every=hybrid_every,
        rope_theta=rope_theta, partial_rotary_factor=partial_rotary_factor,
        rope_scaling=rope_scaling, eps=eps,
        qkv_bias=has_qkv_bias, vsa_qkv_bias=has_qkv_bias,
        qk_norm=has_qk_norm, has_qwen35_lin=has_qwen35_lin,
        packed_q_gate=packed_q_gate,
        donor_layer_types=donor_layer_types,
        layer_types=layer_types,
        linear_conv_kernel_dim=int(hc.get("linear_conv_kernel_dim", 4)),
        linear_num_value_heads=int(hc.get("linear_num_value_heads", 32)),
        linear_num_key_heads=int(hc.get("linear_num_key_heads", 16)),
        linear_value_head_dim=int(hc.get("linear_value_head_dim", 128)),
        linear_key_head_dim=int(hc.get("linear_key_head_dim", 128)),
        vsa_num_heads=vsa_num_heads, vsa_num_kv_heads=vsa_num_kv_heads,
        vsa_head_dim=vsa_head_dim,
        tie_word_embeddings=tie_emb,
        vsa_untie_gqa=True, vsa_local_conv=True,
    )


def map_any_hf_to_student(hf_sd: dict[str, torch.Tensor], cfg: dict, hybrid_every: int) -> dict[str, torch.Tensor]:
    sd = {}
    L = cfg["num_layers"]
    layer_types = cfg.get("layer_types") or compute_layer_types(L, hybrid_every, cfg.get("donor_layer_types"), pin_endpoints=bool(cfg.get("pin_endpoints", False)))
    H, HKV, HD = cfg["num_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    vH, vHKV, vHD = cfg["vsa_num_heads"], cfg["vsa_num_kv_heads"], cfg["vsa_head_dim"]
    D = cfg["dim"]

    for k, t in hf_sd.items():
        if k == "model.embed_tokens.weight":
            sd["embed_tokens.weight"] = t.detach().cpu().clone()
        elif k == "model.norm.weight":
            sd["norm.weight"] = t.detach().cpu().clone()
        elif k == "lm_head.weight":
            sd["lm_head.weight"] = t.detach().cpu().clone()
        elif k.startswith("model.layers."):
            parts = k.split(".")
            li, sub = int(parts[2]), ".".join(parts[3:])
            ltype = layer_types[li]
            core = "attn" if ltype == "softmax" else "vsa"

            if sub.startswith("self_attn."):
                proj = sub.split(".", 1)[1]
                w = t.detach().cpu().clone()
                if ltype == "vsa":
                    if proj == "q_proj.weight" and w.shape[0] == 2 * H * HD:
                        q_part, g_part = torch.chunk(w.view(H, 2, HD, D), 2, dim=1)
                        sd[f"blocks.{li}.vsa.q_proj.weight"] = q_part.reshape(vH * vHD, D).clone()
                        sd[f"blocks.{li}.vsa.g_proj.weight"] = g_part.reshape(vH * vHD, D).clone()
                    elif proj in ("q_norm.weight", "k_norm.weight") and vHD == HD:
                        sd[f"blocks.{li}.vsa.{proj}"] = w
                    elif proj in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                                  "q_proj.bias", "k_proj.bias", "v_proj.bias"):
                        sd[f"blocks.{li}.vsa.{proj}"] = w
                else:
                    if proj in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                                "q_proj.bias", "k_proj.bias", "v_proj.bias", "q_norm.weight", "k_norm.weight"):
                        sd[f"blocks.{li}.attn.{proj}"] = w

            elif sub.startswith("linear_attn.") and ltype == "gated_deltanet":
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

    if cfg["tie_word_embeddings"] or "lm_head.weight" not in sd:
        sd["lm_head.weight"] = sd["embed_tokens.weight"]
    return sd
