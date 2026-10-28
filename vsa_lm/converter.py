"""Universal Donor-to-VSA Converter (Llama, Qwen, DeepSeek-Distill, Mistral, SmolLM).

Dynamically inspects any HuggingFace causal LM's config.json and safetensors,
supports arbitrary parameter scales (135M to 8B+), handles both bias=False
(Llama/Mistral/SmolLM) and qkv_bias=True (Qwen2/Qwen2.5/DeepSeek-R1-Distill),
handles tied and untied lm_head weights, and performs de-clamped Step-0 VSA
surgery with GQA head un-tying and unit-RMS scale absorption baked in.
"""
from __future__ import annotations

import glob
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .surgery import (
    LlamaRMSNorm,
    LlamaRoPE,
    LlamaMLP,
    repeat_kv,
    _vsa_apply,
)


# ------------------------------------------------------------------ Universal Attention & Model

class UniversalSoftmaxAttn(nn.Module):
    """Donor softmax attention supporting arbitrary GQA ratios and optional QKV biases (Qwen)."""

    def __init__(self, cfg: dict):
        super().__init__()
        D, H, HKV = cfg["dim"], cfg["num_heads"], cfg["num_kv_heads"]
        HD = cfg.get("head_dim") or (D // H)
        qkv_bias = bool(cfg.get("qkv_bias", False))
        self.H, self.HKV, self.HD, self.G = H, HKV, HD, H // HKV
        self.q_proj = nn.Linear(D, H * HD, bias=qkv_bias)
        self.k_proj = nn.Linear(D, HKV * HD, bias=qkv_bias)
        self.v_proj = nn.Linear(D, HKV * HD, bias=qkv_bias)
        self.o_proj = nn.Linear(H * HD, D, bias=False)
        self.rope = LlamaRoPE(HD, cfg["rope_theta"], cfg.get("rope_scaling"))

    def head_qkv(self, x: torch.Tensor):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.H, self.HD).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.HKV, self.HD).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.HKV, self.HD).transpose(1, 2)
        return q, k, v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.head_qkv(x)
        q, k = self.rope(q), self.rope(k)
        k, v = repeat_kv(k, self.G), repeat_kv(v, self.G)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        B, _, T, _ = y.shape
        return self.o_proj(y.transpose(1, 2).reshape(B, T, self.H * self.HD))


class UniversalVSAAttn(nn.Module):
    """Universal VSA attention supporting QKV biases (Qwen), GQA head un-tying,
    output gate (g_proj), and local value bypass (v_self_scale)."""

    def __init__(self, cfg: dict, max_T: int, chunk_size: int = 32):
        super().__init__()
        D, H, HKV = cfg["dim"], cfg["num_heads"], cfg["num_kv_heads"]
        HD = cfg.get("head_dim") or (D // H)
        qkv_bias = bool(cfg.get("qkv_bias", False))
        self.H, self.HKV, self.HD, self.G = H, HKV, HD, H // HKV
        self.q_proj = nn.Linear(D, H * HD, bias=qkv_bias)
        self.k_proj = nn.Linear(D, HKV * HD, bias=qkv_bias)
        self.v_proj = nn.Linear(D, HKV * HD, bias=qkv_bias)
        self.o_proj = nn.Linear(H * HD, D, bias=False)
        self.rope = LlamaRoPE(HD, cfg["rope_theta"], cfg.get("rope_scaling"))
        self.pos_code = nn.Parameter(torch.ones(max_T, HD))
        self.log_gamma = nn.Parameter(torch.zeros(H))
        with torch.no_grad():
            horizons = torch.linspace(8.0, 200.0, H)
            self.log_gamma.copy_(torch.logit((1.0 - 1.0 / horizons) / 0.9999))
        self.chunk_size = chunk_size

        if cfg.get("vsa_untie_gqa", True):
            eye = torch.eye(HD).unsqueeze(0).repeat(H, 1, 1)
            self.k_head_mix = nn.Parameter(eye.clone())
            self.v_head_mix = nn.Parameter(eye.clone())
            self.head_pos_delta = nn.Parameter(torch.zeros(H, max_T, HD))
            self.g_proj = nn.Linear(D, H * HD, bias=False)
            nn.init.zeros_(self.g_proj.weight)
            self.v_self_scale = nn.Parameter(torch.zeros(H))

    def head_qkv(self, x: torch.Tensor):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.H, self.HD).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.HKV, self.HD).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.HKV, self.HD).transpose(1, 2)
        return q, k, v

    def gamma(self) -> torch.Tensor:
        return (0.9999 * torch.sigmoid(self.log_gamma)).clamp(1e-4, 0.9999).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.head_qkv(x)
        q, k = self.rope(q), self.rope(k)
        T_use = min(T, self.pos_code.shape[0])
        pc = self.pos_code[:T_use].view(1, 1, T_use, self.HD).to(k.dtype)
        if T > T_use:
            pad = torch.ones(1, 1, T - T_use, self.HD, device=k.device, dtype=k.dtype)
            pc = torch.cat([pc, pad], dim=2)
        kc = k * pc
        kx, vx = repeat_kv(kc, self.G), repeat_kv(v, self.G)
        if hasattr(self, "k_head_mix"):
            kx = torch.einsum("bhtd,hde->bhte", kx, self.k_head_mix.to(kx.dtype))
            if T <= self.head_pos_delta.shape[1]:
                kx = kx * (1.0 + 0.5 * torch.tanh(self.head_pos_delta[:, :T, :])).unsqueeze(0).to(kx.dtype)
            vx = torch.einsum("bhtd,hde->bhte", vx, self.v_head_mix.to(vx.dtype))
        y = _vsa_apply(q, kx, vx, self.gamma(), self.chunk_size)
        if hasattr(self, "v_self_scale"):
            y = y + self.v_self_scale.view(1, self.H, 1, 1).to(y.dtype) * vx
        if hasattr(self, "g_proj"):
            g = 2.0 * torch.sigmoid(self.g_proj(x).view(B, T, self.H, self.HD).transpose(1, 2))
            y = y * g
        return self.o_proj(y.transpose(1, 2).reshape(B, T, self.H * self.HD))


class UniversalBlock(nn.Module):
    def __init__(self, cfg: dict, attn_type: str, max_T: int, chunk_size: int):
        super().__init__()
        self.norm1 = LlamaRMSNorm(cfg["dim"], cfg["eps"])
        if attn_type == "vsa":
            self.vsa = UniversalVSAAttn(cfg, max_T, chunk_size)
        else:
            self.attn = UniversalSoftmaxAttn(cfg)
        self.norm2 = LlamaRMSNorm(cfg["dim"], cfg["eps"])
        self.mlp = LlamaMLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.vsa if hasattr(self, "vsa") else self.attn
        x = x + attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class UniversalVSALM(nn.Module):
    """Universal VSA-LM supporting Llama, Qwen, DeepSeek-Distill, Mistral, and SmolLM."""

    def __init__(self, vocab_size: int, dim: int, num_layers: int, num_heads: int,
                 num_kv_heads: int, intermediate: int, max_T: int = 1024,
                 chunk_size: int = 32, hybrid_every: int = 4,
                 rope_theta: float = 10000.0, rope_scaling: dict | None = None,
                 eps: float = 1e-5, head_dim: int | None = None,
                 qkv_bias: bool = False, tie_word_embeddings: bool = True,
                 vsa_untie_gqa: bool = True, **extra_cfg):
        super().__init__()
        head_dim = head_dim or (dim // num_heads)
        self.config = dict(
            vocab_size=vocab_size, dim=dim, num_layers=num_layers,
            num_heads=num_heads, num_kv_heads=num_kv_heads,
            intermediate=intermediate, max_T=max_T, chunk_size=chunk_size,
            hybrid_every=hybrid_every, rope_theta=rope_theta,
            rope_scaling=rope_scaling, eps=eps, head_dim=head_dim,
            qkv_bias=qkv_bias, tie_word_embeddings=tie_word_embeddings,
            vsa_untie_gqa=vsa_untie_gqa, **extra_cfg,
        )
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList()
        for li in range(num_layers):
            stays_soft = hybrid_every > 0 and (li + 1) % hybrid_every == 0
            t = "softmax" if stays_soft else "vsa"
            self.blocks.append(UniversalBlock(self.config, t, max_T, chunk_size))
        self.norm = LlamaRMSNorm(dim, eps)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        if tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.grad_checkpoint = False

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return self.norm(x)


# ------------------------------------------------------------------ Dynamic Loader & Surgery

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
        sd.update(load_file(sh))
    return sd


def inspect_donor_config(donor_dir: str, hf_sd: dict[str, torch.Tensor],
                         max_T: int = 1024, chunk_size: int = 32,
                         hybrid_every: int = 4) -> dict:
    with open(os.path.join(donor_dir, "config.json")) as f:
        hc = json.load(f)

    dim = int(hc["hidden_size"])
    num_layers = int(hc["num_hidden_layers"])
    num_heads = int(hc["num_attention_heads"])
    num_kv_heads = int(hc.get("num_key_value_heads", num_heads))
    head_dim = int(hc.get("head_dim") or (dim // num_heads))
    intermediate = int(hc["intermediate_size"])
    vocab_size = int(hc["vocab_size"])
    eps = float(hc.get("rms_norm_eps", 1e-5))
    rope_theta = float(hc.get("rope_theta", 10000.0))
    rope_scaling = hc.get("rope_scaling")
    tie_emb = bool(hc.get("tie_word_embeddings", "lm_head.weight" not in hf_sd))

    # Auto-detect Qwen-style QKV biases from state_dict keys
    has_qkv_bias = "model.layers.0.self_attn.q_proj.bias" in hf_sd

    return dict(
        donor_arch=hc.get("model_type", "unknown"),
        vocab_size=vocab_size, dim=dim, num_layers=num_layers,
        num_heads=num_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, intermediate=intermediate,
        max_T=max_T, chunk_size=chunk_size, hybrid_every=hybrid_every,
        rope_theta=rope_theta, rope_scaling=rope_scaling, eps=eps,
        qkv_bias=has_qkv_bias, tie_word_embeddings=tie_emb,
        vsa_untie_gqa=True,
    )


def map_any_hf_to_student(hf_sd: dict[str, torch.Tensor], cfg: dict, hybrid_every: int) -> dict[str, torch.Tensor]:
    sd = {}
    for k, t in hf_sd.items():
        if k == "model.embed_tokens.weight":
            sd["embed_tokens.weight"] = t.detach().cpu().clone()
        elif k == "model.norm.weight":
            sd["norm.weight"] = t.detach().cpu().clone()
        elif k == "lm_head.weight":
            if not cfg["tie_word_embeddings"]:
                sd["lm_head.weight"] = t.detach().cpu().clone()
        elif k.startswith("model.layers."):
            parts = k.split(".")
            li, sub = int(parts[2]), ".".join(parts[3:])
            stays_soft = hybrid_every > 0 and (li + 1) % hybrid_every == 0
            core = "attn" if stays_soft else "vsa"
            if sub.startswith("self_attn."):
                proj = sub.split(".", 1)[1]
                if proj in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                            "q_proj.bias", "k_proj.bias", "v_proj.bias"):
                    sd[f"blocks.{li}.{core}.{proj}"] = t.detach().cpu().clone()
            elif sub == "input_layernorm.weight":
                sd[f"blocks.{li}.norm1.weight"] = t.detach().cpu().clone()
            elif sub == "post_attention_layernorm.weight":
                sd[f"blocks.{li}.norm2.weight"] = t.detach().cpu().clone()
            elif sub in ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"):
                sd[f"blocks.{li}.{sub}"] = t.detach().cpu().clone()

    if cfg["tie_word_embeddings"]:
        sd["lm_head.weight"] = sd["embed_tokens.weight"]
    return sd


def convert_donor_model(
    donor_id_or_dir: str,
    out_ckpt: str,
    *,
    hybrid_every: int = 4,
    max_T: int = 1024,
    chunk_size: int = 32,
    gamma_cap: float = 0.995,
    hf_token: str | None = None,
) -> str:
    """Download/load any Llama, Qwen, DeepSeek-Distill, or SmolLM model and
    build a ready-to-heal VSA hybrid checkpoint with de-clamped horizons
    (gamma <= 0.995), unit-RMS pos_code, GQA head un-tying, output gate, and
    direct value bypass."""
    t0 = time.perf_counter()
    donor_dir = fetch_any_donor(donor_id_or_dir, hf_token=hf_token)
    hf_sd = load_all_safetensors(donor_dir)
    cfg = inspect_donor_config(donor_dir, hf_sd, max_T=max_T, chunk_size=chunk_size, hybrid_every=hybrid_every)

    L = cfg["num_layers"]
    n_soft = (L // hybrid_every) if hybrid_every > 0 else 0
    n_vsa = L - n_soft
    print("=" * 80)
    print(f"UNIVERSAL VSA-LM CONVERTER: {donor_id_or_dir}")
    print("=" * 80)
    print(f"  Detected family     : {cfg['donor_arch']}  (qkv_bias={cfg['qkv_bias']}, tied_emb={cfg['tie_word_embeddings']})")
    print(f"  Dimensions          : dim={cfg['dim']}, L={L}, H={cfg['num_heads']}, HKV={cfg['num_kv_heads']} (GQA {cfg['num_heads']//cfg['num_kv_heads']}:1), HD={cfg['head_dim']}")
    print(f"  Target layout       : {n_vsa} VSA layers + {n_soft} Softmax layers (hybrid_every={hybrid_every})")

    stu = UniversalVSALM(**cfg)
    sd = map_any_hf_to_student(hf_sd, cfg, hybrid_every)
    del hf_sd

    H, HD, D = cfg["num_heads"], cfg["head_dim"], cfg["dim"]
    dtype = sd["embed_tokens.weight"].dtype

    # Initialize VSA parameters cleanly (unit-RMS pos_code, de-clamped gamma <= gamma_cap,
    # identity GQA head un-tiers, zero-init output gate & local value bypass)
    horizons = torch.linspace(8.0, 1.0 / max(1e-4, 1.0 - gamma_cap), H)
    g_init = (1.0 - 1.0 / horizons).clamp(0.5, gamma_cap)
    sig = (g_init / 0.9999).clamp(1e-6, 1.0 - 1e-6)
    lg_init = torch.log(sig / (1.0 - sig)).float()

    for li in range(L):
        if hybrid_every > 0 and (li + 1) % hybrid_every == 0:
            continue
        sd[f"blocks.{li}.vsa.pos_code"] = torch.ones(max_T, HD, dtype=torch.float32)
        sd[f"blocks.{li}.vsa.log_gamma"] = lg_init.clone()
        eye = torch.eye(HD, dtype=dtype).unsqueeze(0).repeat(H, 1, 1)
        sd[f"blocks.{li}.vsa.k_head_mix"] = eye.clone()
        sd[f"blocks.{li}.vsa.v_head_mix"] = eye.clone()
        sd[f"blocks.{li}.vsa.head_pos_delta"] = torch.zeros(H, max_T, HD, dtype=torch.float32)
        sd[f"blocks.{li}.vsa.g_proj.weight"] = torch.zeros(H * HD, D, dtype=dtype)
        sd[f"blocks.{li}.vsa.v_self_scale"] = torch.zeros(H, dtype=torch.float32)

    stu.load_state_dict(sd, strict=True)
    n_params = sum(p.numel() for p in stu.parameters())
    tmp = out_ckpt + ".tmp"
    torch.save({
        "model": stu.state_dict(),
        "step": 0,
        "config": stu.config,
        "injection": {"donor": donor_id_or_dir, "converter": "vsa_lm.converter v2"},
    }, tmp)
    os.replace(tmp, out_ckpt)
    print(f"  Strict weight load  : PASS ({n_params/1e9:.3f}B params)")
    print(f"  Saved VSA student   : {out_ckpt}  ({time.perf_counter() - t0:.1f}s)")
    print("=" * 80)
    return out_ckpt
