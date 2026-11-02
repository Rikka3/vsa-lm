"""Universal Donor-to-VSA Converter & Pure-KD FastDistill Healer (vsa_lm.converter).

Supports:
  - Qwen3.5 Multimodal/Hybrid models (e.g. `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-9B`):
    * Preserves Vision Tower tensors in a `.safetensors` sidecar (`0 MB` GPU VRAM during healing)
      and re-attaches them after healing.
    * Uses exact `Qwen35ZeroCenteredRMSNorm` (`output = x_norm * (1.0 + weight)`).
    * Uses exact `Qwen35SoftmaxAttn` (`q_proj` packed query+gate chunking, `q_norm`/`k_norm`,
      `partial_rotary_factor=0.25` at `rope_theta=10,000,000`, `y * sigmoid(gate)`).
    * Uses exact `Qwen35VSAAttn` preserving `in_proj_qkv`, `conv1d + silu`, L2-normalized Q/K,
      `in_proj_b` write gate, `A_log`/`dt_bias` horizon init, `Qwen35RMSNormGated(z)`, and `out_proj`.
  - Qwen3 (`Qwen/Qwen3-8B`), Qwen2.5, DeepSeek-R1-Distill-Qwen, Llama-3, Mistral, SmolLM.
  - Strictly disjoint train/eval caches (0% overlap, zero benchmark contamination).
  - Phase 4 Validated Speed & Memory Optimizations (E1 + E2 + E3 + E5 + E7) + Auto Zombie-VRAM Purge.
"""
from __future__ import annotations

import gc
import glob
import json
import math
import os
import sys
import threading
import time
import traceback
from array import array

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .surgery import (
    LlamaRMSNorm,
    LlamaRoPE,
    LlamaMLP,
    repeat_kv,
    _vsa_apply,
)


def _purge_zombie_vram():
    """Clears Python/IPython/Marimo exception tracebacks and forces CUDA garbage collection
    so a previously stopped or crashed cell never leaks 20-40 GiB of zombie tensors.
    """
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


class Qwen35ZeroCenteredRMSNorm(nn.Module):
    """Exact Qwen3_5RMSNorm: output = x_norm * (1.0 + weight)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * (1.0 + self.weight.float())).to(dt)


class Qwen35RMSNormGated(nn.Module):
    """Exact Qwen3_5RMSNormGated: weight * rsqrt(var + eps) * x * silu(z)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        xf = x.float()
        zf = F.silu(z.float())
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float() * zf).to(dt)


class Qwen35PartialRoPE(nn.Module):
    """Partial RoPE for Qwen3.5 full-attention layers (rotates first rotary_dim dims)."""

    def __init__(self, head_dim: int, partial_factor: float = 0.25, theta: float = 10_000_000.0):
        super().__init__()
        self.rotary_dim = int(head_dim * partial_factor)
        inv_freq = 1.0 / (theta ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, pos_start: int = 0) -> torch.Tensor:
        T = x.shape[2]
        t = torch.arange(pos_start, pos_start + T, device=x.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(x.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(x.dtype)[None, None]
        sin = emb.sin().to(x.dtype)[None, None]
        x_rot, x_pass = x[..., :self.rotary_dim], x[..., self.rotary_dim:]
        x1, x2 = x_rot[..., :self.rotary_dim // 2], x_rot[..., self.rotary_dim // 2:]
        x_rot_out = x_rot * cos + torch.cat((-x2, x1), dim=-1) * sin
        return torch.cat((x_rot_out, x_pass), dim=-1)


class Qwen35VSAAttn(nn.Module):
    """Exact VSA replacement for Qwen3_5GatedDeltaNet:
    Preserves pretrained in_proj_qkv, conv1d + silu, L2-normalized Q/K,
    in_proj_b write gate, A_log/dt_bias horizon init, RMSNormGated(z), and out_proj,
    while replacing the recurrent state with fixed-state VSA outer-product memory.
    """

    def __init__(self, cfg: dict, max_T: int, chunk_size: int = 32):
        super().__init__()
        D = cfg["dim"]
        HK = cfg.get("vsa_num_kv_heads", 16)
        HV = cfg.get("vsa_num_heads", 32)
        HD = cfg.get("vsa_head_dim", 128)
        conv_k = int(cfg.get("linear_conv_kernel_dim", 4))
        eps = float(cfg.get("eps", 1e-6))

        self.H = HV
        self.HKV = HK
        self.HD = HD
        self.G = HV // HK
        self.key_dim = HK * HD
        self.val_dim = HV * HD
        self.conv_dim = self.key_dim * 2 + self.val_dim
        self.chunk_size = chunk_size

        self.in_proj_qkv = nn.Linear(D, self.conv_dim, bias=False)
        self.conv1d = nn.Conv1d(
            self.conv_dim, self.conv_dim, kernel_size=conv_k,
            groups=self.conv_dim, padding=conv_k - 1, bias=False,
        )
        self.in_proj_z = nn.Linear(D, self.val_dim, bias=False)
        self.in_proj_b = nn.Linear(D, HV, bias=False)
        self.in_proj_a = nn.Linear(D, HV, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(HV))
        self.A_log = nn.Parameter(torch.zeros(HV))
        self.norm = Qwen35RMSNormGated(HD, eps=eps)
        self.out_proj = nn.Linear(self.val_dim, D, bias=False)

        self.pos_code = nn.Parameter(torch.ones(max_T, HD))
        self.log_gamma = nn.Parameter(torch.zeros(HV))
        eye = torch.eye(HD).unsqueeze(0).repeat(HV, 1, 1)
        self.k_head_mix = nn.Parameter(eye.clone())
        self.v_head_mix = nn.Parameter(eye.clone())
        self.head_pos_delta = nn.Parameter(torch.zeros(HV, max_T, HD))
        self.v_self_scale = nn.Parameter(torch.zeros(HV))

    def gamma(self) -> torch.Tensor:
        return (0.9999 * torch.sigmoid(self.log_gamma)).clamp(1e-4, 0.9999).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        mixed_qkv = self.in_proj_qkv(x).transpose(1, 2)
        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[..., :T]).transpose(1, 2)
        q, k, v = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.val_dim], dim=-1)

        q = F.normalize(q.view(B, T, self.HKV, self.HD).float(), p=2.0, dim=-1).to(x.dtype)
        k = F.normalize(k.view(B, T, self.HKV, self.HD).float(), p=2.0, dim=-1).to(x.dtype)
        v = v.view(B, T, self.H, self.HD)

        beta = torch.sigmoid(self.in_proj_b(x)).unsqueeze(-1)
        v = v * beta

        q = q.repeat_interleave(self.G, dim=2).transpose(1, 2).contiguous()
        k = k.repeat_interleave(self.G, dim=2).transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        T_use = min(T, self.pos_code.shape[0])
        pc = self.pos_code[:T_use].view(1, 1, T_use, self.HD).to(k.dtype)
        if T > T_use:
            pad = torch.ones(1, 1, T - T_use, self.HD, device=k.device, dtype=k.dtype)
            pc = torch.cat([pc, pad], dim=2)
        kc = k * pc

        kx = torch.einsum("bhtd,hde->bhte", kc, self.k_head_mix.to(kc.dtype))
        if T <= self.head_pos_delta.shape[1]:
            kx = kx * (1.0 + 0.5 * torch.tanh(self.head_pos_delta[:, :T, :])).unsqueeze(0).to(kx.dtype)
        vx = torch.einsum("bhtd,hde->bhte", v, self.v_head_mix.to(v.dtype))

        gam = self.gamma()
        y = _vsa_apply(q, kx, vx, gam, self.chunk_size)
        y = y + self.v_self_scale.view(1, self.H, 1, 1).to(y.dtype) * vx

        y_flat = y.transpose(1, 2).reshape(-1, self.HD)
        z_flat = self.in_proj_z(x).reshape(-1, self.HD)
        y_normed = self.norm(y_flat, z_flat).reshape(B, T, self.val_dim)
        return self.out_proj(y_normed)


class UniversalSoftmaxAttn(nn.Module):
    """Donor softmax attention supporting arbitrary GQA ratios, optional QKV biases (Qwen2),
    optional per-head q_norm/k_norm (Qwen3 / Qwen3.5), packed Qwen3.5 q_proj gate, and partial RoPE."""

    def __init__(self, cfg: dict):
        super().__init__()
        D, H, HKV = cfg["dim"], cfg["num_heads"], cfg["num_kv_heads"]
        HD = cfg.get("head_dim") or (D // H)
        qkv_bias = bool(cfg.get("qkv_bias", False))
        self.H, self.HKV, self.HD, self.G = H, HKV, HD, H // HKV
        self.packed_q_gate = bool(cfg.get("packed_q_gate", False))
        self.has_qwen35_lin = bool(cfg.get("has_qwen35_lin", False))

        q_out_dim = (H * HD * 2) if self.packed_q_gate else (H * HD)
        self.q_proj = nn.Linear(D, q_out_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(D, HKV * HD, bias=qkv_bias)
        self.v_proj = nn.Linear(D, HKV * HD, bias=qkv_bias)
        self.o_proj = nn.Linear(H * HD, D, bias=False)

        norm_cls = Qwen35ZeroCenteredRMSNorm if self.has_qwen35_lin else LlamaRMSNorm
        if cfg.get("qk_norm", False):
            self.q_norm = norm_cls(HD, cfg["eps"])
            self.k_norm = norm_cls(HD, cfg["eps"])
        else:
            self.q_norm = None
            self.k_norm = None

        if cfg.get("softmax_gate", False) and not self.packed_q_gate:
            self.g_proj = nn.Linear(D, H * HD, bias=False)
        else:
            self.g_proj = None

        partial_factor = float(cfg.get("partial_rotary_factor", 1.0))
        if partial_factor < 1.0:
            self.rope = Qwen35PartialRoPE(HD, partial_factor=partial_factor, theta=cfg["rope_theta"])
        else:
            self.rope = LlamaRoPE(HD, cfg["rope_theta"], cfg.get("rope_scaling"))

    def head_qkv(self, x: torch.Tensor):
        B, T, _ = x.shape
        if self.packed_q_gate:
            q_and_gate = self.q_proj(x).view(B, T, self.H, self.HD * 2)
            q, _ = torch.chunk(q_and_gate, 2, dim=-1)
        else:
            q = self.q_proj(x).view(B, T, self.H, self.HD)
        k = self.k_proj(x).view(B, T, self.HKV, self.HD)
        v = self.v_proj(x).view(B, T, self.HKV, self.HD)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        if self.packed_q_gate:
            q_and_gate = self.q_proj(x).view(B, T, self.H, self.HD * 2)
            q, gate = torch.chunk(q_and_gate, 2, dim=-1)
            gate = gate.reshape(B, T, self.H * self.HD)
            k = self.k_proj(x).view(B, T, self.HKV, self.HD)
            v = self.v_proj(x).view(B, T, self.HKV, self.HD)
            if self.q_norm is not None:
                q = self.q_norm(q)
                k = self.k_norm(k)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            q, k = self.rope(q), self.rope(k)
            k, v = repeat_kv(k, self.G), repeat_kv(v, self.G)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2).reshape(B, T, self.H * self.HD) * torch.sigmoid(gate)
            return self.o_proj(y)

        q, k, v = self.head_qkv(x)
        q, k = self.rope(q), self.rope(k)
        k, v = repeat_kv(k, self.G), repeat_kv(v, self.G)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        if self.g_proj is not None:
            g = torch.sigmoid(self.g_proj(x).view(B, T, self.H, self.HD).transpose(1, 2))
            y = y * g
        return self.o_proj(y.transpose(1, 2).reshape(B, T, self.H * self.HD))


class UniversalVSAAttn(nn.Module):
    """Universal VSA attention supporting QKV biases, Qwen3/2.5/Llama/DeepSeek dimensions,
    GQA head un-tying, 8-tap causal local value/key conv, Q/K positive sharpener,
    and output gate. Caps VSA head_dim at <= 128 so Triton kernels always fit in SRAM."""

    def __init__(self, cfg: dict, max_T: int, chunk_size: int = 32):
        super().__init__()
        D = cfg["dim"]
        H = cfg.get("vsa_num_heads") or cfg["num_heads"]
        HKV = cfg.get("vsa_num_kv_heads") or cfg["num_kv_heads"]
        HD = cfg.get("vsa_head_dim") or cfg.get("head_dim") or (D // H)
        while HD > 128 and H * 2 <= 64:
            H *= 2
            HKV *= 2
            HD //= 2

        qkv_bias = bool(cfg.get("vsa_qkv_bias", cfg.get("qkv_bias", False)))
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

        if cfg.get("vsa_local_conv", True):
            self.v_conv = nn.Conv1d(H * HD, H * HD, kernel_size=8, padding=7, groups=H * HD, bias=False)
            nn.init.zeros_(self.v_conv.weight)
            self.k_conv = nn.Conv1d(H * HD, H * HD, kernel_size=4, padding=3, groups=H * HD, bias=False)
            nn.init.zeros_(self.k_conv.weight)
            self.local_q_gate = nn.Parameter(torch.zeros(H, HD))
            self.q_relu_mix = nn.Parameter(torch.zeros(H, 1, 1))
            self.k_relu_mix = nn.Parameter(torch.zeros(H, 1, 1))

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

        if hasattr(self, "k_conv"):
            kx_flat = kx.permute(0, 1, 3, 2).reshape(B, self.H * self.HD, T)
            k_loc = self.k_conv(kx_flat)[..., :T].view(B, self.H, self.HD, T).permute(0, 1, 3, 2)
            kx = kx + k_loc
            q = q + self.q_relu_mix.to(q.dtype) * F.relu(q)
            kx = kx + self.k_relu_mix.to(kx.dtype) * F.relu(kx)

        y = _vsa_apply(q, kx, vx, self.gamma(), self.chunk_size)

        if hasattr(self, "v_self_scale"):
            y = y + self.v_self_scale.view(1, self.H, 1, 1).to(y.dtype) * vx

        if hasattr(self, "v_conv"):
            vx_flat = vx.permute(0, 1, 3, 2).reshape(B, self.H * self.HD, T)
            v_loc = self.v_conv(vx_flat)[..., :T].view(B, self.H, self.HD, T).permute(0, 1, 3, 2)
            q_mod = torch.sigmoid((q * self.local_q_gate.view(1, self.H, 1, self.HD).to(q.dtype)).sum(-1, keepdim=True))
            y = y + (2.0 * q_mod) * v_loc

        if hasattr(self, "g_proj"):
            g = 2.0 * torch.sigmoid(self.g_proj(x).view(B, T, self.H, self.HD).transpose(1, 2))
            y = y * g

        return self.o_proj(y.transpose(1, 2).reshape(B, T, self.H * self.HD))


class UniversalBlock(nn.Module):
    def __init__(self, cfg: dict, attn_type: str, max_T: int, chunk_size: int):
        super().__init__()
        norm_cls = Qwen35ZeroCenteredRMSNorm if cfg.get("has_qwen35_lin", False) else LlamaRMSNorm
        self.norm1 = norm_cls(cfg["dim"], cfg["eps"])
        if attn_type == "vsa":
            if cfg.get("has_qwen35_lin", False):
                self.vsa = Qwen35VSAAttn(cfg, max_T, chunk_size)
            else:
                self.vsa = UniversalVSAAttn(cfg, max_T, chunk_size)
        else:
            self.attn = UniversalSoftmaxAttn(cfg)
        self.norm2 = norm_cls(cfg["dim"], cfg["eps"])
        self.mlp = LlamaMLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.vsa if hasattr(self, "vsa") else self.attn
        x = x + attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class UniversalVSALM(nn.Module):
    """Universal VSA-LM supporting Qwen3.5 (with Vision sidecar preservation), Qwen3, Qwen2.5, DeepSeek-R1, Llama, Mistral, SmolLM."""

    def __init__(self, vocab_size: int, dim: int, num_layers: int, num_heads: int,
                 num_kv_heads: int, intermediate: int, max_T: int = 1024,
                 chunk_size: int = 32, hybrid_every: int = 4,
                 rope_theta: float = 10000.0, rope_scaling: dict | None = None,
                 eps: float = 1e-5, head_dim: int | None = None,
                 qkv_bias: bool = False, tie_word_embeddings: bool = True,
                 vsa_untie_gqa: bool = True, vsa_local_conv: bool = True, **extra_cfg):
        super().__init__()
        head_dim = head_dim or (dim // num_heads)
        self.config = dict(
            vocab_size=vocab_size, dim=dim, num_layers=num_layers,
            num_heads=num_heads, num_kv_heads=num_kv_heads,
            intermediate=intermediate, max_T=max_T, chunk_size=chunk_size,
            hybrid_every=hybrid_every, rope_theta=rope_theta,
            rope_scaling=rope_scaling, eps=eps, head_dim=head_dim,
            qkv_bias=qkv_bias, tie_word_embeddings=tie_word_embeddings,
            vsa_untie_gqa=vsa_untie_gqa, vsa_local_conv=vsa_local_conv, **extra_cfg,
        )
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList()
        for li in range(num_layers):
            stays_soft = hybrid_every > 0 and (li + 1) % hybrid_every == 0
            t = "softmax" if stays_soft else "vsa"
            self.blocks.append(UniversalBlock(self.config, t, max_T, chunk_size))
        norm_cls = Qwen35ZeroCenteredRMSNorm if self.config.get("has_qwen35_lin", False) else LlamaRMSNorm
        self.norm = norm_cls(dim, eps)
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


def preserve_visual_tower(donor_dir: str, out_sidecar_path: str) -> tuple[int, float]:
    """Extracts and saves all `visual.*` / `model.visual.*` tensors into a `.safetensors`
    sidecar file on disk so Vision is 100% preserved while using 0 MB GPU VRAM during text healing.
    """
    from safetensors.torch import load_file, save_file
    if os.path.isfile(out_sidecar_path):
        vis_sd = load_file(out_sidecar_path)
        mb = sum(v.numel() * v.element_size() for v in vis_sd.values()) / (1024 ** 2)
        return len(vis_sd), mb

    shards = sorted(glob.glob(os.path.join(donor_dir, "*.safetensors")))
    vis_sd = {}
    for sh in shards:
        for k, v in load_file(sh).items():
            if k.startswith("visual.") or k.startswith("model.visual.") or ".visual." in k:
                vis_sd[k] = v
    if vis_sd:
        save_file(vis_sd, out_sidecar_path)
        mb = sum(v.numel() * v.element_size() for v in vis_sd.values()) / (1024 ** 2)
        print(f"[vsa-vision] Preserved {len(vis_sd)} Vision Tower tensors ({mb:.1f} MB) -> {out_sidecar_path}")
        return len(vis_sd), mb
    return 0, 0.0


def reattach_visual_tower(healed_ckpt_path: str, visual_sidecar_path: str, out_multimodal_ckpt_path: str | None = None) -> str:
    """Re-attaches the preserved Vision Tower `.safetensors` sidecar into the healed VSA checkpoint."""
    from safetensors.torch import load_file
    if not os.path.isfile(visual_sidecar_path):
        raise FileNotFoundError(f"Vision sidecar not found: {visual_sidecar_path}")
    vis_sd = load_file(visual_sidecar_path)
    ck = torch.load(healed_ckpt_path, map_location="cpu", weights_only=False)
    ck["visual_tower"] = vis_sd
    ck.setdefault("config", {})["has_visual_tower"] = True
    ck["config"]["visual_tensors_count"] = len(vis_sd)
    dst = out_multimodal_ckpt_path or healed_ckpt_path
    tmp = dst + ".tmp"
    torch.save(ck, tmp)
    os.replace(tmp, dst)
    mb = sum(v.numel() * v.element_size() for v in vis_sd.values()) / (1024 ** 2)
    print(f"[vsa-vision] Re-attached {len(vis_sd)} Vision Tower tensors ({mb:.1f} MB) into {dst}!")
    return dst


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
                            "q_proj.bias", "k_proj.bias", "v_proj.bias",
                            "q_norm.weight", "k_norm.weight"):
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

    H = cfg["vsa_num_heads"]
    HD = cfg["vsa_head_dim"]
    D = cfg["dim"]
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
            A_log = sd[f"{p}.A_log"].float()
            dt_bias = sd[f"{p}.dt_bias"].float()
            g_static = torch.exp(-A_log.exp() * F.softplus(dt_bias)).clamp(0.20, gamma_cap)
            sd[f"{p}.log_gamma"] = torch.logit(g_static / 0.9999).float()
        else:
            sd[f"{p}.log_gamma"] = lg_init.clone()

        eye = torch.eye(HD, dtype=torch.float32).unsqueeze(0).repeat(H, 1, 1)
        sd[f"{p}.k_head_mix"] = eye.clone()
        sd[f"{p}.v_head_mix"] = eye.clone()
        sd[f"{p}.head_pos_delta"] = torch.zeros(H, max_T, HD, dtype=torch.float32)
        sd[f"{p}.v_self_scale"] = torch.zeros(H, dtype=torch.float32)

        if not cfg.get("has_qwen35_lin", False):
            sd.setdefault(f"{p}.g_proj.weight", torch.zeros(H * HD, D, dtype=dtype))
            sd[f"{p}.v_conv.weight"] = torch.zeros(H * HD, 1, 8, dtype=dtype)
            sd[f"{p}.k_conv.weight"] = torch.zeros(H * HD, 1, 4, dtype=dtype)
            sd[f"{p}.local_q_gate"] = torch.zeros(H, HD, dtype=torch.float32)
            sd[f"{p}.q_relu_mix"] = torch.zeros(H, 1, 1, dtype=torch.float32)
            sd[f"{p}.k_relu_mix"] = torch.zeros(H, 1, 1, dtype=torch.float32)

    stu.load_state_dict(sd, strict=True)
    n_params = sum(p.numel() for p in stu.parameters())
    tmp = out_ckpt + ".tmp"
    torch.save({
        "model": stu.state_dict(),
        "step": 0,
        "config": stu.config,
        "injection": {
            "donor": donor_id_or_dir,
            "converter": "vsa_lm.converter v7-qwen35-exact",
            "visual_sidecar_path": vis_path if n_vis > 0 else None,
        },
    }, tmp)
    os.replace(tmp, out_ckpt)
    print(f"  Strict weight load  : PASS ({n_params/1e9:.3f}B text-model params)")
    print(f"  Saved VSA student   : {out_ckpt}  ({time.perf_counter() - t0:.1f}s)")
    print("=" * 80)
    return out_ckpt


def _build_reasoning_and_eval_caches(work: str, donor_dir: str, donor_repo: str, target_tokens: int):
    """Builds strictly disjoint `train_tokens` and `eval_tokens`:
      - Splits raw documents into `eval_docs` and `train_docs` BEFORE tokenizing
        or repeating, guaranteeing 0.0% overlap between train and eval!
      - Never includes evaluation benchmarks (GSM8K, MATH-500, AIME, MMLU, GPQA)
        in the training stream.
    """
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
    eval_docs = all_texts[:n_eval_docs]
    train_docs = all_texts[n_eval_docs:]

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
    """Runs the HF teacher's inner language model backbone WITHOUT computing the
    248,320-vocab lm_head logits tensor on the full batch, and immediately frees
    all non-anchor layer hidden states.
    """
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
    T = stu.config["max_T"]
    V = stu.config["vocab_size"]
    avail = len(eval_tokens) - 1
    n_b = max(1, min(n_tokens // (bs * T), avail // (bs * T)))
    ar = torch.arange(T)
    nll_s, nll_d, kl_sum, top1_ok, cnt = 0.0, 0.0, 0.0, 0, 0
    d_lm_head = donor.get_output_embeddings() if is_hf_donor else donor.lm_head

    for i in range(n_b):
        s = i * bs * T
        base = s + torch.arange(bs)[:, None] * T + ar[None, :]
        x = eval_tokens[base].long().to(device)
        y = eval_tokens[base + 1].long().to(device)
        with torch.autocast("cuda", torch.bfloat16, enabled=(device.type == "cuda")):
            hs = stu(x)
            if is_hf_donor:
                hd, _ = _forward_hf_teacher_body(donor, x, stu.config["num_layers"], stu.config["hybrid_every"])
            else:
                hd = donor(x)
        for c in range(bs):
            slog = stu.lm_head(hs[c:c+1]).float()
            dlog = d_lm_head(hd[c:c+1]).float()
            yv = y[c:c+1].reshape(-1)
            nll_s += F.cross_entropy(slog.reshape(-1, V), yv, reduction="sum").item()
            nll_d += F.cross_entropy(dlog.reshape(-1, V), yv, reduction="sum").item()

            lsm_s = F.log_softmax(slog, dim=-1)
            lsm_d = F.log_softmax(dlog, dim=-1)
            p_d = lsm_d.exp()
            kl_sum += float((p_d * (lsm_d - lsm_s)).sum().item())
            top1_ok += int((slog.argmax(dim=-1) == dlog.argmax(dim=-1)).sum().item())
            cnt += yv.numel()
            del slog, dlog, lsm_s, lsm_d, p_d
        del hs, hd, x, y

    stu_ppl = math.exp(nll_s / max(1, cnt))
    don_ppl = math.exp(nll_d / max(1, cnt))
    kl_nats = kl_sum / max(1, cnt)
    top1_pct = (top1_ok / max(1, cnt)) * 100.0
    return stu_ppl, don_ppl, kl_nats, top1_pct


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
    stu.load_state_dict(ck["model"], strict=True)
    stu.to(device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    for name, p in stu.named_parameters():
        if any(k in name for k in ("pos_code", "log_gamma", "k_head_mix", "v_head_mix", "head_pos_delta", "v_self_scale", "local_q_gate", "q_relu_mix", "k_relu_mix", "A_log", "dt_bias")):
            p.data = p.data.float()

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

    # E5 Optimization: Share frozen Teacher & Student Embedding/LM-Head storage
    if is_hf_donor:
        t_in_emb = donor.get_input_embeddings()
        t_out_emb = donor.get_output_embeddings()
        if t_in_emb is not None and t_in_emb.weight.shape == stu.embed_tokens.weight.shape:
            t_in_emb.weight = stu.embed_tokens.weight
        if t_out_emb is not None and t_out_emb.weight.shape == stu.lm_head.weight.shape:
            t_out_emb.weight = stu.lm_head.weight
    else:
        donor.embed_tokens.weight = stu.embed_tokens.weight
        donor.lm_head.weight = stu.lm_head.weight
    _purge_zombie_vram()

    w_lm_t = donor.get_output_embeddings().weight if is_hf_donor else donor.lm_head.weight

    train_tokens, eval_tokens = _build_reasoning_and_eval_caches(
        work, donor_dir, donor_repo, max(data_tokens, int(tokens_budget * 1.05))
    )
    start_ppl, donor_ppl, start_kl, start_top1 = _eval_reasoning_metrics(stu, donor, is_hf_donor, eval_tokens, device, bs=2)

    total_steps = max(1, math.ceil(tokens_budget / (bs * T)))
    stage1_steps = min(400, total_steps // 6) if start_step == 0 else 0

    print("=" * 84)
    print(f"VSA-FASTDISTILL PURE-TEACHER HEALER: {donor_repo} (kd={kd})")
    print("=" * 84)
    print(f"  Memory Mode    : {mode_str} ({n_params/1e9:.2f}B Student, E1+E2+E3+E5+E7 Active)")
    print(f"  Donor Eval PPL : {donor_ppl:.3f}  |  Student Start PPL: {start_ppl:.3f} ({start_ppl/donor_ppl:.3f}x) | KL: {start_kl:.3f} | Top1: {start_top1:.1f}%")
    print(f"  Budget         : {tokens_budget/1e6:.1f}M Reasoning/Chat tokens ({total_steps} steps, resume={start_step})")

vsa_conv, vsa_core, vsa_proj = [], [], []
    for n_, p_ in stu.named_parameters():
        if any(k in n_ for k in ("v_conv", "k_conv", "conv1d", "local_q_gate", "q_relu_mix", "k_relu_mix")):
            p_.requires_grad_(True)
            vsa_conv.append(p_)
        elif any(k in n_ for k in ("pos_code", "log_gamma", "k_head_mix", "v_head_mix", "head_pos_delta", "v_self_scale", "A_log", "dt_bias")):
            p_.requires_grad_(True)
            vsa_core.append(p_)
        elif ".vsa." in n_:
            p_.requires_grad_(True)
            vsa_proj.append(p_)
        else:
            # Keep donor's pretrained Embeddings, Softmax layers, MLPs, and RMSNorms 100% frozen!
            p_.requires_grad_(False)

    opt_groups = [
        {"params": vsa_conv, "weight_decay": 0.0, "lr_scale": 5.0},
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
            m *= min(1.0, (s - start_step + 1) / 100.0)
        for g in opt.param_groups:
            if g.get("is_mlp", False):
                if s < stage1_steps:
                    g["lr"] = 0.0
                else:
                    mlp_warm = min(1.0, (s - stage1_steps + 1) / 150.0)
                    g["lr"] = lr * m * g.get("lr_scale", 1.0) * mlp_warm
            else:
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
        if not mlp_unfrozen and step >= stage1_steps:
            for p_ in mlp_params:
                p_.requires_grad_(True)
            opt.add_param_group({"params": mlp_params, "weight_decay": 0.01, "lr_scale": mlp_lr_scale, "is_mlp": True, "lr": 0.0})
            trainable = vsa_conv + vsa_core + vsa_proj + mlp_params
            mlp_unfrozen = True
            set_step_lr(step)

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
            hs_2d = h_student.reshape(N_m, D_m)
            ht_2d = h_teacher.reshape(N_m, D_m)
            ym_1d = ym.reshape(N_m)
            dh_2d = torch.empty_like(hs_2d)
            w_lm_s = stu.lm_head.weight

            with torch.no_grad():
                for t_lo in range(0, N_m, tok_slice):
                    t_hi = min(t_lo + tok_slice, N_m)
                    hs_c = hs_2d[t_lo:t_hi]
                    ht_c = ht_2d[t_lo:t_hi]
                    ym_c = ym_1d[t_lo:t_hi]

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
            loss_val = float(loss_gpu.item())
            gn = float(gn_tensor.item())
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
