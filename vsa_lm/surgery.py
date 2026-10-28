"""Unified Donor Surgery & Anchored Healing Engine (1B+ Llama/GQA -> VSA-LM).
"""
from __future__ import annotations

import contextlib
import gc
import json
import math
import os
import threading
import time
from array import array

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .functional import VSAChunkFunction, vsa_parallel_attention
from .train import auto_fit_hardware


# ------------------------------------------------------------------ Architecture

class LlamaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(dt)


def llama_inv_freq(head_dim: int, theta: float, rope_scaling: dict | None, device: torch.device):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64,
                                        device=device).float() / head_dim))
    if rope_scaling and rope_scaling.get("rope_type") == "llama3":
        old_ctx = float(rope_scaling["original_max_position_embeddings"])
        factor = float(rope_scaling["factor"])
        low = float(rope_scaling["low_freq_factor"])
        high = float(rope_scaling["high_freq_factor"])
        low_wl = old_ctx / low
        high_wl = old_ctx / high
        wavelen = 2.0 * math.pi / inv
        inv_a = torch.where(wavelen > low_wl, inv / factor, inv)
        smooth = (old_ctx / wavelen - low) / (high - low)
        smoothed = (1.0 - smooth) * inv_a / factor + smooth * inv_a
        medium = ~(wavelen < high_wl) & ~(wavelen > low_wl)
        inv = torch.where(medium, smoothed, inv_a)
    return inv


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class LlamaRoPE(nn.Module):
    def __init__(self, head_dim: int, theta: float, rope_scaling: dict | None):
        super().__init__()
        buf = llama_inv_freq(head_dim, theta, rope_scaling, torch.device("cpu"))
        self.register_buffer("inv_freq", buf, persistent=False)

    def forward(self, x: torch.Tensor, pos_start: int = 0) -> torch.Tensor:
        T = x.shape[2]
        t = torch.arange(pos_start, pos_start + T, device=x.device, dtype=torch.float32)
        freqs = t[:, None] * self.inv_freq.to(x.device)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(x.dtype)[None, None]
        sin = emb.sin().to(x.dtype)[None, None]
        return x * cos + rotate_half(x) * sin


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    B, H, T, HD = x.shape
    return x[:, :, None, :, :].expand(B, H, n_rep, T, HD).reshape(B, H * n_rep, T, HD)


def _vsa_apply(q, k, v, gamma, chunk_size):
    if q.is_cuda:
        return VSAChunkFunction.apply(q, k, v, gamma, chunk_size)
    return vsa_parallel_attention(q.float(), k.float(), v.float(), gamma.float()).to(q.dtype)


class LlamaSoftmaxAttn(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        D, H, HKV = cfg["dim"], cfg["num_heads"], cfg["num_kv_heads"]
        HD = cfg.get("head_dim") or (D // H)
        self.H, self.HKV, self.HD, self.G = H, HKV, HD, H // HKV
        self.q_proj = nn.Linear(D, H * HD, bias=False)
        self.k_proj = nn.Linear(D, HKV * HD, bias=False)
        self.v_proj = nn.Linear(D, HKV * HD, bias=False)
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


class LlamaVSAAttn(nn.Module):
    """VSA attention with baked-in GQA 32-head un-tying, output gate (g_proj),
    and direct value bypass (v_self_scale) to prevent (1 - gamma) local attenuation."""

    def __init__(self, cfg: dict, max_T: int, chunk_size: int = 32):
        super().__init__()
        D, H, HKV = cfg["dim"], cfg["num_heads"], cfg["num_kv_heads"]
        HD = cfg.get("head_dim") or (D // H)
        self.H, self.HKV, self.HD, self.G = H, HKV, HD, H // HKV
        self.q_proj = nn.Linear(D, H * HD, bias=False)
        self.k_proj = nn.Linear(D, HKV * HD, bias=False)
        self.v_proj = nn.Linear(D, HKV * HD, bias=False)
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


class LlamaMLP(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        D, I = cfg["dim"], cfg["intermediate"]
        self.gate_proj = nn.Linear(D, I, bias=False)
        self.up_proj = nn.Linear(D, I, bias=False)
        self.down_proj = nn.Linear(I, D, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaBlock(nn.Module):
    def __init__(self, cfg: dict, attn_type: str, max_T: int, chunk_size: int):
        super().__init__()
        self.norm1 = LlamaRMSNorm(cfg["dim"], cfg["eps"])
        if attn_type == "vsa":
            self.vsa = LlamaVSAAttn(cfg, max_T, chunk_size)
        else:
            self.attn = LlamaSoftmaxAttn(cfg)
        self.norm2 = LlamaRMSNorm(cfg["dim"], cfg["eps"])
        self.mlp = LlamaMLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.vsa if hasattr(self, "vsa") else self.attn
        x = x + attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class LlamaVSALM(nn.Module):
    """Unified GQA / Llama VSA-LM with **kwargs compatibility and activation checkpointing."""

    def __init__(self, vocab_size: int, dim: int, num_layers: int, num_heads: int,
                 num_kv_heads: int, intermediate: int, max_T: int = 1024,
                 chunk_size: int = 32, hybrid_every: int = 4,
                 rope_theta: float = 500000.0, rope_scaling: dict | None = None,
                 eps: float = 1e-5, head_dim: int | None = None,
                 vsa_untie_gqa: bool = True, **extra_cfg):
        super().__init__()
        head_dim = head_dim or (dim // num_heads)
        self.config = dict(
            vocab_size=vocab_size, dim=dim, num_layers=num_layers,
            num_heads=num_heads, num_kv_heads=num_kv_heads,
            intermediate=intermediate, max_T=max_T, chunk_size=chunk_size,
            hybrid_every=hybrid_every, rope_theta=rope_theta,
            rope_scaling=rope_scaling, eps=eps, head_dim=head_dim,
            vsa_untie_gqa=vsa_untie_gqa, **extra_cfg,
        )
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList()
        for li in range(num_layers):
            stays_soft = hybrid_every > 0 and (li + 1) % hybrid_every == 0
            t = "softmax" if stays_soft else "vsa"
            self.blocks.append(LlamaBlock(self.config, t, max_T, chunk_size))
        self.norm = LlamaRMSNorm(dim, eps)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
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


# ------------------------------------------------------------------ Diagnostics Scan

def scan_checkpoint(ckpt_path: str) -> dict:
    """Fast CPU capacity scan of a VSA checkpoint (gamma saturation & pos_code profile)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model"]
    cfg = ck.get("config", {})
    num_layers = cfg.get("num_layers", 16)
    he = cfg.get("hybrid_every", 4)

    print("=" * 78)
    print(f"VSA-LM CAPACITY SCAN: {ckpt_path} (step {ck.get('step', '?')})")
    print("=" * 78)
    print(f"  {'L':>3} | {'g_min':>6} {'g_med':>6} {'g_max':>7} {'#>0.999':>7} | "
          f"{'pos_rms':>7} {'max|p|':>7} {'late/early':>10}")
    print("  " + "-" * 66)

    out = {}
    for li in range(num_layers):
        if he > 0 and (li + 1) % he == 0:
            continue
        lg = sd[f"blocks.{li}.vsa.log_gamma"].float()
        g = (0.9999 * torch.sigmoid(lg)).clamp(1e-4, 0.9999)
        pos = sd[f"blocks.{li}.vsa.pos_code"].float()
        T = pos.shape[0]
        q_len = max(1, T // 4)
        le = float(pos[T - q_len:].pow(2).mean().sqrt() / pos[:q_len].pow(2).mean().sqrt().clamp_min(1e-9))
        prms = float(pos.pow(2).mean().sqrt())
        pmax = float(pos.abs().max())
        n_c = int((g >= 0.999).sum())
        print(f"  L{li:<2d} | {float(g.min()):6.4f} {float(g.median()):6.4f} "
              f"{float(g.max()):7.5f} {n_c:>4d}/{g.numel():<2d} | "
              f"{prms:7.2f} {pmax:7.2f} {le:9.3f}x")
        out[str(li)] = dict(g_min=float(g.min()), g_med=float(g.median()), g_max=float(g.max()),
                            n_clamped=n_c, pos_rms=prms, pos_max=pmax, late_early=le)
    return out
