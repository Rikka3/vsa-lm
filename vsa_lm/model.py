"""Model architecture: VSA layers, softmax layers, transformer blocks, LM head."""
from __future__ import annotations

import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .functional import VSAChunkFunction


def _rope(x: torch.Tensor, pos_start: int = 0, theta: float = 10000.0) -> torch.Tensor:
    """Rotary position embedding for (B, H, T, HD) tensors."""
    T, HD = x.shape[2], x.shape[3]
    inv_freq = 1.0 / (theta ** (torch.arange(0, HD, 2, device=x.device, dtype=torch.float32) / HD))
    t = torch.arange(pos_start, pos_start + T, device=x.device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos, sin = freqs.cos()[None, None], freqs.sin()[None, None]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand (B, HKV, T, HD) -> (B, H, T, HD) for Grouped-Query Attention."""
    if n_rep == 1:
        return x
    B, HKV, T, HD = x.shape
    return x[:, :, None, :, :].expand(B, HKV, n_rep, T, HD).reshape(B, HKV * n_rep, T, HD)


class TritonVSALayer(nn.Module):
    """Fixed-state decayed linear attention with per-head learnable decay,
    bounded unit-RMS pos_code, output gate, and automatic GQA head un-tying."""

    def __init__(self, dim: int, num_heads: int, max_T: int, chunk_size: int = 32,
                 num_kv_heads: int | None = None, gamma_max: float = 0.995,
                 stabilize_pos: bool = True, use_rope: bool = False,
                 rope_theta: float = 10000.0, use_head_norm: bool = False,
                 vsa_untie_gqa: bool = True, **kwargs):
        super().__init__()
        self.H = num_heads
        self.HKV = num_kv_heads or num_heads
        self.HD = dim // num_heads
        self.G = self.H // self.HKV
        self.max_T = max_T
        self.chunk_size = chunk_size
        self.gamma_max = gamma_max
        self.stabilize_pos = stabilize_pos
        self.use_rope = use_rope
        self.rope_theta = rope_theta

        self.to_q = nn.Linear(dim, self.H * self.HD, bias=False)
        self.to_k = nn.Linear(dim, self.HKV * self.HD, bias=False)
        self.to_v = nn.Linear(dim, self.HKV * self.HD, bias=False)
        self.to_g = nn.Linear(dim, self.H * self.HD, bias=False)
        self.to_out = nn.Linear(self.H * self.HD, dim, bias=False)
        self.head_norm = nn.RMSNorm(self.HD) if use_head_norm else None

        # If GQA is used (HKV < H), un-tie the KV heads so all H heads get
        # independent (HD x HD) memory matrices instead of duplicate copies.
        if self.G > 1 and vsa_untie_gqa:
            eye = torch.eye(self.HD).unsqueeze(0).repeat(self.H, 1, 1)
            self.k_head_mix = nn.Parameter(eye.clone())
            self.v_head_mix = nn.Parameter(eye.clone())
        else:
            self.register_parameter("k_head_mix", None)
            self.register_parameter("v_head_mix", None)

        self.pos_code = nn.Parameter(
            torch.ones(max_T, self.HD) + torch.randn(max_T, self.HD) * (0.1 / self.HD**0.5)
        )
        self.log_gamma = nn.Parameter(torch.zeros(num_heads))
        with torch.no_grad():
            max_horizon = min(float(max_T), 1.0 / max(1e-4, 1.0 - gamma_max))
            horizons = torch.linspace(8.0, max_horizon, num_heads)
            target_g = (1.0 - 1.0 / horizons).clamp(1e-4, gamma_max - 1e-4)
            self.log_gamma.copy_(torch.logit(target_g / gamma_max))

    def gamma(self) -> torch.Tensor:
        return (self.gamma_max * torch.sigmoid(self.log_gamma)).clamp(1e-4, self.gamma_max).float()

    def effective_pos_code(self, T: int, dtype: torch.dtype) -> torch.Tensor:
        """Returns (1, 1, T, HD) position code safe for any length T (even > max_T)."""
        T_use = min(T, self.pos_code.shape[0])
        pc = self.pos_code[:T_use].float()
        if self.stabilize_pos:
            row_rms = pc.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
            pc = pc / row_rms
            over = pc.abs() > 3.0
            if over.any():
                pc = torch.where(
                    over,
                    pc.sign() * 3.0 * (1.0 + torch.tanh((pc.abs() - 3.0) / 3.0)),
                    pc,
                )
        if T > T_use:
            pad = torch.ones(T - T_use, self.HD, device=pc.device, dtype=pc.dtype)
            pc = torch.cat([pc, pad], dim=0)
        return pc.view(1, 1, T, self.HD).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q = self.to_q(x).view(B, T, self.H, self.HD).transpose(1, 2)
        k = self.to_k(x).view(B, T, self.HKV, self.HD).transpose(1, 2)
        v = self.to_v(x).view(B, T, self.HKV, self.HD).transpose(1, 2)
        g = torch.sigmoid(self.to_g(x).view(B, T, self.H, self.HD).transpose(1, 2))
        if self.use_rope:
            q = _rope(q, theta=self.rope_theta)
            k = _rope(k, theta=self.rope_theta)
        k = k * self.effective_pos_code(T, k.dtype)
        kx, vx = _repeat_kv(k, self.G), _repeat_kv(v, self.G)
        if self.k_head_mix is not None:
            kx = torch.einsum("bhtd,hde->bhte", kx, self.k_head_mix.to(kx.dtype))
            vx = torch.einsum("bhtd,hde->bhte", vx, self.v_head_mix.to(vx.dtype))
        y = VSAChunkFunction.apply(q, kx, vx, self.gamma(), self.chunk_size)
        if self.head_norm is not None:
            y = self.head_norm(y)
        return self.to_out((y * g).transpose(1, 2).reshape(B, T, D))


class SoftmaxAttention(nn.Module):
    """Standard causal softmax attention with RoPE, GQA support, and an output gate."""

    def __init__(self, dim: int, num_heads: int, num_kv_heads: int | None = None,
                 rope_theta: float = 10000.0, **kwargs):
        super().__init__()
        self.H = num_heads
        self.HKV = num_kv_heads or num_heads
        self.HD = dim // num_heads
        self.G = self.H // self.HKV
        self.rope_theta = rope_theta
        self.to_q = nn.Linear(dim, self.H * self.HD, bias=False)
        self.to_k = nn.Linear(dim, self.HKV * self.HD, bias=False)
        self.to_v = nn.Linear(dim, self.HKV * self.HD, bias=False)
        self.to_g = nn.Linear(dim, self.H * self.HD, bias=False)
        self.to_out = nn.Linear(self.H * self.HD, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q = self.to_q(x).view(B, T, self.H, self.HD).transpose(1, 2)
        k = self.to_k(x).view(B, T, self.HKV, self.HD).transpose(1, 2)
        v = self.to_v(x).view(B, T, self.HKV, self.HD).transpose(1, 2)
        g = torch.sigmoid(self.to_g(x).view(B, T, self.H, self.HD).transpose(1, 2))
        q = _rope(q, theta=self.rope_theta)
        k = _rope(k, theta=self.rope_theta)
        kx, vx = _repeat_kv(k, self.G), _repeat_kv(v, self.G)
        y = F.scaled_dot_product_attention(q, kx, vx, is_causal=True)
        return self.to_out((y * g).transpose(1, 2).reshape(B, T, D))


class VSABlock(nn.Module):
    """Pre-norm block: attention (VSA or softmax) + gated SwiGLU FFN."""

    def __init__(self, dim, num_heads, max_T, attn_type="vsa", chunk_size=32,
                 num_kv_heads=None, intermediate=None, gamma_max=0.995,
                 stabilize_pos=True, use_rope=False, rope_theta=10000.0,
                 use_head_norm=False, vsa_untie_gqa=True, eps=1e-5, **kwargs):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=eps)
        if attn_type == "vsa":
            self.vsa = TritonVSALayer(
                dim, num_heads, max_T, chunk_size,
                num_kv_heads=num_kv_heads, gamma_max=gamma_max,
                stabilize_pos=stabilize_pos, use_rope=use_rope,
                rope_theta=rope_theta, use_head_norm=use_head_norm,
                vsa_untie_gqa=vsa_untie_gqa,
            )
        else:
            self.softmax_attn = SoftmaxAttention(
                dim, num_heads, num_kv_heads=num_kv_heads, rope_theta=rope_theta
            )
        self.norm2 = nn.RMSNorm(dim, eps=eps)
        hidden = intermediate or int(dim * 8 / 3)
        self.ffn_up = nn.Linear(dim, hidden * 2, bias=False)
        self.ffn_down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.vsa if hasattr(self, "vsa") else self.softmax_attn
        x = x + attn(self.norm1(x))
        u = self.ffn_up(self.norm2(x))
        h = u.shape[-1] // 2
        return x + self.ffn_down(F.silu(u[..., :h]) * u[..., h:])


class VSALanguageModel(nn.Module):
    """Decoder-only LM; `hybrid_every` softmax layers, rest fixed-state VSA.
    Accepts **kwargs so future config keys never break checkpoint loading."""

    def __init__(self, vocab_size=50257, dim=768, num_heads=12, num_layers=8,
                 max_T=512, chunk_size=32, hybrid_every=0, num_kv_heads=None,
                 intermediate=None, gamma_max=0.995, stabilize_pos=True,
                 use_rope=False, rope_theta=10000.0, use_head_norm=False,
                 vsa_untie_gqa=True, eps=1e-5, **extra_cfg):
        super().__init__()
        self.config = dict(
            vocab_size=vocab_size, dim=dim, num_heads=num_heads,
            num_kv_heads=num_kv_heads or num_heads,
            intermediate=intermediate or int(dim * 8 / 3),
            num_layers=num_layers, max_T=max_T, chunk_size=chunk_size,
            hybrid_every=hybrid_every, gamma_max=gamma_max,
            stabilize_pos=stabilize_pos, use_rope=use_rope,
            rope_theta=rope_theta, use_head_norm=use_head_norm,
            vsa_untie_gqa=vsa_untie_gqa, eps=eps, **extra_cfg,
        )
        self.embed = nn.Embedding(vocab_size, dim)
        self.embed.weight.data.normal_(mean=0.0, std=0.02)
        self.blocks = nn.ModuleList()
        for li in range(num_layers):
            t = "softmax" if (hybrid_every > 0 and (li + 1) % hybrid_every == 0) else "vsa"
            self.blocks.append(
                VSABlock(dim, num_heads, max_T, t, chunk_size,
                         num_kv_heads=num_kv_heads, intermediate=intermediate,
                         gamma_max=gamma_max, stabilize_pos=stabilize_pos,
                         use_rope=use_rope, rope_theta=rope_theta,
                         use_head_norm=use_head_norm,
                         vsa_untie_gqa=vsa_untie_gqa, eps=eps)
            )
        self.norm_f = nn.RMSNorm(dim, eps=eps)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        scale = 0.02 / math.sqrt(2 * num_layers)
        for blk in self.blocks:
            proj = blk.vsa.to_out if hasattr(blk, "vsa") else blk.softmax_attn.to_out
            proj.weight.data.normal_(0.0, scale)
            blk.ffn_down.weight.data.normal_(0.0, scale)
        self.lm_head.weight = self.embed.weight

    @classmethod
    def from_config(cls, cfg: dict) -> "VSALanguageModel":
        """Safe constructor that never fails on unknown/legacy config keys."""
        sig = inspect.signature(cls.__init__)
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD
                         for p in sig.parameters.values())
        if has_var_kw:
            return cls(**cfg)
        valid = {k: v for k, v in cfg.items() if k in sig.parameters}
        return cls(**valid)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.norm_f(x)
