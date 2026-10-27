"""Model architecture: VSA layers, softmax layers, transformer blocks, LM head."""
from __future__ import annotations

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


class TritonVSALayer(nn.Module):
    """Fixed-state decayed linear attention with per-head learnable decay.

    Upgrades (v2):
      - gamma_max (default 0.995 for new models, 0.9999 legacy) prevents the
        sigmoid saturation trap where (1 - gamma) -> 1e-4 freezes gradients.
      - stabilize_pos (default True) bounds pos_code spikes and normalizes
        per-position RMS so late context positions are never drowned out, and
        gracefully extends past max_T with 1.0 (unlocking long-context decode).
      - use_rope / use_head_norm (optional) add relative position + per-head
        RMSNorm before output projection.
    """

    def __init__(self, dim: int, num_heads: int, max_T: int, chunk_size: int = 32,
                 gamma_max: float = 0.995, stabilize_pos: bool = True,
                 use_rope: bool = False, use_head_norm: bool = False):
        super().__init__()
        self.H, self.HD = num_heads, dim // num_heads
        self.max_T = max_T
        self.chunk_size = chunk_size
        self.gamma_max = gamma_max
        self.stabilize_pos = stabilize_pos
        self.use_rope = use_rope
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_g = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.head_norm = nn.RMSNorm(self.HD) if use_head_norm else None
        # Initialize pos_code near 1.0 (multiplicative identity) + small noise
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
            # 1. Equalize per-position row RMS to 1.0 (prevents late-context blindness)
            row_rms = pc.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
            pc = pc / row_rms
            # 2. Smoothly clip extreme spikes > 3.0 RMS (prevents 27,426x explosion)
            over = pc.abs() > 3.0
            if over.any():
                pc = torch.where(
                    over,
                    pc.sign() * 3.0 * (1.0 + torch.tanh((pc.abs() - 3.0) / 3.0)),
                    pc,
                )
        if T > T_use:
            # Past max_T: neutral 1.0 multiplicative code instead of crashing/wrapping
            pad = torch.ones(T - T_use, self.HD, device=pc.device, dtype=pc.dtype)
            pc = torch.cat([pc, pad], dim=0)
        return pc.view(1, 1, T, self.HD).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, HD = self.H, self.HD
        q = self.to_q(x).view(B, T, H, HD).transpose(1, 2)
        k = self.to_k(x).view(B, T, H, HD).transpose(1, 2)
        v = self.to_v(x).view(B, T, H, HD).transpose(1, 2)
        g = torch.sigmoid(self.to_g(x).view(B, T, H, HD).transpose(1, 2))
        if self.use_rope:
            q, k = _rope(q), _rope(k)
        k = k * self.effective_pos_code(T, k.dtype)
        y = VSAChunkFunction.apply(q, k, v, self.gamma(), self.chunk_size)
        if self.head_norm is not None:
            y = self.head_norm(y)
        return self.to_out((y * g).transpose(1, 2).reshape(B, T, D))


class SoftmaxAttention(nn.Module):
    """Standard causal softmax attention with RoPE and an output gate."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.H, self.HD = num_heads, dim // num_heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_g = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, HD = self.H, self.HD
        q = self.to_q(x).view(B, T, H, HD).transpose(1, 2)
        k = self.to_k(x).view(B, T, H, HD).transpose(1, 2)
        v = self.to_v(x).view(B, T, H, HD).transpose(1, 2)
        g = torch.sigmoid(self.to_g(x).view(B, T, H, HD).transpose(1, 2))
        q, k = _rope(q), _rope(k)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.to_out((y * g).transpose(1, 2).reshape(B, T, D))


class VSABlock(nn.Module):
    """Pre-norm block: attention (VSA or softmax) + gated SwiGLU FFN."""

    def __init__(self, dim, num_heads, max_T, attn_type="vsa", chunk_size=32,
                 gamma_max=0.995, stabilize_pos=True, use_rope=False,
                 use_head_norm=False):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        if attn_type == "vsa":
            self.vsa = TritonVSALayer(
                dim, num_heads, max_T, chunk_size,
                gamma_max=gamma_max, stabilize_pos=stabilize_pos,
                use_rope=use_rope, use_head_norm=use_head_norm,
            )
        else:
            self.softmax_attn = SoftmaxAttention(dim, num_heads)
        self.norm2 = nn.RMSNorm(dim)
        hidden = int(dim * 8 / 3)
        self.ffn_up = nn.Linear(dim, hidden * 2, bias=False)
        self.ffn_down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.vsa if hasattr(self, "vsa") else self.softmax_attn
        x = x + attn(self.norm1(x))
        u = self.ffn_up(self.norm2(x))
        h = u.shape[-1] // 2
        return x + self.ffn_down(F.silu(u[..., :h]) * u[..., h:])


class VSALanguageModel(nn.Module):
    """Decoder-only LM; `hybrid_every` softmax layers, rest fixed-state VSA."""

    def __init__(self, vocab_size=50257, dim=768, num_heads=12, num_layers=8,
                 max_T=512, chunk_size=32, hybrid_every=0,
                 gamma_max=0.995, stabilize_pos=True, use_rope=False,
                 use_head_norm=False):
        super().__init__()
        self.config = dict(
            vocab_size=vocab_size, dim=dim, num_heads=num_heads,
            num_layers=num_layers, max_T=max_T, chunk_size=chunk_size,
            hybrid_every=hybrid_every, gamma_max=gamma_max,
            stabilize_pos=stabilize_pos, use_rope=use_rope,
            use_head_norm=use_head_norm,
        )
        self.embed = nn.Embedding(vocab_size, dim)
        self.embed.weight.data.normal_(mean=0.0, std=0.02)
        self.blocks = nn.ModuleList()
        for li in range(num_layers):
            t = "softmax" if (hybrid_every > 0 and (li + 1) % hybrid_every == 0) else "vsa"
            self.blocks.append(
                VSABlock(dim, num_heads, max_T, t, chunk_size,
                         gamma_max=gamma_max, stabilize_pos=stabilize_pos,
                         use_rope=use_rope, use_head_norm=use_head_norm)
            )
        self.norm_f = nn.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        scale = 0.02 / math.sqrt(2 * num_layers)   # residual-branch init
        for blk in self.blocks:
            proj = blk.vsa.to_out if hasattr(blk, "vsa") else blk.softmax_attn.to_out
            proj.weight.data.normal_(0.0, scale)
            blk.ffn_down.weight.data.normal_(0.0, scale)
        self.lm_head.weight = self.embed.weight    # tied embeddings

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.norm_f(x)
