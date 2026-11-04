"""Universal Softmax, VSA, Block, and LM classes."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from ..surgery import LlamaRMSNorm, LlamaRoPE, LlamaMLP, repeat_kv, _vsa_apply
from .inspector import compute_layer_types
from .qwen35 import Qwen35ZeroCenteredRMSNorm, Qwen35PartialRoPE, Qwen35VSAAttn


class UniversalSoftmaxAttn(nn.Module):
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
            self.q_norm, self.k_norm = None, None

        self.g_proj = nn.Linear(D, H * HD, bias=False) if (cfg.get("softmax_gate", False) and not self.packed_q_gate) else None
        partial_factor = float(cfg.get("partial_rotary_factor", 1.0))
        if partial_factor < 1.0:
            self.rope = Qwen35PartialRoPE(HD, partial_factor=partial_factor, theta=cfg["rope_theta"])
        else:
            self.rope = LlamaRoPE(HD, cfg["rope_theta"], cfg.get("rope_scaling"))

    def head_qkv(self, x: torch.Tensor):
        B, T, _ = x.shape
        if self.packed_q_gate:
            q, _ = torch.chunk(self.q_proj(x).view(B, T, self.H, self.HD * 2), 2, dim=-1)
        else:
            q = self.q_proj(x).view(B, T, self.H, self.HD)
        k = self.k_proj(x).view(B, T, self.HKV, self.HD)
        v = self.v_proj(x).view(B, T, self.HKV, self.HD)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        if self.packed_q_gate:
            q, gate = torch.chunk(self.q_proj(x).view(B, T, self.H, self.HD * 2), 2, dim=-1)
            gate = gate.reshape(B, T, self.H * self.HD)
            k = self.k_proj(x).view(B, T, self.HKV, self.HD)
            v = self.v_proj(x).view(B, T, self.HKV, self.HD)
            if self.q_norm is not None:
                q, k = self.q_norm(q), self.k_norm(k)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            q, k = self.rope(q), self.rope(k)
            k, v = repeat_kv(k, self.G), repeat_kv(v, self.G)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.o_proj(y.transpose(1, 2).reshape(B, T, self.H * self.HD) * torch.sigmoid(gate))

        q, k, v = self.head_qkv(x)
        q, k = self.rope(q), self.rope(k)
        k, v = repeat_kv(k, self.G), repeat_kv(v, self.G)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        if self.g_proj is not None:
            y = y * torch.sigmoid(self.g_proj(x).view(B, T, self.H, self.HD).transpose(1, 2))
        return self.o_proj(y.transpose(1, 2).reshape(B, T, self.H * self.HD))


class UniversalVSAAttn(nn.Module):
    def __init__(self, cfg: dict, max_T: int, chunk_size: int = 32):
        super().__init__()
        D = cfg["dim"]
        H = cfg.get("vsa_num_heads") or cfg["num_heads"]
        HKV = cfg.get("vsa_num_kv_heads") or cfg["num_kv_heads"]
        HD = cfg.get("vsa_head_dim") or cfg.get("head_dim") or (D // H)
        while HD > 128 and H * 2 <= 64:
            H, HKV, HD = H * 2, HKV * 2, HD // 2

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
            horizons = torch.logspace(math.log10(4.0), math.log10(2000.0), H)
            self.log_gamma.copy_(torch.logit((1.0 - 1.0 / horizons).clamp(0.5, 0.9995) / 0.9999))
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
        B, T, _ = x.shape
        q, k, v = self.head_qkv(x)
        q, k = self.rope(q), self.rope(k)

        T_use = min(T, self.pos_code.shape[0])
        pc = self.pos_code[:T_use].view(1, 1, T_use, self.HD).to(k.dtype)
        if T > T_use:
            pc = torch.cat([pc, torch.ones(1, 1, T - T_use, self.HD, device=k.device, dtype=k.dtype)], dim=2)
        kc = k * pc
        kx, vx = repeat_kv(kc, self.G), repeat_kv(v, self.G)

        if hasattr(self, "k_head_mix"):
            kx = torch.einsum("bhtd,hde->bhte", kx, self.k_head_mix.to(kx.dtype))
            T_d = min(T, self.head_pos_delta.shape[1])
            delta = (1.0 + 0.5 * torch.tanh(self.head_pos_delta[:, :T_d, :])).unsqueeze(0).to(kx.dtype)
            if T > T_d:
                delta = torch.cat([delta, torch.ones(1, self.H, T - T_d, self.HD, device=kx.device, dtype=kx.dtype)], dim=2)
            kx = kx * delta
            vx = torch.einsum("bhtd,hde->bhte", vx, self.v_head_mix.to(vx.dtype))

        if hasattr(self, "k_conv"):
            kx_flat = kx.permute(0, 1, 3, 2).reshape(B, self.H * self.HD, T)
            k_loc = self.k_conv(kx_flat)[..., :T].view(B, self.H, self.HD, T).permute(0, 1, 3, 2)
            kx = kx + k_loc
            q = q + self.q_relu_mix.to(q.dtype) * F.relu(q)
            kx = kx + self.k_relu_mix.to(kx.dtype) * F.relu(kx)

        gam = self.gamma()
        y = _vsa_apply(q, kx, vx, gam, self.chunk_size)
        if hasattr(self, "v_self_scale"):
            y = y + self.v_self_scale.view(1, self.H, 1, 1).to(y.dtype) * vx
        if hasattr(self, "v_conv"):
            vx_flat = vx.permute(0, 1, 3, 2).reshape(B, self.H * self.HD, T)
            v_loc = self.v_conv(vx_flat)[..., :T].view(B, self.H, self.HD, T).permute(0, 1, 3, 2)
            q_mod = torch.sigmoid((q * self.local_q_gate.view(1, self.H, 1, self.HD).to(q.dtype)).sum(-1, keepdim=True))
            y = y + (2.0 * q_mod) * v_loc
        if hasattr(self, "g_proj"):
            y = y * (2.0 * torch.sigmoid(self.g_proj(x).view(B, T, self.H, self.HD).transpose(1, 2)))
        return self.o_proj(y.transpose(1, 2).reshape(B, T, self.H * self.HD))


class UniversalBlock(nn.Module):
    def __init__(self, cfg: dict, attn_type: str, max_T: int, chunk_size: int):
        super().__init__()
        norm_cls = Qwen35ZeroCenteredRMSNorm if cfg.get("has_qwen35_lin", False) else LlamaRMSNorm
        self.norm1 = norm_cls(cfg["dim"], cfg["eps"])
        if attn_type == "gated_deltanet":
            self.vsa = Qwen35VSAAttn(cfg, max_T, chunk_size, mode="gated_deltanet")
        elif attn_type == "vsa":
            if cfg.get("has_qwen35_lin", False) and not cfg.get("donor_layer_types"):
                self.vsa = Qwen35VSAAttn(cfg, max_T, chunk_size, mode="vsa")
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
    def __init__(self, vocab_size: int, dim: int, num_layers: int, num_heads: int,
                 num_kv_heads: int, intermediate: int, max_T: int = 1024,
                 chunk_size: int = 32, hybrid_every: int = 4,
                 rope_theta: float = 10000.0, rope_scaling: dict | None = None,
                 eps: float = 1e-5, head_dim: int | None = None,
                 qkv_bias: bool = False, tie_word_embeddings: bool = True,
                 vsa_untie_gqa: bool = True, vsa_local_conv: bool = True, **extra_cfg):
        super().__init__()
        head_dim = head_dim or (dim // num_heads)
        layer_types = extra_cfg.pop("layer_types", None) or compute_layer_types(
            num_layers, hybrid_every, extra_cfg.get("donor_layer_types")
        )
        self.config = dict(
            vocab_size=vocab_size, dim=dim, num_layers=num_layers,
            num_heads=num_heads, num_kv_heads=num_kv_heads,
            intermediate=intermediate, max_T=max_T, chunk_size=chunk_size,
            hybrid_every=hybrid_every, rope_theta=rope_theta,
            rope_scaling=rope_scaling, eps=eps, head_dim=head_dim,
            qkv_bias=qkv_bias, tie_word_embeddings=tie_word_embeddings,
            vsa_untie_gqa=vsa_untie_gqa, vsa_local_conv=vsa_local_conv,
            layer_types=layer_types, **extra_cfg,
        )
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList()
        for li in range(num_layers):
            self.blocks.append(UniversalBlock(self.config, layer_types[li], max_T, chunk_size))
        norm_cls = Qwen35ZeroCenteredRMSNorm if self.config.get("has_qwen35_lin", False) else LlamaRMSNorm
        self.norm = norm_cls(dim, eps)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        if tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.grad_checkpoint = False

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for blk in self.blocks:
            x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False) if (self.grad_checkpoint and self.training) else blk(x)
        return self.norm(x)
