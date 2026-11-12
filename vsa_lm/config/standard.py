"""Universal Softmax, VSA (with Abelian Page-Slot Bank & Mix-First RoPE), Block, and LM classes."""
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
            try:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=(self.G > 1))
            except TypeError:
                y = F.scaled_dot_product_attention(q, repeat_kv(k, self.G), repeat_kv(v, self.G), is_causal=True)
            return self.o_proj(y.transpose(1, 2).reshape(B, T, self.H * self.HD) * torch.sigmoid(gate))

        q, k, v = self.head_qkv(x)
        q, k = self.rope(q), self.rope(k)
        try:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=(self.G > 1))
        except TypeError:
            y = F.scaled_dot_product_attention(q, repeat_kv(k, self.G), repeat_kv(v, self.G), is_causal=True)
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
        if HD & (HD - 1) != 0:
            raise ValueError(f"VSA head_dim must be power of two, got {HD}")

        qkv_bias = bool(cfg.get("vsa_qkv_bias", cfg.get("qkv_bias", False)))
        self.H, self.HKV, self.HD, self.G = H, HKV, HD, H // HKV
        self.max_T = max_T
        self.mix_before_rope = bool(cfg.get("vsa_mix_before_rope", False))
        self.q_proj = nn.Linear(D, H * HD, bias=qkv_bias)
        self.k_proj = nn.Linear(D, HKV * HD, bias=qkv_bias)
        self.v_proj = nn.Linear(D, HKV * HD, bias=qkv_bias)
        self.o_proj = nn.Linear(H * HD, D, bias=False)

        norm_cls = Qwen35ZeroCenteredRMSNorm if cfg.get("has_qwen35_lin", False) else LlamaRMSNorm
        if cfg.get("qk_norm", False) and HD == (cfg.get("head_dim") or (D // cfg["num_heads"])):
            self.q_norm = norm_cls(HD, cfg["eps"])
            self.k_norm = norm_cls(HD, cfg["eps"])
        else:
            self.q_norm, self.k_norm = None, None

        partial_factor = float(cfg.get("partial_rotary_factor", 1.0))
        if partial_factor < 1.0:
            self.rope = Qwen35PartialRoPE(HD, partial_factor=partial_factor, theta=cfg["rope_theta"])
        else:
            self.rope = LlamaRoPE(HD, cfg["rope_theta"], cfg.get("rope_scaling"))
        self.out_norm = LlamaRMSNorm(HD, cfg["eps"])

        self.log_gamma = nn.Parameter(torch.zeros(H))
        with torch.no_grad():
            max_h = float(max(256, max_T))
            horizons = torch.logspace(math.log10(4.0), math.log10(max_h), H)
            self.log_gamma.copy_(torch.logit((1.0 - 1.0 / horizons).clamp(0.5, 0.9995) / 0.9999))
        self.chunk_size = chunk_size

        if cfg.get("vsa_untie_gqa", True):
            eye = torch.eye(HD).unsqueeze(0).repeat(H, 1, 1)
            self.k_head_mix = nn.Parameter(eye.clone())
            self.v_head_mix = nn.Parameter(eye.clone())
            self.g_proj = nn.Linear(D, H * HD, bias=False)
            nn.init.zeros_(self.g_proj.weight)
            self.v_self_scale = nn.Parameter(torch.zeros(H))

        self.swa_window = int(cfg.get("vsa_swa_window", 0) or 0)
        if self.swa_window > 0:
            self.swa_mix_logit = nn.Parameter(torch.full((H,), 2.0))

        if cfg.get("vsa_local_conv", True):
            self.v_conv = nn.Conv1d(H * HD, H * HD, kernel_size=8, padding=7, groups=H * HD, bias=False)
            nn.init.zeros_(self.v_conv.weight)
            self.k_conv = nn.Conv1d(H * HD, H * HD, kernel_size=4, padding=3, groups=H * HD, bias=False)
            nn.init.zeros_(self.k_conv.weight)
            self.local_q_gate = nn.Parameter(torch.zeros(H, HD))
            self.q_relu_mix = nn.Parameter(torch.zeros(H, 1, 1))
            self.k_relu_mix = nn.Parameter(torch.zeros(H, 1, 1))

        # Native Abelian Page-Slot Bank (vsa_num_slots=32 -> 8 pages x 4 slots per KV head)
        self.num_slots = int(cfg.get("vsa_num_slots", 0) or 0)
        self.num_pages = int(cfg.get("vsa_num_pages", 8) or 8)
        if self.num_slots > 0:
            self.slot_router = nn.Linear(D, HKV * self.num_slots, bias=True)
            nn.init.normal_(self.slot_router.weight, std=0.01)
            nn.init.constant_(self.slot_router.bias, -3.0)
            self.slot_gate = nn.Parameter(torch.zeros(H))

    def head_qkv(self, x: torch.Tensor):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.H, self.HD)
        k = self.k_proj(x).view(B, T, self.HKV, self.HD)
        v = self.v_proj(x).view(B, T, self.HKV, self.HD)
        if getattr(self, "q_norm", None) is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def gamma(self) -> torch.Tensor:
        return (0.9999 * torch.sigmoid(self.log_gamma)).clamp(1e-4, 0.9999).float()

    def _slot_bank_forward(self, x: torch.Tensor, q_raw: torch.Tensor, k_raw: torch.Tensor, v_raw: torch.Tensor) -> torch.Tensor:
        """Fast bf16 chunk-vectorized Abelian Page-Slot prefix sum (3x faster training)."""
        B, T, _ = x.shape
        S = self.num_slots
        dt = x.dtype
        r_raw = self.slot_router(x).view(B, T, self.HKV, S).permute(0, 2, 1, 3).float()
        w = (F.softmax(r_raw, dim=-1) * torch.sigmoid(r_raw)).to(dt)
        k_c = F.normalize(k_raw.float(), p=2.0, dim=-1).to(dt)
        q_c = F.normalize(q_raw.float(), p=2.0, dim=-1).to(dt)
        v_c = v_raw.to(dt)

        C_t = (w.unsqueeze(-1) * k_c.unsqueeze(-2)).cumsum(dim=2)
        U_t = (w.unsqueeze(-1) * v_c.unsqueeze(-2)).cumsum(dim=2)
        Z_t = (w.float().cumsum(dim=2) + 1e-5).to(dt).unsqueeze(-1)

        K_slot = C_t / Z_t
        V_slot = U_t / Z_t
        qb = q_c.view(B, self.HKV, self.G, T, self.HD)
        scores = torch.einsum("bhgtd,bhtsd->bhgts", qb, K_slot).float() * (self.HD ** 0.5)
        probs = F.softmax(scores, dim=-1).to(dt)
        return torch.einsum("bhgts,bhtsd->bhgtd", probs, V_slot).reshape(B, self.H, T, self.HD)

    def window_attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        T = q.shape[2]
        pos = torch.arange(T, device=q.device)
        dist = pos[:, None] - pos[None, :]
        mask = (dist >= 0) & ((dist < self.swa_window) | (pos[None, :] == 0))
        kr, vr = repeat_kv(k, self.G), repeat_kv(v, self.G)
        if T <= self.max_T or not hasattr(self.rope, "inv_freq"):
            return F.scaled_dot_product_attention(q, kr, vr, attn_mask=mask)
        shift = (pos + 1 - self.max_T).clamp(min=0).float()
        freqs = torch.outer(-shift, self.rope.inv_freq.to(q.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().to(q.dtype)[None, None], emb.sin().to(q.dtype)[None, None]
        q_Half = q.shape[-1] // 2
        q_sink = q * cos + torch.cat((-q[..., q_Half:], q[..., :q_Half]), dim=-1) * sin
        win_mask = (dist >= 0) & (dist < self.swa_window) & (pos[None, :] > 0)
        scores = torch.matmul(q.float(), kr.float().transpose(-1, -2)) * (self.HD ** -0.5)
        scores = scores.masked_fill(~win_mask, float("-inf"))
        sink_score = (q_sink.float() * kr[:, :, :1].float()).sum(dim=-1, keepdim=True) * (self.HD ** -0.5)
        scores[:, :, :, :1] = sink_score
        probs = F.softmax(scores, dim=-1).to(vr.dtype)
        return torch.matmul(probs, vr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q_raw, k_raw, v_raw = self.head_qkv(x)
        y_slot = self._slot_bank_forward(x, q_raw, k_raw, v_raw) if self.num_slots > 0 else None

        q, k = self.rope(q_raw), self.rope(k_raw)
        y_swa = self.window_attn(q, k, v_raw) if self.swa_window > 0 else None

        if self.mix_before_rope and hasattr(self, "k_head_mix"):
            kx = torch.einsum("bhtd,hde->bhte", repeat_kv(k_raw, self.G), self.k_head_mix.to(k_raw.dtype))
            kx = self.rope(kx)
            vx = torch.einsum("bhtd,hde->bhte", repeat_kv(v_raw, self.G), self.v_head_mix.to(v_raw.dtype))
        else:
            kx, vx = repeat_kv(k, self.G), repeat_kv(v_raw, self.G)
            if hasattr(self, "k_head_mix"):
                kx = torch.einsum("bhtd,hde->bhte", kx, self.k_head_mix.to(kx.dtype))
                vx = torch.einsum("bhtd,hde->bhte", vx, self.v_head_mix.to(vx.dtype))

        if hasattr(self, "k_conv"):
            kx_flat = kx.permute(0, 1, 3, 2).reshape(B, self.H * self.HD, T)
            k_loc = self.k_conv(kx_flat)[..., :T].view(B, self.H, self.HD, T).permute(0, 1, 3, 2)
            kx = kx + k_loc
            q = q + self.q_relu_mix.to(q.dtype) * F.relu(q)
            kx = kx + self.k_relu_mix.to(kx.dtype) * F.relu(kx)

        q = F.normalize(q.float(), p=2.0, dim=-1).to(x.dtype) * (self.HD ** 0.5)
        kx = F.normalize(kx.float(), p=2.0, dim=-1).to(x.dtype)

        gam = self.gamma()
        y = _vsa_apply(q, kx, vx, gam, self.chunk_size)
        y = self.out_norm(y)

        if hasattr(self, "v_self_scale"):
            y = y + self.v_self_scale.view(1, self.H, 1, 1).to(y.dtype) * vx
        if hasattr(self, "v_conv"):
            vx_flat = vx.permute(0, 1, 3, 2).reshape(B, self.H * self.HD, T)
            v_loc = self.v_conv(vx_flat)[..., :T].view(B, self.H, self.HD, T).permute(0, 1, 3, 2)
            q_mod = torch.sigmoid((q * self.local_q_gate.view(1, self.H, 1, self.HD).to(q.dtype)).sum(-1, keepdim=True))
            y = y + (2.0 * q_mod) * v_loc
        if y_swa is not None:
            a = torch.sigmoid(self.swa_mix_logit.float()).view(1, self.H, 1, 1).to(y.dtype)
            y = y + a * (y_swa.to(y.dtype) - y)
        if y_slot is not None:
            sg = torch.tanh(self.slot_gate.float()).view(1, self.H, 1, 1).to(y.dtype)
            y = y + sg * y_slot
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
                 num_kv_heads: int, intermediate: int, max_T: int = 2048,
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
