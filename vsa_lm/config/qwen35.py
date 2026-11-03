"""Qwen3.5 hybrid linear-attention, zero-centered RMSNorm, partial RoPE, and Vision sidecar utilities."""
from __future__ import annotations

import glob
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..surgery import _vsa_apply


class Qwen35ZeroCenteredRMSNorm(nn.Module):
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
    def __init__(self, cfg: dict, max_T: int, chunk_size: int = 32):
        super().__init__()
        D = cfg["dim"]
        HK = cfg.get("vsa_num_kv_heads", 16)
        HV = cfg.get("vsa_num_heads", 32)
        HD = cfg.get("vsa_head_dim", 128)
        conv_k = int(cfg.get("linear_conv_kernel_dim", 4))
        eps = float(cfg.get("eps", 1e-6))

        self.H, self.HKV, self.HD, self.G = HV, HK, HD, HV // HK
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
        v = v.view(B, T, self.H, self.HD) * torch.sigmoid(self.in_proj_b(x)).unsqueeze(-1)

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

        y = _vsa_apply(q, kx, vx, self.gamma(), self.chunk_size)
        y = y + self.v_self_scale.view(1, self.H, 1, 1).to(y.dtype) * vx
        y_flat = y.transpose(1, 2).reshape(-1, self.HD)
        z_flat = self.in_proj_z(x).reshape(-1, self.HD)
        return self.out_proj(self.norm(y_flat, z_flat).reshape(B, T, self.val_dim))


def preserve_visual_tower(donor_dir: str, out_sidecar_path: str) -> tuple[int, float]:
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
