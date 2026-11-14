"""Autograd wrapper and fp32/CPU reference implementations for VSA attention."""
import torch
import torch.nn.functional as F

from .kernels import HAS_TRITON, vsa_chunk_forward_kernel, vsa_chunk_backward_kernel, vsa_recurrent_step_kernel


class VSAChunkFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, gamma, chunk_size):
        if not HAS_TRITON or not q.is_cuda:
            return vsa_parallel_attention(q.float(), k.float(), v.float(), gamma.float()).to(q.dtype)
        B, H, T, D = q.shape
        if q.dtype in (torch.bfloat16, torch.float16) and T >= 32:
            eff_chunk = 32 if chunk_size >= 32 else chunk_size
        else:
            eff_chunk = 16 if (D >= 128 and chunk_size >= 16 and T >= 16) else chunk_size
        if T < 1:
            raise ValueError(f"sequence length {T} must be positive")
        q_flat = q.reshape(B*H, T, D).contiguous()
        k_flat = k.reshape(B*H, T, D).contiguous()
        v_flat = v.reshape(B*H, T, D).contiguous()
        y_flat = torch.empty(B*H, T, D, device=q.device, dtype=q.dtype)
        num_chunks = (T + eff_chunk - 1) // eff_chunk
        m_dtype = torch.bfloat16 if q.dtype == torch.bfloat16 else torch.float32
        M_saved = torch.empty(B*H, max(1, num_chunks - 1), D, D,
                              device=q.device, dtype=m_dtype)
        grid = (B * H,)
        vsa_chunk_forward_kernel[grid](
            q_flat, k_flat, v_flat, y_flat, M_saved, gamma,
            q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
            M_saved.stride(0), M_saved.stride(1), M_saved.stride(2),
            B, H, T, D, BLOCK_SIZE=eff_chunk, num_warps=4, num_stages=(1 if D >= 128 else 2))
        y_out = y_flat.view(B, H, T, D)
        ctx.save_for_backward(q, k, v, y_out, gamma, M_saved)
        ctx.chunk_size = eff_chunk
        return y_out

    @staticmethod
    def backward(ctx, dy):
        q, k, v, y_saved, gamma, M_saved = ctx.saved_tensors
        B, H, T, D = q.shape
        chunk_size = ctx.chunk_size
        q_flat = q.reshape(B*H, T, D).contiguous()
        k_flat = k.reshape(B*H, T, D).contiguous()
        v_flat = v.reshape(B*H, T, D).contiguous()
        y_flat = y_saved.reshape(B*H, T, D).contiguous()
        dy_flat = dy.reshape(B*H, T, D).contiguous()
        dq_flat = torch.empty(B*H, T, D, device=q.device, dtype=q.dtype)
        dk_flat = torch.empty_like(dq_flat)
        dv_flat = torch.empty_like(dq_flat)
        dgamma_flat = torch.empty(B*H, device=q.device, dtype=torch.float32)
        grid = (B * H,)
        vsa_chunk_backward_kernel[grid](
            q_flat, k_flat, v_flat, y_flat, dy_flat, M_saved, gamma,
            dq_flat, dk_flat, dv_flat, dgamma_flat,
            q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
            M_saved.stride(0), M_saved.stride(1), M_saved.stride(2),
            B, H, T, D, BLOCK_SIZE=chunk_size, num_warps=4, num_stages=(1 if D >= 128 else 2))
        dgamma = dgamma_flat.view(B, H).sum(0)
        return dq_flat.view(B, H, T, D), dk_flat.view(B, H, T, D), dv_flat.view(B, H, T, D), dgamma, None


def fused_vsa_recurrent_step(q: torch.Tensor, kx: torch.Tensor, vx: torch.Tensor,
                             M: torch.Tensor, gam: torch.Tensor, omg_scale: torch.Tensor,
                             v_self: torch.Tensor, q_relu: torch.Tensor | None = None,
                             k_relu: torch.Tensor | None = None, norm_w: torch.Tensor | None = None,
                             v_loc: torch.Tensor | None = None, local_q_gate: torch.Tensor | None = None,
                             k_bar: torch.Tensor | None = None, v_bar: torch.Tensor | None = None,
                             z_bar: torch.Tensor | None = None, center_alpha: torch.Tensor | None = None, norm_eps: float = 1e-6) -> torch.Tensor:
    B, H, _, D = q.shape
    has_center = (k_bar is not None and v_bar is not None and z_bar is not None and center_alpha is not None)
    if not HAS_TRITON or not q.is_cuda:
        qf, kf, vf = q.float(), kx.float(), vx.float()
        if q_relu is not None:
            qf = qf + q_relu.view(1, H, 1, 1) * F.relu(qf)
        if k_relu is not None:
            kf = kf + k_relu.view(1, H, 1, 1) * F.relu(kf)
        qf = F.normalize(qf, p=2.0, dim=-1) * (D ** 0.5)
        kf = F.normalize(kf, p=2.0, dim=-1)
        M.mul_(gam.view(1, H, 1, 1)).add_(torch.matmul(kf.transpose(-1, -2), vf))
        y = torch.matmul(qf, M) * omg_scale.view(1, H, 1, 1)
        if has_center:
            k_bar.mul_(gam.view(1, H, 1, 1)).add_(kf)
            v_bar.mul_(gam.view(1, H, 1, 1)).add_(vf)
            z_bar.mul_(gam.view(1, H, 1, 1)).add_(1.0)
            kq_dot = (k_bar * qf).sum(dim=-1, keepdim=True)
            y = y - center_alpha.float() * omg_scale.view(1, H, 1, 1) * (kq_dot / (z_bar + 1e-6)) * v_bar
        if norm_w is not None:
            y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + norm_eps) * norm_w.view(1, 1, 1, D)
        y = y + v_self.view(1, H, 1, 1) * vf
        if v_loc is not None and local_q_gate is not None:
            q_mod = torch.sigmoid((qf * local_q_gate.view(1, H, 1, D)).sum(-1, keepdim=True))
            y = y + (2.0 * q_mod) * v_loc.float()
        return y.to(q.dtype)

    out = torch.empty_like(q)
    z_h = torch.zeros_like(gam)
    qr = q_relu.view(-1) if q_relu is not None else z_h
    kr = k_relu.view(-1) if k_relu is not None else z_h
    has_norm = norm_w is not None
    nw = norm_w if has_norm else z_h
    has_vloc = (v_loc is not None and local_q_gate is not None)
    vl = v_loc.contiguous() if has_vloc else vx
    lqg = local_q_gate if has_vloc else z_h
    kb_p = k_bar if has_center else q
    vb_p = v_bar if has_center else vx
    zb_p = z_bar if has_center else z_h
    ca_p = center_alpha.contiguous() if has_center else z_h
    vsa_recurrent_step_kernel[(B * H,)](
        q, kx, vx, vl, M, out, gam, omg_scale, v_self, qr, kr, nw, lqg,
        kb_p, vb_p, zb_p, ca_p,
        HAS_NORM=has_norm, HAS_VLOC=has_vloc, HAS_CENTER=has_center, H=H, D=D, EPS=float(norm_eps), num_warps=4,
    )
    return out


def vsa_parallel_attention(q, k, v, gamma):
    B, H, T, Dh = q.shape
    t = torch.arange(T, device=q.device, dtype=torch.float32)
    diff = (t[:, None] - t[None, :]).clamp_min(0)
    dec = gamma.view(1, H, 1, 1) ** diff.view(1, 1, T, T)
    attn = (q @ k.transpose(-1, -2)) / (Dh ** 0.5)
    attn = attn * dec * (t[:, None] >= t[None, :])
    return (1.0 - gamma).view(1, H, 1, 1) * (attn @ v)


def vsa_recurrent_reference(q, k, v, gamma):
    B, H, T, Dh = q.shape
    M = torch.zeros(B, H, Dh, Dh, device=q.device, dtype=torch.float32)
    ys = []
    for t in range(T):
        M = gamma.view(1, H, 1, 1) * M + k[:, :, t, :, None] * v[:, :, t, None, :]
        ys.append(torch.einsum('bhe,bhed->bhd', q[:, :, t], M).unsqueeze(2) / (Dh ** 0.5))
    return torch.cat(ys, dim=2) * (1.0 - gamma).view(1, H, 1, 1), M
