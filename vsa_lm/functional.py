"""Autograd wrapper and fp32 reference implementations for VSA attention."""
import torch

from .kernels import vsa_chunk_forward_kernel, vsa_chunk_backward_kernel


class VSAChunkFunction(torch.autograd.Function):
    """Chunkwise VSA attention with exact gradients, including d/dgamma."""

    @staticmethod
    def forward(ctx, q, k, v, gamma, chunk_size):
        B, H, T, D = q.shape
        if T % chunk_size or T < chunk_size:
            raise ValueError(f"sequence length {T} must be a positive multiple of {chunk_size}")
        q_flat = q.reshape(B*H, T, D).contiguous()
        k_flat = k.reshape(B*H, T, D).contiguous()
        v_flat = v.reshape(B*H, T, D).contiguous()
        y_flat = torch.empty(B*H, T, D, device=q.device, dtype=q.dtype)
        num_chunks = T // chunk_size
        # Store M_saved in fp32 when input is fp32 (for 1e-4 oracle tests), or bf16/fp32 during AMP
        m_dtype = torch.bfloat16 if q.dtype == torch.bfloat16 else torch.float32
        M_saved = torch.empty(B*H, max(1, num_chunks - 1), D, D,
                              device=q.device, dtype=m_dtype)
        grid = (B * H,)
        vsa_chunk_forward_kernel[grid](
            q_flat, k_flat, v_flat, y_flat, M_saved, gamma,
            q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
            M_saved.stride(0), M_saved.stride(1), M_saved.stride(2),
            B, H, T, D, BLOCK_SIZE=chunk_size, num_warps=4, num_stages=2)
        ctx.save_for_backward(q, k, v, gamma, M_saved)
        ctx.chunk_size = chunk_size
        return y_flat.view(B, H, T, D)

    @staticmethod
    def backward(ctx, dy):
        q, k, v, gamma, M_saved = ctx.saved_tensors
        B, H, T, D = q.shape
        chunk_size = ctx.chunk_size
        q_flat = q.reshape(B*H, T, D).contiguous()
        k_flat = k.reshape(B*H, T, D).contiguous()
        v_flat = v.reshape(B*H, T, D).contiguous()
        dy_flat = dy.reshape(B*H, T, D).contiguous()
        dq_flat = torch.empty(B*H, T, D, device=q.device, dtype=q.dtype)
        dk_flat = torch.empty_like(dq_flat)
        dv_flat = torch.empty_like(dq_flat)
        dgamma_flat = torch.empty(B*H, device=q.device, dtype=torch.float32)
        grid = (B * H,)
        vsa_chunk_backward_kernel[grid](
            q_flat, k_flat, v_flat, dy_flat, M_saved, gamma,
            dq_flat, dk_flat, dv_flat, dgamma_flat,
            q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
            M_saved.stride(0), M_saved.stride(1), M_saved.stride(2),
            B, H, T, D, BLOCK_SIZE=chunk_size, num_warps=4, num_stages=2)
        dgamma = dgamma_flat.view(B, H).sum(0)
        return dq_flat.view(B, H, T, D), dk_flat.view(B, H, T, D), dv_flat.view(B, H, T, D), dgamma, None


def vsa_parallel_attention(q, k, v, gamma):
    """Dense fp32 reference: decayed causal attention with (1-gamma) read scaling."""
    B, H, T, Dh = q.shape
    t = torch.arange(T, device=q.device, dtype=torch.float32)
    diff = (t[:, None] - t[None, :]).clamp_min(0)
    dec = gamma.view(1, H, 1, 1) ** diff.view(1, 1, T, T)
    attn = (q @ k.transpose(-1, -2)) / (Dh ** 0.5)
    attn = attn * dec * (t[:, None] >= t[None, :])
    return (1.0 - gamma).view(1, H, 1, 1) * (attn @ v)


def vsa_recurrent_reference(q, k, v, gamma):
    """Step-by-step fp32 reference; returns (outputs, final state)."""
    B, H, T, Dh = q.shape
    M = torch.zeros(B, H, Dh, Dh, device=q.device, dtype=torch.float32)
    ys = []
    for t in range(T):
        M = gamma.view(1, H, 1, 1) * M + k[:, :, t, :, None] * v[:, :, t, None, :]
        ys.append(torch.einsum('bhe,bhed->bhd', q[:, :, t], M).unsqueeze(2) / (Dh ** 0.5))
    return torch.cat(ys, dim=2) * (1.0 - gamma).view(1, H, 1, 1), M
