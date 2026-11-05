"""Triton kernels for chunkwise VSA attention (BF16 TensorCore-accelerated forward scan, Y_saved adjoint backward, and O(1) decode)."""
import triton
import triton.language as tl


@triton.jit
def vsa_chunk_forward_kernel(
    Q, K, V, Y, M_out, GAMMA,
    stride_qbh, stride_qt, stride_qd,
    stride_mbh, stride_mc, stride_mr,
    B, H, T,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_bh = tl.program_id(axis=0)
    bh_offset = pid_bh * stride_qbh
    M_offset = pid_bh * stride_mbh
    head_idx = pid_bh % H
    gamma = tl.load(GAMMA + head_idx).to(tl.float32)
    one_minus_gamma = 1.0 - gamma

    M = tl.zeros((D, D), dtype=tl.float32)
    log_gamma = tl.log(gamma)
    t_offsets_base = tl.arange(0, BLOCK_SIZE)
    d_offsets = tl.arange(0, D)
    local_t = tl.arange(0, BLOCK_SIZE)

    decay_powers = tl.cast(BLOCK_SIZE - 1 - local_t, tl.float32)
    chunk_decay = tl.exp(decay_powers * log_gamma)
    q_decay_powers = tl.cast(local_t + 1, tl.float32)
    local_decay_q = tl.exp(q_decay_powers * log_gamma)
    diff = local_t[:, None] - local_t[None, :]
    diff_clamped = tl.where(diff >= 0, diff, 0).to(tl.float32)
    intra_decay = tl.exp(diff_clamped * log_gamma)
    chunk_total_decay = tl.exp(tl.cast(BLOCK_SIZE, tl.float32) * log_gamma)
    inv_sqrt_d = 1.0 / (D ** 0.5)

    for i in range(0, T, BLOCK_SIZE):
        t_offsets = i + t_offsets_base
        t_mask = t_offsets < T
        load_mask = t_mask[:, None]

        q_raw = tl.load(Q + bh_offset + t_offsets[:, None]*stride_qt + d_offsets[None,:]*stride_qd, mask=load_mask, other=0.0)
        k_raw = tl.load(K + bh_offset + t_offsets[:, None]*stride_qt + d_offsets[None,:]*stride_qd, mask=load_mask, other=0.0)
        v_raw = tl.load(V + bh_offset + t_offsets[:, None]*stride_qt + d_offsets[None,:]*stride_qd, mask=load_mask, other=0.0)

        # Use native dtype (bf16/fp16 on TensorCores, fp32 in fp32 tests) for tl.dot with fp32 accumulation
        M_dot = M.to(q_raw.dtype)
        y_carry = tl.dot(q_raw, M_dot) * (local_decay_q[:, None] * inv_sqrt_d)
        attn = tl.dot(q_raw, tl.trans(k_raw)) * inv_sqrt_d
        causal_mask = (diff >= 0) & (t_mask[:, None] & t_mask[None, :])
        attn_masked = tl.where(causal_mask, attn * intra_decay, 0.0)
        y_intra = tl.dot(attn_masked.to(v_raw.dtype), v_raw)

        y = (y_carry + y_intra) * one_minus_gamma
        y_ptrs = Y + bh_offset + t_offsets[:, None]*stride_qt + d_offsets[None,:]*stride_qd
        tl.store(y_ptrs, y.to(Y.dtype.element_ty), mask=load_mask)

        v_weighted = (v_raw.to(tl.float32) * chunk_decay[:, None]).to(k_raw.dtype)
        delta_M = tl.dot(tl.trans(k_raw), v_weighted)
        M = chunk_total_decay * M + delta_M

        if i + BLOCK_SIZE < T:
            M_ptrs = M_out + M_offset + (i // BLOCK_SIZE) * stride_mc + d_offsets[:, None]*stride_mr + d_offsets[None,:]
            tl.store(M_ptrs, M.to(M_out.dtype.element_ty))


@triton.jit
def vsa_chunk_backward_kernel(
    Q, K, V, Y_saved, dY, M_saved, GAMMA,
    dQ, dK, dV, dGAMMA,
    stride_qbh, stride_qt, stride_qd,
    stride_mbh, stride_mc, stride_mr,
    B, H, T,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_bh = tl.program_id(axis=0)
    bh_offset = pid_bh * stride_qbh
    M_offset = pid_bh * stride_mbh
    head_idx = pid_bh % H
    gamma = tl.load(GAMMA + head_idx).to(tl.float32)
    one_minus_gamma = 1.0 - gamma
    inv_omg = 1.0 / tl.maximum(one_minus_gamma, 1e-7)

    dM_acc = tl.zeros((D, D), dtype=tl.float32)
    dg_acc = 0.0

    log_gamma = tl.log(gamma)
    t_offsets_base = tl.arange(0, BLOCK_SIZE)
    d_offsets = tl.arange(0, D)
    local_t = tl.arange(0, BLOCK_SIZE)

    decay_powers = tl.cast(BLOCK_SIZE - 1 - local_t, tl.float32)
    chunk_decay = tl.exp(decay_powers * log_gamma)
    chunk_total_decay = tl.exp(tl.cast(BLOCK_SIZE, tl.float32) * log_gamma)
    chunk_total_decay_dot = tl.cast(BLOCK_SIZE, tl.float32) * \
        tl.exp((tl.cast(BLOCK_SIZE, tl.float32) - 1.0) * log_gamma)
    cp = tl.cast(BLOCK_SIZE - 1 - local_t, tl.float32)
    chunk_decay_dot = tl.where(cp >= 1.0, cp * tl.exp((cp - 1.0) * log_gamma), 0.0)

    q_decay_powers = tl.cast(local_t + 1, tl.float32)
    local_decay_q = tl.exp(q_decay_powers * log_gamma)

    diff = local_t[:, None] - local_t[None, :]
    diff_clamped = tl.where(diff >= 0, diff, 0).to(tl.float32)
    intra_decay = tl.exp(diff_clamped * log_gamma)
    intra_decay_dot = tl.where(diff >= 1, diff_clamped * tl.exp((diff_clamped - 1.0) * log_gamma), 0.0)

    num_chunks = T // BLOCK_SIZE
    inv_sqrt_d = 1.0 / (D ** 0.5)

    for c_idx in range(num_chunks - 1, -1, -1):
        i = c_idx * BLOCK_SIZE
        t_offsets = i + t_offsets_base
        t_mask = t_offsets < T
        load_mask = t_mask[:, None]

        q_raw = tl.load(Q + bh_offset + t_offsets[:, None]*stride_qt + d_offsets[None,:]*stride_qd, mask=load_mask, other=0.0)
        k_raw = tl.load(K + bh_offset + t_offsets[:, None]*stride_qt + d_offsets[None,:]*stride_qd, mask=load_mask, other=0.0)
        v_raw = tl.load(V + bh_offset + t_offsets[:, None]*stride_qt + d_offsets[None,:]*stride_qd, mask=load_mask, other=0.0)
        y_sav = tl.load(Y_saved + bh_offset + t_offsets[:, None]*stride_qt + d_offsets[None,:]*stride_qd, mask=load_mask, other=0.0).to(tl.float32)
        dy_out = tl.load(dY + bh_offset + t_offsets[:, None]*stride_qt + d_offsets[None,:]*stride_qd, mask=load_mask, other=0.0).to(tl.float32)
        dy = dy_out * one_minus_gamma
        dy_dot = dy.to(q_raw.dtype)

        if c_idx == 0:
            M = tl.zeros((D, D), dtype=tl.float32)
        else:
            M = tl.load(M_saved + M_offset + (c_idx - 1) * stride_mc + d_offsets[:, None]*stride_mr + d_offsets[None,:]).to(tl.float32)
        M_dot = M.to(q_raw.dtype)
        dM_acc_dot = dM_acc.to(q_raw.dtype)

        attn = tl.dot(q_raw, tl.trans(k_raw)) * inv_sqrt_d
        causal_mask = (diff >= 0) & (t_mask[:, None] & t_mask[None, :])
        attn_masked = tl.where(causal_mask, attn * intra_decay, 0.0)
        v_weighted = (v_raw.to(tl.float32) * chunk_decay[:, None]).to(q_raw.dtype)

        dy_scaled = (dy * local_decay_q[:, None]).to(q_raw.dtype)
        dM_direct = tl.dot(tl.trans(q_raw), dy_scaled) * inv_sqrt_d
        dM = dM_direct + chunk_total_decay * dM_acc

        dQ_carry = tl.dot(dy_dot, tl.trans(M_dot)) * (local_decay_q[:, None] * inv_sqrt_d)
        dA = tl.dot(dy_dot, tl.trans(v_raw))
        dA_masked = tl.where(causal_mask, dA * intra_decay, 0.0)
        dA_masked_dot = dA_masked.to(q_raw.dtype)
        dQ_intra = tl.dot(dA_masked_dot, k_raw) * inv_sqrt_d
        dQ_chunk = dQ_carry + dQ_intra

        dK_intra = tl.dot(tl.trans(dA_masked_dot), q_raw) * inv_sqrt_d
        dK_rec = tl.dot(v_weighted, tl.trans(dM_acc_dot))
        dK_chunk = dK_intra + dK_rec

        dV_intra = tl.dot(tl.trans(attn_masked.to(q_raw.dtype)), dy_dot)
        k_weighted = (k_raw.to(tl.float32) * chunk_decay[:, None]).to(q_raw.dtype)
        dV_rec = tl.dot(k_weighted, dM_acc_dot)
        dV_chunk = dV_intra + dV_rec

        dg_acc += tl.sum(dQ_carry * q_raw.to(tl.float32) * (q_decay_powers / gamma)[:, None])
        attn_dot = tl.where(causal_mask, attn * intra_decay_dot, 0.0)
        dg_acc += tl.sum(attn_dot * dA)
        dg_acc += chunk_total_decay_dot * tl.sum(dM_acc * M)
        v_dot_w = (v_raw.to(tl.float32) * chunk_decay_dot[:, None]).to(q_raw.dtype)
        dg_acc += tl.sum(tl.dot(tl.trans(k_raw), v_dot_w) * dM_acc)
        # E7 optimization: reuse Y_saved * inv_omg instead of recomputing y_carry + y_intra!
        dg_acc -= tl.sum(dy_out * (y_sav * inv_omg))

        tl.store(dQ + bh_offset + t_offsets[:, None]*stride_qt + d_offsets[None,:]*stride_qd,
                 dQ_chunk.to(dQ.dtype.element_ty), mask=load_mask)
        tl.store(dK + bh_offset + t_offsets[:, None]*stride_qt + d_offsets[None,:]*stride_qd,
                 dK_chunk.to(dK.dtype.element_ty), mask=load_mask)
        tl.store(dV + bh_offset + t_offsets[:, None]*stride_qt + d_offsets[None,:]*stride_qd,
                 dV_chunk.to(dV.dtype.element_ty), mask=load_mask)

        dM_acc = dM

    tl.store(dGAMMA + pid_bh, dg_acc)


@triton.jit
def vsa_recurrent_step_kernel(
    Q_ptr, K_ptr, V_ptr, M_ptr, Out_ptr,
    GAM_ptr, OMG_SCALE_ptr, VSELF_ptr,
    Q_RELU_ptr, K_RELU_ptr, NORM_W_ptr,
    HAS_NORM: tl.constexpr,
    H: tl.constexpr, D: tl.constexpr,
):
    """Single-launch fused O(1) VSA step: ReLU sharpener + L2-norm + M update + RMSNorm readout."""
    pid = tl.program_id(0)
    h_idx = pid % H

    d_idx = tl.arange(0, D)
    vec_off = pid * D + d_idx
    mat_off = pid * D * D + d_idx[:, None] * D + d_idx[None, :]

    q = tl.load(Q_ptr + vec_off).to(tl.float32)
    k = tl.load(K_ptr + vec_off).to(tl.float32)
    v = tl.load(V_ptr + vec_off).to(tl.float32)

    qr = tl.load(Q_RELU_ptr + h_idx).to(tl.float32)
    kr = tl.load(K_RELU_ptr + h_idx).to(tl.float32)
    q = q + qr * tl.maximum(q, 0.0)
    k = k + kr * tl.maximum(k, 0.0)

    q_inv = tl.rsqrt(tl.sum(q * q, axis=0) + 1e-12) * (D ** 0.5)
    k_inv = tl.rsqrt(tl.sum(k * k, axis=0) + 1e-12)
    q = q * q_inv
    k = k * k_inv

    gam = tl.load(GAM_ptr + h_idx).to(tl.float32)
    omg_s = tl.load(OMG_SCALE_ptr + h_idx).to(tl.float32)
    v_self = tl.load(VSELF_ptr + h_idx).to(tl.float32)

    M = tl.load(M_ptr + mat_off).to(tl.float32)
    M_new = gam * M + k[:, None] * v[None, :]
    tl.store(M_ptr + mat_off, M_new.to(M_ptr.dtype.element_ty))

    y = tl.sum(M_new * q[:, None], axis=0) * omg_s
    if HAS_NORM:
        nw = tl.load(NORM_W_ptr + d_idx).to(tl.float32)
        y = y * tl.rsqrt(tl.sum(y * y, axis=0) / D + 1e-6) * nw
    y = y + v_self * v
    tl.store(Out_ptr + vec_off, y.to(Out_ptr.dtype.element_ty))
