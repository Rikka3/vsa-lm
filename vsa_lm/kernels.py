"""Triton kernels for chunkwise VSA attention."""
"""Triton kernels for chunkwise VSA attention (with CPU no-Triton fallback)."""
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    class _DummyTriton:
        @staticmethod
        def jit(fn):
            return fn
    class _DummyTL:
        constexpr = int
        float32 = float
    triton = _DummyTriton()
    tl = _DummyTL()

@triton.jit
def _dot(x, y):
    if x.dtype == tl.float32:
        return tl.dot(x, y, input_precision="ieee")
    return tl.dot(x, y)


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

    q_decay_powers = tl.cast(local_t + 1, tl.float32)
    local_decay_q = tl.exp(q_decay_powers * log_gamma)
    diff = local_t[:, None] - local_t[None, :]
    diff_clamped = tl.where(diff >= 0, diff, 0).to(tl.float32)
    intra_decay = tl.exp(diff_clamped * log_gamma)
    inv_sqrt_d = 1.0 / (D ** 0.5)

    for i in range(0, T, BLOCK_SIZE):
        c_len = tl.minimum(BLOCK_SIZE, T - i)
        t_offsets = i + t_offsets_base
        t_mask = t_offsets < T
        load_mask = t_mask[:, None]

        q_raw = tl.load(Q + bh_offset + t_offsets[:, None]*stride_qt + d_offsets[None,:]*stride_qd, mask=load_mask, other=0.0)
        k_raw = tl.load(K + bh_offset + t_offsets[:, None]*stride_qt + d_offsets[None,:]*stride_qd, mask=load_mask, other=0.0)
        v_raw = tl.load(V + bh_offset + t_offsets[:, None]*stride_qt + d_offsets[None,:]*stride_qd, mask=load_mask, other=0.0)

        M_dot = M.to(q_raw.dtype)
        y_carry = _dot(q_raw, M_dot) * (local_decay_q[:, None] * inv_sqrt_d)
        attn = _dot(q_raw, tl.trans(k_raw)) * inv_sqrt_d
        causal_mask = (diff >= 0) & (t_mask[:, None] & t_mask[None, :])
        attn_masked = tl.where(causal_mask, attn * intra_decay, 0.0)
        y_intra = _dot(attn_masked.to(v_raw.dtype), v_raw)

        y = (y_carry + y_intra) * one_minus_gamma
        y_ptrs = Y + bh_offset + t_offsets[:, None]*stride_qt + d_offsets[None,:]*stride_qd
        tl.store(y_ptrs, y.to(Y.dtype.element_ty), mask=load_mask)

        tail_powers = tl.maximum(0.0, tl.cast(c_len - 1 - local_t, tl.float32))
        c_decay = tl.where(t_mask, tl.exp(tail_powers * log_gamma), 0.0)
        c_total_decay = tl.exp(tl.cast(c_len, tl.float32) * log_gamma)
        v_weighted = (v_raw.to(tl.float32) * c_decay[:, None]).to(k_raw.dtype)
        delta_M = _dot(tl.trans(k_raw), v_weighted)
        M = c_total_decay * M + delta_M

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

    q_decay_powers = tl.cast(local_t + 1, tl.float32)
    local_decay_q = tl.exp(q_decay_powers * log_gamma)

    diff = local_t[:, None] - local_t[None, :]
    diff_clamped = tl.where(diff >= 0, diff, 0).to(tl.float32)
    intra_decay = tl.exp(diff_clamped * log_gamma)
    intra_decay_dot = tl.where(diff >= 1, diff_clamped * tl.exp((diff_clamped - 1.0) * log_gamma), 0.0)

    num_chunks = tl.cdiv(T, BLOCK_SIZE)
    inv_sqrt_d = 1.0 / (D ** 0.5)

    for c_idx in range(num_chunks - 1, -1, -1):
        i = c_idx * BLOCK_SIZE
        c_len = tl.minimum(BLOCK_SIZE, T - i)
        t_offsets = i + t_offsets_base
        t_mask = t_offsets < T
        load_mask = t_mask[:, None]

        cp_c = tl.maximum(0.0, tl.cast(c_len - 1 - local_t, tl.float32))
        chunk_decay = tl.where(t_mask, tl.exp(cp_c * log_gamma), 0.0)
        chunk_total_decay = tl.exp(tl.cast(c_len, tl.float32) * log_gamma)
        chunk_total_decay_dot = tl.cast(c_len, tl.float32) * tl.exp((tl.cast(c_len, tl.float32) - 1.0) * log_gamma)
        chunk_decay_dot = tl.where(t_mask & (cp_c >= 1.0), cp_c * tl.exp((cp_c - 1.0) * log_gamma), 0.0)

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

        attn = _dot(q_raw, tl.trans(k_raw)) * inv_sqrt_d
        causal_mask = (diff >= 0) & (t_mask[:, None] & t_mask[None, :])
        attn_masked = tl.where(causal_mask, attn * intra_decay, 0.0)
        v_weighted = (v_raw.to(tl.float32) * chunk_decay[:, None]).to(q_raw.dtype)

        dy_scaled = (dy * local_decay_q[:, None]).to(q_raw.dtype)
        dM_direct = _dot(tl.trans(q_raw), dy_scaled) * inv_sqrt_d
        dM = dM_direct + chunk_total_decay * dM_acc

        dQ_carry = _dot(dy_dot, tl.trans(M_dot)) * (local_decay_q[:, None] * inv_sqrt_d)
        dA = _dot(dy_dot, tl.trans(v_raw))
        dA_masked = tl.where(causal_mask, dA * intra_decay, 0.0)
        dA_masked_dot = dA_masked.to(q_raw.dtype)
        dQ_intra = _dot(dA_masked_dot, k_raw) * inv_sqrt_d
        dQ_chunk = dQ_carry + dQ_intra

        dK_intra = _dot(tl.trans(dA_masked_dot), q_raw) * inv_sqrt_d
        dK_rec = _dot(v_weighted, tl.trans(dM_acc_dot))
        dK_chunk = dK_intra + dK_rec

        dV_intra = _dot(tl.trans(attn_masked.to(q_raw.dtype)), dy_dot)
        k_weighted = (k_raw.to(tl.float32) * chunk_decay[:, None]).to(q_raw.dtype)
        dV_rec = _dot(k_weighted, dM_acc_dot)
        dV_chunk = dV_intra + dV_rec

        dg_acc += tl.sum(dQ_carry * q_raw.to(tl.float32) * (q_decay_powers / gamma)[:, None])
        attn_dot = tl.where(causal_mask, attn * intra_decay_dot, 0.0)
        dg_acc += tl.sum(attn_dot * dA)
        dg_acc += chunk_total_decay_dot * tl.sum(dM_acc * M)
        v_dot_w = (v_raw.to(tl.float32) * chunk_decay_dot[:, None]).to(q_raw.dtype)
        dg_acc += tl.sum(_dot(tl.trans(k_raw), v_dot_w) * dM_acc)
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
    Q_ptr, K_ptr, V_ptr, VLOC_ptr, M_ptr, Out_ptr,
    GAM_ptr, OMG_SCALE_ptr, VSELF_ptr,
    Q_RELU_ptr, K_RELU_ptr, NORM_W_ptr, LQG_ptr,
    HAS_NORM: tl.constexpr, HAS_VLOC: tl.constexpr,
    H: tl.constexpr, D: tl.constexpr,
):
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

    if HAS_VLOC:
        v_loc = tl.load(VLOC_ptr + vec_off).to(tl.float32)
        lqg = tl.load(LQG_ptr + h_idx * D + d_idx).to(tl.float32)
        q_mod = tl.sigmoid(tl.sum(q * lqg, axis=0))
        y = y + (2.0 * q_mod) * v_loc

    tl.store(Out_ptr + vec_off, y.to(Out_ptr.dtype.element_ty))
