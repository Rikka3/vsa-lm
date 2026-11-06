"""O(1)-state recurrent decoder with StaticKV, enable_gqa=True SDPA, and fully fused Triton step."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .functional import vsa_parallel_attention, fused_vsa_recurrent_step
from .model import _rope


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    B, H, T, HD = x.shape
    return x[:, :, None, :, :].expand(B, H, n_rep, T, HD).reshape(B, H * n_rep, T, HD)


def _apply_cached_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


class StaticKV:
    def __init__(self, B: int, HKV: int, Tmax: int, HD: int, dev, dt=torch.bfloat16):
        self.k = torch.empty(B, HKV, Tmax, HD, device=dev, dtype=dt)
        self.v = torch.empty_like(self.k)
        self.n = 0

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor):
        L = k_new.shape[2]
        if self.n + L > self.k.shape[2]:
            new_max = max(self.n + L + 1024, self.k.shape[2] * 2)
            nk = torch.empty(self.k.shape[0], self.k.shape[1], new_max, self.k.shape[3], device=self.k.device, dtype=self.k.dtype)
            nv = torch.empty_like(nk)
            nk[:, :, :self.n] = self.k[:, :, :self.n]
            nv[:, :, :self.n] = self.v[:, :, :self.n]
            self.k, self.v = nk, nv
        self.k[:, :, self.n : self.n + L] = k_new
        self.v[:, :, self.n : self.n + L] = v_new
        self.n += L
        return self.k[:, :, :self.n], self.v[:, :, :self.n]


class GraphKV:
    """Fixed-capacity softmax KV cache for CUDA-graph decoding: the new key/value is written at a device-side
    position and attention runs over the whole buffer with a validity mask, so every step has the same shapes.
    SDPA's fused kernel accumulates scores in fp32, matching the eager SDPA path."""

    def __init__(self, B: int, HKV: int, Tmax: int, HD: int, dev, dt=torch.bfloat16):
        self.k = torch.zeros(B, HKV, Tmax, HD, device=dev, dtype=dt)
        self.v = torch.zeros_like(self.k)
        self.ar = torch.arange(Tmax, device=dev)

    def attend(self, q: torch.Tensor, k_new: torch.Tensor, v_new: torch.Tensor, pos: torch.Tensor, G: int) -> torch.Tensor:
        self.k.index_copy_(2, pos, k_new.to(self.k.dtype))
        self.v.index_copy_(2, pos, v_new.to(self.v.dtype))
        B, HKV, T, HD = self.k.shape
        # The G query heads sharing a KV head become G "query positions" of that head, so SDPA needs no GQA
        # expansion, and reading the bf16 buffer directly lets its fused kernel accumulate scores in fp32.
        # (Upcasting the whole buffer to fp32 each step cost ~2 GB of traffic per step at B=32, T=4096.)
        qg = q.to(self.k.dtype).reshape(B, HKV, G, HD)
        mask = (self.ar <= pos).view(1, 1, 1, T)
        y = F.scaled_dot_product_attention(qg, self.k, self.v, attn_mask=mask)
        return y.reshape(B, HKV * G, 1, HD).to(q.dtype)


class _CUDAGraphStep:
    """One captured decode step (token -> logits) for a fixed batch size and KV capacity. Replaying it launches
    the whole 28-layer step at once instead of ~1000 small kernels, which is what limits batch-1 decoding.
    Only valid for generation that starts from an empty state at position 0 (generate / generate_batch)."""

    def __init__(self, dec: "VSARecurrentDecoder", B: int, Tmax: int, device):
        m = dec.model
        self.states, self.kv = dec.blank_state(m, B, device, max_kv_len=Tmax, graph=True)
        self.tok = torch.zeros(B, 1, dtype=torch.long, device=device)
        self.pos = torch.zeros(1, dtype=torch.long, device=device)
        # Keep the packed decode tables this graph reads alive, even if a later decoder rebuilds them.
        self._keep = [t for blk in m.blocks for ly in (getattr(blk, "vsa", None), getattr(blk, "attn", None))
                      if ly is not None for t in vars(ly).values() if torch.is_tensor(t)]

        def run():
            with torch.no_grad():  # no autograd bookkeeping may be recorded into the graph
                h, _, _ = dec.step(self.tok, self.pos, self.states, self.kv)
                return m.lm_head(h[:, -1, :])

        side = torch.cuda.Stream(device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):  # warm-up: Triton compile + allocator settle, outside the capture
            for _ in range(2):
                run()
        torch.cuda.current_stream(device).wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.logits = run()
        self.reset()

    def reset(self):
        VSARecurrentDecoder.reset_state(self.states, self.kv)

    def __call__(self, tok: torch.Tensor, pos: int) -> torch.Tensor:
        self.tok.copy_(tok)
        self.pos.fill_(pos)
        self.graph.replay()
        return self.logits


class VSARecurrentDecoder:
    def __init__(self, model, wrap_positions: bool = False, max_cache_pos: int = 65536, use_cuda_graph: bool = True):
        self.model = model.eval()
        self.wrap_positions = wrap_positions
        self.max_T = model.config["max_T"]
        self.is_universal = hasattr(model, "embed_tokens")
        self.max_cache_pos = max_cache_pos
        self._prepare_fast_decode_tables(max_cache_pos)
        self.use_cuda_graph = use_cuda_graph and self._graph_supported()
        self._graphs: dict = {}

    def _graph_supported(self) -> bool:
        """CUDA graphs need every layer on the packed fast path with cached RoPE (no host-side branching)."""
        if not self.is_universal or not torch.cuda.is_available():
            return False
        for blk in self.model.blocks:
            ly = blk.vsa if hasattr(blk, "vsa") else getattr(blk, "attn", None)
            if ly is None or not hasattr(ly, "_wqkv_packed") or not hasattr(ly, "_rope_cos"):
                return False
        return True

    def _graph_runner(self, B: int, need_len: int, device):
        """A reset CUDA-graph step for batch B with room for need_len tokens, or None (eager fallback).
        Capacity is rounded up to 1024 so one capture serves many prompts; only the two most recent
        captures are kept, since each owns a full KV buffer."""
        if not self.use_cuda_graph or device.type != "cuda":
            return None
        Tmax = -(-need_len // 1024) * 1024
        if Tmax > self.max_cache_pos:
            return None
        key = (B, Tmax)
        runner = self._graphs.pop(key, None)
        if runner is None:
            while len(self._graphs) >= 2:
                self._graphs.pop(next(iter(self._graphs)))
            try:
                runner = _CUDAGraphStep(self, B, Tmax, device)
            except Exception as e:  # never fail generation over a capture problem
                print(f"[vsa-decode] CUDA graph capture failed ({type(e).__name__}: {e}); using eager decode.", flush=True)
                self.use_cuda_graph = False
                return None
        else:
            runner.reset()
        self._graphs[key] = runner
        return runner

    def _start(self, ids: torch.Tensor, max_new_tokens: int):
        """Prefill `ids` from an empty state. Returns (logits for the next token, feed(tok, pos) -> logits),
        through a CUDA graph when possible and the eager step otherwise."""
        B, T = ids.shape
        runner = self._graph_runner(B, T + max_new_tokens + 1, ids.device)
        if runner is not None:
            logits = None
            for t in range(T):
                logits = runner(ids[:, t : t + 1], t)
            return logits, runner
        x, states, kv = self.prefill(ids, max_kv_len=T + max_new_tokens + 64)

        def feed(tok, pos):
            h, _, _ = self.step(tok, pos, states, kv)
            return self.model.lm_head(h[:, -1, :])

        return self.model.lm_head(x[:, -1, :]), feed

    @torch.no_grad()
    def _prepare_fast_decode_tables(self, max_pos: int):
        if not self.is_universal:
            return
        device = next(self.model.parameters()).device
        dt = torch.bfloat16 if device.type == "cuda" else torch.float32
        self.model.lm_head.weight.data = self.model.lm_head.weight.data.to(dt)
        t = torch.arange(max_pos, device=device, dtype=torch.float32)

        rope_cache = {}
        for blk in self.model.blocks:
            ly = blk.vsa if hasattr(blk, "vsa") else blk.attn
            if hasattr(ly, "rope") and hasattr(ly.rope, "inv_freq"):
                key = (ly.HD, id(ly.rope.inv_freq))
                if key not in rope_cache:
                    freqs = torch.outer(t, ly.rope.inv_freq.to(device))
                    emb = torch.cat((freqs, freqs), dim=-1)
                    rope_cache[key] = (emb.cos().to(dt), emb.sin().to(dt))
                ly._rope_cos, ly._rope_sin = rope_cache[key]

            if hasattr(blk, "vsa"):
                H, HKV, HD, G = ly.H, ly.HKV, ly.HD, ly.G
                gam = ly.gamma().detach()
                ly._cached_gam = gam
                ly._cached_omg = (1.0 - gam) / (HD ** 0.5)
                ly._cached_vself = ly.v_self_scale.detach().float() if hasattr(ly, "v_self_scale") else torch.zeros_like(gam)
                ly._cached_qr = ly.q_relu_mix.detach().float().view(-1) if hasattr(ly, "q_relu_mix") else torch.zeros_like(gam)
                ly._cached_kr = ly.k_relu_mix.detach().float().view(-1) if hasattr(ly, "k_relu_mix") else torch.zeros_like(gam)
                ly._cached_nw = ly.out_norm.weight.detach().float() if hasattr(ly, "out_norm") else None
                ly._cached_lqg = ly.local_q_gate.detach().float().contiguous() if hasattr(ly, "local_q_gate") else None
                ly._cached_swa_a = torch.sigmoid(ly.swa_mix_logit.detach().float()) if getattr(ly, "swa_window", 0) > 0 else None

                if hasattr(ly, "v_head_mix") and hasattr(ly, "v_proj"):
                    wv = ly.v_proj.weight.detach().to(torch.float32).view(HKV, HD, -1)
                    wv_rep = wv[:, None, :, :].expand(HKV, G, HD, -1).reshape(H, HD, -1)
                    vm_T = ly.v_head_mix.detach().to(torch.float32).transpose(1, 2)
                    wv_fused = torch.bmm(vm_T, wv_rep).reshape(H * HD, -1).to(dt)
                    wq = ly.q_proj.weight.detach().to(dt)
                    wk = ly.k_proj.weight.detach().to(dt)
                    # Skip a branch only when it is exactly zero (zero-init, never trained). A magnitude
                    # threshold silently drops small-but-real weights: g_proj sums over D inputs, so even
                    # max|w| <= 1e-4 moves the 2*sigmoid gate by several percent (seen as ~14% error in layer 0).
                    g_act = bool(hasattr(ly, "g_proj") and ly.g_proj.weight.any().item())
                    ly._g_active = g_act
                    # The sliding window reads the raw (un-mixed) v; packing it here saves one matmul per VSA layer.
                    raw_v = getattr(ly, "swa_window", 0) > 0
                    w_parts, sizes = [wq, wk, wv_fused], [H * HD, HKV * HD, H * HD]
                    if g_act:
                        w_parts.append(ly.g_proj.weight.detach().to(dt))
                        sizes.append(H * HD)
                    if raw_v:
                        w_parts.append(ly.v_proj.weight.detach().to(dt))
                        sizes.append(HKV * HD)
                    ly._wqkv_packed = torch.cat(w_parts, dim=0).contiguous()
                    ly._qkv_sizes = sizes
                    ly._raw_v_packed = raw_v

                    if ly.q_proj.bias is not None:
                        bq = ly.q_proj.bias.detach().to(dt)
                        bk = ly.k_proj.bias.detach().to(dt)
                        bv = ly.v_proj.bias.detach().to(torch.float32).view(HKV, 1, HD).expand(HKV, G, HD).reshape(H, 1, HD)
                        bv_fused = torch.bmm(bv, ly.v_head_mix.detach().to(torch.float32)).reshape(H * HD).to(dt)
                        b_parts = [bq, bk, bv_fused]
                        if g_act:
                            b_parts.append(torch.zeros(H * HD, device=device, dtype=dt))
                        if raw_v:
                            b_parts.append(ly.v_proj.bias.detach().to(dt))
                        ly._bqkv_packed = torch.cat(b_parts, dim=0).contiguous()
                    else:
                        ly._bqkv_packed = None
                    ly._km_dt = ly.k_head_mix.detach().to(dt).contiguous()

                if hasattr(ly, "k_conv"):
                    ly._k_conv_w = ly.k_conv.weight.detach().to(dt).view(1, H, 1, HD, 4)
                    ly._v_conv_w = ly.v_conv.weight.detach().to(dt).view(1, H, 1, HD, 8)
                    ly._k_conv_active = bool(ly.k_conv.weight.any().item())
                    ly._v_conv_active = bool(ly.v_conv.weight.any().item())
            else:
                if hasattr(ly, "q_proj") and not getattr(ly, "packed_q_gate", False):
                    wq = ly.q_proj.weight.detach().to(dt)
                    wk = ly.k_proj.weight.detach().to(dt)
                    wv = ly.v_proj.weight.detach().to(dt)
                    ly._wqkv_packed = torch.cat([wq, wk, wv], dim=0).contiguous()
                    if ly.q_proj.bias is not None:
                        ly._bqkv_packed = torch.cat([ly.q_proj.bias.detach().to(dt), ly.k_proj.bias.detach().to(dt), ly.v_proj.bias.detach().to(dt)], dim=0).contiguous()
                    else:
                        ly._bqkv_packed = None

    @staticmethod
    def state_bytes(model):
        vsa_bytes, kv_per_token = 0, 0
        for blk in model.blocks:
            if hasattr(blk, "vsa"):
                v = blk.vsa
                vsa_bytes += v.H * v.HD * v.HD * 4
                if getattr(v, "swa_window", 0) > 0:
                    vsa_bytes += 2 * (v.swa_window + 1) * v.HKV * v.HD * 2
            else:
                s = blk.attn if hasattr(blk, "attn") else blk.softmax_attn
                hk = getattr(s, "HKV", s.H)
                kv_per_token += 2 * hk * s.HD * 2
        return vsa_bytes, kv_per_token

    @staticmethod
    def blank_state(model, batch_size: int, device, max_kv_len: int = 4096, state_dtype=torch.float32, graph: bool = False):
        """Per-layer decode state. graph=True gives softmax layers a fixed-capacity GraphKV (CUDA-graph safe)."""
        states, kv_caches = [None] * len(model.blocks), [None] * len(model.blocks)
        dt = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        for li, blk in enumerate(model.blocks):
            if hasattr(blk, "vsa"):
                v = blk.vsa
                states[li] = torch.zeros(batch_size, v.H, v.HD, v.HD, device=device, dtype=state_dtype)
                cache = {}
                if hasattr(v, "k_conv") and hasattr(v, "v_conv"):
                    cache["k_buf"] = torch.zeros(batch_size, v.H, 1, v.HD, 3, device=device, dtype=dt)
                    cache["v_buf"] = torch.zeros(batch_size, v.H, 1, v.HD, 7, device=device, dtype=dt)
                if getattr(v, "swa_window", 0) > 0:
                    W = v.swa_window
                    cache["swa_k"] = torch.zeros(batch_size, v.HKV, W + 1, v.HD, device=device, dtype=dt)
                    cache["swa_v"] = torch.zeros_like(cache["swa_k"])
                    # Tokens seen so far, kept on device so a step has no host-side branching (CUDA-graph safe).
                    cache["swa_n"] = torch.zeros(1, dtype=torch.long, device=device)
                    cache["swa_ar"] = torch.arange(W + 1, device=device)
                kv_caches[li] = cache or None
            else:
                s = blk.attn if hasattr(blk, "attn") else blk.softmax_attn
                hk = getattr(s, "HKV", s.H)
                kv_cls = GraphKV if graph else StaticKV
                kv_caches[li] = kv_cls(batch_size, hk, max_kv_len, s.HD, device, dt=dt)
        return states, kv_caches

    @staticmethod
    def reset_state(states, kv_caches):
        """Zero a decode state in place (keeps every buffer's address, as a captured CUDA graph requires)."""
        for s in states:
            if s is not None:
                s.zero_()
        for c in kv_caches:
            if isinstance(c, dict):
                for key, t in c.items():
                    if torch.is_tensor(t) and key != "swa_ar":
                        t.zero_()
            elif isinstance(c, GraphKV):
                c.k.zero_()
                c.v.zero_()
            elif isinstance(c, StaticKV):
                c.n = 0

    def _gamma(self, layer):
        return getattr(layer, "_cached_gam", None) or layer.gamma()

    def _proj(self, blk, h, B, T):
        layer = blk.vsa if hasattr(blk, "vsa") else blk.softmax_attn
        H, HD = layer.H, layer.HD
        q = layer.to_q(h).view(B, T, H, HD).transpose(1, 2)
        k = layer.to_k(h).view(B, T, H, HD).transpose(1, 2)
        v = layer.to_v(h).view(B, T, H, HD).transpose(1, 2)
        g = torch.sigmoid(layer.to_g(h).view(B, T, H, HD).transpose(1, 2))
        return layer, q, k, v, g

    @staticmethod
    def _sample(logits, temperature: float, top_k: int = 40, top_p: float = 0.95):
        if temperature <= 1e-5:
            return logits.argmax(-1, keepdim=True)
        probs = F.softmax(logits.float() / temperature, dim=-1)
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = cum_probs - sorted_probs > top_p
            sorted_probs[mask] = 0.0
            sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12))
            next_idx = torch.multinomial(sorted_probs, 1)
            return sorted_indices.gather(-1, next_idx)
        topv, topi = torch.topk(logits.float() / temperature, min(top_k, logits.size(-1)), dim=-1)
        return topi.gather(-1, torch.multinomial(F.softmax(topv, -1), 1))

    @staticmethod
    def _window_attn_step(cache: dict, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, W: int, G: int) -> torch.Tensor:
        """Zero-copy 65-token sliding-window + sink attention (slot 0 = sink, slots 1..W = ring buffer).
        The token counter lives on device and unfilled slots are masked, so there are no data-dependent
        shapes or host syncs (CUDA-graph safe)."""
        n = cache["swa_n"]
        if not torch.is_tensor(n):  # states saved before the counter moved to device
            n = cache["swa_n"] = torch.tensor([int(n)], dtype=torch.long, device=k.device)
            cache["swa_ar"] = torch.arange(W + 1, device=k.device)
        slot = ((n - 1).remainder(W) + 1) * (n != 0)  # token 0 -> sink slot 0, token p>=1 -> 1 + (p-1) % W
        cache["swa_k"].index_copy_(2, slot, k.to(cache["swa_k"].dtype))
        cache["swa_v"].index_copy_(2, slot, v.to(cache["swa_v"].dtype))
        n.add_(1)
        valid = cache["swa_ar"] < n  # slots 0..min(n, W+1)-1 hold real tokens
        pk, pv = cache["swa_k"], cache["swa_v"]
        # For m <= 65, direct GQA broadcast matmul is 2.5x faster than SDPA dispatcher overhead!
        # Scores and the weighted sum run in fp32 with autocast off: training's SDPA keeps scores in fp32,
        # while a bf16 matmul rounds them BEFORE softmax. Layer 0's q.k logits reach the hundreds (attention
        # sink), where bf16 steps are 1-2 units, so the old path gave a different softmax there.
        B, HKV, _, HD = pk.shape
        with torch.autocast(q.device.type, enabled=False):
            qg = q.float().reshape(B, HKV, G, 1, HD)
            scores = torch.matmul(qg, pk.float().unsqueeze(2).transpose(-1, -2)) * (HD ** -0.5)
            probs = F.softmax(scores.masked_fill(~valid, float("-inf")), dim=-1)
            out = torch.matmul(probs, pv.float().unsqueeze(2)).view(B, HKV * G, 1, HD)
        return out.to(q.dtype)

    def _step_universal_block(self, li: int, blk, x: torch.Tensor, pos: int, states, kv_caches, record=None, write_scale: float = 1.0):
        B = x.shape[0]
        h = blk.norm1(x)
        if hasattr(blk, "vsa"):
            ly = blk.vsa
            H, HKV, HD, G = ly.H, ly.HKV, ly.HD, ly.G
            g_proj_out = None
            v_raw = None
            if hasattr(ly, "_wqkv_packed"):
                parts = torch.split(F.linear(h, ly._wqkv_packed, ly._bqkv_packed), ly._qkv_sizes, dim=-1)
                q, k, vx = parts[0], parts[1], parts[2]
                nxt = 3
                if ly._g_active:
                    g_proj_out = parts[nxt]
                    nxt += 1
                if ly._raw_v_packed:
                    v_raw = parts[nxt].view(B, 1, HKV, HD).transpose(1, 2)
                q = q.view(B, 1, H, HD).transpose(1, 2)
                k = k.view(B, 1, HKV, HD).transpose(1, 2)
                vx = vx.view(B, 1, H, HD).transpose(1, 2)
            else:
                q, k, v = ly.head_qkv(h)
                vx = repeat_kv(v, G)
                v_raw = v

            # pos is an int (eager) or a 1-element device tensor (CUDA graph: always inside the rope table).
            if hasattr(ly, "_rope_cos") and (torch.is_tensor(pos) or pos < ly._rope_cos.shape[0]):
                cos = ly._rope_cos[pos].view(1, 1, 1, HD)
                sin = ly._rope_sin[pos].view(1, 1, 1, HD)
                q = _apply_cached_rope(q, cos, sin)
                k = _apply_cached_rope(k, cos, sin)
            else:
                q, k = ly.rope(q, pos_start=pos), ly.rope(k, pos_start=pos)

            y_swa = None
            if getattr(ly, "swa_window", 0) > 0:
                if v_raw is None:
                    v_raw = ly.v_proj(h).view(B, 1, HKV, HD).transpose(1, 2)
                y_swa = self._window_attn_step(kv_caches[li], q, k, v_raw, ly.swa_window, G)

            kx = repeat_kv(k, G)
            if hasattr(ly, "_km_dt"):
                kx = torch.matmul(kx, ly._km_dt)

            if hasattr(ly, "k_conv") and isinstance(kv_caches[li], dict) and getattr(ly, "_k_conv_active", False):
                k_in = torch.cat([kv_caches[li]["k_buf"], kx.unsqueeze(-1)], dim=-1)
                kv_caches[li]["k_buf"].copy_(k_in[..., -3:])  # in place: buffer address must stay fixed
                kx = kx + (k_in * ly._k_conv_w).sum(dim=-1)

            if record is not None:
                record[li].append((pos, kx.detach().clone(), vx.detach().clone()))

            v_loc = None
            if hasattr(ly, "v_conv") and isinstance(kv_caches[li], dict) and getattr(ly, "_v_conv_active", False):
                v_in = torch.cat([kv_caches[li]["v_buf"], vx.unsqueeze(-1)], dim=-1)
                kv_caches[li]["v_buf"].copy_(v_in[..., -7:])
                v_loc = (v_in * ly._v_conv_w).sum(dim=-1)

            y = fused_vsa_recurrent_step(
                q.contiguous(), kx.contiguous(), vx.contiguous(),
                states[li], ly._cached_gam, ly._cached_omg, ly._cached_vself,
                q_relu=ly._cached_qr, k_relu=ly._cached_kr, norm_w=ly._cached_nw,
                v_loc=v_loc, local_q_gate=ly._cached_lqg,
            )

            if y_swa is not None:
                a = ly._cached_swa_a.view(1, H, 1, 1).to(y.dtype)
                y = y + a * (y_swa.to(y.dtype) - y)

            if g_proj_out is not None:
                y = y * (2.0 * torch.sigmoid(g_proj_out.view(B, 1, H, HD).transpose(1, 2)))
            x = x + ly.o_proj(y.transpose(1, 2).reshape(B, 1, H * HD))
        else:
            ly = blk.attn
            H, HKV, HD = ly.H, ly.HKV, ly.HD
            if getattr(ly, "packed_q_gate", False):
                q, gate = torch.chunk(ly.q_proj(h).view(B, 1, H, HD * 2), 2, dim=-1)
                gate = gate.reshape(B, 1, H * HD)
                k = ly.k_proj(h).view(B, 1, HKV, HD)
                v = ly.v_proj(h).view(B, 1, HKV, HD)
                if ly.q_norm is not None:
                    q, k = ly.q_norm(q), ly.k_norm(k)
                q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                q, k = ly.rope(q, pos_start=pos), ly.rope(k, pos_start=pos)
            elif hasattr(ly, "_wqkv_packed"):
                qkv = F.linear(h, ly._wqkv_packed, ly._bqkv_packed)
                q, k, v = torch.split(qkv, [H * HD, HKV * HD, HKV * HD], dim=-1)
                q = q.view(B, 1, H, HD).transpose(1, 2)
                k = k.view(B, 1, HKV, HD).transpose(1, 2)
                v = v.view(B, 1, HKV, HD).transpose(1, 2)
                gate = None
                if hasattr(ly, "_rope_cos") and (torch.is_tensor(pos) or pos < ly._rope_cos.shape[0]):
                    cos = ly._rope_cos[pos].view(1, 1, 1, HD)
                    sin = ly._rope_sin[pos].view(1, 1, 1, HD)
                    q = _apply_cached_rope(q, cos, sin)
                    k = _apply_cached_rope(k, cos, sin)
                else:
                    q, k = ly.rope(q, pos_start=pos), ly.rope(k, pos_start=pos)
            else:
                q, k, v = ly.head_qkv(h)
                gate = None
                q, k = ly.rope(q, pos_start=pos), ly.rope(k, pos_start=pos)

            cache_obj = kv_caches[li]
            if isinstance(cache_obj, GraphKV):
                y = cache_obj.attend(q, k, v, pos, ly.G)
            else:
                if isinstance(cache_obj, StaticKV):
                    pk, pv = cache_obj.append(k, v)
                else:
                    pk, pv = cache_obj
                    pk, pv = torch.cat([pk, k.to(pk.dtype)], dim=2), torch.cat([pv, v.to(pv.dtype)], dim=2)
                    kv_caches[li] = (pk, pv)
                try:
                    y = F.scaled_dot_product_attention(q, pk, pv, enable_gqa=(ly.G > 1))
                except TypeError:
                    y = F.scaled_dot_product_attention(q, repeat_kv(pk, ly.G), repeat_kv(pv, ly.G))

            if gate is not None:
                y_out = y.transpose(1, 2).reshape(B, 1, H * HD) * torch.sigmoid(gate)
            elif getattr(ly, "g_proj", None) is not None:
                y_out = (y * torch.sigmoid(ly.g_proj(h).view(B, 1, H, HD).transpose(1, 2))).transpose(1, 2).reshape(B, 1, H * HD)
            else:
                y_out = y.transpose(1, 2).reshape(B, 1, H * HD)
            x = x + ly.o_proj(y_out)

        return x + blk.mlp(blk.norm2(x))

    @torch.no_grad()
    def prefill(self, ids: torch.Tensor, states=None, kv_caches=None, start_pos: int = 0, max_kv_len: int | None = None):
        m = self.model
        B, T = ids.shape
        if states is None or kv_caches is None:
            states, kv_caches = self.blank_state(m, B, ids.device, max_kv_len=max(max_kv_len or 4096, start_pos + T + 1024))
        if self.is_universal:
            hidden = None
            pos = start_pos
            for t in range(T):
                hidden, states, kv_caches = self.step(ids[:, t : t + 1], pos, states, kv_caches)
                pos += 1
            return hidden, states, kv_caches

        x = m.embed(ids)
        for li, block in enumerate(m.blocks):
            layer, q, k, v, g = self._proj(block, block.norm1(x), B, T)
            if hasattr(block, "vsa"):
                HD = layer.HD
                if getattr(layer, "use_rope", False):
                    q, k = _rope(q), _rope(k)
                k = k * (layer.effective_pos_code(T, k.dtype) if hasattr(layer, "effective_pos_code") else layer.pos_code[:T].view(1, 1, T, HD).to(k.dtype))
                gamma = self._gamma(layer)
                y = vsa_parallel_attention(q.float(), k.float(), v.float(), gamma)
                if getattr(layer, "head_norm", None) is not None:
                    y = layer.head_norm(y.to(x.dtype))
                w = gamma.view(1, layer.H, 1) ** torch.arange(T - 1, -1, -1, device=x.device, dtype=torch.float32)
                states[li] = torch.einsum("bhte,bhtd->bhed", k.float() * w.unsqueeze(-1), v.float())
            else:
                q, k = _rope(q), _rope(k)
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                kv_caches[li].append(k, v)
            y = (y.to(x.dtype) * g).transpose(1, 2).reshape(B, T, -1)
            x = x + layer.to_out(y)
            u = block.ffn_up(block.norm2(x))
            hu = block.ffn_up.out_features // 2
            x = x + block.ffn_down(F.silu(u[..., :hu]) * u[..., hu:])
        return m.norm_f(x)[:, -1:], states, kv_caches

    @torch.no_grad()
    def step(self, token: torch.Tensor, pos: int, states, kv_caches, record=None, write_scale: float = 1.0):
        m = self.model
        if self.is_universal:
            # cache_enabled=False: required inside CUDA-graph capture, and a no-op cost here since the
            # autocast cache never outlived a single step anyway.
            with torch.autocast("cuda", torch.bfloat16, enabled=token.is_cuda, cache_enabled=False):
                x = m.embed_tokens(token)
                for li, blk in enumerate(m.blocks):
                    x = self._step_universal_block(li, blk, x, pos, states, kv_caches, record=record, write_scale=write_scale)
                return m.norm(x), states, kv_caches

        x = m.embed(token)
        B = x.shape[0]
        for li, block in enumerate(m.blocks):
            layer, q, k, v, g = self._proj(block, block.norm1(x), B, 1)
            if hasattr(block, "vsa"):
                H, HD = layer.H, layer.HD
                if getattr(layer, "use_rope", False):
                    q, k = _rope(q, pos), _rope(k, pos)
                if hasattr(layer, "effective_pos_code"):
                    if pos < layer.pos_code.shape[0]:
                        pc = layer.effective_pos_code(pos + 1, k.dtype)[:, :, pos : pos + 1, :]
                    elif self.wrap_positions:
                        p = pos % layer.pos_code.shape[0]
                        pc = layer.effective_pos_code(p + 1, k.dtype)[:, :, p : p + 1, :]
                    else:
                        pc = torch.ones(1, 1, 1, HD, device=k.device, dtype=k.dtype)
                    k = k * pc
                else:
                    p = pos % layer.pos_code.shape[0] if self.wrap_positions else min(pos, layer.pos_code.shape[0] - 1)
                    k = k * (layer.pos_code[p].view(1, 1, 1, HD).to(k.dtype) if pos < layer.pos_code.shape[0] or self.wrap_positions else 1.0)
                gamma = self._gamma(layer)
                omg = (1.0 - gamma).view(1, H, 1, 1)
                qf, kf, vf = q.float(), k.float(), v.float()
                M = states[li]
                carry = torch.einsum("bhed,bhe->bhd", M, qf.squeeze(2))
                y = (gamma.view(1, H, 1, 1) * carry.unsqueeze(2) + write_scale * (qf * kf).sum(-1, keepdim=True) * vf) * omg / (HD ** 0.5)
                if getattr(layer, "head_norm", None) is not None:
                    y = layer.head_norm(y.to(x.dtype))
                states[li] = gamma.view(1, H, 1, 1) * M + write_scale * torch.einsum("bhe,bhd->bhed", kf.squeeze(2), vf.squeeze(2))
                if record is not None:
                    record[li].append((pos, (write_scale * kf.squeeze(2)[0]).clone(), vf.squeeze(2)[0].clone()))
            else:
                q, k = _rope(q, pos), _rope(k, pos)
                pk, pv = kv_caches[li].append(k, v)
                y = F.scaled_dot_product_attention(q, pk, pv)
            y = (y.to(x.dtype) * g).transpose(1, 2).reshape(B, 1, -1)
            x = x + layer.to_out(y)
            u = block.ffn_up(block.norm2(x))
            hu = block.ffn_up.out_features // 2
            x = x + block.ffn_down(F.silu(u[..., :hu]) * u[..., hu:])
        return m.norm_f(x), states, kv_caches

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new_tokens: int, temperature: float = 0.0, top_k: int = 40, top_p: float = 0.95, eos_ids: set[int] | None = None):
        out = prompt.clone()
        logits, feed = self._start(prompt, max_new_tokens)
        pos = prompt.shape[1] - 1
        for _ in range(max_new_tokens):
            nxt = self._sample(logits, temperature, top_k=top_k, top_p=top_p)
            out = torch.cat([out, nxt], dim=1)
            if eos_ids and int(nxt[0, 0].item()) in eos_ids:
                break
            pos += 1
            logits = feed(nxt, pos)
        return out

    @torch.no_grad()
    def generate_batch(self, ids: torch.Tensor, max_new_tokens: int, temperature: float = 0.6, top_k: int = 40,
                       top_p: float = 0.95, eos_ids: set[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample B continuations of B prompts that all have the same length (so `pos` and the window
        counter stay shared scalars). Returns (tokens [B, n], lengths [B]); lengths count generated tokens
        up to and including each row's first EOS, or max_new_tokens if it never stopped. Tokens after a
        row's EOS are filler and must be dropped by the caller."""
        B, T = ids.shape
        logits, feed = self._start(ids, max_new_tokens)
        eos = torch.tensor(sorted(eos_ids or ()), device=ids.device, dtype=torch.long)
        done = torch.zeros(B, dtype=torch.bool, device=ids.device)
        lengths = torch.full((B,), max_new_tokens, dtype=torch.long, device=ids.device)
        out, pos = [], T - 1
        for i in range(max_new_tokens):
            nxt = self._sample(logits, temperature, top_k=top_k, top_p=top_p)
            out.append(nxt)
            if eos.numel():
                hit = torch.isin(nxt[:, 0], eos) & ~done
                lengths[hit] = i + 1
                done |= hit
                if i % 16 == 15 and bool(done.all()):
                    break
            pos += 1
            logits = feed(nxt, pos)
        return torch.cat(out, dim=1), lengths

    @torch.no_grad()
    def continue_generate(self, ids: torch.Tensor, states, kv_caches, start_pos: int,
                          max_new_tokens: int, temperature: float = 0.0, top_k: int = 40, top_p: float = 0.95, eos_ids: set[int] | None = None):
        out, pos, logits = [], start_pos, None
        for t in range(ids.shape[1]):
            x, states, kv_caches = self.step(ids[:, t : t + 1], pos, states, kv_caches)
            logits = self.model.lm_head(x[:, -1, :])
            pos += 1
        for _ in range(max_new_tokens):
            nxt = self._sample(logits, temperature, top_k=top_k, top_p=top_p)
            out.append(nxt)
            if eos_ids and int(nxt[0, 0].item()) in eos_ids:
                break
            x, states, kv_caches = self.step(nxt, pos, states, kv_caches)
            logits = self.model.lm_head(x[:, -1, :])
            pos += 1
        toks = torch.cat(out, dim=1) if out else torch.empty(ids.shape[0], 0, dtype=torch.long, device=ids.device)
        return toks, states, kv_caches, pos

    @torch.no_grad()
    def memorize(self, ids: torch.Tensor, states, kv_caches, start_pos: int):
        traces = [[] for _ in self.model.blocks]
        pos = start_pos
        for t in range(ids.shape[1]):
            _, states, kv_caches = self.step(ids[:, t : t + 1], pos, states, kv_caches, record=traces)
            pos += 1
        return states, kv_caches, pos, traces

    @torch.no_grad()
    def forget(self, traces, states, next_pos: int, kv_caches=None, pre_kv_lens: list[int] | None = None):
        for li, trace in enumerate(traces):
            if not trace:
                continue
            ly = self.model.blocks[li].vsa
            M_doc = torch.zeros_like(states[li])
            last_p = trace[-1][0]
            for (p, k, v) in trace:
                kt = k.to(states[li].device)
                vt = v.to(states[li].device)
                if kt.ndim == 2:
                    kt = kt.unsqueeze(0).unsqueeze(2)
                    vt = vt.unsqueeze(0).unsqueeze(2)
                fused_vsa_recurrent_step(
                    kt.contiguous(), kt.contiguous(), vt.contiguous(),
                    M_doc, ly._cached_gam, ly._cached_omg, ly._cached_vself,
                    q_relu=ly._cached_qr, k_relu=ly._cached_kr, norm_w=ly._cached_nw,
                )
            rem_steps = float(next_pos - 1 - last_p)
            if rem_steps > 0:
                M_doc = M_doc * (ly._cached_gam.view(1, -1, 1, 1) ** rem_steps)
            states[li] = states[li] - M_doc
        return states
