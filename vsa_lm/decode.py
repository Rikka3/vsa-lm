from __future__ import annotations

import math

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


def _rope_cos_sin(inv_freq64: torch.Tensor, pos: torch.Tensor, dt) -> tuple[torch.Tensor, torch.Tensor]:
    ang = torch.remainder(pos.to(torch.float64).unsqueeze(-1) * inv_freq64, 2.0 * math.pi)
    emb = torch.cat((ang, ang), dim=-1)
    return emb.cos().to(dt), emb.sin().to(dt)


def _split_kv_attend(q: torch.Tensor, k_buf: torch.Tensor, v_buf: torch.Tensor, valid_mask: torch.Tensor, G: int) -> torch.Tensor:
    B, HKV, T, HD = k_buf.shape
    qg = q.to(k_buf.dtype).reshape(B, HKV, G, HD)
    if T <= 2048 or T % 512 != 0:
        y = F.scaled_dot_product_attention(qg, k_buf, v_buf, attn_mask=valid_mask.view(1, 1, 1, T))
        return y.reshape(B, HKV * G, 1, HD).to(q.dtype)
    S = T // 512
    ks = k_buf.view(B, HKV, S, 512, HD)
    vs = v_buf.view(B, HKV, S, 512, HD)
    with torch.autocast(q.device.type, enabled=False):
        scores = torch.matmul(qg.unsqueeze(2), ks.transpose(-1, -2)).float() * (HD ** -0.5)
        scores.masked_fill_(~valid_mask.view(1, 1, S, 1, 512), float("-inf"))
        m = scores.amax(dim=(-1, 2), keepdim=True)
        w = torch.exp(scores - m)
        probs = (w / w.sum(dim=(-1, 2), keepdim=True)).to(v_buf.dtype)
        y = torch.matmul(probs, vs).sum(dim=2)
    return y.reshape(B, HKV * G, 1, HD).to(q.dtype)


class StreamKV:
    def __init__(self, B: int, HKV: int, sinks: int, window: int, HD: int, dev, dt=torch.bfloat16):
        self.S, self.W = sinks, window
        T_cap = sinks + window
        T_buf = -(-T_cap // 512) * 512 if T_cap > 2048 else T_cap
        self.k = torch.zeros(B, HKV, T_buf, HD, device=dev, dtype=dt)
        self.v = torch.zeros_like(self.k)
        self.sink_k = torch.zeros(B, HKV, sinks, HD, device=dev, dtype=dt)
        self.n = torch.zeros(1, dtype=torch.long, device=dev)
        self.ar = torch.arange(T_buf, device=dev)

    def reset(self):
        self.k.zero_()
        self.v.zero_()
        self.sink_k.zero_()
        self.n.zero_()

    def attend(self, q: torch.Tensor, k_new: torch.Tensor, v_new: torch.Tensor, G: int, inv_freq64: torch.Tensor) -> torch.Tensor:
        S, W, n = self.S, self.W, self.n
        k_new, v_new = k_new.to(self.k.dtype), v_new.to(self.v.dtype)
        slot = torch.where(n < S, n, S + (n - S).remainder(W))
        self.k.index_copy_(2, slot, k_new)
        self.v.index_copy_(2, slot, v_new)
        s_idx = n.clamp(max=S - 1)
        self.sink_k.index_copy_(2, s_idx, torch.where(n < S, k_new, self.sink_k.index_select(2, s_idx)))
        n.add_(1)
        cos, sin = _rope_cos_sin(inv_freq64, (n - S - W).clamp(min=0), torch.float32)
        self.k[:, :, :S].copy_(_apply_cached_rope(self.sink_k.float(), cos.view(1, 1, 1, -1), sin.view(1, 1, 1, -1)))
        valid = self.ar < n.clamp(max=S + W)
        return _split_kv_attend(q, self.k, self.v, valid, G)


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
    def __init__(self, B: int, HKV: int, Tmax: int, HD: int, dev, dt=torch.bfloat16):
        Tmax = -(-max(512, Tmax) // 512) * 512
        self.k = torch.zeros(B, HKV, Tmax, HD, device=dev, dtype=dt)
        self.v = torch.zeros_like(self.k)
        self.ar = torch.arange(Tmax, device=dev)

    def attend(self, q: torch.Tensor, k_new: torch.Tensor, v_new: torch.Tensor, pos: torch.Tensor, G: int) -> torch.Tensor:
        self.k.index_copy_(2, pos, k_new.to(self.k.dtype))
        self.v.index_copy_(2, pos, v_new.to(self.v.dtype))
        return _split_kv_attend(q, self.k, self.v, self.ar <= pos, G)


class _CUDAGraphStep:
    def __init__(self, dec: "VSARecurrentDecoder", B: int, Tmax: int, device):
        m = dec.model
        self.states, self.kv = dec.blank_state(m, B, device, max_kv_len=Tmax, graph=True, stream=dec.stream)
        self.tok = torch.zeros(B, 1, dtype=torch.long, device=device)
        self.pos = torch.zeros(1, dtype=torch.long, device=device)
        self._keep = [t for blk in m.blocks for ly in (getattr(blk, "vsa", None), getattr(blk, "attn", None))
                      if ly is not None for t in vars(ly).values() if torch.is_tensor(t)]

        def run():
            with torch.no_grad():
                h, _, _ = dec.step(self.tok, self.pos, self.states, self.kv)
                return m.lm_head(h[:, -1, :])

        side = torch.cuda.Stream(device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
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
    def __init__(self, model, wrap_positions: bool = False, max_cache_pos: int = 65536, use_cuda_graph: bool = True,
                 stream_window: int | None = None, stream_sinks: int = 4):
        self.model = model.eval()
        self.wrap_positions = wrap_positions
        self.max_T = model.config["max_T"]
        self.is_universal = hasattr(model, "embed_tokens")
        self.max_cache_pos = max_cache_pos
        self.stream = (int(stream_sinks), int(stream_window)) if stream_window else None
        self._step_rope = None
        self._inv64, self._rope_tables = {}, {}
        self._prepare_fast_decode_tables(max_cache_pos)
        if self.stream is not None:
            if not self.is_universal or stream_sinks < 1:
                raise ValueError("Streaming needs UniversalVSALM checkpoint and stream_sinks >= 1")
            for blk in self.model.blocks:
                ly = blk.vsa if hasattr(blk, "vsa") else blk.attn
                if not hasattr(ly, "_rope_key") or getattr(ly, "packed_q_gate", False):
                    raise ValueError(f"Streaming needs full RoPE on every layer ({type(ly).__name__} has none/partial)")
        self.use_cuda_graph = use_cuda_graph and self._graph_supported()
        self._graphs: dict = {}

    def _graph_supported(self) -> bool:
        if not self.is_universal or not torch.cuda.is_available():
            return False
        device = next(self.model.parameters()).device
        if device.type != "cuda":
            return False
        for blk in self.model.blocks:
            ly = blk.vsa if hasattr(blk, "vsa") else getattr(blk, "attn", None)
            if ly is None or not hasattr(ly, "_wqkv_packed") or not hasattr(ly, "_rope_cos"):
                return False
        return True

    def _graph_runner(self, B: int, need_len: int, device):
        if not self.use_cuda_graph or device.type != "cuda":
            return None
        if self.stream is not None:
            Tmax = 0
        else:
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
            except Exception as e:
                print(f"[vsa-decode] CUDA graph capture failed ({type(e).__name__}: {e}); using eager decode.", flush=True)
                self.use_cuda_graph = False
                return None
        else:
            runner.reset()
        self._graphs[key] = runner
        return runner

    def _start(self, ids: torch.Tensor, max_new_tokens: int):
        B, T = ids.shape
        runner = self._graph_runner(B, T + max_new_tokens + 1, ids.device)
        if runner is not None:
            logits = None
            for t in range(T):
                logits = runner(ids[:, t : t + 1], t)
            return logits, runner
        x, states, kv = self.prefill(ids, max_kv_len=T + max_new_tokens + 64)

        @torch.no_grad()
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
        if isinstance(getattr(self.model.lm_head, "weight", None), torch.Tensor):
            self.model.lm_head.weight.data = self.model.lm_head.weight.data.to(dt)
        t = torch.arange(max_pos, device=device, dtype=torch.float32)

        rope_cache = self._rope_tables = {}
        self._inv64 = {}
        for blk in self.model.blocks:
            ly = blk.vsa if hasattr(blk, "vsa") else blk.attn
            if hasattr(ly, "rope") and hasattr(ly.rope, "inv_freq"):
                inv = ly.rope.inv_freq.to(device)
                key = (ly.HD, tuple(inv.tolist()))
                if key not in rope_cache:
                    freqs = torch.outer(t, inv)
                    emb = torch.cat((freqs, freqs), dim=-1)
                    rope_cache[key] = (emb.cos().to(dt), emb.sin().to(dt))
                ly._rope_cos, ly._rope_sin = rope_cache[key]
                if 2 * inv.numel() == ly.HD:
                    self._inv64[key] = inv.double()
                    ly._rope_key = key

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
                ly._slot_active = bool(getattr(ly, "num_slots", 0) > 0 and hasattr(ly, "slot_gate") and ly.slot_gate.detach().abs().max().item() > 0.05)
                ly._cached_slot_g = torch.tanh(ly.slot_gate.detach().float()).view(1, H, 1, 1).to(dt) if ly._slot_active else None

                if hasattr(ly, "v_head_mix"):
                    ly._km_dt = ly.k_head_mix.detach().to(dt).contiguous()

                if hasattr(ly, "v_head_mix") and hasattr(ly, "v_proj") and isinstance(getattr(ly.q_proj, "weight", None), torch.Tensor):
                    wv = ly.v_proj.weight.detach().to(torch.float32).view(HKV, HD, -1)
                    wv_rep = wv[:, None, :, :].expand(HKV, G, HD, -1).reshape(H, HD, -1)
                    vm_T = ly.v_head_mix.detach().to(torch.float32).transpose(1, 2)
                    wv_fused = torch.bmm(vm_T, wv_rep).reshape(H * HD, -1).to(dt)
                    wq = ly.q_proj.weight.detach().to(dt)
                    wk = ly.k_proj.weight.detach().to(dt)
                    g_act = bool(hasattr(ly, "g_proj") and isinstance(getattr(ly.g_proj, "weight", None), torch.Tensor) and ly.g_proj.weight.any().item())
                    ly._g_active = g_act
                    raw_v = getattr(ly, "swa_window", 0) > 0 or ly._slot_active
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

                if hasattr(ly, "k_conv"):
                    ly._k_conv_w = ly.k_conv.weight.detach().to(dt).view(1, H, 1, HD, 4)
                    ly._v_conv_w = ly.v_conv.weight.detach().to(dt).view(1, H, 1, HD, 8)
                    ly._k_conv_active = bool(ly.k_conv.weight.any().item())
                    ly._v_conv_active = bool(ly.v_conv.weight.any().item())
            else:
                if hasattr(ly, "q_proj") and not getattr(ly, "packed_q_gate", False) and isinstance(getattr(ly.q_proj, "weight", None), torch.Tensor):
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
    def blank_state(model, batch_size: int, device, max_kv_len: int = 4096, state_dtype=torch.float32, graph: bool = False,
                    stream: tuple[int, int] | None = None):
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
                    cache["swa_n"] = torch.zeros(1, dtype=torch.long, device=device)
                    cache["swa_ar"] = torch.arange(W + 1, device=device)
                    cache["swa_k0"] = torch.zeros(batch_size, v.HKV, 1, v.HD, device=device, dtype=dt)
                if getattr(v, "num_slots", 0) > 0:
                    S = v.num_slots
                    cache["slot_C"] = torch.zeros(batch_size, v.HKV, S, v.HD, device=device, dtype=torch.float32)
                    cache["slot_U"] = torch.zeros(batch_size, v.HKV, S, v.HD, device=device, dtype=torch.float32)
                    cache["slot_Z"] = torch.full((batch_size, v.HKV, S, 1), 1e-6, device=device, dtype=torch.float32)
                kv_caches[li] = cache or None
            else:
                s = blk.attn if hasattr(blk, "attn") else blk.softmax_attn
                hk = getattr(s, "HKV", s.H)
                if stream is not None:
                    kv_caches[li] = StreamKV(batch_size, hk, stream[0], stream[1], s.HD, device, dt=dt)
                else:
                    kv_cls = GraphKV if graph else StaticKV
                    kv_caches[li] = kv_cls(batch_size, hk, max_kv_len, s.HD, device, dt=dt)
        return states, kv_caches

    @staticmethod
    def reset_state(states, kv_caches):
        for s in states:
            if s is not None:
                s.zero_()
        for c in kv_caches:
            if isinstance(c, dict):
                for key, t in c.items():
                    if torch.is_tensor(t) and key != "swa_ar":
                        if key == "slot_Z":
                            t.fill_(1e-6)
                        else:
                            t.zero_()
            elif isinstance(c, GraphKV):
                c.k.zero_()
                c.v.zero_()
            elif isinstance(c, StreamKV):
                c.reset()
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
    def _window_attn_step(cache: dict, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, W: int, G: int,
                          sink_cap: int | None = None, inv_freq64: torch.Tensor | None = None) -> torch.Tensor:
        n = cache["swa_n"]
        if not torch.is_tensor(n):
            n = cache["swa_n"] = torch.tensor([int(n)], dtype=torch.long, device=k.device)
            cache["swa_ar"] = torch.arange(W + 1, device=k.device)
        slot = ((n - 1).remainder(W) + 1) * (n != 0)
        cache["swa_k"].index_copy_(2, slot, k.to(cache["swa_k"].dtype))
        cache["swa_v"].index_copy_(2, slot, v.to(cache["swa_v"].dtype))
        if sink_cap is not None and "swa_k0" in cache:
            k0 = cache["swa_k0"]
            k0.copy_(torch.where(n == 0, k.to(k0.dtype), k0))
        n.add_(1)
        if sink_cap is not None and "swa_k0" in cache and inv_freq64 is not None:
            cos, sin = _rope_cos_sin(inv_freq64, (n - sink_cap).clamp(min=0), torch.float32)
            cache["swa_k"][:, :, :1].copy_(_apply_cached_rope(cache["swa_k0"].float(), cos.view(1, 1, 1, -1), sin.view(1, 1, 1, -1)))
        valid = cache["swa_ar"] < n
        pk, pv = cache["swa_k"], cache["swa_v"]
        B, HKV, _, HD = pk.shape
        with torch.autocast(q.device.type, enabled=False):
            qg = q.float().reshape(B, HKV, G, 1, HD)
            scores = torch.matmul(qg, pk.float().unsqueeze(2).transpose(-1, -2)) * (HD ** -0.5)
            probs = F.softmax(scores.masked_fill(~valid, float("-inf")), dim=-1)
            out = torch.matmul(probs, pv.float().unsqueeze(2)).view(B, HKV * G, 1, HD)
        return out.to(q.dtype)

    def _step_rope_for(self, pos, device) -> dict:
        p = pos if torch.is_tensor(pos) else torch.tensor([pos], device=device)
        out = {}
        for key, inv in self._inv64.items():
            cos_t, sin_t = self._rope_tables[key]
            L = cos_t.shape[0]
            c64, s64 = _rope_cos_sin(inv, p, cos_t.dtype)
            inside = (p < L).view(-1, 1)
            idx = p.clamp(max=L - 1)
            out[key] = (torch.where(inside, cos_t[idx], c64), torch.where(inside, sin_t[idx], s64))
        return out

    def _rope_qk(self, ly, q: torch.Tensor, k: torch.Tensor, pos):
        key = getattr(ly, "_rope_key", None)
        if key is None:
            return ly.rope(q, pos_start=pos), ly.rope(k, pos_start=pos)
        if self._step_rope is not None:
            cos, sin = self._step_rope[key]
        elif torch.is_tensor(pos) or pos < ly._rope_cos.shape[0]:
            cos, sin = ly._rope_cos[pos], ly._rope_sin[pos]
        else:
            cos, sin = _rope_cos_sin(self._inv64[key], torch.tensor([pos], device=q.device), ly._rope_cos.dtype)
        cos, sin = cos.view(1, 1, 1, -1), sin.view(1, 1, 1, -1)
        return _apply_cached_rope(q, cos, sin), _apply_cached_rope(k, cos, sin)

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
                if hasattr(ly, "v_head_mix"):
                    vx = torch.matmul(vx, ly.v_head_mix.to(vx.dtype))
                v_raw = v
                if hasattr(ly, "g_proj"):
                    g_proj_out = ly.g_proj(h)

            if getattr(ly, "q_norm", None) is not None and hasattr(ly, "_wqkv_packed"):
                q = ly.q_norm(q.transpose(1, 2)).transpose(1, 2)
                k = ly.k_norm(k.transpose(1, 2)).transpose(1, 2)

            y_slot = None
            if getattr(ly, "_slot_active", False) and isinstance(kv_caches[li], dict) and "slot_C" in kv_caches[li]:
                if v_raw is None:
                    v_raw = ly.v_proj(h).view(B, 1, HKV, HD).transpose(1, 2)
                S = ly.num_slots
                w_s = torch.sigmoid(ly.slot_router(h)).view(B, 1, HKV, S).permute(0, 2, 3, 1).float()
                k_cl = F.normalize(k.float(), p=2.0, dim=-1).squeeze(2).unsqueeze(2)
                q_cl = F.normalize(q.float(), p=2.0, dim=-1).view(B, HKV, G, 1, HD)
                v_cl = v_raw.float().squeeze(2).unsqueeze(2)
                sc, su, sz = kv_caches[li]["slot_C"], kv_caches[li]["slot_U"], kv_caches[li]["slot_Z"]
                sc.add_(w_s * k_cl)
                su.add_(w_s * v_cl)
                sz.add_(w_s)
                K_sl = sc / sz
                V_sl = su / sz
                s_scores = torch.matmul(q_cl, K_sl.unsqueeze(2).transpose(-1, -2)) * (HD ** 0.5)
                s_probs = F.softmax(s_scores, dim=-1)
                y_slot = torch.matmul(s_probs, V_sl.unsqueeze(2)).view(B, H, 1, HD).to(q.dtype)

            if getattr(ly, "mix_before_rope", False) and hasattr(ly, "_km_dt"):
                kx_pre = torch.matmul(repeat_kv(k, G), ly._km_dt.to(k.dtype))
                q, k = self._rope_qk(ly, q, k, pos)
                _, kx = self._rope_qk(ly, q, kx_pre, pos)
            else:
                q, k = self._rope_qk(ly, q, k, pos)
                kx = repeat_kv(k, G)
                if hasattr(ly, "_km_dt"):
                    kx = torch.matmul(kx, ly._km_dt.to(kx.dtype))

            y_swa = None
            if getattr(ly, "swa_window", 0) > 0:
                if v_raw is None:
                    v_raw = ly.v_proj(h).view(B, 1, HKV, HD).transpose(1, 2)
                inv64 = self._inv64.get(getattr(ly, "_rope_key", None))
                y_swa = self._window_attn_step(
                    kv_caches[li], q, k, v_raw, ly.swa_window, G,
                    sink_cap=self.max_T, inv_freq64=inv64,
                )

            if hasattr(ly, "k_conv") and isinstance(kv_caches[li], dict) and getattr(ly, "_k_conv_active", False):
                k_in = torch.cat([kv_caches[li]["k_buf"], kx.unsqueeze(-1)], dim=-1)
                kv_caches[li]["k_buf"].copy_(k_in[..., -3:])
                kx = kx + (k_in * ly._k_conv_w.to(kx.dtype)).sum(dim=-1)

            if record is not None:
                record[li].append((pos, kx.detach().clone(), vx.detach().clone()))

            v_loc = None
            if hasattr(ly, "v_conv") and isinstance(kv_caches[li], dict) and getattr(ly, "_v_conv_active", False):
                v_in = torch.cat([kv_caches[li]["v_buf"], vx.unsqueeze(-1)], dim=-1)
                kv_caches[li]["v_buf"].copy_(v_in[..., -7:])
                v_loc = (v_in * ly._v_conv_w.to(vx.dtype)).sum(dim=-1)

            y = fused_vsa_recurrent_step(
                q.contiguous(), kx.contiguous(), vx.contiguous(),
                states[li], ly._cached_gam, ly._cached_omg, ly._cached_vself,
                q_relu=ly._cached_qr, k_relu=ly._cached_kr, norm_w=ly._cached_nw,
                v_loc=v_loc, local_q_gate=ly._cached_lqg,
            )

            if y_swa is not None:
                a = ly._cached_swa_a.view(1, H, 1, 1).to(y.dtype)
                y = y + a * (y_swa.to(y.dtype) - y)
            if y_slot is not None:
                y = y + ly._cached_slot_g.to(y.dtype) * y_slot

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
                q, k = self._rope_qk(ly, q, k, pos)
            else:
                q, k, v = ly.head_qkv(h)
                gate = None
                q, k = self._rope_qk(ly, q, k, pos)

            cache_obj = kv_caches[li]
            if isinstance(cache_obj, StreamKV):
                y = cache_obj.attend(q, k, v, ly.G, self._inv64[ly._rope_key])
            elif isinstance(cache_obj, GraphKV):
                y = cache_obj.attend(q, k, v, pos, ly.G)
            else:
                if isinstance(cache_obj, StaticKV):
                    pk, pv = cache_obj.append(k, v)
                else:
                    pk, pv = cache_obj
                    pk, pv = torch.cat([pk, k.to(pk.dtype)], dim=2), torch.cat([pv, v.to(pv.dtype)], dim=2)
                    kv_caches[li] = (pk, pv)
                try:
                    y = F.scaled_dot_product_attention(q, pk.to(q.dtype), pv.to(q.dtype), enable_gqa=(ly.G > 1))
                except TypeError:
                    y = F.scaled_dot_product_attention(q, repeat_kv(pk.to(q.dtype), ly.G), repeat_kv(pv.to(q.dtype), ly.G))

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
            states, kv_caches = self.blank_state(m, B, ids.device, max_kv_len=max(max_kv_len or 4096, start_pos + T + 1024),
                                                 stream=self.stream)
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
            with torch.autocast("cuda", torch.bfloat16, enabled=token.is_cuda, cache_enabled=False):
                self._step_rope = self._step_rope_for(pos, token.device) if self.stream is not None else None
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
                y = F.scaled_dot_product_attention(q, pk.to(q.dtype), pv.to(q.dtype))
            y = (y.to(x.dtype) * g).transpose(1, 2).reshape(B, 1, -1)
            x = x + layer.to_out(y)
            u = block.ffn_up(block.norm2(x))
            hu = block.ffn_up.out_features // 2
            x = x + block.ffn_down(F.silu(u[..., :hu]) * u[..., hu:])
        return m.norm_f(x), states, kv_caches

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new_tokens: int, temperature: float = 0.0, top_k: int = 40, top_p: float = 0.95, eos_ids: set[int] | None = None, close_think_ids: list[int] | None = None):
        out = prompt.clone()
        logits, feed = self._start(prompt, max_new_tokens)
        pos = prompt.shape[1] - 1
        gen_hist: list[int] = []
        seen_blocks: set[tuple[int, ...]] = set()
        end_think_id = close_think_ids[1] if (close_think_ids and len(close_think_ids) > 1) else None
        closed_think = False
        forced_queue: list[int] = []
        budget_trigger = max(64, max_new_tokens - 64)

        for step_i in range(max_new_tokens):
            if forced_queue:
                nxt = torch.tensor([[forced_queue.pop(0)]], dtype=torch.long, device=prompt.device)
            else:
                nxt = self._sample(logits, temperature, top_k=top_k, top_p=top_p)
            tok_id = int(nxt[0, 0].item())
            if end_think_id is not None and tok_id == end_think_id:
                closed_think = True
            gen_hist.append(tok_id)
            out = torch.cat([out, nxt], dim=1)
            if eos_ids and tok_id in eos_ids:
                break

            # In-stream </think> budget exit at max_new_tokens - 64 (never cuts early mid-chain)
            if close_think_ids and not closed_think and not forced_queue and step_i == budget_trigger:
                forced_queue = list(close_think_ids)
                closed_think = True

            pos += 1
            logits = feed(nxt, pos)
        return out

    @torch.no_grad()
    def generate_batch(self, ids: torch.Tensor, max_new_tokens: int, temperature: float = 0.6, top_k: int = 40,
                       top_p: float = 0.95, eos_ids: set[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
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
            if not self.is_universal:
                g = self._gamma(ly).view(1, ly.H, 1, 1)
                delta = sum((g ** (next_pos - 1 - p)) * torch.einsum("he,hd->hed", k, v).unsqueeze(0) for p, k, v in trace)
                states[li] = states[li] - delta
                continue
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

    @staticmethod
    def erase_page(kv_caches: list, page_idx: int, num_pages: int = 8):
        for c in kv_caches:
            if isinstance(c, dict) and "slot_C" in c:
                S = c["slot_C"].shape[2]
                sp = max(1, S // num_pages)
                lo, hi = page_idx * sp, min(S, (page_idx + 1) * sp)
                c["slot_C"][:, :, lo:hi].zero_()
                c["slot_U"][:, :, lo:hi].zero_()
                c["slot_Z"][:, :, lo:hi].fill_(1e-6)
