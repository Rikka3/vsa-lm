"""Ultra-fast O(1)-state recurrent decoder with pre-computed RoPE tables and fused Triton step."""
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


class VSARecurrentDecoder:
    def __init__(self, model, wrap_positions: bool = False, max_cache_pos: int = 65536):
        self.model = model.eval()
        self.wrap_positions = wrap_positions
        self.max_T = model.config["max_T"]
        self.is_universal = hasattr(model, "embed_tokens")
        self._prepare_fast_decode_tables(max_cache_pos)

    @torch.no_grad()
    def _prepare_fast_decode_tables(self, max_pos: int):
        if not self.is_universal:
            return
        device = next(self.model.parameters()).device
        dt = torch.bfloat16 if device.type == "cuda" else torch.float32
        t = torch.arange(max_pos, device=device, dtype=torch.float32)

        # Precompute shared RoPE cos/sin tables once per distinct (HD, inv_freq)
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

                # Pre-fold GQA repeat_kv + v_head_mix into v_proj so V needs 0 repeat_kv and 0 matmul per step!
                if hasattr(ly, "v_head_mix") and hasattr(ly, "v_proj"):
                    wv = ly.v_proj.weight.detach().to(torch.float32).view(HKV, HD, -1)
                    wv_rep = wv[:, None, :, :].expand(HKV, G, HD, -1).reshape(H, HD, -1)
                    # vx = v @ v_head_mix -> wv_fused = v_head_mix^T @ wv_rep
                    vm_T = ly.v_head_mix.detach().to(torch.float32).transpose(1, 2)
                    ly._wv_fused = torch.bmm(vm_T, wv_rep).reshape(H * HD, -1).to(dt)
                    if ly.v_proj.bias is not None:
                        bv = ly.v_proj.bias.detach().to(torch.float32).view(HKV, 1, HD).expand(HKV, G, HD).reshape(H, 1, HD)
                        ly._bv_fused = torch.bmm(bv, ly.v_head_mix.detach().to(torch.float32)).reshape(H * HD).to(dt)
                    else:
                        ly._bv_fused = None

                if hasattr(ly, "k_conv"):
                    ly._k_conv_w = ly.k_conv.weight.detach().to(dt).view(1, H, 1, HD, 4)
                    ly._v_conv_w = ly.v_conv.weight.detach().to(dt).view(1, H, 1, HD, 8)
                    ly._k_conv_active = bool(ly.k_conv.weight.abs().max().item() > 1e-6)
                    ly._v_conv_active = bool(ly.v_conv.weight.abs().max().item() > 1e-6)
                    ly._g_active = bool(hasattr(ly, "g_proj") and ly.g_proj.weight.abs().max().item() > 1e-6)

    @staticmethod
    def state_bytes(model):
        vsa_bytes, kv_per_token = 0, 0
        for blk in model.blocks:
            if hasattr(blk, "vsa"):
                v = blk.vsa
                vsa_bytes += v.H * v.HD * v.HD * 4
            else:
                s = blk.attn if hasattr(blk, "attn") else blk.softmax_attn
                hk = getattr(s, "HKV", s.H)
                kv_per_token += 2 * hk * s.HD * 2
        return vsa_bytes, kv_per_token

    @staticmethod
    def blank_state(model, batch_size: int, device):
        states, kv_caches = [None] * len(model.blocks), [None] * len(model.blocks)
        dt = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        for li, blk in enumerate(model.blocks):
            if hasattr(blk, "vsa"):
                v = blk.vsa
                states[li] = torch.zeros(batch_size, v.H, v.HD, v.HD, device=device, dtype=torch.float32)
                if hasattr(v, "k_conv") and hasattr(v, "v_conv"):
                    kv_caches[li] = {
                        "k_buf": torch.zeros(batch_size, v.H, 1, v.HD, 3, device=device, dtype=dt),
                        "v_buf": torch.zeros(batch_size, v.H, 1, v.HD, 7, device=device, dtype=dt),
                    }
            else:
                s = blk.attn if hasattr(blk, "attn") else blk.softmax_attn
                hk = getattr(s, "HKV", s.H)
                kv_caches[li] = (
                    torch.zeros(batch_size, hk, 0, s.HD, device=device, dtype=dt),
                    torch.zeros(batch_size, hk, 0, s.HD, device=device, dtype=dt),
                )
        return states, kv_caches

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
    def _sample(logits, temperature: float, top_k: int = 40):
        if temperature <= 1e-5:
            return logits.argmax(-1, keepdim=True)
        topv, topi = torch.topk(logits / temperature, min(top_k, logits.size(-1)), dim=-1)
        return topi.gather(-1, torch.multinomial(F.softmax(topv, -1), 1))

    def _step_universal_block(self, li: int, blk, x: torch.Tensor, pos: int, states, kv_caches, record=None, write_scale: float = 1.0):
        B = x.shape[0]
        h = blk.norm1(x)
        if hasattr(blk, "vsa"):
            ly = blk.vsa
            H, HKV, HD, G = ly.H, ly.HKV, ly.HD, ly.G
            q = ly.q_proj(h).view(B, 1, H, HD).transpose(1, 2)
            k = ly.k_proj(h).view(B, 1, HKV, HD).transpose(1, 2)
            if hasattr(ly, "_wv_fused"):
                vx = F.linear(h, ly._wv_fused, ly._bv_fused).view(B, 1, H, HD).transpose(1, 2)
            else:
                vx = repeat_kv(ly.v_proj(h).view(B, 1, HKV, HD).transpose(1, 2), G)

            if hasattr(ly, "_rope_cos") and pos < ly._rope_cos.shape[0]:
                cos = ly._rope_cos[pos].view(1, 1, 1, HD)
                sin = ly._rope_sin[pos].view(1, 1, 1, HD)
                q = _apply_cached_rope(q, cos, sin)
                k = _apply_cached_rope(k, cos, sin)
            else:
                q, k = ly.rope(q, pos_start=pos), ly.rope(k, pos_start=pos)

            if hasattr(ly, "pos_code") and pos < ly.pos_code.shape[0]:
                k = k * ly.pos_code[pos].view(1, 1, 1, HD).to(k.dtype)
            kx = repeat_kv(k, G)

            if hasattr(ly, "k_head_mix"):
                kx = torch.matmul(kx, ly.k_head_mix.to(kx.dtype))
                if hasattr(ly, "head_pos_delta") and pos < ly.head_pos_delta.shape[1]:
                    kx = kx * (1.0 + 0.5 * torch.tanh(ly.head_pos_delta[:, pos : pos + 1, :])).unsqueeze(0).to(kx.dtype)
                if not hasattr(ly, "_wv_fused"):
                    vx = torch.matmul(vx, ly.v_head_mix.to(vx.dtype))

            if hasattr(ly, "k_conv") and isinstance(kv_caches[li], dict) and getattr(ly, "_k_conv_active", False):
                k_in = torch.cat([kv_caches[li]["k_buf"], kx.unsqueeze(-1)], dim=-1)
                kv_caches[li]["k_buf"] = k_in[..., -3:]
                kx = kx + (k_in * ly._k_conv_w).sum(dim=-1)

            if record is not None:
                record[li].append((pos, kx.detach().clone(), vx.detach().clone()))

            # Single-launch Fused Triton O(1) Step (fuses ReLU + L2-norm + M update + out_norm + v_self_scale!)
            y = fused_vsa_recurrent_step(
                q.contiguous(), kx.contiguous(), vx.contiguous(),
                states[li], ly._cached_gam, ly._cached_omg, ly._cached_vself,
                q_relu=ly._cached_qr, k_relu=ly._cached_kr, norm_w=ly._cached_nw,
            )

            if hasattr(ly, "v_conv") and isinstance(kv_caches[li], dict) and getattr(ly, "_v_conv_active", False):
                v_in = torch.cat([kv_caches[li]["v_buf"], vx.unsqueeze(-1)], dim=-1)
                kv_caches[li]["v_buf"] = v_in[..., -7:]
                v_loc = (v_in * ly._v_conv_w).sum(dim=-1)
                q_mod = torch.sigmoid((q * ly.local_q_gate.view(1, H, 1, HD).to(q.dtype)).sum(-1, keepdim=True))
                y = y + (2.0 * q_mod) * v_loc

            if getattr(ly, "_g_active", False):
                y = y * (2.0 * torch.sigmoid(ly.g_proj(h).view(B, 1, H, HD).transpose(1, 2)))
            x = x + ly.o_proj(y.transpose(1, 2).reshape(B, 1, H * HD))
        else:
            ly = blk.attn
            H, HD = ly.H, ly.HD
            if getattr(ly, "packed_q_gate", False):
                q, gate = torch.chunk(ly.q_proj(h).view(B, 1, H, HD * 2), 2, dim=-1)
                gate = gate.reshape(B, 1, H * HD)
                k = ly.k_proj(h).view(B, 1, ly.HKV, HD)
                v = ly.v_proj(h).view(B, 1, ly.HKV, HD)
                if ly.q_norm is not None:
                    q, k = ly.q_norm(q), ly.k_norm(k)
                q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                q, k = ly.rope(q, pos_start=pos), ly.rope(k, pos_start=pos)
            else:
                q, k, v = ly.head_qkv(h)
                gate = None
                if hasattr(ly, "_rope_cos") and pos < ly._rope_cos.shape[0]:
                    cos = ly._rope_cos[pos].view(1, 1, 1, HD)
                    sin = ly._rope_sin[pos].view(1, 1, 1, HD)
                    q = _apply_cached_rope(q, cos, sin)
                    k = _apply_cached_rope(k, cos, sin)
                else:
                    q, k = ly.rope(q, pos_start=pos), ly.rope(k, pos_start=pos)
            pk, pv = kv_caches[li]
            pk, pv = torch.cat([pk, k.to(pk.dtype)], dim=2), torch.cat([pv, v.to(pv.dtype)], dim=2)
            kv_caches[li] = (pk, pv)
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
    def prefill(self, ids: torch.Tensor, states=None, kv_caches=None, start_pos: int = 0):
        m = self.model
        B, T = ids.shape
        if states is None or kv_caches is None:
            states, kv_caches = self.blank_state(m, B, ids.device)

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
                kv_caches[li] = (k, v)
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
            with torch.autocast("cuda", torch.bfloat16, enabled=token.is_cuda):
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
                pk, pv = kv_caches[li]
                pk, pv = torch.cat([pk, k], dim=2), torch.cat([pv, v], dim=2)
                kv_caches[li] = (pk, pv)
                y = F.scaled_dot_product_attention(q, pk, pv)
            y = (y.to(x.dtype) * g).transpose(1, 2).reshape(B, 1, -1)
            x = x + layer.to_out(y)
            u = block.ffn_up(block.norm2(x))
            hu = block.ffn_up.out_features // 2
            x = x + block.ffn_down(F.silu(u[..., :hu]) * u[..., hu:])
        return m.norm_f(x), states, kv_caches

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new_tokens: int, temperature: float = 0.0, top_k: int = 40, eos_ids: set[int] | None = None):
        out = prompt.clone()
        x, states, kv = self.prefill(out)
        logits = F.linear(x[:, -1, :].float(), self.model.lm_head.weight.float())
        pos = prompt.shape[1] - 1
        for _ in range(max_new_tokens):
            nxt = self._sample(logits, temperature, top_k)
            out = torch.cat([out, nxt], dim=1)
            if eos_ids and int(nxt[0, 0].item()) in eos_ids:
                break
            pos += 1
            x, states, kv = self.step(nxt, pos, states, kv)
            logits = F.linear(x[:, -1, :].float(), self.model.lm_head.weight.float())
        return out

    @torch.no_grad()
    def continue_generate(self, ids: torch.Tensor, states, kv_caches, start_pos: int,
                          max_new_tokens: int, temperature: float = 0.0, top_k: int = 40, eos_ids: set[int] | None = None):
        out, pos, logits = [], start_pos, None
        for t in range(ids.shape[1]):
            x, states, kv_caches = self.step(ids[:, t : t + 1], pos, states, kv_caches)
            logits = F.linear(x[:, -1, :].float(), self.model.lm_head.weight.float())
            pos += 1
        for _ in range(max_new_tokens):
            nxt = self._sample(logits, temperature, top_k)
            out.append(nxt)
            if eos_ids and int(nxt[0, 0].item()) in eos_ids:
                break
            x, states, kv_caches = self.step(nxt, pos, states, kv_caches)
            logits = F.linear(x[:, -1, :].float(), self.model.lm_head.weight.float())
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
