"""O(1)-state recurrent decoder with exact memory write/erase operations."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .functional import vsa_parallel_attention
from .model import _rope


class VSARecurrentDecoder:
    """Token-by-token decoder producing outputs identical to the parallel path.

    State layout (full-length lists indexed by block):
        states[li]    -> (B, H, HD, HD) fp32 memory (VSA blocks; None otherwise)
        kv_caches[li] -> (k, v) tensors (softmax blocks; None otherwise)
    """

    def __init__(self, model, wrap_positions: bool = True):
        self.model = model.eval()
        self.wrap_positions = wrap_positions
        self.max_T = model.config["max_T"]

    @staticmethod
    def state_bytes(model):
        """Return (flat_state_bytes, kv_bytes_per_token) at fp16 serving dtype."""
        vsa_bytes, kv_per_token = 0, 0
        for blk in model.blocks:
            if hasattr(blk, "vsa"):
                v = blk.vsa
                vsa_bytes += v.H * v.HD * v.HD * 4
            else:
                s = blk.softmax_attn
                kv_per_token += 2 * s.H * s.HD * 2
        return vsa_bytes, kv_per_token

    @staticmethod
    def blank_state(model, batch_size, device):
        """Zero memory and empty kv caches (safe for memorize-from-blank)."""
        states, kv_caches = [None] * len(model.blocks), [None] * len(model.blocks)
        for li, blk in enumerate(model.blocks):
            if hasattr(blk, "vsa"):
                v = blk.vsa
                states[li] = torch.zeros(batch_size, v.H, v.HD, v.HD,
                                         device=device, dtype=torch.float32)
            else:
                s = blk.softmax_attn
                kv_caches[li] = (torch.zeros(batch_size, s.H, 0, s.HD, device=device),
                                 torch.zeros(batch_size, s.H, 0, s.HD, device=device))
        return states, kv_caches

    def _gamma(self, layer):
        return layer.gamma()

    def _proj(self, blk, h, B, T):
        layer = blk.vsa if hasattr(blk, "vsa") else blk.softmax_attn
        H, HD = layer.H, layer.HD
        q = layer.to_q(h).view(B, T, H, HD).transpose(1, 2)
        k = layer.to_k(h).view(B, T, H, HD).transpose(1, 2)
        v = layer.to_v(h).view(B, T, H, HD).transpose(1, 2)
        g = torch.sigmoid(layer.to_g(h).view(B, T, H, HD).transpose(1, 2))
        return layer, q, k, v, g

    @staticmethod
    def _sample(logits, temperature, top_k):
        if temperature <= 0:
            return logits.argmax(-1, keepdim=True)
        topv, topi = torch.topk(logits / temperature, top_k, dim=-1)
        return topi.gather(-1, torch.multinomial(F.softmax(topv, -1), 1))

    @torch.no_grad()
    def prefill(self, ids):
        m = self.model
        x = m.embed(ids)
        B, T, _ = x.shape
        if T > self.max_T and not getattr(m.blocks[0], "vsa", None):
            raise ValueError(f"prompt length {T} exceeds max_T {self.max_T}")
        states, kv_caches = self.blank_state(m, B, x.device)
        for li, block in enumerate(m.blocks):
            layer, q, k, v, g = self._proj(block, block.norm1(x), B, T)
            if hasattr(block, "vsa"):
                HD = layer.HD
                if getattr(layer, "use_rope", False):
                    q, k = _rope(q), _rope(k)
                if hasattr(layer, "effective_pos_code"):
                    k = k * layer.effective_pos_code(T, k.dtype)
                else:
                    k = k * layer.pos_code[:T].view(1, 1, T, HD).to(k.dtype)
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
    def step(self, token, pos, states, kv_caches, record=None, write_scale: float = 1.0):
        """Process one token at absolute position `pos`; returns (hidden, states, kv).
        `write_scale` (default 1.0) allows gated state writes during latent thinking passes.
        """
        m = self.model
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
                        pc = layer.effective_pos_code(pos + 1, k.dtype)[:, :, pos:pos + 1, :]
                    elif self.wrap_positions:
                        p = pos % layer.pos_code.shape[0]
                        pc = layer.effective_pos_code(p + 1, k.dtype)[:, :, p:p + 1, :]
                    else:
                        pc = torch.ones(1, 1, 1, HD, device=k.device, dtype=k.dtype)
                    k = k * pc
                else:
                    p = pos % layer.pos_code.shape[0] if self.wrap_positions else pos
                    k = k * layer.pos_code[p].view(1, 1, 1, HD).to(k.dtype)
                gamma = self._gamma(layer)
                omg = (1.0 - gamma).view(1, H, 1, 1)
                qf, kf, vf = q.float(), k.float(), v.float()
                M = states[li]
                carry = torch.einsum("bhed,bhe->bhd", M, qf.squeeze(2))
                y = (gamma.view(1, H, 1, 1) * carry.unsqueeze(2)
                     + write_scale * (qf * kf).sum(-1, keepdim=True) * vf) * omg / (HD ** 0.5)
                if getattr(layer, "head_norm", None) is not None:
                    y = layer.head_norm(y.to(x.dtype))
                states[li] = (gamma.view(1, H, 1, 1) * M
                              + write_scale * torch.einsum("bhe,bhd->bhed", kf.squeeze(2), vf.squeeze(2)))
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
    def generate(self, prompt, max_new_tokens, temperature=0.8, top_k=40):
        out = prompt.clone()
        x, states, kv = self.prefill(out)
        logits = self.model.lm_head(x)[:, -1, :]
        pos = prompt.shape[1] - 1
        for _ in range(max_new_tokens):
            nxt = self._sample(logits, temperature, top_k)
            out = torch.cat([out, nxt], dim=1)
            pos += 1
            x, states, kv = self.step(nxt, pos, states, kv)
            logits = self.model.lm_head(x)[:, -1, :]
        return out

    @torch.no_grad()
    def continue_generate(self, ids, states, kv_caches, start_pos,
                          max_new_tokens, temperature=0.8, top_k=40):
        """Generate from an existing state; the memory clock keeps running."""
        out, pos, logits = [], start_pos, None
        for t in range(ids.shape[1]):
            x, states, kv_caches = self.step(ids[:, t:t+1], pos, states, kv_caches)
            logits = self.model.lm_head(x)[:, -1, :]
            pos += 1
        for _ in range(max_new_tokens):
            nxt = self._sample(logits, temperature, top_k)
            out.append(nxt)
            x, states, kv_caches = self.step(nxt, pos, states, kv_caches)
            logits = self.model.lm_head(x)[:, -1, :]
            pos += 1
        toks = torch.cat(out, dim=1) if out else torch.empty(ids.shape[0], 0, dtype=torch.long, device=ids.device)
        return toks, states, kv_caches, pos

    @torch.no_grad()
    def memorize(self, ids, states, kv_caches, start_pos):
        """Write text into memory in a single pass (batch size 1).

        Returns (states, kv_caches, next_pos, traces). Traces are O(doc length)
        and enable exact erasure later.
        """
        traces = [[] for _ in self.model.blocks]
        pos = start_pos
        for t in range(ids.shape[1]):
            _, states, kv_caches = self.step(ids[:, t:t+1], pos, states, kv_caches, record=traces)
            pos += 1
        return states, kv_caches, pos, traces

    @torch.no_grad()
    def forget(self, traces, states, next_pos):
        """Exactly erase a memorized document from the VSA-layer states.

        The state is linear in the writes, so subtracting the decay-weighted
        outer products is exact. Softmax-layer caches are not covered.
        """
        for li, trace in enumerate(traces):
            if not trace:
                continue
            gamma = self._gamma(self.model.blocks[li].vsa)
            for (p, k, v) in trace:
                w = gamma.view(-1, 1, 1) ** float(next_pos - 1 - p)
                states[li] -= w * torch.einsum("he,hd->hed", k, v).unsqueeze(0)
        return states
