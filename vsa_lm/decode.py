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
                states[li] = torch.einsum('bhte,bhtd->bhed', k.float() * w.unsqueeze(-1), v.float())
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
                carry = torch.einsum('bhed,bhe->bhd', M, qf.squeeze(2))
                y = (gamma.view(1, H, 1, 1) * carry.unsqueeze(2)
                     + write_scale * (qf * kf).sum(-1, keepdim=True) * vf) * omg / (HD ** 0.5)
                if getattr(layer, "head_norm", None) is not None:
                    y = layer.head_norm(y.to(x.dtype))
                states[li] = (gamma.view(1, H, 1, 1) * M
                              + write_scale * torch.einsum('bhe,bhd->bhed', kf.squeeze(2), vf.squeeze(2)))
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
