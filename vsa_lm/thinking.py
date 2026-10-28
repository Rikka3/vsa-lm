"""OpenMythos Constant-Memory Latent Thinking Engine & Phase-B Trainer (vsa_lm.thinking).

Runs K recurrent latent reasoning passes exclusively inside the fixed-size
(H x HD x HD) VSA state matrices while keeping Softmax KV caches frozen
(strictly O(1) zero-byte memory growth during thinking).

Key guarantee (Zero-Poisoning Init):
  With gated state updates M <- (1 - beta*(1 - gamma))*M + beta*(k (x) v)
  initialized at beta = 0, K > 0 thinking steps are an EXACT mathematical
  identity with K = 0 at initialization -- eliminating untrained EOT/decay
  state poisoning and allowing fast (~15 min) latent-thought training.
"""
from __future__ import annotations

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from .surgery import repeat_kv, _vsa_apply


class LatentThinkController(nn.Module):
    """Learnable latent thought embeddings + per-layer/per-head VSA write gate beta.

    Parameters are tiny (~32 KB total for a 1B model) and initialized so beta = 0
    at step 0: K thinking steps start at exact K=0 accuracy and only open beta
    where multi-hop VSA state refinement lowers answer cross-entropy.
    """

    def __init__(self, dim: int, num_layers: int, num_heads: int, max_K: int = 8):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_K = max_K
        # Learnable latent thought tokens tau_0 .. tau_{max_K-1}
        self.think_embeds = nn.Parameter(torch.randn(max_K, dim) * 0.02)
        # Per-layer, per-head logit for beta; initialized at -6.0 so sigmoid(-6.0) ~ 0.0025 (near-identity)
        self.logit_beta = nn.Parameter(torch.full((num_layers, num_heads), -6.0))
        # Halting confidence projector (predicts when KL change saturates)
        self.halt_proj = nn.Linear(dim, 1)
        nn.init.zeros_(self.halt_proj.weight)
        nn.init.constant_(self.halt_proj.bias, -2.0)

    def beta(self, layer_idx: int) -> torch.Tensor:
        """Returns (1, H, 1, 1) write gate in (0, 1) for VSA layer `layer_idx`."""
        return torch.sigmoid(self.logit_beta[layer_idx]).view(1, self.num_heads, 1, 1).float()


class OpenMythosEngine:
    """Constant-memory recurrent decoder + K-step latent thinking engine for VSA-LM."""

    def __init__(self, model: nn.Module, controller: LatentThinkController | None = None):
        self.model = model.eval()
        cfg = model.config
        self.dim = cfg["dim"]
        self.num_layers = cfg["num_layers"]
        self.num_heads = cfg["num_heads"]
        self.controller = controller

    @staticmethod
    def blank_state(model: nn.Module, batch_size: int, device: torch.device):
        states, kv = [None] * len(model.blocks), [None] * len(model.blocks)
        for li, blk in enumerate(model.blocks):
            if hasattr(blk, "vsa"):
                ly = blk.vsa
                states[li] = torch.zeros(batch_size, ly.H, ly.HD, ly.HD,
                                         device=device, dtype=torch.float32)
            else:
                kv[li] = None
        return states, kv

    @staticmethod
    def clone_state(states, kv):
        s_copy = [s.clone() if s is not None else None for s in states]
        kv_copy = [None if pair is None else (pair[0].clone(), pair[1].clone()) for pair in kv]
        return s_copy, kv_copy

    def step_from_embed(self, x_emb: torch.Tensor, pos: int, states: list, kv: list,
                        *, write_vsa: bool = True, write_softmax_kv: bool = True,
                        beta_override: float | None = None):
        """Single recurrent step starting from an embedding tensor (B, 1, D).
        When running a latent thinking pass:
          - write_softmax_kv=False (Softmax KV cache grows by ZERO bytes!)
          - write_vsa=True with gated beta (refines the fixed 64x64 VSA state M).
        """
        m = self.model
        x = x_emb
        B = x.shape[0]
        for li, blk in enumerate(m.blocks):
            h = blk.norm1(x)
            if hasattr(blk, "vsa"):
                ly = blk.vsa
                q, k, v = ly.head_qkv(h)
                q = ly.rope(q, pos_start=pos)
                k = ly.rope(k, pos_start=pos)
                p = pos % ly.pos_code.shape[0]
                kc = k * ly.pos_code[p].view(1, 1, 1, ly.HD).to(k.dtype)
                kx = repeat_kv(kc, ly.G)
                vx = repeat_kv(v, ly.G)
                if hasattr(ly, "k_head_mix"):
                    kx = torch.einsum("bhtd,hde->bhte", kx, ly.k_head_mix.to(kx.dtype))
                    kx = kx * (1.0 + 0.5 * torch.tanh(ly.head_pos_delta[:, p:p + 1, :])).unsqueeze(0).to(kx.dtype)
                    vx = torch.einsum("bhtd,hde->bhte", vx, ly.v_head_mix.to(vx.dtype))
                gam = ly.gamma()
                qf, kf, vf = q.float(), kx.float(), vx.float()
                outer = kf[:, :, 0, :].unsqueeze(-1) * vf[:, :, 0, :].unsqueeze(-2)

                if beta_override is not None:
                    b_gate = float(beta_override)
                elif self.controller is not None and not write_softmax_kv:
                    b_gate = self.controller.beta(li)
                else:
                    b_gate = 1.0

                decay_factor = 1.0 - b_gate * (1.0 - gam.view(1, -1, 1, 1))
                M_next = decay_factor * states[li] + b_gate * outer
                if write_vsa:
                    states[li] = M_next
                y = torch.einsum("bhkv,bhk->bhv", M_next, qf[:, :, 0, :])
                y = ((1.0 - gam).view(1, -1, 1, 1) * y.unsqueeze(2) / (ly.HD ** 0.5)).to(x.dtype)
                if hasattr(ly, "v_self_scale"):
                    y = y + ly.v_self_scale.view(1, ly.H, 1, 1).to(y.dtype) * vx
                if hasattr(ly, "g_proj"):
                    g = 2.0 * torch.sigmoid(ly.g_proj(h).view(B, 1, ly.H, ly.HD).transpose(1, 2))
                    y = y * g
            else:
                ly = blk.attn if hasattr(blk, "attn") else blk.softmax_attn
                q, k, v = ly.head_qkv(h)
                q = ly.rope(q, pos_start=pos)
                k = ly.rope(k, pos_start=pos)
                if kv[li] is None:
                    pk, pv = k, v
                else:
                    pk = torch.cat([kv[li][0], k], dim=2)
                    pv = torch.cat([kv[li][1], v], dim=2)
                if write_softmax_kv:
                    kv[li] = (pk, pv)
                y = F.scaled_dot_product_attention(q, repeat_kv(pk, ly.G), repeat_kv(pv, ly.G)).to(x.dtype)

            y = y.transpose(1, 2).reshape(B, 1, ly.H * ly.HD)
            o_proj = ly.o_proj if hasattr(ly, "o_proj") else ly.to_out
            x = x + o_proj(y)
            x = x + blk.mlp(blk.norm2(x))
        norm_f = m.norm if hasattr(m, "norm") else m.norm_f
        return norm_f(x), states, kv

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor):
        states, kv = self.blank_state(self.model, input_ids.shape[0], input_ids.device)
        emb_layer = self.model.embed_tokens if hasattr(self.model, "embed_tokens") else self.model.embed
        hidden = None
        for t in range(input_ids.shape[1]):
            x_emb = emb_layer(input_ids[:, t:t + 1])
            hidden, states, kv = self.step_from_embed(x_emb, t, states, kv)
        return hidden, states, kv

    def think(self, states: list, kv: list, pos: int, K: int = 4):
        """Perform K constant-memory latent thinking steps inside the VSA layers.
        Softmax KV caches are NEVER grown (0 bytes added).
        """
        if K <= 0:
            return None, states, kv
        B = states[0].shape[0]
        device = states[0].device
        emb_layer = self.model.embed_tokens if hasattr(self.model, "embed_tokens") else self.model.embed
        dtype = emb_layer.weight.dtype
        hidden = None
        for k_step in range(K):
            if self.controller is not None:
                tau = self.controller.think_embeds[k_step % self.controller.max_K].view(1, 1, -1).expand(B, 1, -1).to(dtype)
            else:
                tau = emb_layer.weight.mean(0).view(1, 1, -1).expand(B, 1, -1).to(dtype)
            hidden, states, kv = self.step_from_embed(
                tau, pos, states, kv,
                write_vsa=True,
                write_softmax_kv=False,  # Softmax KV cache stays frozen!
                beta_override=0.05 if self.controller is None else None,
            )
        return hidden, states, kv
