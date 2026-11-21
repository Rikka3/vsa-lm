"""Bolt-3 Dual-Field Attention: Diversity Softmax Attractors + Gershgorin Eviction-to-Delta M_res."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..decode import StaticKV
from ..surgery import LlamaRMSNorm, LlamaRoPE, repeat_kv


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def _apply_rope_pos(x: torch.Tensor, pos: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    freqs = pos.unsqueeze(-1).float() * inv_freq.view(1, 1, 1, -1)
    emb = torch.cat((freqs, freqs), dim=-1)
    return (x * emb.cos().to(x.dtype)) + (_rotate_half(x) * emb.sin().to(x.dtype))


class Bolt3Attn(nn.Module):
    def __init__(self, cfg: dict, max_T: int = 2048, S: int = 128, C: int = 128):
        super().__init__()
        D, H, HKV = int(cfg["dim"]), int(cfg["num_heads"]), int(cfg["num_kv_heads"])
        HD = int(cfg.get("head_dim") or (D // H))
        qkv_bias = bool(cfg.get("qkv_bias", False))
        self.H, self.HKV, self.HD, self.G = H, HKV, HD, H // HKV
        self.S, self.C, self.n_sinks = S, C, 4
        self.sim_k, self.sim_v, self.gamma_u = 0.88, 0.80, 0.995

        self.q_proj = nn.Linear(D, H * HD, bias=qkv_bias)
        self.k_proj = nn.Linear(D, HKV * HD, bias=qkv_bias)
        self.v_proj = nn.Linear(D, HKV * HD, bias=qkv_bias)
        self.o_proj = nn.Linear(H * HD, D, bias=False)
        if cfg.get("qk_norm", False):
            self.q_norm = LlamaRMSNorm(HD, cfg["eps"])
            self.k_norm = LlamaRMSNorm(HD, cfg["eps"])
        else:
            self.q_norm, self.k_norm = None, None
        self.rope = LlamaRoPE(HD, cfg["rope_theta"], cfg.get("rope_scaling"))

        self.kappa = nn.Parameter(torch.ones(HKV))
        self.log_gamma = nn.Parameter(torch.full((HKV,), 3.5))
        self.beta_proj = nn.Linear(D, HKV, bias=True)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.constant_(self.beta_proj.bias, -1.0)
        self.eta_proj = nn.Linear(D, H, bias=True)
        nn.init.zeros_(self.eta_proj.weight)
        nn.init.zeros_(self.eta_proj.bias)
        self.res_norm = LlamaRMSNorm(HD, cfg["eps"])
        with torch.no_grad():
            self.res_norm.weight.fill_(0.05)

    def gamma_delta(self) -> torch.Tensor:
        return (0.999 * torch.sigmoid(self.log_gamma)).clamp(1e-3, 0.999).float()

    def head_qkv(self, x: torch.Tensor):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.H, self.HD)
        k = self.k_proj(x).view(b, t, self.HKV, self.HD)
        v = self.v_proj(x).view(b, t, self.HKV, self.HD)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def _consolidate_functional(
        self, ak, av, mass, util, pm, act, m_res,
        k_raw, k_rope, v_old, pos_old, beta_old, inv_freq, diag_eye, gam,
        n_filled: int = 0,
    ):
        b, hkv, l_tok, hd = k_raw.shape
        bh = b * hkv
        dt = k_raw.dtype
        kr_f = k_raw.reshape(bh, l_tok, hd)
        kp_f = k_rope.reshape(bh, l_tok, hd)
        v_f = v_old.reshape(bh, l_tok, hd)
        p_f = pos_old.expand(b, hkv, l_tok).reshape(bh, l_tok).float()
        b_f = beta_old.reshape(bh, l_tok)
        scale = hd ** -0.5

        if ak is None:
            dev = k_raw.device
            ak = torch.zeros(bh, self.S, hd, device=dev, dtype=dt)
            av = torch.zeros(bh, self.S, hd, device=dev, dtype=dt)
            mass = torch.ones(bh, self.S, device=dev, dtype=torch.float32)
            util = torch.zeros(bh, self.S, device=dev, dtype=torch.float32)
            pm = torch.zeros(bh, self.S, device=dev, dtype=torch.float32)
            act = torch.zeros(bh, self.S, device=dev, dtype=torch.bool)
            m_res = torch.zeros(bh, hd, hd, device=dev, dtype=torch.float32)
            n_filled = 0

        util = util * self.gamma_u

        if n_filled >= self.S:
            k_read = _apply_rope_pos(ak.view(b, hkv, self.S, hd), pm.view(b, hkv, self.S), inv_freq).view(bh, self.S, hd)
            kap = self.kappa.float().repeat(b).view(bh, 1, 1)
            with torch.no_grad():
                sc = torch.bmm(kp_f, k_read.transpose(-1, -2)).float() * scale + kap * torch.log(mass.clamp_min(1.0)).unsqueeze(1)
                v_hat = torch.bmm(F.softmax(sc, dim=-1).to(dt), av).float()
                v_f32 = v_f.float()
                rel_err = (v_f32 - v_hat).norm(dim=-1) / (v_f32.norm(dim=-1) + 1e-6)
                ak_n = F.normalize(ak.float(), dim=-1)
                cos_k = torch.bmm(F.normalize(kr_f.float(), dim=-1), ak_n.transpose(-1, -2))
                best_sim_k, best_slot = cos_k.max(dim=-1)
                matched_av = torch.gather(av.float(), 1, best_slot.unsqueeze(-1).expand(-1, -1, hd))
                sim_v = F.cosine_similarity(v_f32, matched_av, dim=-1)
                score_j = rel_err * (1.0 - best_sim_k.clamp(0.0, 1.0))
                merge_mask = (best_sim_k >= self.sim_k) & (sim_v >= self.sim_v)

            # Branchless merge (no CPU-GPU sync)
            m_oh = F.one_hot(best_slot, num_classes=self.S).float() * merge_mask.float().unsqueeze(-1)
            slot_cnt = m_oh.sum(dim=1)
            k_sum = torch.bmm(m_oh.transpose(1, 2), kr_f.float())
            v_sum = torch.bmm(m_oh.transpose(1, 2), v_f.float())
            p_sum = torch.bmm(m_oh.transpose(1, 2), p_f.unsqueeze(-1)).squeeze(-1)
            hit = slot_cnt > 0
            new_m = mass + slot_cnt
            ak = torch.where(hit.unsqueeze(-1), ((mass.unsqueeze(-1) * ak.float() + k_sum) / new_m.unsqueeze(-1)).to(dt), ak)
            av = torch.where(hit.unsqueeze(-1), ((mass.unsqueeze(-1) * av.float() + v_sum) / new_m.unsqueeze(-1)).to(dt), av)
            pm = torch.where(hit, (mass * pm + p_sum) / new_m, pm)
            mass = torch.where(hit, new_m, mass)

            # Branchless eviction + delta update (no CPU-GPU sync)
            ev_mask = ~merge_mask
            with torch.no_grad():
                ak_n2 = F.normalize(ak.float(), dim=-1)
                pair_cos = torch.bmm(ak_n2, ak_n2.transpose(-1, -2)).masked_fill(diag_eye, -1.0)
                keep_val = util * (1.0 - pair_cos.max(dim=-1).values.clamp(0.0, 0.98))
                n_scores = torch.where(ev_mask, score_j, torch.full_like(score_j, -1e9))
                k_cand = min(l_tok, self.S)
                top_vals, top_toks = torch.topk(n_scores, k=k_cand, dim=-1, largest=True)
                bot_vals, bot_slots = torch.topk(keep_val, k=k_cand, dim=-1, largest=False)
                win = (top_vals > bot_vals) & (top_vals > -1e8)
                ins_mask = torch.zeros_like(ev_mask).scatter(1, top_toks, win)
                delta_mask = (ev_mask & (~ins_mask)).float()

            k_g = torch.gather(kr_f, 1, top_toks.unsqueeze(-1).expand(-1, -1, hd))
            v_g = torch.gather(v_f, 1, top_toks.unsqueeze(-1).expand(-1, -1, hd))
            p_g = torch.gather(p_f, 1, top_toks)
            k_o = torch.gather(ak, 1, bot_slots.unsqueeze(-1).expand(-1, -1, hd))
            v_o = torch.gather(av, 1, bot_slots.unsqueeze(-1).expand(-1, -1, hd))
            p_o = torch.gather(pm, 1, bot_slots)
            m_o = torch.gather(mass, 1, bot_slots)
            u_o = torch.gather(util, 1, bot_slots)
            w3 = win.unsqueeze(-1)
            ak = ak.scatter(1, bot_slots.unsqueeze(-1).expand(-1, -1, hd), torch.where(w3, k_g, k_o))
            av = av.scatter(1, bot_slots.unsqueeze(-1).expand(-1, -1, hd), torch.where(w3, v_g, v_o))
            pm = pm.scatter(1, bot_slots, torch.where(win, p_g, p_o))
            mass = mass.scatter(1, bot_slots, torch.where(win, torch.ones_like(m_o), m_o))
            util = util.scatter(1, bot_slots, torch.where(win, torch.gather(rel_err, 1, top_toks), u_o))

            k_ev = F.normalize(kr_f.float(), dim=-1)
            w_ev = b_f.float() * delta_mask
            gram = torch.bmm(k_ev, k_ev.transpose(-1, -2)).abs()
            gersh = torch.bmm(gram, w_ev.unsqueeze(-1)).squeeze(-1).clamp_min(1.0)
            beta_safe = (w_ev / gersh).unsqueeze(-1)
            v_pred = torch.bmm(k_ev, m_res)
            err_v = (v_f.float() - v_pred) * beta_safe
            gam_bh = gam.repeat(b).view(bh, 1, 1)
            m_res = gam_bh * m_res + torch.bmm(k_ev.transpose(1, 2), err_v)
            return ak, av, mass, util, pm, act, m_res, n_filled

        with torch.no_grad():
            n_empty = self.S - n_filled
            m = min(n_empty, l_tok)
            if n_filled > 0:
                k_read = _apply_rope_pos(ak.view(b, hkv, self.S, hd), pm.view(b, hkv, self.S), inv_freq).view(bh, self.S, hd)
                sc = torch.bmm(kp_f, k_read.transpose(-1, -2)).float() * scale + torch.log(mass.clamp_min(1.0)).unsqueeze(1)
                sc.masked_fill_(~act.unsqueeze(1), -1e9)
                v_hat = torch.bmm(F.softmax(sc, dim=-1).to(dt), av).float()
                v_f32 = v_f.float()
                rel_err = (v_f32 - v_hat).norm(dim=-1) / (v_f32.norm(dim=-1) + 1e-6)
                cos_k = torch.bmm(F.normalize(kr_f.float(), dim=-1), F.normalize(ak.float(), dim=-1).transpose(-1, -2))
                cos_k.masked_fill_(~act.unsqueeze(1), 0.0)
                score_j = rel_err * (1.0 - cos_k.max(dim=-1).values.clamp(0.0, 1.0))
                init_u = rel_err
            else:
                score_j = F.normalize(v_f.float(), dim=-1).var(dim=-1)
                init_u = torch.full((bh, l_tok), 0.15, device=k_raw.device, dtype=torch.float32)

            top_toks = torch.topk(score_j, k=m, dim=-1, largest=True).indices
            slot_ids = torch.arange(n_filled, n_filled + m, device=k_raw.device).unsqueeze(0).expand(bh, -1)

        ak = ak.scatter(1, slot_ids.unsqueeze(-1).expand(-1, -1, hd), torch.gather(kr_f, 1, top_toks.unsqueeze(-1).expand(-1, -1, hd)))
        av = av.scatter(1, slot_ids.unsqueeze(-1).expand(-1, -1, hd), torch.gather(v_f, 1, top_toks.unsqueeze(-1).expand(-1, -1, hd)))
        pm = pm.scatter(1, slot_ids, torch.gather(p_f, 1, top_toks))
        mass = mass.scatter(1, slot_ids, torch.ones(bh, m, device=k_raw.device, dtype=torch.float32))
        util = util.scatter(1, slot_ids, torch.gather(init_u, 1, top_toks))
        act = act.scatter(1, slot_ids, torch.ones(bh, m, device=k_raw.device, dtype=torch.bool))
        return ak, av, mass, util, pm, act, m_res, n_filled + m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, T, _ = x.shape
        q_raw, k_raw, v_all_in = self.head_qkv(x)
        q_rope, k_rope = self.rope(q_raw), self.rope(k_raw)
        beta_all = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)
        eta_all = torch.tanh(self.eta_proj(x)).transpose(1, 2).unsqueeze(-1)
        gam = self.gamma_delta()
        inv_freq = self.rope.inv_freq.to(x.device)
        diag_eye = torch.eye(self.S, device=x.device, dtype=torch.bool).unsqueeze(0)
        scale = self.HD ** -0.5

        sink_k = sink_v = ak = av = mass = util = pm = act = m_res = None
        prev_kr = prev_kp = prev_v = prev_pos = prev_b = None
        n_filled = 0
        ys = []

        for s in range(0, T, self.C):
            e = min(T, s + self.C)
            tc = e - s
            qc_r, qc_p = q_raw[:, :, s:e], q_rope[:, :, s:e]
            kc_r, kc_p = k_raw[:, :, s:e], k_rope[:, :, s:e]
            vc, bc = v_all_in[:, :, s:e], beta_all[:, :, s:e]
            ec = eta_all[:, :, s:e]
            pos_c = torch.arange(s, e, device=x.device, dtype=torch.float32)[None, None, :]

            if prev_kp is not None:
                loc_k = torch.cat([prev_kp, kc_p], dim=2)
                loc_v = torch.cat([prev_v, vc], dim=2)
                prev_len = prev_kp.shape[2]
            else:
                loc_k, loc_v, prev_len = kc_p, vc, 0

            s_loc = torch.matmul(qc_p, repeat_kv(loc_k, self.G).transpose(-1, -2)).float() * scale
            qi = torch.arange(tc, device=x.device)[:, None]
            kj = torch.arange(loc_k.shape[2], device=x.device)[None, :]
            s_loc = s_loc.masked_fill(~(kj <= (prev_len + qi)).view(1, 1, tc, -1), float("-inf"))

            sp, vp = [], []
            if sink_k is not None:
                sp.append(torch.matmul(qc_p, repeat_kv(sink_k, self.G).transpose(-1, -2)).float() * scale)
                vp.append(repeat_kv(sink_v, self.G))
            if ak is not None:
                ak_eff = _apply_rope_pos(ak.view(b, self.HKV, self.S, self.HD), pm.view(b, self.HKV, self.S), inv_freq)
                s_att = torch.matmul(qc_p, repeat_kv(ak_eff, self.G).transpose(-1, -2)).float() * scale
                kap = self.kappa.float().view(1, self.HKV, 1, 1)
                m_bias = repeat_kv(kap * torch.log(mass.view(b, self.HKV, 1, self.S).clamp_min(1.0)), self.G)
                act_m = repeat_kv(act.view(b, self.HKV, 1, self.S), self.G)
                s_att = (s_att + m_bias).masked_fill(~act_m, float("-inf"))
                sp.append(s_att)
                vp.append(repeat_kv(av.view(b, self.HKV, self.S, self.HD), self.G))

            sp.append(s_loc)
            vp.append(repeat_kv(loc_v, self.G))
            scores = torch.cat(sp, dim=-1) if len(sp) > 1 else sp[0]
            v_cat = torch.cat(vp, dim=2) if len(vp) > 1 else vp[0]
            yc = torch.matmul(F.softmax(scores, dim=-1).to(v_cat.dtype), v_cat)

            if m_res is not None:
                q_rn = F.normalize(qc_r.view(b, self.HKV, self.G, tc, self.HD).float(), dim=-1)
                ym = torch.einsum("bhgtc,bhcd->bhgtd", q_rn, m_res.view(b, self.HKV, self.HD, self.HD))
                ym = ym * ((1.0 - gam).view(1, self.HKV, 1, 1, 1) / (self.HD ** 0.5))
                ym_n = self.res_norm(ym.reshape(b, self.H, tc, self.HD))
                yc = yc + (ec * ym_n).to(yc.dtype)
            elif e == T:
                yc = yc + 0.0 * (ec.sum() + self.res_norm.weight.sum() + self.kappa.sum() + gam.sum() + bc.sum())

            ys.append(yc)

            if tc == self.C and e < T:
                if prev_kp is not None:
                    p_kr, p_kp, p_v, p_pos, p_b = prev_kr, prev_kp, prev_v, prev_pos, prev_b
                    if sink_k is None:
                        ns = min(self.n_sinks, p_kp.shape[2])
                        sink_k, sink_v = p_kp[:, :, :ns], p_v[:, :, :ns]
                        p_kr, p_kp, p_v, p_pos, p_b = p_kr[:, :, ns:], p_kp[:, :, ns:], p_v[:, :, ns:], p_pos[:, :, ns:], p_b[:, :, ns:]
                    ak, av, mass, util, pm, act, m_res, n_filled = self._consolidate_functional(
                        ak, av, mass, util, pm, act, m_res, p_kr, p_kp, p_v, p_pos, p_b, inv_freq, diag_eye, gam,
                        n_filled=n_filled,
                    )
                prev_kr, prev_kp, prev_v, prev_pos, prev_b = kc_r, kc_p, vc, pos_c, bc

        y_out = torch.cat(ys, dim=2)
        return self.o_proj(y_out.transpose(1, 2).reshape(b, T, self.H * self.HD))


class Bolt3Decoder:
    def __init__(self, model, wrap_positions: bool = False, max_cache_pos: int = 65536, **kwargs):
        self.model = model.eval()
        self.max_T = model.config["max_T"]
        self.max_cache_pos = max_cache_pos
        self._diag_eye_cache = {}
        self.C = int(model.config.get("bolt3_C", 128))

    def _graph_runner(self, B: int, need_len: int, device):
        return None

    @staticmethod
    def state_bytes(model):
        flat, per_tok = 0, 0
        for blk in model.blocks:
            if hasattr(blk, "vsa"):
                ly = blk.vsa
                flat += ly.HKV * ly.HD * ly.HD * 4 + 2 * ly.S * ly.HKV * ly.HD * 2 + 4 * ly.C * ly.HKV * ly.HD * 2
            else:
                s = blk.attn
                per_tok += 2 * s.HKV * s.HD * 2
        return flat, per_tok

    @staticmethod
    def blank_state(model, batch_size: int, device, max_kv_len: int = 4096, state_dtype=torch.float32, **kwargs):
        states, kv_caches = [None] * len(model.blocks), [None] * len(model.blocks)
        dt = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        for li, blk in enumerate(model.blocks):
            if hasattr(blk, "vsa"):
                ly = blk.vsa
                states[li] = torch.zeros(batch_size, ly.HKV, ly.HD, ly.HD, device=device, dtype=state_dtype)
                kv_caches[li] = {
                    "ak": None, "av": None, "mass": None, "util": None, "pm": None, "act": None,
                    "sink_k": None, "sink_v": None,
                    "prev_kr": None, "prev_kp": None, "prev_v": None, "prev_pos": None, "prev_b": None,
                    "cur_kr": None, "cur_kp": None, "cur_v": None, "cur_pos": None, "cur_b": None,
                }
            else:
                s = blk.attn
                kv_caches[li] = StaticKV(batch_size, s.HKV, max_kv_len, s.HD, device, dt=dt)
        return states, kv_caches

    @staticmethod
    def _sample(logits, temperature: float, top_k: int = 40, top_p: float = 0.95):
        if temperature <= 1e-5:
            return logits.argmax(-1, keepdim=True)
        probs = F.softmax(logits.float() / temperature, dim=-1)
        if top_p < 1.0:
            sp, si = torch.sort(probs, descending=True, dim=-1)
            cp = torch.cumsum(sp, dim=-1)
            sp[cp - sp > top_p] = 0.0
            sp.div_(sp.sum(dim=-1, keepdim=True).clamp_min(1e-12))
            return si.gather(-1, torch.multinomial(sp, 1))
        tv, ti = torch.topk(logits.float() / temperature, min(top_k, logits.size(-1)), dim=-1)
        return ti.gather(-1, torch.multinomial(F.softmax(tv, -1), 1))

    @torch.no_grad()
    def step(self, tokens: torch.Tensor, pos_start: int, states, kv_caches, record=None):
        m = self.model
        b, t_s = tokens.shape
        with torch.autocast("cuda", torch.bfloat16, enabled=tokens.is_cuda):
            x = m.embed_tokens(tokens)
            for li, blk in enumerate(m.blocks):
                h = blk.norm1(x)
                if hasattr(blk, "vsa"):
                    ly: Bolt3Attn = blk.vsa
                    sc: dict = kv_caches[li]
                    q_r, k_r, v_n = ly.head_qkv(h)
                    q_p, k_p = ly.rope(q_r, pos_start=pos_start), ly.rope(k_r, pos_start=pos_start)
                    beta_n = torch.sigmoid(ly.beta_proj(h)).transpose(1, 2)
                    eta_n = torch.tanh(ly.eta_proj(h)).transpose(1, 2).unsqueeze(-1)
                    pos_n = torch.arange(pos_start, pos_start + t_s, device=tokens.device, dtype=torch.float32)[None, None, :]
                    gam = ly.gamma_delta()
                    inv_freq = ly.rope.inv_freq.to(tokens.device)
                    scale = ly.HD ** -0.5

                    len_before = 0 if sc["cur_kp"] is None else sc["cur_kp"].shape[2]
                    if sc["cur_kp"] is None:
                        sc["cur_kr"], sc["cur_kp"], sc["cur_v"], sc["cur_pos"], sc["cur_b"] = k_r, k_p, v_n, pos_n, beta_n
                    else:
                        sc["cur_kr"] = torch.cat([sc["cur_kr"], k_r], dim=2)
                        sc["cur_kp"] = torch.cat([sc["cur_kp"], k_p], dim=2)
                        sc["cur_v"] = torch.cat([sc["cur_v"], v_n], dim=2)
                        sc["cur_pos"] = torch.cat([sc["cur_pos"], pos_n], dim=2)
                        sc["cur_b"] = torch.cat([sc["cur_b"], beta_n], dim=2)

                    if sc["prev_kp"] is not None:
                        loc_k = torch.cat([sc["prev_kp"], sc["cur_kp"]], dim=2)
                        loc_v = torch.cat([sc["prev_v"], sc["cur_v"]], dim=2)
                        prev_len = sc["prev_kp"].shape[2]
                    else:
                        loc_k, loc_v, prev_len = sc["cur_kp"], sc["cur_v"], 0

                    s_loc = torch.matmul(q_p.float(), repeat_kv(loc_k, ly.G).float().transpose(-1, -2)) * scale
                    if t_s > 1:
                        qi = torch.arange(t_s, device=tokens.device)[:, None]
                        kj = torch.arange(loc_k.shape[2], device=tokens.device)[None, :]
                        s_loc.masked_fill_(~(kj <= (prev_len + len_before + qi)).view(1, 1, t_s, -1), float("-inf"))

                    sp, vp = [], []
                    if sc["sink_k"] is not None:
                        sp.append(torch.matmul(q_p.float(), repeat_kv(sc["sink_k"], ly.G).float().transpose(-1, -2)) * scale)
                        vp.append(repeat_kv(sc["sink_v"], ly.G))
                    if sc["ak"] is not None:
                        ak_eff = _apply_rope_pos(sc["ak"].view(b, ly.HKV, ly.S, ly.HD), sc["pm"].view(b, ly.HKV, ly.S), inv_freq)
                        s_att = torch.matmul(q_p.float(), repeat_kv(ak_eff, ly.G).float().transpose(-1, -2)) * scale
                        kap = ly.kappa.float().view(1, ly.HKV, 1, 1)
                        mb = repeat_kv(kap * torch.log(sc["mass"].view(b, ly.HKV, 1, ly.S).clamp_min(1.0)), ly.G)
                        am = repeat_kv(sc["act"].view(b, ly.HKV, 1, ly.S), ly.G)
                        s_att = (s_att + mb).masked_fill(~am, float("-inf"))
                        sp.append(s_att)
                        vp.append(repeat_kv(sc["av"].view(b, ly.HKV, ly.S, ly.HD), ly.G))

                    sp.append(s_loc)
                    vp.append(repeat_kv(loc_v, ly.G))
                    scores = torch.cat(sp, dim=-1) if len(sp) > 1 else sp[0]
                    v_cat = torch.cat(vp, dim=2) if len(vp) > 1 else vp[0]
                    y = torch.matmul(F.softmax(scores, dim=-1).to(v_cat.dtype), v_cat)

                    if states[li] is not None:
                        q_rn = F.normalize(q_r.view(b, ly.HKV, ly.G, t_s, ly.HD).float(), dim=-1)
                        ym = torch.einsum("bhgtc,bhcd->bhgtd", q_rn, states[li]) * ((1.0 - gam).view(1, ly.HKV, 1, 1, 1) / (ly.HD ** 0.5))
                        ym_n = ly.res_norm(ym.reshape(b, ly.H, t_s, ly.HD))
                        y = y + (eta_n * ym_n).to(y.dtype)

                    x = x + ly.o_proj(y.transpose(1, 2).reshape(b, t_s, ly.H * ly.HD))

                    if sc["cur_kp"].shape[2] >= ly.C:
                        if sc["prev_kp"] is not None:
                            p_kr, p_kp, p_v, p_pos, p_b = sc["prev_kr"], sc["prev_kp"], sc["prev_v"], sc["prev_pos"], sc["prev_b"]
                            if sc["sink_k"] is None:
                                ns = min(ly.n_sinks, p_kp.shape[2])
                                sc["sink_k"], sc["sink_v"] = p_kp[:, :, :ns].clone(), p_v[:, :, :ns].clone()
                                p_kr, p_kp, p_v, p_pos, p_b = p_kr[:, :, ns:], p_kp[:, :, ns:], p_v[:, :, ns:], p_pos[:, :, ns:], p_b[:, :, ns:]
                            eye = self._diag_eye_cache.get(tokens.device)
                            if eye is None:
                                eye = torch.eye(ly.S, device=tokens.device, dtype=torch.bool).unsqueeze(0)
                                self._diag_eye_cache[tokens.device] = eye
                            m_in = states[li].view(b * ly.HKV, ly.HD, ly.HD)
                            sc["ak"], sc["av"], sc["mass"], sc["util"], sc["pm"], sc["act"], m_out, sc["n_filled"] = ly._consolidate_functional(
                                sc["ak"], sc["av"], sc["mass"], sc["util"], sc["pm"], sc["act"], m_in,
                                p_kr, p_kp, p_v, p_pos, p_b, inv_freq, eye, gam,
                                n_filled=sc.get("n_filled", 0),
                            )
                            states[li] = m_out.view(b, ly.HKV, ly.HD, ly.HD)
                        sc["prev_kr"], sc["prev_kp"], sc["prev_v"], sc["prev_pos"], sc["prev_b"] = sc["cur_kr"], sc["cur_kp"], sc["cur_v"], sc["cur_pos"], sc["cur_b"]
                        sc["cur_kr"] = sc["cur_kp"] = sc["cur_v"] = sc["cur_pos"] = sc["cur_b"] = None
                else:
                    ly = blk.attn
                    q, k, v = ly.head_qkv(h)
                    q, k = ly.rope(q, pos_start=pos_start), ly.rope(k, pos_start=pos_start)
                    n_prev = kv_caches[li].n
                    pk, pv = kv_caches[li].append(k, v)
                    if t_s > 1:
                        qi = torch.arange(t_s, device=tokens.device)[:, None]
                        kj = torch.arange(n_prev + t_s, device=tokens.device)[None, :]
                        cmask = (kj <= (n_prev + qi)).view(1, 1, t_s, n_prev + t_s)
                        try:
                            y = F.scaled_dot_product_attention(q, pk.to(q.dtype), pv.to(q.dtype), attn_mask=cmask, enable_gqa=(ly.G > 1))
                        except TypeError:
                            y = F.scaled_dot_product_attention(q, repeat_kv(pk.to(q.dtype), ly.G), repeat_kv(pv.to(q.dtype), ly.G), attn_mask=cmask)
                    else:
                        try:
                            y = F.scaled_dot_product_attention(q, pk.to(q.dtype), pv.to(q.dtype), enable_gqa=(ly.G > 1))
                        except TypeError:
                            y = F.scaled_dot_product_attention(q, repeat_kv(pk.to(q.dtype), ly.G), repeat_kv(pv.to(q.dtype), ly.G))
                    x = x + ly.o_proj(y.transpose(1, 2).reshape(b, t_s, ly.H * ly.HD))
                x = x + blk.mlp(blk.norm2(x))
            return m.norm(x), states, kv_caches

    @torch.no_grad()
    def prefill(self, ids: torch.Tensor, max_kv_len: int = 4096):
        b, t = ids.shape
        states, kv = self.blank_state(self.model, b, ids.device, max_kv_len=max(max_kv_len, t + 1024))
        h = None
        for s in range(0, t, self.C):
            e = min(t, s + self.C)
            h, states, kv = self.step(ids[:, s:e], s, states, kv)
        return h, states, kv

    @torch.no_grad()
    def generate_batch(self, ids: torch.Tensor, max_new_tokens: int, temperature: float = 0.6, top_k: int = 40,
                       top_p: float = 0.95, eos_ids: set[int] | None = None,
                       close_think_ids: list[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = ids.shape
        h, states, kv = self.prefill(ids, max_kv_len=T + max_new_tokens + 64)
        logits = self.model.lm_head(h[:, -1, :])
        eos = torch.tensor(sorted(eos_ids or ()), device=ids.device, dtype=torch.long)
        done = torch.zeros(B, dtype=torch.bool, device=ids.device)
        lengths = torch.full((B,), max_new_tokens, dtype=torch.long, device=ids.device)
        end_think_id = close_think_ids[1] if (close_think_ids and len(close_think_ids) > 1) else None
        closed_think = torch.zeros(B, dtype=torch.bool, device=ids.device)
        q_pos = torch.full((B,), -1, dtype=torch.long, device=ids.device)
        budget_trigger = max(64, max_new_tokens - 64)
        out, pos = [], T
        for i in range(max_new_tokens):
            nxt = self._sample(logits, temperature, top_k=top_k, top_p=top_p)
            if close_think_ids:
                if i == budget_trigger:
                    need = (~closed_think) & (~done) & (q_pos < 0)
                    q_pos[need] = 0
                active_q = q_pos >= 0
                if bool(active_q.any()):
                    for r in torch.nonzero(active_q, as_tuple=False).view(-1).tolist():
                        nxt[r, 0] = close_think_ids[int(q_pos[r])]
                        q_pos[r] += 1
                        if q_pos[r] >= len(close_think_ids):
                            q_pos[r] = -1
                            closed_think[r] = True
                if end_think_id is not None:
                    closed_think |= (nxt[:, 0] == end_think_id)
            out.append(nxt)
            if eos.numel():
                hit = torch.isin(nxt[:, 0], eos) & ~done
                lengths[hit] = i + 1
                done |= hit
                if i % 16 == 15 and bool(done.all()):
                    break
            h, states, kv = self.step(nxt, pos, states, kv)
            logits = self.model.lm_head(h[:, -1, :])
            pos += 1
        return torch.cat(out, dim=1), lengths

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new_tokens: int, temperature: float = 0.0, top_k: int = 40,
                 top_p: float = 0.95, eos_ids: set[int] | None = None, close_think_ids: list[int] | None = None):
        toks, n = self.generate_batch(prompt, max_new_tokens, temperature, top_k, top_p, eos_ids, close_think_ids)
        return torch.cat([prompt, toks[:, : int(n[0].item())]], dim=1)

    @torch.no_grad()
    def continue_generate(self, ids: torch.Tensor, states, kv_caches, start_pos: int, max_new_tokens: int,
                          temperature: float = 0.0, top_k: int = 40, top_p: float = 0.95, eos_ids: set[int] | None = None):
        out, pos, h = [], start_pos, None
        for s in range(0, ids.shape[1], self.C):
            e = min(ids.shape[1], s + self.C)
            h, states, kv_caches = self.step(ids[:, s:e], pos, states, kv_caches)
            pos += (e - s)
        logits = self.model.lm_head(h[:, -1, :])
        for _ in range(max_new_tokens):
            nxt = self._sample(logits, temperature, top_k=top_k, top_p=top_p)
            out.append(nxt)
            if eos_ids and int(nxt[0, 0].item()) in eos_ids:
                break
            h, states, kv_caches = self.step(nxt, pos, states, kv_caches)
            logits = self.model.lm_head(h[:, -1, :])
            pos += 1
        toks = torch.cat(out, dim=1) if out else torch.empty(ids.shape[0], 0, dtype=torch.long, device=ids.device)
        return toks, states, kv_caches, pos

    @torch.no_grad()
    def memorize(self, ids: torch.Tensor, states, kv_caches, start_pos: int):
        pre_s = [s.clone() if s is not None else None for s in states]
        pos = start_pos
        for s in range(0, ids.shape[1], self.C):
            e = min(ids.shape[1], s + self.C)
            _, states, kv_caches = self.step(ids[:, s:e], pos, states, kv_caches)
            pos += (e - s)
        n_tok = ids.shape[1]
        traces = []
        for li, blk in enumerate(self.model.blocks):
            if hasattr(blk, "vsa"):
                gam = blk.vsa.gamma_delta().view(1, -1, 1, 1)
                n_chunks = n_tok // blk.vsa.C
                delta = states[li] - (gam ** n_chunks) * pre_s[li]
                traces.append((pos, delta))
            else:
                traces.append(None)
        return states, kv_caches, pos, traces

    @torch.no_grad()
    def forget(self, traces, states, next_pos: int, kv_caches=None, pre_kv_lens=None):
        for li, tr in enumerate(traces):
            if tr is not None and states[li] is not None:
                end_p, delta = tr
                blk = self.model.blocks[li]
                gam = blk.vsa.gamma_delta().view(1, -1, 1, 1)
                n_chunks = max(0, (next_pos - end_p) // blk.vsa.C)
                states[li] = states[li] - (gam ** n_chunks) * delta
        return states
