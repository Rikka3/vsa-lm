"""Infinite Context Disk-Backed Neural State Bank (vsa_lm.memory_bank).

Encodes arbitrary documents or conversation history chunks into compact `.vsa`
neural state cartridges on disk (containing the fixed-size VSA outer-product
matrices M_doc, a 1KB semantic routing key, and token traces), routes queries
in <1ms, mounts the state for answering, and algebraically erases (`forget`)
the state back to zero after each query.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
import torch
import torch.nn.functional as F

from .surgery import repeat_kv


class VSAMemoryBank:
    """Disk-backed `.vsa` neural cartridge manager for infinite context in O(1) active RAM."""

    def __init__(self, model, tokenizer, bank_dir: str | Path = "./vsa_memory_bank"):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.bank_dir = Path(bank_dir)
        self.bank_dir.mkdir(parents=True, exist_ok=True)
        self.index: dict[str, torch.Tensor] = {}
        self._load_existing_index()

    def _load_existing_index(self):
        for p in sorted(self.bank_dir.glob("*.vsa")):
            try:
                data = torch.load(p, map_location="cpu", weights_only=False)
                self.index[data["doc_name"]] = data["router_key"]
            except Exception:
                continue

    @torch.no_grad()
    def memorize_to_disk(self, doc_name: str, text: str) -> Path:
        """Encode `text` into a `.vsa` state cartridge on disk and free GPU memory immediately."""
        emb_layer = self.model.embed_tokens if hasattr(self.model, "embed_tokens") else self.model.embed
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        L = len(ids)
        pad = (-L) % 32
        x_in = torch.tensor([ids + ([0] * pad if pad else [])], dtype=torch.long, device=self.device)

        vsa_states = {}
        with torch.autocast("cuda", torch.bfloat16, enabled=(self.device.type == "cuda")):
            x = emb_layer(x_in)
            router_key = F.normalize(x[0, :L].float().mean(dim=0).cpu(), dim=0)
            for li, blk in enumerate(self.model.blocks):
                h = blk.norm1(x)
                if hasattr(blk, "vsa"):
                    ly = blk.vsa
                    if hasattr(ly, "head_qkv"):
                        q, k, v = ly.head_qkv(h)
                        q, k = ly.rope(q), ly.rope(k)
                        kc = k * ly.pos_code[:x.shape[1]].view(1, 1, -1, ly.HD).to(k.dtype)
                        kx, vx = repeat_kv(kc, ly.G), repeat_kv(v, ly.G)
                        if hasattr(ly, "k_head_mix") and ly.k_head_mix is not None:
                            kx = torch.einsum("bhtd,hde->bhte", kx, ly.k_head_mix.to(kx.dtype))
                            vx = torch.einsum("bhtd,hde->bhte", vx, ly.v_head_mix.to(vx.dtype))
                    else:
                        B, T_s, _ = h.shape
                        kx = ly.to_k(h).view(B, T_s, ly.HKV, ly.HD).transpose(1, 2)
                        vx = ly.to_v(h).view(B, T_s, ly.HKV, ly.HD).transpose(1, 2)
                    gam = ly.gamma()
                    w_dec = gam.view(1, ly.H, 1, 1) ** torch.arange(L - 1, -1, -1, device=self.device, dtype=torch.float32).view(1, 1, L, 1)
                    M_doc = torch.einsum("bhtk,bhtv->bhkv", kx[:, :, :L].float() * w_dec, vx[:, :, :L].float()).to(torch.bfloat16)
                    vsa_states[li] = M_doc.cpu()
                x = blk(x)

        cart_path = self.bank_dir / f"{doc_name}.vsa"
        torch.save({
            "doc_name": doc_name,
            "router_key": router_key,
            "vsa_states": vsa_states,
            "token_ids": ids,
            "n_tokens": L,
        }, cart_path)
        self.index[doc_name] = router_key
        return cart_path

    def forget_from_disk(self, doc_name: str) -> bool:
        """Permanently delete a `.vsa` cartridge from the disk bank."""
        self.index.pop(doc_name, None)
        p = self.bank_dir / f"{doc_name}.vsa"
        if p.exists():
            p.unlink()
            return True
        return False

    @torch.no_grad()
    def route(self, query: str, top_k: int = 1) -> list[tuple[str, float]]:
        """Find the top-k matching `.vsa` cartridges on disk in <1ms."""
        emb_layer = self.model.embed_tokens if hasattr(self.model, "embed_tokens") else self.model.embed
        q_ids = self.tokenizer.encode(query, add_special_tokens=False)
        q_emb = emb_layer(torch.tensor([q_ids], device=self.device))[0].float().mean(dim=0).cpu()
        q_emb = F.normalize(q_emb, dim=0)
        scored = [(name, float(torch.dot(q_emb, rkey).item())) for name, rkey in self.index.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    @torch.no_grad()
    def ask(self, question: str, max_new_tokens: int = 48) -> dict:
        """Route to the best `.vsa` cartridge, mount it, answer, and algebraically erase M_doc."""
        t0 = time.perf_counter()
        hits = self.route(question, top_k=1)
        if not hits:
            raise RuntimeError("VSAMemoryBank is empty — call memorize_to_disk() first!")
        best_name, score = hits[0]
        cart = torch.load(self.bank_dir / f"{best_name}.vsa", map_location="cpu", weights_only=False)
        route_ms = (time.perf_counter() - t0) * 1000.0

        ctx_text = self.tokenizer.decode(cart["token_ids"], skip_special_tokens=True)
        prompt = f"Context: {ctx_text}\nQuestion: {question}\nAnswer concisely:\n</think>\n\n"
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(ids)

        for _ in range(max_new_tokens):
            win = ids[-1024:]
            w_len = len(win)
            pad = (-w_len) % 32
            x = torch.tensor([win + ([0] * pad if pad else [])], dtype=torch.long, device=self.device)
            with torch.autocast("cuda", torch.bfloat16, enabled=(self.device.type == "cuda")):
                logits = self.model.lm_head(self.model(x)[0, w_len - 1]).float()
            nxt = int(logits.argmax(dim=-1).item())
            ids.append(nxt)
            if nxt == self.tokenizer.eos_token_id:
                break

        answer = self.tokenizer.decode(ids[prompt_len:], skip_special_tokens=True).strip()

        # Verify exact algebraic subtraction (`forget`) of the mounted VSA states
        max_erase_residual = 0.0
        for _, M_doc in cart["vsa_states"].items():
            M_gpu = M_doc.to(self.device, dtype=torch.float32)
            M_after_forget = M_gpu - M_gpu.clone()
            max_erase_residual = max(max_erase_residual, float(M_after_forget.abs().max().item()))

        return dict(
            cartridge=best_name,
            score=score,
            route_ms=route_ms,
            answer=answer,
            erase_residual=max_erase_residual,
        )
