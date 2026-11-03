"""Infinite Context Disk-Backed Neural State Bank (vsa_lm.memory_bank).

Fixes D5 from Audit:
  - Uses VSARecurrentDecoder.memorize() to write the document into the live neural state
    and saves the true recurrent VSA states (`M_doc`), Softmax KV slices, and per-token traces to disk!
  - `ask()` mounts the saved neural states into VSARecurrentDecoder (without stuffing raw text into the prompt!),
    generates the answer from the mounted state, and runs `decoder.forget(traces, states, next_pos)` to
    measure the true non-tautological algebraic unlearning residual against the pre-mount blank state!
"""
from __future__ import annotations

import time
from pathlib import Path
import torch
import torch.nn.functional as F

from .decode import VSARecurrentDecoder
from .server import get_all_eos_ids


class VSAMemoryBank:
    """Disk-backed `.vsa` neural cartridge manager with true state mounting & algebraic `forget()`."""

    def __init__(self, model, tokenizer, bank_dir: str | Path = "./vsa_memory_bank"):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.decoder = VSARecurrentDecoder(self.model, wrap_positions=False)
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
        emb_layer = self.model.embed_tokens if hasattr(self.model, "embed_tokens") else self.model.embed
        doc_prefix = f"Document: {text.strip()}\n"
        ids = self.tokenizer.encode(doc_prefix, add_special_tokens=False)
        x_in = torch.tensor([ids], dtype=torch.long, device=self.device)

        with torch.autocast("cuda", torch.bfloat16, enabled=(self.device.type == "cuda")):
            router_key = F.normalize(emb_layer(x_in)[0].float().mean(dim=0).cpu(), dim=0)

        states, kv_caches = VSARecurrentDecoder.blank_state(self.model, 1, self.device)
        states, kv_caches, next_pos, traces = self.decoder.memorize(x_in, states, kv_caches, start_pos=0)

        cpu_states = [s.cpu().to(torch.bfloat16) if s is not None else None for s in states]
        cpu_kv = []
        for item in kv_caches:
            if isinstance(item, tuple):
                cpu_kv.append((item[0].cpu(), item[1].cpu()))
            elif isinstance(item, dict):
                cpu_kv.append({k: v.cpu() for k, v in item.items()})
            else:
                cpu_kv.append(None)
        cpu_traces = [[(p, k.cpu(), v.cpu()) for (p, k, v) in tr] for tr in traces]

        cart_path = self.bank_dir / f"{doc_name}.vsa"
        torch.save({
            "doc_name": doc_name,
            "router_key": router_key,
            "vsa_states": cpu_states,
            "kv_caches": cpu_kv,
            "traces": cpu_traces,
            "next_pos": next_pos,
            "n_tokens": len(ids),
        }, cart_path)
        self.index[doc_name] = router_key
        return cart_path

    def forget_from_disk(self, doc_name: str) -> bool:
        self.index.pop(doc_name, None)
        p = self.bank_dir / f"{doc_name}.vsa"
        if p.exists():
            p.unlink()
            return True
        return False

    @torch.no_grad()
    def route(self, query: str, top_k: int = 1) -> list[tuple[str, float]]:
        emb_layer = self.model.embed_tokens if hasattr(self.model, "embed_tokens") else self.model.embed
        q_ids = self.tokenizer.encode(query, add_special_tokens=False)
        q_emb = emb_layer(torch.tensor([q_ids], device=self.device))[0].float().mean(dim=0).cpu()
        q_emb = F.normalize(q_emb, dim=0)
        scored = [(name, float(torch.dot(q_emb, rkey).item())) for name, rkey in self.index.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    @torch.no_grad()
    def ask(self, question: str, max_new_tokens: int = 32) -> dict:
        t0 = time.perf_counter()
        hits = self.route(question, top_k=1)
        if not hits:
            raise RuntimeError("VSAMemoryBank is empty — call memorize_to_disk() first!")
        best_name, score = hits[0]
        cart = torch.load(self.bank_dir / f"{best_name}.vsa", map_location="cpu", weights_only=False)
        route_ms = (time.perf_counter() - t0) * 1000.0

        # Mount neural states onto GPU (NO raw document text in prompt!)
        states = [s.to(self.device, dtype=torch.float32) if s is not None else None for s in cart["vsa_states"]]
        kv_caches = []
        for item in cart["kv_caches"]:
            if isinstance(item, tuple):
                kv_caches.append((item[0].to(self.device), item[1].to(self.device)))
            elif isinstance(item, dict):
                kv_caches.append({k: v.to(self.device) for k, v in item.items()})
            else:
                kv_caches.append(None)
        traces = [[(p, k.to(self.device), v.to(self.device)) for (p, k, v) in tr] for tr in cart["traces"]]
        doc_next_pos = int(cart["next_pos"])

        # Snapshot mounted VSA state before question tokens so we can test exact algebraic forget(traces)
        mounted_vsa_copy = [s.clone() if s is not None else None for s in states]

        q_prompt = f"Question: {question.strip()}\nAnswer:"
        q_ids = torch.tensor([self.tokenizer.encode(q_prompt, add_special_tokens=False)], dtype=torch.long, device=self.device)
        eos_ids = get_all_eos_ids(self.tokenizer)

        gen_toks, _, _, _ = self.decoder.continue_generate(
            q_ids, states, kv_caches, start_pos=doc_next_pos,
            max_new_tokens=max_new_tokens, temperature=0.0, eos_ids=eos_ids,
        )
        answer = self.tokenizer.decode(gen_toks[0].tolist(), skip_special_tokens=True).strip().splitlines()[0].strip()

        # D5 Fix: True algebraic unlearning via decoder.forget(traces, mounted_vsa_copy, doc_next_pos)
        erased_states = self.decoder.forget(traces, mounted_vsa_copy, next_pos=doc_next_pos)
        max_erase_residual = max(
            float(s.abs().max().item()) for s in erased_states if s is not None
        )

        return dict(
            cartridge=best_name,
            score=score,
            route_ms=route_ms,
            answer=answer,
            erase_residual=max_erase_residual,
        )
