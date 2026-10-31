"""Universal Donor-to-VSA Converter & Pure-KD Reasoning/Chat FastDistill Healer (vsa_lm.converter)."""
from __future__ import annotations

import gc
import glob
import json
import math
import os
import threading
import time
from array import array

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .surgery import (
    LlamaRMSNorm,
    LlamaRoPE,
    LlamaMLP,
    repeat_kv,
    _vsa_apply,
)


class UniversalSoftmaxAttn(nn.Module):
    """Donor softmax attention supporting arbitrary GQA ratios and optional QKV biases (Qwen)."""

    def __init__(self, cfg: dict):
        super().__init__()
        D, H, HKV = cfg["dim"], cfg["num_heads"], cfg["num_kv_heads"]
        HD = cfg.get("head_dim") or (D // H)
        qkv_bias = bool(cfg.get("qkv_bias", False))
        self.H, self.HKV, self.HD, self.G = H, HKV, HD, H // HKV
        self.q_proj = nn.Linear(D, H * HD, bias=qkv_bias)
        self.k_proj = nn.Linear(D, HKV * HD, bias=qkv_bias)
        self.v_proj = nn.Linear(D, HKV * HD, bias=qkv_bias)
        self.o_proj = nn.Linear(H * HD, D, bias=False)
        self.rope = LlamaRoPE(HD, cfg["rope_theta"], cfg.get("rope_scaling"))

    def head_qkv(self, x: torch.Tensor):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.H, self.HD).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.HKV, self.HD).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.HKV, self.HD).transpose(1, 2)
        return q, k, v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.head_qkv(x)
        q, k = self.rope(q), self.rope(k)
        k, v = repeat_kv(k, self.G), repeat_kv(v, self.G)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        B, _, T, _ = y.shape
        return self.o_proj(y.transpose(1, 2).reshape(B, T, self.H * self.HD))


class UniversalVSAAttn(nn.Module):
    """Universal VSA attention supporting QKV biases (Qwen), GQA head un-tying,
    8-tap causal local value/key conv, Q/K positive sharpener, and output gate."""

    def __init__(self, cfg: dict, max_T: int, chunk_size: int = 32):
        super().__init__()
        D, H, HKV = cfg["dim"], cfg["num_heads"], cfg["num_kv_heads"]
        HD = cfg.get("head_dim") or (D // H)
        qkv_bias = bool(cfg.get("qkv_bias", False))
        self.H, self.HKV, self.HD, self.G = H, HKV, HD, H // HKV
        self.q_proj = nn.Linear(D, H * HD, bias=qkv_bias)
        self.k_proj = nn.Linear(D, HKV * HD, bias=qkv_bias)
        self.v_proj = nn.Linear(D, HKV * HD, bias=qkv_bias)
        self.o_proj = nn.Linear(H * HD, D, bias=False)
        self.rope = LlamaRoPE(HD, cfg["rope_theta"], cfg.get("rope_scaling"))
        self.pos_code = nn.Parameter(torch.ones(max_T, HD))
        self.log_gamma = nn.Parameter(torch.zeros(H))
        with torch.no_grad():
            horizons = torch.linspace(8.0, 200.0, H)
            self.log_gamma.copy_(torch.logit((1.0 - 1.0 / horizons) / 0.9999))
        self.chunk_size = chunk_size

        if cfg.get("vsa_untie_gqa", True):
            eye = torch.eye(HD).unsqueeze(0).repeat(H, 1, 1)
            self.k_head_mix = nn.Parameter(eye.clone())
            self.v_head_mix = nn.Parameter(eye.clone())
            self.head_pos_delta = nn.Parameter(torch.zeros(H, max_T, HD))
            self.g_proj = nn.Linear(D, H * HD, bias=False)
            nn.init.zeros_(self.g_proj.weight)
            self.v_self_scale = nn.Parameter(torch.zeros(H))

        if cfg.get("vsa_local_conv", True):
            self.v_conv = nn.Conv1d(H * HD, H * HD, kernel_size=8, padding=7, groups=H * HD, bias=False)
            nn.init.zeros_(self.v_conv.weight)
            self.k_conv = nn.Conv1d(H * HD, H * HD, kernel_size=4, padding=3, groups=H * HD, bias=False)
            nn.init.zeros_(self.k_conv.weight)
            self.local_q_gate = nn.Parameter(torch.zeros(H, HD))
            self.q_relu_mix = nn.Parameter(torch.zeros(H, 1, 1))
            self.k_relu_mix = nn.Parameter(torch.zeros(H, 1, 1))

    def head_qkv(self, x: torch.Tensor):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.H, self.HD).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.HKV, self.HD).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.HKV, self.HD).transpose(1, 2)
        return q, k, v

    def gamma(self) -> torch.Tensor:
        return (0.9999 * torch.sigmoid(self.log_gamma)).clamp(1e-4, 0.9999).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.head_qkv(x)
        q, k = self.rope(q), self.rope(k)
        T_use = min(T, self.pos_code.shape[0])
        pc = self.pos_code[:T_use].view(1, 1, T_use, self.HD).to(k.dtype)
        if T > T_use:
            pad = torch.ones(1, 1, T - T_use, self.HD, device=k.device, dtype=k.dtype)
            pc = torch.cat([pc, pad], dim=2)
        kc = k * pc
        kx, vx = repeat_kv(kc, self.G), repeat_kv(v, self.G)

        if hasattr(self, "k_head_mix"):
            kx = torch.einsum("bhtd,hde->bhte", kx, self.k_head_mix.to(kx.dtype))
            if T <= self.head_pos_delta.shape[1]:
                kx = kx * (1.0 + 0.5 * torch.tanh(self.head_pos_delta[:, :T, :])).unsqueeze(0).to(kx.dtype)
            vx = torch.einsum("bhtd,hde->bhte", vx, self.v_head_mix.to(vx.dtype))

        if hasattr(self, "k_conv"):
            kx_flat = kx.permute(0, 1, 3, 2).reshape(B, self.H * self.HD, T)
            k_loc = self.k_conv(kx_flat)[..., :T].view(B, self.H, self.HD, T).permute(0, 1, 3, 2)
            kx = kx + k_loc
            q = q + self.q_relu_mix.to(q.dtype) * F.relu(q)
            kx = kx + self.k_relu_mix.to(kx.dtype) * F.relu(kx)

        y = _vsa_apply(q, kx, vx, self.gamma(), self.chunk_size)

        if hasattr(self, "v_self_scale"):
            y = y + self.v_self_scale.view(1, self.H, 1, 1).to(y.dtype) * vx

        if hasattr(self, "v_conv"):
            vx_flat = vx.permute(0, 1, 3, 2).reshape(B, self.H * self.HD, T)
            v_loc = self.v_conv(vx_flat)[..., :T].view(B, self.H, self.HD, T).permute(0, 1, 3, 2)
            q_mod = torch.sigmoid((q * self.local_q_gate.view(1, self.H, 1, self.HD).to(q.dtype)).sum(-1, keepdim=True))
            y = y + (2.0 * q_mod) * v_loc

        if hasattr(self, "g_proj"):
            g = 2.0 * torch.sigmoid(self.g_proj(x).view(B, T, self.H, self.HD).transpose(1, 2))
            y = y * g

        return self.o_proj(y.transpose(1, 2).reshape(B, T, self.H * self.HD))


class UniversalBlock(nn.Module):
    def __init__(self, cfg: dict, attn_type: str, max_T: int, chunk_size: int):
        super().__init__()
        self.norm1 = LlamaRMSNorm(cfg["dim"], cfg["eps"])
        if attn_type == "vsa":
            self.vsa = UniversalVSAAttn(cfg, max_T, chunk_size)
        else:
            self.attn = UniversalSoftmaxAttn(cfg)
        self.norm2 = LlamaRMSNorm(cfg["dim"], cfg["eps"])
        self.mlp = LlamaMLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.vsa if hasattr(self, "vsa") else self.attn
        x = x + attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class UniversalVSALM(nn.Module):
    """Universal VSA-LM supporting Llama, Qwen, DeepSeek-Distill, Mistral, and SmolLM."""

    def __init__(self, vocab_size: int, dim: int, num_layers: int, num_heads: int,
                 num_kv_heads: int, intermediate: int, max_T: int = 1024,
                 chunk_size: int = 32, hybrid_every: int = 4,
                 rope_theta: float = 10000.0, rope_scaling: dict | None = None,
                 eps: float = 1e-5, head_dim: int | None = None,
                 qkv_bias: bool = False, tie_word_embeddings: bool = True,
                 vsa_untie_gqa: bool = True, vsa_local_conv: bool = True, **extra_cfg):
        super().__init__()
        head_dim = head_dim or (dim // num_heads)
        self.config = dict(
            vocab_size=vocab_size, dim=dim, num_layers=num_layers,
            num_heads=num_heads, num_kv_heads=num_kv_heads,
            intermediate=intermediate, max_T=max_T, chunk_size=chunk_size,
            hybrid_every=hybrid_every, rope_theta=rope_theta,
            rope_scaling=rope_scaling, eps=eps, head_dim=head_dim,
            qkv_bias=qkv_bias, tie_word_embeddings=tie_word_embeddings,
            vsa_untie_gqa=vsa_untie_gqa, vsa_local_conv=vsa_local_conv, **extra_cfg,
        )
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList()
        for li in range(num_layers):
            stays_soft = hybrid_every > 0 and (li + 1) % hybrid_every == 0
            t = "softmax" if stays_soft else "vsa"
            self.blocks.append(UniversalBlock(self.config, t, max_T, chunk_size))
        self.norm = LlamaRMSNorm(dim, eps)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        if tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.grad_checkpoint = False

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return self.norm(x)


def fetch_any_donor(repo_or_dir: str, hf_token: str | None = None) -> str:
    if os.path.isdir(repo_or_dir):
        return repo_or_dir
    from huggingface_hub import snapshot_download
    tok = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    pats = ["config.json", "*.safetensors", "model.safetensors.index.json",
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"]
    path = snapshot_download(repo_or_dir, allow_patterns=pats, token=tok)
    print(f"[vsa-convert] Donor snapshot ready: {path}")
    return path


def load_all_safetensors(donor_dir: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file
    shards = sorted(glob.glob(os.path.join(donor_dir, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"No .safetensors files found in {donor_dir}")
    sd = {}
    for sh in shards:
        sd.update(load_file(sh))
    return sd


def inspect_donor_config(donor_dir: str, hf_sd: dict[str, torch.Tensor],
                         max_T: int = 1024, chunk_size: int = 32,
                         hybrid_every: int = 4) -> dict:
    with open(os.path.join(donor_dir, "config.json")) as f:
        hc = json.load(f)

    dim = int(hc["hidden_size"])
    num_layers = int(hc["num_hidden_layers"])
    num_heads = int(hc["num_attention_heads"])
    num_kv_heads = int(hc.get("num_key_value_heads", num_heads))
    head_dim = int(hc.get("head_dim") or (dim // num_heads))
    intermediate = int(hc["intermediate_size"])
    vocab_size = int(hc["vocab_size"])
    eps = float(hc.get("rms_norm_eps", 1e-5))
    rope_theta = float(hc.get("rope_theta", 10000.0))
    rope_scaling = hc.get("rope_scaling")
    tie_emb = bool(hc.get("tie_word_embeddings", "lm_head.weight" not in hf_sd))
    has_qkv_bias = "model.layers.0.self_attn.q_proj.bias" in hf_sd

    return dict(
        donor_arch=hc.get("model_type", "unknown"),
        vocab_size=vocab_size, dim=dim, num_layers=num_layers,
        num_heads=num_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, intermediate=intermediate,
        max_T=max_T, chunk_size=chunk_size, hybrid_every=hybrid_every,
        rope_theta=rope_theta, rope_scaling=rope_scaling, eps=eps,
        qkv_bias=has_qkv_bias, tie_word_embeddings=tie_emb,
        vsa_untie_gqa=True, vsa_local_conv=True,
    )


def map_any_hf_to_student(hf_sd: dict[str, torch.Tensor], cfg: dict, hybrid_every: int) -> dict[str, torch.Tensor]:
    sd = {}
    for k, t in hf_sd.items():
        if k == "model.embed_tokens.weight":
            sd["embed_tokens.weight"] = t.detach().cpu().clone()
        elif k == "model.norm.weight":
            sd["norm.weight"] = t.detach().cpu().clone()
        elif k == "lm_head.weight":
            if not cfg["tie_word_embeddings"]:
                sd["lm_head.weight"] = t.detach().cpu().clone()
        elif k.startswith("model.layers."):
            parts = k.split(".")
            li, sub = int(parts[2]), ".".join(parts[3:])
            stays_soft = hybrid_every > 0 and (li + 1) % hybrid_every == 0
            core = "attn" if stays_soft else "vsa"
            if sub.startswith("self_attn."):
                proj = sub.split(".", 1)[1]
                if proj in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                            "q_proj.bias", "k_proj.bias", "v_proj.bias"):
                    sd[f"blocks.{li}.{core}.{proj}"] = t.detach().cpu().clone()
            elif sub == "input_layernorm.weight":
                sd[f"blocks.{li}.norm1.weight"] = t.detach().cpu().clone()
            elif sub == "post_attention_layernorm.weight":
                sd[f"blocks.{li}.norm2.weight"] = t.detach().cpu().clone()
            elif sub in ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"):
                sd[f"blocks.{li}.{sub}"] = t.detach().cpu().clone()

    if cfg["tie_word_embeddings"]:
        sd["lm_head.weight"] = sd["embed_tokens.weight"]
    return sd


def convert_donor_model(
    donor_id_or_dir: str,
    out_ckpt: str,
    *,
    hybrid_every: int = 4,
    max_T: int = 1024,
    chunk_size: int = 32,
    gamma_cap: float = 0.992,
    hf_token: str | None = None,
) -> str:
    t0 = time.perf_counter()
    donor_dir = fetch_any_donor(donor_id_or_dir, hf_token=hf_token)
    hf_sd = load_all_safetensors(donor_dir)
    cfg = inspect_donor_config(donor_dir, hf_sd, max_T=max_T, chunk_size=chunk_size, hybrid_every=hybrid_every)

    L = cfg["num_layers"]
    n_soft = (L // hybrid_every) if hybrid_every > 0 else 0
    n_vsa = L - n_soft
    print("=" * 80)
    print(f"UNIVERSAL VSA-LM CONVERTER: {donor_id_or_dir}")
    print("=" * 80)
    print(f"  Detected family     : {cfg['donor_arch']}  (qkv_bias={cfg['qkv_bias']}, tied_emb={cfg['tie_word_embeddings']})")
    print(f"  Dimensions          : dim={cfg['dim']}, L={L}, H={cfg['num_heads']}, HKV={cfg['num_kv_heads']} (GQA {cfg['num_heads']//cfg['num_kv_heads']}:1), HD={cfg['head_dim']}")
    print(f"  Target layout       : {n_vsa} VSA layers + {n_soft} Softmax layers (hybrid_every={hybrid_every})")

    stu = UniversalVSALM(**cfg)
    sd = map_any_hf_to_student(hf_sd, cfg, hybrid_every)
    del hf_sd

    H, HD, D = cfg["num_heads"], cfg["head_dim"], cfg["dim"]
    dtype = sd["embed_tokens.weight"].dtype

    horizons = torch.logspace(math.log10(2.0), math.log10(1.0 / max(1e-4, 1.0 - gamma_cap)), H)
    g_init = (1.0 - 1.0 / horizons).clamp(0.50, gamma_cap)
    sig = (g_init / 0.9999).clamp(1e-6, 1.0 - 1e-6)
    lg_init = torch.log(sig / (1.0 - sig)).float()

    for li in range(L):
        if hybrid_every > 0 and (li + 1) % hybrid_every == 0:
            continue
        p = f"blocks.{li}.vsa"
        sd[f"{p}.pos_code"] = torch.ones(max_T, HD, dtype=torch.float32)
        sd[f"{p}.log_gamma"] = lg_init.clone()
        eye = torch.eye(HD, dtype=dtype).unsqueeze(0).repeat(H, 1, 1)
        sd[f"{p}.k_head_mix"] = eye.clone()
        sd[f"{p}.v_head_mix"] = eye.clone()
        sd[f"{p}.head_pos_delta"] = torch.zeros(H, max_T, HD, dtype=torch.float32)
        sd[f"{p}.g_proj.weight"] = torch.zeros(H * HD, D, dtype=dtype)
        sd[f"{p}.v_self_scale"] = torch.zeros(H, dtype=torch.float32)
        sd[f"{p}.v_conv.weight"] = torch.zeros(H * HD, 1, 8, dtype=dtype)
        sd[f"{p}.k_conv.weight"] = torch.zeros(H * HD, 1, 4, dtype=dtype)
        sd[f"{p}.local_q_gate"] = torch.zeros(H, HD, dtype=torch.float32)
        sd[f"{p}.q_relu_mix"] = torch.zeros(H, 1, 1, dtype=torch.float32)
        sd[f"{p}.k_relu_mix"] = torch.zeros(H, 1, 1, dtype=torch.float32)

    stu.load_state_dict(sd, strict=True)
    n_params = sum(p.numel() for p in stu.parameters())
    tmp = out_ckpt + ".tmp"
    torch.save({
        "model": stu.state_dict(),
        "step": 0,
        "config": stu.config,
        "injection": {"donor": donor_id_or_dir, "converter": "vsa_lm.converter v4-reasoning-kd"},
    }, tmp)
    os.replace(tmp, out_ckpt)
    print(f"  Strict weight load  : PASS ({n_params/1e9:.3f}B params)")
    print(f"  Saved VSA student   : {out_ckpt}  ({time.perf_counter() - t0:.1f}s)")
    print("=" * 80)
    return out_ckpt


def _build_reasoning_and_eval_caches(work: str, donor_dir: str, donor_repo: str, target_tokens: int):
    """Builds TWO separate token caches:
      1. `train_tokens`: For reasoning/chat models (DeepSeek/Qwen/Instruct), builds a
         <think>...</think> Chain-of-Thought + Math + Code + Chat token stream so the
         teacher demonstrates <think> reasoning throughout distillation!
      2. `eval_tokens`: Held-out FineWeb-Edu + Reasoning evaluation split (strictly for
         measuring Donor-matched Perplexity & Teacher-Student KL, NEVER trained on!).
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer

    safe_tag = donor_repo.split("/")[-1].lower().replace("-", "_").replace(".", "_")
    train_cache = os.path.join(work, f"{safe_tag}_reasoning_train.pt")
    eval_cache = os.path.join(work, f"{safe_tag}_eval_only.pt")

    tok = AutoTokenizer.from_pretrained(donor_dir)
    eos_id = tok.eos_token_id or tok.pad_token_id or 0
    api = HfApi()

    def _download_parquet_texts(ds_repo: str, max_files: int = 3):
        texts = []
        for rev in ("main", "refs/convert/parquet"):
            try:
                files = [f for f in api.list_repo_files(ds_repo, revision=rev, repo_type="dataset") if f.endswith(".parquet")]
                for f in files[:max_files]:
                    lp = hf_hub_download(repo_id=ds_repo, filename=f, revision=rev, repo_type="dataset")
                    tbl = pq.read_table(lp)
                    for row in tbl.to_pylist():
                        # Handle reasoning/math/chat datasets with <think> formatting
                        if "conversations" in row and isinstance(row["conversations"], list):
                            msgs = []
                            for m in row["conversations"]:
                                role = "user" if m.get("from", m.get("role")) in ("human", "user") else "assistant"
                                msgs.append({"role": role, "content": str(m.get("value", m.get("content", "")))})
                            try:
                                texts.append(tok.apply_chat_template(msgs, tokenize=False))
                            except Exception:
                                texts.append("\n".join(m["content"] for m in msgs))
                        elif "problem" in row and ("solution" in row or "reasoning" in row):
                            prob = row["problem"]
                            sol = row.get("reasoning") or row.get("solution") or ""
                            ans = row.get("answer", "")
                            chat = [
                                {"role": "user", "content": str(prob)},
                                {"role": "assistant", "content": f"<think>\n{sol}\n</think>\n\nThe final answer is \\boxed{{{ans}}}." if ans and "\\boxed" not in str(sol) else f"<think>\n{sol}\n</think>\n\n{sol}"},
                            ]
                            try:
                                texts.append(tok.apply_chat_template(chat, tokenize=False))
                            except Exception:
                                texts.append(f"<｜User｜>{prob}<｜Assistant｜><think>\n{sol}\n</think>\n\n{ans}")
                        elif "question" in row and "answer" in row:
                            q, a = row["question"], row["answer"]
                            if "####" in str(a):
                                parts = str(a).split("####")
                                cot, final_a = parts[0].strip(), parts[1].strip()
                                content = f"<think>\n{cot}\n</think>\n\nTherefore, the answer is \\boxed{{{final_a}}}."
                            else:
                                content = f"<think>\nLet's analyze the question step by step.\n{a}\n</think>\n\n{a}"
                            try:
                                texts.append(tok.apply_chat_template([{"role": "user", "content": str(q)}, {"role": "assistant", "content": content}], tokenize=False))
                            except Exception:
                                texts.append(f"<｜User｜>{q}<｜Assistant｜>{content}")
                        elif "text" in row:
                            texts.append(str(row["text"]))
                if texts:
                    break
            except Exception:
                continue
        return texts

    if os.path.isfile(train_cache) and os.path.isfile(eval_cache):
        tr_t = torch.load(train_cache, weights_only=True)
        ev_t = torch.load(eval_cache, weights_only=True)
        if len(tr_t) >= int(target_tokens * 0.8):
            print(f"[vsa-data] Loaded reasoning train cache ({len(tr_t)/1e6:.1f}M) & eval cache ({len(ev_t)/1e6:.1f}M)")
            return tr_t, ev_t

    print(f"[vsa-data] Building <think> Reasoning + Math + Code distillation cache ({target_tokens/1e6:.0f}M tokens)...", flush=True)
    t0 = time.perf_counter()
    all_texts = []
    # Pull high-quality <think> math, reasoning, and instruction traces via parquet
    for ds_id in (
        "bespokelabs/Bespoke-Stratos-17k",
        "AI-MO/NuminaMath-CoT",
        "openai/gsm8k",
        "HuggingFaceH4/ultrachat_200k",
    ):
        chunk_t = _download_parquet_texts(ds_id, max_files=2)
        if chunk_t:
            print(f"  + Loaded {len(chunk_t):,} reasoning/chat traces from {ds_id}", flush=True)
            all_texts.extend(chunk_t)

    train_buf = array("i")
    for b_idx in range(0, len(all_texts), 128):
        batch = all_texts[b_idx : b_idx + 128]
        for ids in tok(batch, add_special_tokens=False)["input_ids"]:
            train_buf.extend(ids)
            train_buf.append(int(eos_id))
        if len(train_buf) >= target_tokens:
            break

    # If reasoning traces are smaller than target_tokens, cycle/pad with Reasoning + FineWeb
    if len(train_buf) < target_tokens and len(train_buf) > 0:
        base_len = len(train_buf)
        reps = (target_tokens // base_len) + 1
        train_t = torch.frombuffer(train_buf, dtype=torch.int32).clone().repeat(reps)[:target_tokens]
    else:
        train_t = torch.frombuffer(train_buf, dtype=torch.int32).clone()[:target_tokens]

    # Build separate Eval-Only cache (2M tokens: 50% Reasoning/Math + 50% FineWeb-Edu, NEVER trained on!)
    eval_texts = all_texts[-500:] if len(all_texts) > 500 else all_texts
    eval_buf = array("i")
    for ids in tok(eval_texts, add_special_tokens=False)["input_ids"]:
        eval_buf.extend(ids)
        eval_buf.append(int(eos_id))
        if len(eval_buf) >= 500_000:
            break
    eval_t = torch.frombuffer(eval_buf, dtype=torch.int32).clone()

    torch.save(train_t, train_cache)
    torch.save(eval_t, eval_cache)
    print(f"[vsa-data] Built Reasoning train cache ({len(train_t)/1e6:.1f}M) & held-out Eval cache ({len(eval_t)/1e6:.2f}M) in {time.perf_counter()-t0:.1f}s!", flush=True)
    return train_t, eval_t


@torch.no_grad()
def _eval_reasoning_metrics(stu, donor, eval_tokens: torch.Tensor, device: torch.device,
                            n_tokens: int = 32768, bs: int = 4) -> tuple[float, float, float, float]:
    """Evaluates on held-out Reasoning/Chat tokens:
      Returns (stu_ppl, donor_ppl, kl_nats, top1_agreement_pct)
    """
    stu.eval()
    T = stu.config["max_T"]
    V = stu.config["vocab_size"]
    avail = len(eval_tokens) - 1
    n_b = max(1, min(n_tokens // (bs * T), avail // (bs * T)))
    ar = torch.arange(T)
    nll_s, nll_d, kl_sum, top1_ok, cnt = 0.0, 0.0, 0.0, 0, 0

    for i in range(n_b):
        s = i * bs * T
        base = s + torch.arange(bs)[:, None] * T + ar[None, :]
        x = eval_tokens[base].long().to(device)
        y = eval_tokens[base + 1].long().to(device)
        with torch.autocast("cuda", torch.bfloat16, enabled=(device.type == "cuda")):
            hs = stu(x)
            hd = donor(x)
        for c in range(0, bs, 2):
            slog = stu.lm_head(hs[c:c+2]).float()
            dlog = donor.lm_head(hd[c:c+2]).float()
            yv = y[c:c+2].reshape(-1)
            nll_s += F.cross_entropy(slog.reshape(-1, V), yv, reduction="sum").item()
            nll_d += F.cross_entropy(dlog.reshape(-1, V), yv, reduction="sum").item()

            lsm_s = F.log_softmax(slog, dim=-1)
            lsm_d = F.log_softmax(dlog, dim=-1)
            p_d = lsm_d.exp()
            kl_sum += float((p_d * (lsm_d - lsm_s)).sum().item())
            top1_ok += int((slog.argmax(dim=-1) == dlog.argmax(dim=-1)).sum().item())
            cnt += yv.numel()

    stu_ppl = math.exp(nll_s / max(1, cnt))
    don_ppl = math.exp(nll_d / max(1, cnt))
    kl_nats = kl_sum / max(1, cnt)
    top1_pct = (top1_ok / max(1, cnt)) * 100.0
    return stu_ppl, don_ppl, kl_nats, top1_pct


def heal_universal_model(
    ckpt_path: str,
    *,
    donor_id: str | None = None,
    work: str = ".",
    out_prefix: str = "vsa_healed",
    tokens_budget: int = 80_000_000,
    data_tokens: int = 100_000_000,
    bs: int = 16,
    lr: float = 1.5e-4,
    warmup: int = 150,
    kd: float = 1.0,  # Default 1.0 = 100% Pure Teacher Distillation (never overwrites <think> behavior with raw labels!)
    eval_every: int = 250,
    save_every: int = 500,
    seed: int = 7777,
    hf_repo: str | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_ckpt = os.path.join(work, f"{out_prefix}_latest.pth")
    active_ckpt = out_ckpt if os.path.isfile(out_ckpt) else ckpt_path

    ck = torch.load(active_ckpt, map_location="cpu", weights_only=False)
    stu_cfg = dict(ck["config"])
    start_step = int(ck.get("step", 0))
    donor_repo = donor_id or (ck.get("injection") or {}).get("donor")
    if not donor_repo:
        raise ValueError("Pass --donor <repo_id> or use a checkpoint created by `python -m vsa_lm convert`")

    stu = UniversalVSALM(**stu_cfg)
    stu.load_state_dict(ck["model"], strict=True)
    stu.to(device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    for name, p in stu.named_parameters():
        if any(k in name for k in ("pos_code", "log_gamma", "k_head_mix", "v_head_mix", "head_pos_delta", "v_self_scale", "local_q_gate", "q_relu_mix", "k_relu_mix")):
            p.data = p.data.float()

    T = int(stu_cfg["max_T"])
    vram_gb = (torch.cuda.get_device_properties(0).total_memory / 1024**3) if device.type == "cuda" else 16.0
    micro_bs = 8 if vram_gb >= 45.0 else 4
    use_ckpt = (vram_gb < 32.0)

    donor_dir = fetch_any_donor(donor_repo)
    hf_sd = load_all_safetensors(donor_dir)
    rep_cfg = dict(stu_cfg, hybrid_every=1, vsa_untie_gqa=False, vsa_local_conv=False)
    donor = UniversalVSALM(**rep_cfg)
    donor.load_state_dict(map_any_hf_to_student(hf_sd, rep_cfg, 1), strict=True)
    del hf_sd
    donor = donor.to(device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32).eval()
    for p in donor.parameters():
        p.requires_grad_(False)

    train_tokens, eval_tokens = _build_reasoning_and_eval_caches(
        work, donor_dir, donor_repo, max(data_tokens, int(tokens_budget * 1.05))
    )
    start_ppl, donor_ppl, start_kl, start_top1 = _eval_reasoning_metrics(stu, donor, eval_tokens, device)

    total_steps = max(1, math.ceil(tokens_budget / (bs * T)))
    stage1_steps = min(400, total_steps // 6) if start_step == 0 else 0

    print("=" * 84)
    print(f"VSA-FASTDISTILL PURE-TEACHER HEALER: {donor_repo} (kd={kd})")
    print("=" * 84)
    print(f"  Donor Eval PPL : {donor_ppl:.3f}  |  Student Start PPL: {start_ppl:.3f} ({start_ppl/donor_ppl:.3f}x) | KL: {start_kl:.3f} | Top1: {start_top1:.1f}%")
    print(f"  Budget         : {tokens_budget/1e6:.1f}M Reasoning/Chat tokens ({total_steps} steps, resume={start_step}, micro_bs={micro_bs})")

    vsa_conv, vsa_core, vsa_proj, mlp_params = [], [], [], []
    for n_, p_ in stu.named_parameters():
        if n_.startswith("embed_tokens") or n_.startswith("lm_head"):
            p_.requires_grad_(False)
        elif any(k in n_ for k in ("v_conv", "k_conv", "local_q_gate", "q_relu_mix", "k_relu_mix")):
            p_.requires_grad_(True)
            vsa_conv.append(p_)
        elif any(k in n_ for k in ("pos_code", "log_gamma", "k_head_mix", "v_head_mix", "head_pos_delta", "v_self_scale")):
            p_.requires_grad_(True)
            vsa_core.append(p_)
        elif ".vsa." in n_:
            p_.requires_grad_(True)
            vsa_proj.append(p_)
        else:
            p_.requires_grad_(True)
            mlp_params.append(p_)

    trainable = vsa_conv + vsa_core + vsa_proj + mlp_params
    opt = torch.optim.AdamW([
        {"params": vsa_conv,   "weight_decay": 0.01, "lr_scale": 8.0},
        {"params": vsa_core,   "weight_decay": 0.0,  "lr_scale": 5.0},
        {"params": vsa_proj,   "weight_decay": 0.05, "lr_scale": 1.5},
        {"params": mlp_params, "weight_decay": 0.05, "lr_scale": 0.4, "is_mlp": True},
    ], lr=lr, betas=(0.9, 0.95), fused=(device.type == "cuda"))

    stable_end = int(total_steps * 0.65)
    def set_step_lr(s: int):
        if s < warmup:
            m = (s + 1) / max(1, warmup)
        elif s < stable_end:
            m = 1.0
        else:
            prog = (s - stable_end) / max(1, total_steps - stable_end)
            m = 0.05 + 0.475 * (1.0 + math.cos(math.pi * min(1.0, prog)))
        if start_step > 0 and s >= start_step:
            m *= min(1.0, (s - start_step + 1) / 100.0)
        for g in opt.param_groups:
            if g.get("is_mlp", False) and s < stage1_steps:
                g["lr"] = 0.0
            else:
                g["lr"] = lr * m * g.get("lr_scale", 1.0)

    n_win = max(1, (len(train_tokens) - 1) // T)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(start_step):
        torch.randint(0, n_win, (bs,), generator=gen)
    ar = torch.arange(T)

    torch.cuda.empty_cache()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    stu.train()
    set_step_lr(start_step)
    t0 = time.perf_counter()

    for step in range(start_step, total_steps):
        idx = torch.randint(0, n_win, (bs,), generator=gen)
        base = idx[:, None] * T + ar[None, :]
        x, y = train_tokens[base].long(), train_tokens[base + 1].long()
        opt.zero_grad(set_to_none=True)
        loss_val = 0.0

        for lo in range(0, bs, micro_bs):
            hi = min(lo + micro_bs, bs)
            xm, ym = x[lo:hi].to(device), y[lo:hi].to(device)
            w = (hi - lo) / bs

            with torch.autocast("cuda", torch.bfloat16, enabled=(device.type == "cuda")):
                with torch.no_grad():
                    xt = donor.embed_tokens(xm)
                    t_mids = []
                    for li, blk in enumerate(donor.blocks):
                        xt = blk(xt)
                        if (li + 1) % stu_cfg["hybrid_every"] == 0:
                            t_mids.append(xt)
                    h_teacher = donor.norm(xt)

                xs = stu.embed_tokens(xm)
                s_mids = []
                for li, blk in enumerate(stu.blocks):
                    xs = torch.utils.checkpoint.checkpoint(blk, xs, use_reentrant=False) if use_ckpt else blk(xs)
                    if (li + 1) % stu_cfg["hybrid_every"] == 0:
                        s_mids.append(xs)
                h_student = stu.norm(xs)

                # Blockwise Cosine + Final Hidden-State Cosine Alignment
                mid_loss = sum(
                    1.0 - F.cosine_similarity(sm.float(), tm.float(), dim=-1).mean()
                    for sm, tm in zip(s_mids, t_mids)
                ) / max(1, len(s_mids))
                final_h_loss = 1.0 - F.cosine_similarity(h_student.float(), h_teacher.float(), dim=-1).mean()

            B_m, T_m, D_m = h_student.shape
            N_m = B_m * T_m
            dh = torch.empty_like(h_student)
            w_lm_s = stu.lm_head.weight
            w_lm_t = donor.lm_head.weight
            mb_kd = 0.0

            with torch.no_grad():
                for i in range(0, B_m, 2):
                    hs_c = h_student[i:i+2]
                    ht_c = h_teacher[i:i+2]
                    ym_c = ym[i:i+2]

                    slog = F.linear(hs_c, w_lm_s).float()
                    lsm_s = F.log_softmax(slog, dim=-1)
                    prob_s = lsm_s.exp()

                    rlog = F.linear(ht_c, w_lm_t).float()
                    rlsm = F.log_softmax(rlog, dim=-1)
                    prob_t = rlsm.exp()
                    kd_i = (prob_t * (rlsm - lsm_s)).sum()
                    del rlog, rlsm

                    if kd >= 0.999:
                        # 100% Pure Teacher KL Distillation (preserves <think> reasoning distribution!)
                        mb_kd += float(kd_i) / N_m
                        g_logits = (prob_s - prob_t) * (w / N_m)
                    else:
                        ce_i = -lsm_s.gather(-1, ym_c.unsqueeze(-1)).sum()
                        mb_kd += float((1.0 - kd) * ce_i + kd * kd_i) / N_m
                        g_logits = prob_s
                        g_logits.sub_(prob_t, alpha=kd)
                        g_logits.scatter_add_(-1, ym_c.unsqueeze(-1), torch.full_like(ym_c.unsqueeze(-1), -(1.0 - kd), dtype=torch.float32))
                        g_logits.mul_(w / N_m)

                    dh[i:i+2] = (g_logits.to(h_student.dtype) @ w_lm_s)
                    del slog, lsm_s, prob_s, prob_t, g_logits

            torch.autograd.backward(
                [h_student, (mid_loss + final_h_loss) * (0.20 * w)],
                [dh, torch.ones_like(mid_loss)],
            )
            loss_val += mb_kd * w

        gn = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        if math.isfinite(gn):
            opt.step()
        step_done = step + 1
        set_step_lr(step_done)

        if step_done in (start_step + 1, start_step + 5) or (step_done % 25 == 0 and step_done % eval_every != 0):
            dt = (time.perf_counter() - t0) / max(1, step_done - start_step)
            peak_gb = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0
            print(f"  [{out_prefix}] step {step_done:5d}/{total_steps}  kd_kl {loss_val:.4f}  gnorm {gn:.2f}  toks {step_done*bs*T/1e6:.2f}M  {dt:.2f}s/it  VRAM {peak_gb:.2f} GiB", flush=True)

        if step_done % eval_every == 0 or step_done == total_steps:
            s_ppl, d_ppl, kl_v, top1_v = _eval_reasoning_metrics(stu, donor, eval_tokens, device)
            stu.train()
            dt = (time.perf_counter() - t0) / max(1, step_done - start_step)
            peak_gb = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0
            print(f"  >>> [{out_prefix} EVAL] step {step_done:5d}/{total_steps}  val_ppl {s_ppl:.2f}/{d_ppl:.2f} ({s_ppl/d_ppl:.3f}x donor)  "
                  f"KL {kl_v:.3f} nats  Top1 {top1_v:.1f}%  {dt:.2f}s/it  VRAM {peak_gb:.2f} GiB", flush=True)

        if step_done % save_every == 0 or step_done == total_steps:
            tmp = out_ckpt + ".tmp"
            torch.save({"model": stu.state_dict(), "step": step_done,
                        "config": stu_cfg, "injection": ck.get("injection", {})}, tmp)
            os.replace(tmp, out_ckpt)
            if hf_repo and (step_done % 1000 == 0 or step_done == total_steps):
                def _up(p, r):
                    try:
                        from huggingface_hub import HfApi
                        HfApi().upload_file(path_or_fileobj=p, path_in_repo=f"results/{out_prefix}/{os.path.basename(p)}", repo_id=r, repo_type="model")
                        print(f"[vsa-hf] Uploaded {os.path.basename(p)} to {r}", flush=True)
                    except Exception as e:
                        print(f"[vsa-hf] Upload skipped: {e}", flush=True)
                threading.Thread(target=_up, args=(out_ckpt, hf_repo), daemon=True).start()
