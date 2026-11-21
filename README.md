# Bolt-3 Attention (`bolt_attention`)

**Constant-memory Dual-Field Attention** for converting pretrained Transformer LLMs (Qwen3, Llama, Mistral, SmolLM, etc.) into fixed-state hybrid models with minimal distillation.

- **GitHub:** [Rikka3/bolt_attention](https://github.com/Rikka3/bolt_attention)
- **Checkpoints (HF):** [rikkathree/bolt-qwen3-4b](https://huggingface.co/rikkathree/bolt-qwen3-4b)

---

## Architecture: Bolt-3 Dual-Field Attention

In a converted hybrid model, every `hybrid_every`-th layer (default: every 4th) remains standard Softmax attention as a global anchor, while all other layers are replaced with **`Bolt3Attn`** (`vsa_lm/config/bolt3.py`).

`Bolt3Attn` processes sequences in chunks of size `C` (default `128`) and attends over **three fixed-size memory fields** per KV head, keeping inference state constant regardless of context length:

1. **Attention Sinks + Local Window**
   - Keeps the first `n_sinks = 4` tokens plus the previous and current chunks (up to `2C` tokens) with exact causal Softmax attention.
2. **Diversity Softmax Attractors (`S = 128` slots)**
   - As older chunks leave the local window, tokens are consolidated into `S` fixed attractor slots storing keys (`ak`), values (`av`), running `mass`, mean RoPE position (`pm`), and utility (`util`).
   - **Merge:** Tokens matching an existing slot (`cos_k >= 0.88` and `cos_v >= 0.80`) merge into it, increasing its `mass` (which adds a learned `kappa * log(mass)` attention bias) and updating its mean position.
   - **Diversity Eviction:** Unmerged tokens are scored by value prediction error (`rel_err`) and key novelty (`1 - best_sim_k`), replacing low-utility / redundant slots.
3. **Gershgorin Eviction-to-Delta Residual Memory (`M_res`)**
   - Tokens that neither merge nor win an attractor slot are written into a fixed `[HKV, HD, HD]` associative state `M_res` using a Gershgorin-normalized delta rule (`beta_safe`) with learned per-KV-head decay (`gamma_delta`).
   - Readout is normalized via `res_norm` (initialized at `0.05` for near-identity surgery) and gated into the attention output via a learned per-head `eta_proj`.

---

## Quick Start

### 1. Convert a Pretrained Donor to Bolt-3
```bash
python -m vsa_lm convert \
  --donor Qwen/Qwen3-4B \
  --out bolt3_qwen3_4b_init.pth \
  --bolt3
```

### 2. Heal with Knowledge Distillation
For fast, low-VRAM healing (recommended on 48–96 GB GPUs):
```bash
python -m vsa_lm heal \
  --ckpt bolt3_qwen3_4b_init.pth \
  --donor Qwen/Qwen3-4B \
  --out-prefix bolt3_qwen3_4b \
  --tokens 16000000 \
  --bs 8 \
  --lr 8e-5 \
  --kd 1.0 \
  --stage1-tokens 0 \
  --eval-every 250 \
  --save-every 50 \
  --freeze-mlp \
  --highspeed \
  --resume \
  --hf-repo rikkathree/bolt-qwen3-4b
```

Key flags:
- `--freeze-mlp`: Keeps donor MLPs/norms frozen in BF16 and trains only attention/Bolt-3 parameters (cuts ~40+ GB of FP32 master weights + Adam states on 4B models).
- `--highspeed`: Disables activation checkpointing when VRAM allows.
- `--resume`: Resumes from `<out-prefix>_latest.pth` and its optimizer sidecar.
- `--hf-repo`: Automatically creates the Hugging Face repo if needed and uploads checkpoints on every `--save-every` step (uses `HF_TOKEN` from environment).

### 3. Benchmark
```bash
python -m vsa_lm benchmark \
  --ckpt bolt3_qwen3_4b_latest.pth \
  --donor Qwen/Qwen3-4B \
  --limit 200 \
  --gsm8k-limit 200
```

### 4. Serve (OpenAI-Compatible API)
```bash
python -m vsa_lm serve \
  --ckpt bolt3_qwen3_4b_latest.pth \
  --donor Qwen/Qwen3-4B \
  --host 0.0.0.0 \
  --port 8000
```

---

## Repository Layout

- `vsa_lm/config/bolt3.py` — `Bolt3Attn` (chunked training forward) and `Bolt3Decoder` (constant-memory recurrent decoding).
- `vsa_lm/converter.py` — Donor conversion (`convert`), Stage-1/Stage-2 KL + hidden-state healing (`heal`), and Stage-3 on-policy distillation.
- `vsa_lm/benchmark.py` — Evaluation harness (MMLU, ARC, PIQA, HellaSwag, GSM8K, memory recall).
- `vsa_lm/cli.py` — CLI entry point (`convert`, `heal`, `benchmark`, `diagnose`, `scan`, `serve`).
