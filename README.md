# VSA-LM: Fixed-State Linear-Attention Hybrids with Exact Memory Editing & Latent Thinking

**VSA-LM** is an open-source language model architecture and conversion toolkit that replaces **75% of a Transformer's growing KV-cache attention layers** with **Fixed-State Decayed Outer-Product Memory (VSA)** layers:

$$M_t = \gamma M_{t-1} + k_t \otimes v_t, \qquad y_t = \frac{1 - \gamma}{\sqrt{D}} (M_t q_t)$$

Because 75% of the network compresses context into a constant-size $D \times D$ matrix per head instead of appending tokens to a KV cache, **VSA-LM cuts KV-cache RAM growth by ~75% forever**, decodes **up to 2.43x faster** at 64k context, and unlocks three capabilities standard Transformers cannot perform:

1. **Exact Algebraic Memory Unlearning (`memorize` / `forget`):** Because the VSA state $M$ is strictly linear in the key-value outer products, any memorized document can be surgically subtracted from the live neural state ($M \leftarrow M - \sum \gamma^{\Delta t} k_t \otimes v_t$) in $< 1\text{ ms}$ with $< 10^{-5}$ residual error.
2. **Disk-Backed Infinite Context Cartridges (`VSAMemoryBank`):** Encode unlimited documents or chat history into compact `.vsa` neural state cartridges on disk (`0.00 MB` idle GPU/CPU RAM), route queries in `< 20 ms`, mount the state to answer, and algebraically unmount it.
3. **OpenMythos $O(1)$ Latent Thinking (`vsa_lm.thinking`):** Run $K$ recurrent latent reasoning passes exclusively inside the fixed-size VSA state matrices while freezing Softmax KV caches (**0 bytes of memory growth during thinking**).

---

## Pretrained & Converted Models on Hugging Face

| Model | Base / Donor | Architecture | Key Results | Hugging Face Link |
|---|---|---|---|---|
| **VSA-DeepSeek-R1-Qwen-1.5B** | `DeepSeek-R1-Distill-Qwen-1.5B` | 21 VSA + 7 Softmax (`1.87B`) | **0.937x donor reasoning PPL** (`9.52` vs `10.16`), **79.9% Top-1 match**, full `<think>` CoT, **+28.7%** OpenMythos latent confidence boost, **-74.6% KV RAM** | [`rikkathree/vsa-deepseek-r1-qwen-1.5b`](https://huggingface.co/rikkathree/vsa-deepseek-r1-qwen-1.5b) |
| **VSA-Llama-3.2-1B** | `meta-llama/Llama-3.2-1B` | 12 VSA + 4 Softmax (`1.31B`) | **1.213x donor PPL**, **97.1% ARC-Easy** (within $2\sigma$ noise), **95.0% PIQA**, **2.43x faster decode @ 64k**, **-74.9% KV RAM** | [`rikkathree/vsa-llama-3.2-1b`](https://huggingface.co/rikkathree/vsa-llama-3.2-1b) |
| **VSA-LM-100M (From Scratch & A4b)** | Trained from scratch on FineWeb-Edu / DCLM | 6 VSA + 2 Softmax (`100M`) | **0.996x donor PPL** (`42.64` vs `42.83`), **+0.054 loss tie** from scratch on 2B DCLM tokens, **47/48** Binder multi-doc recall | [`rikkathree/vsa-lm`](https://huggingface.co/rikkathree/vsa-lm) |

---

## Measured Benchmarks

### 1. `VSA-DeepSeek-R1-Qwen-1.5B` (21 VSA + 7 Softmax)
Converted in 30 seconds and healed via **Pure-KD `VSA-FastDistill`** in 3,663 steps:
* **Held-Out Reasoning & Math Perplexity:** **`9.52`** vs `10.16` Donor (**`0.937x`**)
* **Top-1 Next-Token Agreement (151,936 vocab):** **`79.9%`** (`0.258 nats` KL divergence)
* **MMLU (Zero-Shot):** **`28.0%`** vs `29.0%` Donor (**96.6% retained**)
* **OpenMythos Latent Thinking (`K=0` $\to$ `K=4`):** Answer CE loss drops **`1.218 -> 0.274`**, True-Token Confidence jumps **`52.95% -> 81.65%` (+28.70%)** with **0 bytes** added to Softmax KV cache.
* **Disk-Backed `.vsa` Cartridge Bank (`VSAMemoryBank`):** **`4/4` (100%)** exact secret retrieval with **`0.0` erase residual** and **`0.00 MB`** idle RAM.

### 2. `VSA-Llama-3.2-1B` Speed & Memory Scaling (Fused Triton $O(1)$ Decode Kernel)

| Context Length | Unquantized Llama-3.2-1B Speed | **VSA-Llama-3.2-1B Speed** | **Speedup** | Llama-3.2-1B Cache | **VSA-Llama-3.2-1B Cache** | **RAM Saved** |
|---:|---:|---:|---:|---:|---:|---:|
| **512** | `148.4 tok/s` | **`149.2 tok/s`** | `1.01x` | `16.00 MB` | **`7.00 MB`** | **56.2%** |
| **8,192** | `148.9 tok/s` | **`148.8 tok/s`** | `1.00x` | `256.00 MB` | **`67.00 MB`** | **73.8%** |
| **16,384** | `138.8 tok/s` | **`149.2 tok/s`** | **`1.08x`** | `512.00 MB` | **`131.00 MB`** | **74.4%** |
| **32,768** | `85.5 tok/s` | **`149.7 tok/s`** | **`1.75x`** | `1024.00 MB` | **`259.00 MB`** | **74.7%** |
| **65,536** | `51.4 tok/s` | **`124.7 tok/s`** | **`2.43x`** | `2048.00 MB` | **`515.00 MB`** | **74.9%** |

---

## Installation

```bash
git clone https://github.com/Rikka3/vsa-lm.git
cd vsa-lm
pip install -r requirements.txt
```

---

## 1. Training a VSA-LM From Scratch (Auto-Hardware-Fitting)

`vsa_lm/train.py` includes `auto_fit_hardware()`, which inspects your GPU VRAM (from an 8 GB consumer GPU or 15 GB Kaggle T4 up to a 96 GB Blackwell) and automatically configures `micro_batch_size`, `grad_accum`, per-block activation checkpointing, `AdamW8bit`, and TensorCore `bfloat16`/`float16` precision while preserving the exact mathematical batch size:

```bash
# Train a 6-VSA + 2-Softmax hybrid language model from scratch:
python -m vsa_lm train --run vsa --max-len 512

# Run the Triton kernel, gradient, recurrent decoder, and algebraic erasure parity suite:
python -m vsa_lm test
```

---

## 2. Converting & Healing Any Pretrained Model (Llama, Qwen, DeepSeek, Mistral, SmolLM)

Convert any Hugging Face causal LM (of any size: `0.5B`, `1B`, `1.5B`, `3B`, `7B`, `8B`) into a 75%-VSA / 25%-Softmax hybrid with **GQA Head Un-Tying (`k_head_mix`, `v_head_mix`)**, **8-Tap Causal Local Convolution**, **Positive Feature Sharpening**, and **Multi-Scale Decay Horizons ($\gamma \in [0.50, 0.992]$)**:

```bash
# Step 1: Surgically convert a donor model into a VSA hybrid checkpoint (~20 seconds):
python -m vsa_lm convert \
    --donor deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --out deepseek_vsa_init.pth \
    --hybrid-every 4

# Step 2: Heal with Pure-Teacher VSA-FastDistill (~14-20 GiB VRAM default; add --highspeed on 60GB+ GPUs):
python -m vsa_lm heal \
    --ckpt deepseek_vsa_init.pth \
    --donor deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --out-prefix deepseek_vsa \
    --tokens 60000000 \
    --lr 1.5e-4 \
    --kd 1.0

# Inspect learned decay horizons (gamma) and position code profiles on CPU:
python -m vsa_lm scan --ckpt deepseek_vsa_latest.pth
```

---

## 3. Infinite Context via Disk-Backed `.vsa` Neural Cartridges (`VSAMemoryBank`)

Store thousands of documents or past chat sessions on your SSD/hard drive as `.vsa` state cartridges with **0.00 MB of idle RAM**, route to the right cartridge in milliseconds, answer questions, and algebraically erase the state:

```python
import torch
from transformers import AutoTokenizer
from vsa_lm.converter import UniversalVSALM
from vsa_lm.memory_bank import VSAMemoryBank

ck = torch.load("vsa_deepseek_r1_qwen_1.5b.pth", map_location="cuda", weights_only=False)
model = UniversalVSALM(**ck["config"]).cuda().to(torch.bfloat16).eval()
model.load_state_dict(ck["model"], strict=True)
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

bank = VSAMemoryBank(model, tokenizer, bank_dir="./my_vsa_bank")

# 1. Encode documents to disk as .vsa state cartridges (0 MB kept in GPU RAM):
bank.memorize_to_disk("orion", "The launch authorization code for Project Orion is TITAN-OMEGA-942.")
bank.memorize_to_disk("hemlock", "The antidote synthesis temperature for Protocol Hemlock is 347 Kelvin.")

# 2. Query across all disk cartridges (routes in <20ms, mounts state, answers, and algebraically erases):
res = bank.ask("What is the launch authorization code for Project Orion?")
print(res["cartridge"], "->", res["answer"], "| Erase residual:", res["erase_residual"])

# 3. Permanently unlearn/delete a cartridge from disk:
bank.forget_from_disk("orion")
```

---

## 4. Connecting to Coding Agents (`opencode`, `Aider`, `Continue.dev`, `Cline`) & Running on CPU-Only Laptops

VSA-LM includes a built-in **OpenAI-Compatible Local API Server** (`http://localhost:8000/v1/chat/completions`). On CPU-only laptops (e.g., 8 GB RAM with no GPU), it automatically applies **Dynamic INT8 Weight Quantization** to cut RAM usage in half and accelerate CPU matrix multiplications:

```bash
# Start the local OpenAI-compatible server (works on CUDA GPUs or CPU-only laptops):
python -m vsa_lm serve \
    --ckpt vsa_deepseek_r1_qwen_1.5b.pth \
    --donor deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --port 8000
```

Then point **`opencode`**, **Aider**, or **Continue.dev** to your local server:

```bash
# Example with Aider / OpenCode environment variables:
export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="local-vsa"
opencode --model openai/vsa-lm
```

---

## Repository Structure

* `vsa_lm/kernels.py` — Custom Triton kernels: TensorCore chunkwise forward scan, exact adjoint backward (including $\partial \mathcal{L} / \partial \gamma$), and the 150 tok/s Fused $O(1)$ Recurrent Decode step kernel.
* `vsa_lm/functional.py` — Autograd wrapper (`VSAChunkFunction`) and dense/recurrent fp32 reference oracles.
* `vsa_lm/model.py` & `vsa_lm/decode.py` — From-scratch `VSALanguageModel` and $O(1)$ `VSARecurrentDecoder` with `memorize()` and `forget()`.
* `vsa_lm/converter.py` & `vsa_lm/surgery.py` — Universal donor converter (`UniversalVSALM`) and Low-VRAM `VSA-FastDistill` healer supporting Llama, Qwen, DeepSeek-Distill, Mistral, and SmolLM.
* `vsa_lm/thinking.py` — `OpenMythosEngine` and `LatentThinkController` for zero-KV-growth latent reasoning inside VSA states.
* `vsa_lm/memory_bank.py` — `VSAMemoryBank` for disk-backed `.vsa` infinite context cartridges.
* `vsa_lm/server.py` — OpenAI-compatible `/v1/chat/completions` server with CPU Dynamic INT8 quantization for coding agents (`opencode`, `aider`).
* `vsa_lm/train.py` & `vsa_lm/tests.py` — Hardware-auto-fitting scratch trainer and 8-gate parity test suite.

## License

MIT License — Copyright (c) 2026 Rikka3.
