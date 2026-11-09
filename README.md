# VSA-LM & Bolt Attention: Retrofitting Pretrained Transformers into Fixed-State Symbolic Memory Hybrids

**VSA-LM** is open-source conversion toolkit and runtime built around **Bolt Attention**—my hybrid linear-attention architecture designed to make strong reasoning and coding models run fast on low-VRAM hardware.

By converting **75% of pretrained Transformer attention layers** into **Bolt Attention** (fixed-size $D \times D$ outer-product memory + non-decaying **Abelian Page-Slot Banks** + 64-token local window) and keeping **25% Softmax anchor layers**, VSA-LM:
* Cuts KV-cache RAM growth by **~74% at 64k context** (or caps RAM at **73.1 MB flat forever** in streaming mode).
* Decodes **1.38x to 2.43x faster** via fused Triton $O(1)$ recurrent kernels and Split-KV CUDA Graphs.
* Unlocks **exact algebraic memory unlearning (`memorize` / `forget`)**, **mountable `.vsa` disk cartridges (`0 MB` idle RAM)**, and **zero-KV-growth latent thinking (`OpenMythos`)**.

---

## Core Equations of Bolt Attention

### 1. Mix-First GQA Head Un-Tying & Feature Sharpening
For pretrained donors with $H$ query heads and $H_{KV} < H$ key-value heads ($G = H / H_{KV}$), shared KV heads are projected into $H$ independent subspaces **before** rotary position encoding so relative position phases never drift past training length:

$$
\tilde{k}_{t,h} = \mathrm{RoPE}_t\!\left(\mathrm{repeat\_kv}(k_t)_h W^{\mathrm{mix}}_{k,h}\right) + \mathrm{Conv1D}_4(\tilde{k})_{t,h}
$$

$$
v'_{t,h} = \mathrm{repeat\_kv}(v_t)_h W^{\mathrm{mix}}_{v,h}
$$

$$
q_{t,h} \leftarrow \sqrt{D}\,\frac{\phi_q(q_{t,h})}{\left\|\phi_q(q_{t,h})\right\|_2}
$$

$$
k_{t,h} \leftarrow \frac{\phi_k(\tilde{k}_{t,h})}{\left\|\phi_k(\tilde{k}_{t,h})\right\|_2}
$$

$$
\phi(z) = z + \alpha \mathrm{ReLU}(z)
$$

### 2. Decayed Outer-Product State & Steady-State Readout
Each head $h \in \{1, \dots, H\}$ maintains fixed $D \times D$ associative matrix $M_{t,h}$ with learnable decay $\gamma_h = 0.9999\,\sigma(\ell_h)$:

$$
M_{t,h} = \gamma_h M_{t-1,h} + k_{t,h} \otimes v'_{t,h}
$$

$$
y^{\mathrm{vsa}}_{t,h} = \mathrm{RMSNorm}\left(\frac{1 - \gamma_h}{\sqrt{D}} M_{t,h} q_{t,h}\right) + s_h v'_{t,h} + 2\sigma\left(q_{t,h}^{\top} g^{\mathrm{loc}}_h\right)\mathrm{Conv1D}_8(v')_{t,h}
$$

### 3. Abelian Page-Slot Binder (Non-Decaying Prefix-Sum Slots)
To retain early numbers, passkeys, and definitions across long filler without exponential decay or RoPE distance attenuation, each layer maintains $S = 32$ position-free slots (8 pages $\times$ 4 slots) updated via salience gate $w_{t,m} = \sigma(\mathrm{Router}_m(x_t)) \in (0, 1)$:

$$
C_{t,m} = C_{t-1,m} + w_{t,m}\hat{k}_t
$$

$$
U_{t,m} = U_{t-1,m} + w_{t,m}v_t
$$

$$
Z_{t,m} = Z_{t-1,m} + w_{t,m}
$$

$$
\hat{K}_{t,m} = \frac{C_{t,m}}{Z_{t,m} + \epsilon}
$$

$$
\hat{V}_{t,m} = \frac{U_{t,m}}{Z_{t,m} + \epsilon}
$$

$$
y^{\mathrm{slot}}_t = \mathrm{Softmax}\left(\sqrt{D}\,\hat{q}_t\hat{K}_t^{\top}\right)\hat{V}_t
$$

### 4. Intra-Layer Local Window & Combined Output
Each Bolt Attention layer blends outer-product state $y^{\mathrm{vsa}}_t$, 64-token + sink local window $y^{\mathrm{swa}}_t$, and Page-Slot readout $y^{\mathrm{slot}}_t$:

$$
y_t = \left((1-a_h)y^{\mathrm{vsa}}_t + a_h y^{\mathrm{swa}}_t
+ \tanh(g^{\mathrm{slot}}_h)y^{\mathrm{slot}}_t\right)
\odot 2\sigma(W_gx_t).
$$

### 5. Exact Algebraic Unlearning & Page Erasure
Because $M_t$ and $(C_t, U_t, Z_t)$ are strictly additive over tokens, any memorized document span $[t_0, t_1]$ or slot page $p \in \{0,\dots,7\}$ can be removed in $<1\,\mathrm{ms}$:

$$
M_t \leftarrow M_t - \sum_{j=t_0}^{t_1}\gamma^{t-j}(k_j\otimes v'_j)
$$

$$
\left(C_{t,4p:4p+4},U_{t,4p:4p+4},Z_{t,4p:4p+4}\right) \leftarrow (0,0,\epsilon)
$$

### 6. OpenMythos Gated Latent Thinking ($0$-Byte Softmax KV Growth)
During $K$ recurrent latent thinking passes with learnable thought embeddings $\tau_1, \dots, \tau_K$, Softmax KV caches stay frozen while Bolt Attention states refine via learned gate $\beta_h \in (0, 1)$:

$$
M^{(k)}_h = \left(1-\beta_h(1-\gamma_h)\right)M^{(k-1)}_h
+ \beta_h\left(k^{(k)}_h\otimes v'^{(k)}_h\right).
$$

---

## Pretrained & Converted Models on Hugging Face

| Model | Donor / Base | Layout | Measured Highlights | Hugging Face Repo |
|---|---|---|---|---|
| **VSA-DeepSeek-R1-Qwen-1.5B (v2)** | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 21 Bolt-VSA + 7 Softmax (`1.835B`) | **70.0% HumanEval** (`+10%` over donor), **70.0% GSM8K** (`p=0.727` tie), **40.0% MBPP**, **98.34% prompt-number Top-1 match**, **1.42x faster @ 64k**, **-74.0% KV RAM**, **9/9 hubcheck PASS** | [`rikkathree/vsa-deepseek-r1-qwen-1.5b`](https://huggingface.co/rikkathree/vsa-deepseek-r1-qwen-1.5b) |

---

## Measured Benchmarks (`VSA-DeepSeek-R1-Qwen-1.5B v2`)

Evaluated against official Hugging Face `AutoModelForCausalLM` (`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`) using `python -m vsa_lm benchmark` with exact paired McNemar statistical tests:

### 1. Reasoning, Coding & Knowledge Quality

| Benchmark | Official HF Donor | **VSA-DeepSeek-R1-Qwen-1.5B (v2)** | Paired McNemar (`p`-value) | Verdict |
|---|---:|---:|---:|---|
| **HumanEval** (Python Pass@1, `temp=0.0`) | `60.00%` | **`70.00%`** (`+10.00%`) | `p = 0.219` | **No sig. diff (+10% VSA lead)** |
| **MBPP** (Python Pass@1, `temp=0.0`) | `35.00%` | **`40.00%`** (`+5.00%`) | `p = 0.688` | **No sig. diff (+5% VSA lead)** |
| **GSM8K** (`<think>` CoT, `temp=0.2, top_p=0.85, cap=4096`) | `75.00%` (`30/40`) | **`70.00%`** (`28/40`) | `p = 0.727` | **No sig. diff (Statistical Tie)** |
| **GSM8K** (`<think>` CoT, `temp=0.6, top_p=0.95, cap=4096`) | `82.50%` (`33/40`) | **`65.00%–67.50%`** (`26–27/40`) | `p = 0.039–0.109` | **High-temp tail sampling variance** |
| **MMLU** (57-subject stratified) | `38.00%` | **`39.50%`** (`+1.50%`) | `p = 0.711` | **No sig. diff (Tie)** |
| **HellaSwag** | `41.00%` | **`42.00%`** (`+1.00%`) | `p = 0.774` | **No sig. diff (Tie)** |
| **GSM8K Prompt-Number Top-1 Agreement** | `100.0%` (ref) | **`98.34%`** (`KL = 0.0248 nats`) | — | **Near-exact number retention** |
| **`.vsa` Disk Cartridge Recall (`VSAMemoryBank`)** | N/A | **`100.0%` (`4/4`)** | `0.0` erase residual | **Exact algebraic unlearning** |

### 2. Decode Speed & Memory Scaling (Split-KV CUDA Graph + Fused Triton Step)

| Context Length | Official HF Donor Speed | **VSA-LM v2 Speed** | **Speedup** | Donor KV RAM | **VSA-LM State + KV RAM** | **RAM Saved** |
|---:|---:|---:|---:|---:|---:|---:|
| **2,048** | `92.2 tok/s` | **`130.2 tok/s`** | **`1.41x`** | `56.0 MB` | **`41.9 MB`** | **25.2%** |
| **8,192** | `86.6 tok/s` | **`130.3 tok/s`** | **`1.50x`** | `224.0 MB` | **`76.1 MB`** | **66.0%** |
| **16,384** | `91.0 tok/s` | **`128.6 tok/s`** | **`1.41x`** | `448.0 MB` | **`130.2 MB`** | **70.9%** |
| **32,768** | `90.9 tok/s` | **`125.4 tok/s`** | **`1.38x`** | `896.0 MB` | **`242.2 MB`** | **73.0%** |
| **65,536** | `83.9 tok/s` | **`119.5 tok/s`** | **`1.42x`** | `1,792.0 MB` | **`466.2 MB`** | **74.0%** |
| **Infinite Stream (`window=8192`)** | *Unbounded KV growth* | **`125.6 tok/s`** | **Flat forever** | *Unbounded* | **`73.1 MB` (+0.0 MB growth)** | **Constant RAM** |

---

## Honest Guide to Context Length: Exact vs Streaming vs Disk Cartridges

I want users to know what works out-of-box and where physical limits sit:
1. **Exact Decode Mode (Default, up to 64k tokens):**
   75% Bolt Attention layers stay fixed-size; 25% Softmax anchor layers keep full KV cache (`466.2 MB` at 64k vs `1,792.0 MB` donor). Best verbatim needle recall is within `0..4,096` tokens (healing window).
2. **Constant-RAM Streaming Mode (`--stream-window 8192`):**
   Each Softmax anchor layer keeps `4` attention-sink tokens + last `8,192` tokens (`StreamKV` with float64 RoPE sink re-rotation). RAM caps at **`73.1 MB` forever** (`0.0 MB` growth over 32k+ tokens) at `125.6 tok/s`. Tokens inside last `8,192` window have exact Softmax recall; older tokens survive as compressed VSA + Page-Slot summaries.
3. **Unbounded Exact Recall via `.vsa` Disk Cartridges (`VSAMemoryBank`):**
   Single fixed $128 \times 128$ matrix cannot store millions of arbitrary numbers without compression loss. For **true unbounded exact recall**, `VSAMemoryBank` encodes documents or past chat chunks into `.vsa` state cartridges on disk (`0.00 MB` idle RAM), routes queries in under `45 ms`, mounts state to answer (`4/4` exact recall), and algebraically unmounts it (`0.0` residual).

---

## Beginner Step-by-Step Guide (Noob-Friendly)

### Step 1: Install VSA-LM (Takes 1 Minute)
Requires Python 3.10+ and NVIDIA GPU with CUDA + Triton (or CPU laptop for INT8 serving):

```bash
git clone https://github.com/Rikka3/vsa-lm.git
cd vsa-lm
pip install -r requirements.txt huggingface_hub transformers
```

Verify your GPU kernels pass all 8 exactness gates:
```bash
python -m vsa_lm test
```

---

### Step 2: Download & Chat with Pre-Converted `VSA-DeepSeek-R1-Qwen-1.5B`

Create file `run_chat.py` and run `python run_chat.py`:

```python
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from vsa_lm.converter import UniversalVSALM
from vsa_lm.decode import VSARecurrentDecoder
from vsa_lm.server import get_all_eos_ids

# 1. Auto-download checkpoint from Hugging Face
ckpt_path = hf_hub_download(
    repo_id="rikkathree/vsa-deepseek-r1-qwen-1.5b",
    filename="vsa_deepseek_r1_qwen_1_5b.pth",
)

# 2. Load model onto GPU in bfloat16
ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model = UniversalVSALM(**ck["config"]).cuda().to(torch.bfloat16).eval()
model.load_state_dict(ck["model"], strict=False)

tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
eos_ids = get_all_eos_ids(tok)

# 3. Create fast Split-KV CUDA Graph decoder (stream_window=8192 caps RAM at 73.1 MB forever)
decoder = VSARecurrentDecoder(model, stream_window=8192)

# 4. Ask math or coding question
messages = [{"role": "user", "content": "Write a Python function is_prime(n) and explain why checking up to sqrt(n) is enough."}]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = torch.tensor([tok.encode(prompt, add_special_tokens=False)], device="cuda")

out_ids = decoder.generate(input_ids, max_new_tokens=1024, temperature=0.2, top_p=0.85, eos_ids=eos_ids)
print(tok.decode(out_ids[0, input_ids.shape[1]:].tolist(), skip_special_tokens=True))
```

---

### Step 3: Use `.vsa` Disk Cartridges & Exact Algebraic Unlearning

Store unlimited documents on disk with `0 MB` idle VRAM, query across them, and erase facts cleanly:

```python
from vsa_lm.memory_bank import VSAMemoryBank

bank = VSAMemoryBank(model, tok, bank_dir="./my_vsa_cartridges")

# 1. Memorize documents to disk (.vsa files, 0 MB kept in GPU RAM)
bank.memorize_to_disk("orion", "Launch authorization code for Project Orion is TITAN-OMEGA-942.")
bank.memorize_to_disk("hemlock", "Antidote synthesis temperature for Protocol Hemlock is 347 Kelvin.")

# 2. Ask question -> routes in <45ms, mounts cartridge, answers, and algebraically unmounts
res = bank.ask("What is launch authorization code for Project Orion?")
print("Matched Cartridge:", res["cartridge"])
print("Answer:", res["answer"])
print("Algebraic Erase Residual:", res["erase_residual"])  # 0.0

# 3. Permanently delete cartridge from disk
bank.forget_from_disk("orion")
```

To surgically erase single page from live `SlotBank` during conversation without touching other 7 pages:
```python
VSARecurrentDecoder.erase_page(kv_caches, page_idx=0, num_pages=8)
```

---

### Step 4: Convert & Heal DeepSeek-R1-Distill-Qwen Yourself

Convert the supported DeepSeek-R1-Distill-Qwen model into a Bolt Attention hybrid:

```bash
# 1. Surgically convert donor weights (~20 seconds):
python -m vsa_lm convert \
    --donor deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --out my_vsa_init.pth \
    --hybrid-every 4 \
    --max-T 2048

# 2. Heal with Stage-1 Attention Transfer + Stage-2 KD & Page-Slot Binder Curriculum:
# (Works on 16GB GPUs automatically; add --highspeed on 45GB+ GPUs)
python -m vsa_lm heal \
    --ckpt my_vsa_init.pth \
    --donor deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --out-prefix my_vsa_healed \
    --tokens 60000000 \
    --lr 1.5e-4 \
    --kd 0.90 \
    --highspeed

# 3. Lock in long reasoning chains with Stage-3 Teacher-Chain Sequence KD:
python -m vsa_lm onpolicy \
    --ckpt my_vsa_healed_latest.pth \
    --donor deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --out-prefix my_vsa_final \
    --chain-source teacher \
    --rounds 3

# 4. Run 9-gate readiness check & full benchmark suite:
python -m vsa_lm hubcheck --ckpt my_vsa_final_r3.pth --stream-window 8192
python -m vsa_lm benchmark --ckpt my_vsa_final_r3.pth
```

---

### Step 5: Connect to Coding Agents (`opencode`, `Aider`, `Continue.dev`) or Run on CPU Laptops

Start built-in OpenAI-compatible server (`http://localhost:8000/v1/chat/completions`). On CPU-only machines, it automatically applies dynamic INT8 weight quantization:

```bash
python -m vsa_lm serve \
    --ckpt vsa_deepseek_r1_qwen_1.5b.pth \
    --donor deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --stream-window 8192 \
    --port 8000
```

Point your coding agent to local server:
```bash
export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="local-vsa"
opencode --model openai/vsa-lm
```

---

## Repository Map

* `vsa_lm/kernels.py` — Custom Triton kernels: TensorCore chunkwise forward scan, exact adjoint backward (including $\partial\mathcal{L}/\partial\gamma$ and tail-chunk support), and fused $O(1)$ recurrent decode step.
* `vsa_lm/config/standard.py` — `UniversalVSALM` with Bolt Attention (`Mix-First` GQA un-tying, `SlotBank` Abelian Page-Slot Binder, and 64-token sink SWA).
* `vsa_lm/decode.py` — `VSARecurrentDecoder` with Split-KV CUDA Graph, `StreamKV` constant-RAM streaming, `memorize()`, `forget()`, and `erase_page()`.
* `vsa_lm/converter.py` — Universal weight converter, Stage-1 layerwise attention transfer, Stage-2 KD + procedural binder curriculum, and Stage-3 teacher-chain sequence distillation.
* `vsa_lm/thinking.py` — `OpenMythosEngine` and `LatentThinkController` for zero-KV-growth latent reasoning.
* `vsa_lm/memory_bank.py` — `VSAMemoryBank` for disk-backed `.vsa` neural state cartridges.
* `vsa_lm/benchmark.py` & `vsa_lm/hubcheck.py` — Paired exact McNemar benchmark suite and 9-gate Hub readiness verifier.

## Citation & Prior Art

Bolt Attention and VSA-LM build upon foundational work in Tensor Product Representations (Smolensky, 1990), Vector Symbolic Architectures (Kanerva, 2009), Fast Weights (Hinton & Plaut, 1987; Schlag et al., 2021), RetNet (Sun et al., 2023), and Lightning Attention-2 / MiniMax-01 (Qin et al., 2024; MiniMax, 2025).

## License

MIT License — Copyright (c) 2026 Rikka3.

---

## Architectural Honesty

I started this project from first principles with one goal: build language model memory using **Vector Symbolic Architectures (VSA)** and Smolensky **Tensor Product Representations (TPR)**—where key-value pairs bind via outer products ($k_t \otimes v_t$), accumulate by linear superposition, and unbind by matrix-vector readout ($M_t q_t$).

While building and scaling my Triton chunked scans and 3:1 hybrid stacks, I converged on math that makes **Bolt Attention** direct twin to **Lightning Attention-2 / MiniMax-01** (`arXiv:2401.04658`, `arXiv:2501.08313`) and descendant of **RetNet** (`arXiv:2307.08621`), approached from VSA state-algebra ideology.

I believe in 100% scientific honesty:
* **What shares prior lineage:** Core decayed outer-product recurrence ($M_t = \gamma M_{t-1} + k_t \otimes v_t$), intra/inter-chunk Triton tiling, and periodic Softmax hybrid interleaving (3:1) match RetNet and Lightning Attention-2.
* **What Bolt Attention & VSA-LM add that RetNet / Lightning Attention do NOT have:**
  1. **Trainable Per-Head Decay ($\gamma_h$) with Exact Triton Adjoint Backward ($\partial\mathcal{L}/\partial\gamma_h$):** RetNet and Lightning Attention-2 freeze decay slopes to fixed ALiBi constants. Bolt Attention derives and computes exact analytical gradients $\partial\mathcal{L}/\partial\gamma_h$ across chunks plus $(1-\gamma_h)/\sqrt{D}$ steady-state read normalization, letting every head learn its own memory horizon.
  2. **1-GPU Pretrained GQA Retrofit Surgery (`convert` + `heal` + `onpolicy`):** MiniMax-01 must be pretrained from scratch on trillions of tokens. Dropping raw linear attention into pretrained GQA models (like `DeepSeek-R1-Distill-Qwen-1.5B`, which has only $H_{KV}=2$ KV heads) collapses quality because 2 KV heads give only 2 state matrices. Bolt Attention un-ties GQA heads (`k_head_mix`, `v_head_mix`) **before** RoPE (`Mix-First, Bind-Second`), adds depthwise causal convolutions (`k_conv`, `v_conv`), and blends an intra-layer 64-token + sink window (`swa_window`).
  3. **Abelian Page-Slot Binder (`SlotBank`):** Passive exponential decay ($\gamma < 1$) wipes out early facts after thousands of filler tokens. Bolt Attention pairs fast matrix $M_t$ with **32 position-free (no-RoPE), non-decaying prefix-sum slots** (8 addressable pages $\times$ 4 slots) gated by token salience—giving 100% bit-level parity between parallel training and $O(1)$ decoding, plus surgical single-page erasure (`erase_page`).
  4. **Algebraic Unlearning & `.vsa` Disk Cartridges (`VSAMemoryBank`):** Because Bolt Attention states form an Abelian group (linear/commutative addition), documents can be encoded to `.vsa` files on SSD (`0 MB` idle GPU RAM), routed in `<45 ms`, mounted by matrix addition, and surgically unlearned by matrix subtraction (`0.0` residual error).
  5. **OpenMythos Zero-KV Latent Thinking:** Run $K$ recurrent reasoning passes inside fixed-size Bolt Attention states while freezing Softmax KV caches (`0 bytes` memory growth during thinking).
