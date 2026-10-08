# Results

Will try to post direct CSV files soon but for now...

## Quality comparison

Both models: 100M params, same data (500M tokens FineWeb-Edu), same training steps, same hardware (Kaggle T4). Evaluated on 5 slices of held-out text.

| Text slice | Transformer | VSA | Gap |
|---|---|---|---|
| spot 1 | 3.776 | 3.853 | +0.077 |
| spot 2 | 4.358 | 4.430 | +0.072 |
| spot 3 | 3.683 | 3.721 | +0.038 |
| spot 4 | 4.167 | 4.232 | +0.065 |
| spot 5 | 3.736 | 3.828 | +0.092 |
| average | 3.944 | 4.013 | +0.069 |
| perplexity | 51.6 | 55.3 | |

Tie. The gap is 0.069 which is small enough that it's within the noise range for this setup. One caveat, the VSA model saw roughly one-third less unique training data due to a resume bug (fixed now), so this number is conservative.

## Memory usage (the actual point)

| Context length | VSA total state | Transformer total state |
|---|---|---|
| 512 tokens | 4.3 MB | 12.6 MB |
| 8,192 tokens | 51 MB | 201 MB |
| 32,768 tokens | 201 MB | 805 MB |

The VSA state grows at 6 KB per token vs 24.6 KB per token. The 6 memory layers stay flat at 1.2 MB forever; only the 2 softmax layers grow.

## Speed

| | VSA | Transformer |
|---|---|---|
| Training | 1.71 s/step | 1.28 s/step |
| Training VRAM | 2.2 GB | 1.2 GB |

The VSA trains about 25% slower. This is the cost of the custom Triton kernels vs PyTorch's optimized softmax. At longer contexts this reverses — VSA attention scales linearly, softmax scales quadratically — but we only trained at 512.

## Memory editing demo

Trained model, greedy decoding:

Question: "The secret code for the vault is"

| Condition | Answer |
|---|---|
| Doc in memory | BLUE-FALCON-77. The secret |
| Doc erased | - The first floor of the building is the first |
| No doc ever | The secret code for the vault is the same as... |

The exact string is recalled when the doc is in memory. After erasure, it's gone. A second document (about Project Atlas) stayed perfectly intact after the first was erased — the erasure is surgical.

## Recall capacity

A secret phrase buried in filler text:

| Document length | Recall |
|---|---|
| 100 tokens | exact |
| 200 tokens | exact |
| 300 tokens | exact |
| 400 tokens | exact |

## Training notes

- Both models plateaued around step 25,000 due to being out of data.
- VSA decay rates (γ): early layers learned longer memory (0.876 → 0.926), late layers kept short-horizon heads (min 0.777)

Results v2: Phase 2 (2B tokens) + binder gates — supersedes the Run 1 post below

Run 1 (500M tokens, +0.069) was the pilot. This is the real result.

Phase 2 — quality (pre-registered tie rule: 5-spot gap ≤ 0.10):

Two arms, same data and recipe (DCLM 2B tokens, batch 32×512, lr 3e-4, cosine), ~100M params each. vsa2 = 6 VSA + 2 softmax layers, base2 = 8 softmax.
	
vsa2
	
base2
5-spot val loss	3.662	3.608
val ppl	38.95	36.89
steps	121,948	121,949
 
 

Gap +0.054 → tie. Verified twice from archived checkpoints, byte-identical logs.

Serving, measured (fp16, Cell 2 bench):
context
	
vsa2 state
	
base2 state
512	4.33 MB	12.61 MB
32k	202 MB	805 MB
 
 

Decode: ~8 ms/tok flat vs 10 → 26.9 ms/tok at 32k (3.3× slower). Peak GPU 1.33 vs 1.93 GB. vsa2 = 1.18 MB flat VSA state + 6.1 KB/token from its 2 softmax layers; base2 = 24.6 KB/token.

The trade: vsa2 trains ~40% slower per step (1.78 vs 1.27 s/it on T4). No quality claims past 512 context — pos_code wraps at max_T, untrained there. The memory/speed numbers past 512 still hold; they're mechanism-level.

Data disclosure: sessions 1–2 had a data-position bug that repeated tokens — ~1.6B/~1.5B unique of 2.0B presentations (vsa2/base2), fixed from session 3. The confound favors vsa2; estimated effect 0.005–0.01, smaller than the tie window. Disclosed rather than re-run.

Binder (adaptive layer), attempt 4, final:

31M model: 4 VSA + 2 softmax + SlotLayer — 8 pages × 64 slots, zero-init (exact no-op at birth). Training shows questions only; memos go to pages, never into context. Everything procedural (topics, names, codes) so nothing is memorizable as a list.

Pre-registered gates, all passed:

     G1 recall: 94% at 4 docs, 88% at 8 vs 6% for the in-context arm at 8 docs (8 docs + 8 Q&As overflow its 512 window — the fixed-state point)
     G2 surgical erase: target gone, co-stored doc intact
     G3 empty-binder CE gap: 5.66 nats

Independent re-verification (separate evaluator code, fresh seeds): 20/20 fresh-secret recall incl. 7/7 random codes; empty binder 0/20; wrong-topic page 0/20 — and asked about X while holding Y's page it answers Y's secret 9/20, so reading is content-driven. Capacity 47/48 across 8 pages. Erase 8/8. Erase-then-rewrite 7/7. CE gap reproduced at 5.45.

Honest limits, measured: 1/8 recall with unseen memo filler, 0/8 with paraphrased questions — this checkpoint reads inside its trained template envelope. Contamination: 0/20 exact (topic, secret) pairs in the training pool; topics themselves appear with different secrets. Template brittleness is a data fix (varied filler + question forms), not architecture — Phase 3.

For the record: first binder run overfit 16 fixed topics (eval 0–3%). Final run crashed at step 2500 on a padding bug in the live-eval path; the watchdog killed both arms as designed, and the rerun completed the same pre-registered design.

Artifacts: Phase 2 checkpoints/logs: rikkathree/vsa-lm-phase2-final · Binder arms + full logs + verification outputs: rikkathree/vsa-lm-binder

Known gaps, not yet run: γ fixed-vs-learned, hybrid ratio (6:2 / 4:4 / 8:0), scaling sweep past 100M, Mamba/RetNet baselines, quality past 512 context. On the list.
- Previous run with smaller memory heads (32 instead of 64) collapsed — γ dropped to 0.73, meaning layers gave up on memory. The 64-dim fix resolved this.
