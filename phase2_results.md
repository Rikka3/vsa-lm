# Results — v2 (Phase 2 + binder)

Supersedes the Run 1 results. Run 1 was the pilot: 500M FineWeb-Edu tokens, gap +0.069, plus a resume bug that fed the VSA arm ~1/3 less unique data. Everything below is Phase 2 (2B DCLM tokens) and the final binder run. Full logs and CSVs are on the HuggingFace repos linked at the bottom.

## Phase 2: quality

Pre-registered tie rule: 5-spot average gap ≤ 0.10 counts as a tie.

Both arms ~100M params, same data (DCLM 2B tokens), same recipe (batch 32×512, lr 3e-4, warmup + cosine, AdamW8bit), same hardware (Kaggle T4, one GPU per arm). vsa2 = 6 VSA + 2 softmax layers. base2 = 8 softmax.

| | vsa2 | base2 |
|---|---|---|
| 5-spot val loss | 3.662 | 3.608 |
| val ppl | 38.95 | 36.89 |
| steps | 121,948 | 121,949 |

Gap +0.054. Tie. Re-verified from archived checkpoints; the re-run produced byte-identical logs.

## Phase 2: serving (measured, fp16)

| Context | vsa2 total state | base2 total state |
|---|---|---|
| 512 | 4.33 MB | 12.61 MB |
| 32k | 202 MB | 805 MB |

vsa2 = 1.18 MB flat VSA state + 6.1 KB/token from its 2 softmax layers. base2 = 24.6 KB/token, all softmax.

Decode speed: vsa2 ~8 ms/token flat. base2 10 → 26.9 ms/token at 32k — 3.3× slower. Peak GPU during decode: 1.33 vs 1.93 GB.

No quality claims past 512 context: pos_code wraps at max_T and was never trained past it. The memory and speed numbers past 512 still hold — they're mechanism-level, not quality-level.

## Data disclosure

Sessions 1–2 had a data-position bug: resume re-fed already-seen batches. Roughly 1.6B (vsa2) / 1.5B (base2) unique tokens out of 2.0B presentations. Fixed from session 3. The confound favors vsa2; estimated effect 0.005–0.01 loss — smaller than the tie window.

## Binder: adaptive memory layer — final run

What it is: a SlotLayer on a 31M hybrid model (4 VSA + 2 softmax, dim 384). 8 pages × 64 slots. A memo is encoded by the model itself; its hidden states become one page. During generation the model reads pages through gated attention. Pages are exactly erasable: zero one page, the others are untouched. The layer is zero-init, so at birth it's an exact no-op.

Training: questions only in context; memos go to pages and never appear in the text. Everything procedural — 26 adjectives × 30 nouns for topics, syllable-generated names, random alphanumeric codes — so no answer list exists to memorize. 25,000 steps per arm, ~5h each on one T4. Live eval on fresh topics every 2,500 steps: 5/8 at step 2,500, 8/8 from step 5,000 on (one 7/8 at 15,000), final 8/8.

Two arms: C = binder (memos in pages). A = same size, no slot layer, memos pasted into context — the normal way.

Gates, pre-registered:

**G1 — recall by doc count** (4 trials per cell, fresh procedural docs):

| docs | A (in context) | C (binder) |
|---|---|---|
| 1 | 4/4 | 4/4 |
| 2 | 8/8 | 8/8 |
| 4 | 16/16 | 15/16 |
| 8 | 2/32 | 28/32 |

Pass bar: C ≥ 80% at 4 docs and C ≥ A at 8 docs. A's collapse at 8 docs is mechanical — 8 docs + 8 Q&As overflow its 512-token window. That's the fixed-state point: the binder's cost doesn't grow with what it holds.

**G2 — surgical erase:** 2 docs, 2 pages. Before: both answers exact. Zero page 1: doc-1's secret gone, doc-2's answer still exact. Pass. (One trial at grading; the independent run below did 8.)

**G3 — channel check:** answer CE with correct pages 0.112, with empty pages 5.771. Gap 5.659 nats — the answers come from the pages, not the question text. A questions-only CE: 7.945. Pass bar was > 3.0.

## Binder: independent re-verification

Separate evaluator code, fresh seeds, and controls the grading never ran:

| Test | Result |
|---|---|
| Fresh-secret recall, single doc | 20/20 (codes 7/7, dates 7/7, names 6/6) |
| Empty binder, same questions | 0/20 |
| Wrong-topic page | true secret 0/20; answered the secret actually in the page 9/20 |
| 8 docs in 8 pages | 47/48 |
| Surgical erase, 8 trials | 8/8 |
| Erase then rewrite the same page | 7/7 |
| Channel CE gap, 32 fresh sequences | 5.45 nats |

The wrong-topic row is the interesting one: asked about topic X while holding topic Y's page, it answers Y's secret 9 times out of 20. Reading is content-driven — it extracts what's on the page, not what the question asks for.

Contamination check: 0/20 exact (topic, secret) pairs appeared in the training pool. But 20/20 topics appeared with different secrets — the demonstrated generalization is over secrets, fillers, and doc IDs, not over topics.

## Binder: limits, measured

- Unseen filler sentences in the memo: 1/8 recall.
- Paraphrased questions ("What is the access code for…?"): 0/8.

This checkpoint reads inside its trained template envelope. The failure is in the data — one filler set, one question form — not in the mechanism. Fix is procedural filler and multiple question forms; Phase 3 work, no architecture change.

## State-level editing on the Phase 2 model

For the record, before the binder existed: post-hoc edits to the VSA recurrent state of the 100M checkpoint. Single-doc recall 4/6 exact — the failures were string-specific corruptions, the same one at two lengths. Multi-doc capacity ~1–2 docs. Exact state erase removes the target but damages co-memorized docs: write-time entanglement plus the softmax layers' KV cache, which a state edit can't touch. Architectural at the state level — which is why the binder uses addressable pages, where erase is exact by construction.

## Artifacts

- Phase 2 checkpoints, logs, RESULTS: https://huggingface.co/rikkathree/vsa-lm-phase2-final
- Binder arms (C_slot.pth / A_slot.pth, step 25,000), full training log, slot_arch.py (exact training + grading code), grading and verification outputs, training pools: https://huggingface.co/rikkathree/vsa-lm-binder
- Code: this repo — `vsa_lm` package, `binder/slot_arch.py`.

## not yet run

- γ fixed vs learned.
- Hybrid ratio: 6:2 vs 4:4 vs 8:0.
- Scaling sweep: 30M / 100M / 180M iso-budget — is the tie 100M-specific?
- Mamba / RetNet baselines on the same data and budget.
- Quality past 512 context
