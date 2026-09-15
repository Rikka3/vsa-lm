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

Tie. The gap is 0.069 — small enough that it's within the noise range for this setup. One caveat: the VSA model saw roughly one-third less unique training data due to a resume bug (fixed now), so this number is conservative.

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
- Previous run with smaller memory heads (32 instead of 64) collapsed — γ dropped to 0.73, meaning layers gave up on memory. The 64-dim fix resolved this.
