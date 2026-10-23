# A4b: I got a VSA hybrid back to donor-level perplexity (0.996x)

I wanted to find out whether I could replace most of a language model's attention layers with VSA, then recover the lost quality without needing another full training run from scratch.

Model Page : https://huggingface.co/rikkathree/vsa-lm/tree/main

For this experiment, I started with my trained 100M-parameter softmax language model, which had seen 500M FineWeb-Edu tokens. I transplanted its weights into a hybrid model where **6 of the 8 attention layers use VSA**. VSA uses a fixed-size memory state with the update `M <- gamma*M + k⊗v`, so that state does not grow with the context length.

After one standard training budget of recovery training, the hybrid reached **0.996x the donor's perplexity** on my 64k-token evaluation. That's effectively at donor level within the noise of this evaluation, while six attention layers use fixed-state memory.

This is a big improvement over my previous A4 run, which stalled at around 1.95x the donor's perplexity. For A4b, I changed two things: I kept the donor's knowledge anchored in the loss and restarted learning at a gentler learning rate. That was enough to avoid the earlier regression.

There is an important caveat. The healed model did **not** pass the pre-registered reasoning certification: it scored **0/5 gates**. The results suggest that the basic thinking channel is still active, but the original test became too easy for the model to show a gain over doing no thinking at all. At the same time, keyed retrieval and binding are still around chance because I had not trained the model to do those tasks. My next step is to teach those behaviours directly in Phase B.

## Main results

All numbers below were measured in this run using the same token budget and the same 64k-token evaluation protocol.

| Model (100M parameters, similar token budget) | Validation perplexity (64k) | Relative to donor |
|---|---:|---:|
| Donor softmax model (8 of 8 softmax layers) | 42.83 | 1.00x |
| A2 snapshot, 6/2 injected model at 3.3% of the budget, KD-anchored | 56.42 | 1.32x |
| **A4b, full recovery training with the donor anchor** | **42.64** | **0.996x** |

Before running the experiment, I set the success threshold from the A3 rule at **1.5x donor perplexity or better**, which works out to 64.25. A4b passed that threshold comfortably.

The main takeaways from the run are:

- **Recovery target passed.** Keeping the donor anchor in the loss appears to have prevented the plateau I saw in A4.
- **The full training budget helped.** A4b improved from 56.42 to 42.64 perplexity rather than regressing from its starting checkpoint.
- **Near-donor recovery was possible.** Weight transplantation followed by anchored recovery training brought the hybrid back to roughly the donor's evaluation performance.

### Cross-check with the 4k-token evaluation

I also checked the models using `train.py`'s official single-window evaluation protocol. This test is noisier because it uses a 4k-token window.

| Model | Spot-A perplexity (4k) |
|---|---:|
| Donor softmax model | 88.07 |
| Scratch-trained 7/1 hybrid (vsa8 arm, full budget) | 106.99 |
| A4 recovery without the donor anchor (full budget) | 83.37 |
| **A4b recovery with the donor anchor (full budget)** | **87.73** |

A4b scored 19.26 perplexity points better than the scratch hybrid on this evaluation, although that comparison is not perfectly matched. The scratch reference is the 7/1 hybrid, which is the only scratch hybrid I trained for the full budget. I did not train a matching 6/2 scratch arm, so this difference should be treated as an upper bound on the improvement in a like-for-like comparison.

The donor itself varied between 39.6 and 75.8 perplexity across five spots in the validation region. That is why I use the contiguous 64k-token result for the main decision. On that evaluation, A4b was effectively at donor level.

## What I changed between A4 and A4b

A4b came directly from the issues I found while reviewing A4.

In A4, I resumed the 56.42-perplexity snapshot with `train.py` unchanged. The run used pure cross-entropy, and the position-correct cosine learning-rate schedule restarted at the donor's peak learning rate of `3e-4`. Perplexity spiked to 134.2 at step 2,000 and eventually plateaued at 83.37, which was worse than the checkpoint I started with.

My post-mortem pointed to two likely causes. First, I had removed the donor anchor even though the checkpoint had been built with knowledge distillation anchoring. Second, restarting at the peak learning rate was too aggressive.

For A4b, I changed those two things and kept the rest of the setup the same:

- I used `0.7 * CE + 0.3 * KL(donor)` throughout the run.
- I used a gentler learning-rate restart, with a `1e-4` peak, 300-step warmup, and cosine decay to a 5% floor.
- I kept the same data stream, seed, skip behaviour, optimizer, gradient clipping setup, and overall training budget.

Training ran from step 1,000 to step 30,395 and used about 498M tokens.

## Recovery curve

| Step | Validation perplexity (64k) | KL to donor | Top-1 agreement |
|---:|---:|---:|---:|
| 1,000, starting A2 snapshot | 56.42 | 0.308 nats | 66.4% |
| 2,000 | 54.64 | 0.290 | 67.9% |
| 10,000 | 47.84 | 0.227 | 73.1% |
| 20,000 | 43.96 | 0.175 | 75.3% |
| 30,395, final | **42.64** | **0.146** | **78.0%** |

Unlike A4, A4b did not spike at step 2,000. It was already below its starting perplexity and continued to improve throughout the run. KL to the donor fell by more than half. The run took 2.46 hours on one GPU.

## What this tells me about the architecture

A4 and A4b started from the same checkpoint and used the same data stream, optimizer, and training budget. The two intentional differences were whether I kept the donor anchor in the loss and how I restarted the learning rate.

A4 started at 1.32x donor perplexity and finished at 1.95x. A4b started from that same 1.32x checkpoint and finished at 0.996x. That is strong evidence that the earlier plateau came from the training setup, not necessarily from a hard limit in the fixed-state layers.

For my planned 1B experiment, the current recipe is straightforward: transplant the weights, keep the donor anchored during recovery training, and use one standard training budget to heal the model. I do not need to add the deeper A5 injection machinery unless recovery at 1B runs into problems.

## Does thinking survive the full recovery run?

This part of the result is more mixed, so I want to be clear about what passed and what did not.

On the A2 snapshot, my earlier `cert3` run marked the thinking mechanism as REAL on all 3 pre-registered gates. It tested hidden-mode decay, mean, and newline thinking at K=16, using exact McNemar tests and a Bonferroni-adjusted significance threshold of `0.05/3`. Accuracy rose from about 65% with K=0 to 85% to 91% with thinking, with `p <= 3e-5`.

A4b came from the same model lineage but received about 30 times more anchored training. I reran the five-gate certification on the recovered model with a fresh seed, the guard enabled, 100 trials per task, and a significance threshold of `0.05/5 = 0.01`.

The certification result was **0/5 gates REAL**. The discovery-run effect did not replicate under the pre-registered test, so the next step is pause-token training. Here is how I interpret the result.

### 1. The original magic-task tests hit a ceiling

The gate asks whether thinking performs better than K=0, meaning no thinking pass. But after recovery training, the model scored **100% at K=0** on the magic task. Since the model was already solving the task in a single forward pass, the accuracy test had no room left to show an improvement.

| Magic task, hidden mode | A2 snapshot at 56.4 perplexity (`cert3`) | A4b at 42.6 perplexity |
|---|---:|---:|
| K=0, no thinking | About 65% | **100%** |
| K=16, decay | 87% | 100% |
| K=16, mean | 91% | 100% |
| K=16, newline | 85% | 76% |

The model changed more than the test did. The additional anchored training taught it to solve the test's in-context pattern in one pass. That makes this particular accuracy gate too easy to certify a further benefit from thinking, although it can still catch cases where thinking makes performance worse.

### 2. The thinking channel still affects the result

One contrast did remain clear. With end-of-thought (`eot`) tokens, thinking at K=16 dropped accuracy from 100% to 39%. With decay or mean thinking tokens, accuracy stayed at 100% across K values. The end-of-thought contrast was significant at `p < 0.0001`, with 61 discordant pairs in one direction and none in the other.

In hidden mode, the evaluation snapshots and restores the softmax KV cache around the thinking passes. That means the readout change must flow through the VSA state. In other words, the channel is writing to state, and the token used for thinking affects what gets written.

The relative quality of the thinking tokens also stayed consistent with `cert3`: decay and mean performed best, newline was weaker, and `eot` was worst.

There is also an interesting result that was not pre-registered. Even though accuracy was saturated, the true-token probability under decay thinking rose as K increased, from 0.704 at K=0 to 0.927 at K=8. So the improvement may now show up in confidence rather than accuracy. The label-free adaptive guard also continued to distinguish the token types: with `eot` tokens it picked K=0 in 98 out of 100 trials, achieving 98% accuracy; with decay tokens it picked around K=6 to 10 and achieved 100% accuracy.

### 3. More recovery training did not teach keyed retrieval or binding

The certification also tested whether the recovered model could do keyed retrieval and binding. These results stayed around chance, even after the full recovery run.

| Task | K=0 | K=16, hidden mode with decay | Chance level | McNemar p-value vs K=0 |
|---|---:|---:|---:|---:|
| Keyed retrieval | 12% | 19% | 12.5% | 0.105 |
| Binding | 16% | 16% | 16.7% | 0.60 |

This suggests that reaching donor-level perplexity is not enough to make the model good at tasks it was never trained on. It also weakens the idea that the earlier chance-level keyed and binding results were only caused by an undertrained student. The next step is to train those behaviours directly, including when the model should think and what it should do during those passes. That is the goal of Phase B, starting with v1 pause-token training.

## A lesson for the next certification tests

A test that requires thinking to beat K=0 needs to leave some headroom for the model to improve. For the next version of the battery, I should use harder tasks or pre-register confidence-based measurements alongside accuracy. Confidence still had room to move here, rising from 0.704 to 0.927 under decay thinking.

The main lesson is simple: once a probe saturates, it can still reveal regressions, but it cannot certify further gains in accuracy.

## Limitations and artifacts

- This is one seed, one donor, and one data mix: my 100M base model trained on 500M FineWeb-Edu tokens. I have not established the result across multiple seeds. The claim is that A4b is within evaluation noise of the donor, not that the two models are a measured statistical tie.
- Recovery used about 97% of one standard training budget, or 498M tokens. By "cheap," I mean no more training than a from-scratch run needs, not free training.
- I skipped 9 of 29,395 steps because the gradient norm was non-finite. The usual skip logic handled them.
- I used a fresh certification seed, and locked the gates before the A4b training run began. The certification artifacts are in Hugging Face at `rikkathree/vsa-lm`, in `a4b_cert_results` (JSON report and console transcript).
- The scratch-hybrid result of 106.99 on spot-A is the **7/1 (vsa8) arm**, which is the only scratch hybrid I trained for the full budget. I did not run a like-for-like 6/2 scratch experiment, so the 19.26-point gap is an upper bound rather than a matched comparison.
https://huggingface.co/rikkathree/vsa-lm/tree/main
- I still need the final validation perplexity for the all-VSA (`vsa0`) arm.
- The A4b artifacts are in Hugging Face at `rikkathree/vsa-lm`, in `a4b_heal_results`. They include the `a4b_latest.pth` checkpoint, baseline JSON, validation and log CSVs, summary, and token cache.

