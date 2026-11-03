# Experimental-design review, 2026-10-20

Scope: existing E1-E4 implementation and the proposed additional ablations.
This review changes neither completed results nor the frozen E4 protocol.
No additional training or hypothesis test is launched by this review.

## Verdict

The completed E4 is a defensible exploratory comparison of three training
procedures under one fixed setting. It does not identify the mechanism of KL's
effect or establish that any route-stability improvement causes error reduction.
The proposed identity-only / amplitude-only training losses do not cleanly
separate the two mechanisms. A fixed-checkpoint factorial intervention should
precede further training.

## High-priority correction: fixed output gate changes gradient access

In run.py:81-94 the prediction is max(p)*h_argmax(p)(x). Identity selection is
hard and has no ordinary derivative. Replacing max(p) with 1 removes the task
loss's gradient to the router. Balancing can still train the router, and adding
KL supplies another gradient, so comparisons could measure gradient availability
rather than amplitude invariance. A local 40-example backward check confirmed
that all four router parameter gradients are None under fixed-gate task loss,
while expert gradients remain present.

Do not introduce a straight-through estimator or soft router only in this arm:
that adds another changed mechanism. A separately designed architecture study
could use the same surrogate across all cells, but would be a different study.

## High-priority correction: losses are not independent switches

KL on the entire softmax affects rank, margins, entropy and selected amplitude.
An argmax-disagreement loss has no useful ordinary gradient. A cross-entropy
surrogate for teacher identity changes the probabilities, including amplitude.
A selected-amplitude consistency loss can also change expert ranking. Therefore
the labels "identity only" and "amplitude only" would overstate identification.
If studied, call them different regularizers and measure their cross-effects.
Equal numerical coefficients across different losses do not imply equal strength.

## Recommended next experiment: frozen 2x2 intervention

Fix model parameters and the corrupted expert input z. Cache clean and corrupted
routes: identity j0,j1 and selected amplitude w0,w1. Evaluate all four predictions

    pred_ab = w_b * h_{j_a}(z), a,b in {0,1}
    M_ab = mean((pred_ab - y)^2)

- M11: ordinary corrupted prediction.
- M01: clean identity, corrupted amplitude (existing ID-only diagnostic).
- M10: corrupted identity, clean amplitude (missing amplitude-only diagnostic).
- M00: clean identity and clean amplitude (existing full restoration).

Report identity restoration M11-M01, amplitude restoration M11-M10, joint
restoration M11-M00, and interaction M11-M10-M01+M00. These are controlled,
path-sensitive diagnostics, not additive unique causal shares. Pair IDs and
weights explicitly; amplitude in a crossed cell is the cached scalar, not the
new expert's probability. Crossed routes can be off the learned support.

Use nuisance noise SD 1 as the designated next diagnostic, both profiles and all
five seeds. Analyze clean-training, augmentation and KL models separately.
Use identical source items, perturbation draws, backend and precision for all
four cells. Existing checkpoints suffice; retraining is unnecessary. Near-tied
hard routing can change across CPU/GPU, so recompute all cells consistently and
do not mix newly replayed cells with archived GPU baseline predictions.

Clean routes remain privileged. The experiment identifies sensitivity to an
intervention, not an attainable inference-time repair or an information floor.
Keep missing-feature/cue conditions secondary and interpret their lost information
separately. This is a new exploratory follow-up, not a retrospective primary test.

## Training comparison improvements, if another round is justified

1. Select checkpoints for the intended target distribution. Current E4 selects
   clean validation MSE; in the shortcut profile validation retains the shortcut,
   while the main test removes it. This is not test leakage, but it may select
   a checkpoint that is poor for the stated robustness objective. A new protocol
   should define a validation distribution and a fixed clean/noisy score or a
   clean-error constraint before inspecting new outcomes. Preserve current E4
   selection/results as historical; do not silently reselect on the test set.
2. Treat lambda=0.1 as one setting, not a tuned method optimum. If tuning is
   necessary, use a small declared validation-only grid, count tuning costs,
   and use a fresh held-out test generator for a subsequent confirmatory claim.
3. To distinguish route-specific regularization from generic smoothing, consider
   an augmentation-plus-output-consistency control in addition to augmentation
   and router KL. This tests specificity but still is not a pure causal mediator
   analysis. Match views, initialization, batches, steps and selection rule.
4. Do not add seeds until a preferred interval excludes zero. Freeze the number
   and analysis first. Five optimization seeds on one dataset do not establish
   generality across training datasets. New independently generated training
   datasets are more informative for that separate question.

## Endpoint and inference checks

- Existing primary endpoint is corrupted-input MSE, KL minus augmentation.
  Both profile-specific intervals include zero. Lower group means cannot be
  described as a confirmed incremental advantage.
- Separately compare each model's noise-induced degradation, MSE(noisy)-MSE(clean).
  Compute paired difference-of-differences from per-source losses, not by
  subtracting interval endpoints. This diagnostic was added after scalar-mean
  inspection in E4 and remains exploratory; it must not replace the primary test.
- Route switch rate alone misses amplitude changes and has no monotonic
  relationship with prediction quality. Include gate changes, clean performance,
  task-conditioned errors and expert-load checks alongside it.
- Preserve perturbation/source/seed pairing. Repeated noise draws are not
  independent new training replications. Pointwise bootstrap intervals do not
  adjust for all inspected conditions, and five seeds give uncertain tails.
- Expert removal establishes dependence, not specialization by itself. Stronger
  specialization claims need task-relative expert advantages and a separate
  selection/evaluation split, not only the dominant expert selected on test data.

## Recommended order

First fill the missing M10 cell using existing checkpoints. Then decide whether
the interaction pattern warrants a new matched training study. Do not launch the
previously suggested fixed-gate-from-scratch comparison as a clean mechanism
ablation. Existing E4 findings remain valid within their documented limited scope.
