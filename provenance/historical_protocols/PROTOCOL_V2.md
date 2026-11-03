# Protocol v2: controlled MoE routing and specialization

## Implemented experiments

E1: learned MoE, linear-MAC-matched dense and parameter-matched dense; route
frequencies, mutual information, forced expert identity and expert removal with
the original gate weight fixed. Expert IDs are not aligned across seeds.

E2: nuisance Gaussian noise, mean shifts, relevant-feature masking with/without
indicators, task-cue removal, and a separate shortcut training profile. Stochastic
conditions have three matched perturbation replicates; deterministic shifts have
one. Original sample IDs, targets and perturbations are shared across models and
training seeds.

E3: four clean/corrupt input/routing cells, full routing restoration, identity-only
restoration, random identity with fixed selected gate weight, and random identity
with that expert's gate probability. Preserve signed effects. Clean-routing
restoration uses privileged information; it is not a deployable algorithm.

## Phase-2 configuration and budgets

Independent and shortcut profiles; five training seeds each; 80,000 train, 10,000
validation, 20,000 test. Four experts of width 128; 100 epochs, batch 1,024;
AdamW lr .002, weight decay .0001, balance .1. No capacity limit or dropped inputs.
All models have the same data/optimizer/step count, but minibatch orders are not
explicitly synchronized between architectures.

MoE: 85,640 parameters; 24,192 active linear MACs per item.
MAC-matched dense: 24,477 parameters; exactly 24,192 linear MACs.
Parameter-matched dense: 85,639 parameters; 85,049 linear MACs.
MAC counts exclude activation, dispatch, memory and selection overhead; they do
not establish equal latency. The phase-2 configuration was chosen after pilots,
not an independent preregistered confirmatory experiment.

## Reproduction

```
python -m unittest discover -s work/moe_study -p "test_*.py"
python work/moe_study/run.py --out work/my_run --device cpu --epochs 2 --train 400 --val 200 --test 200 --width 16
python work/moe_study/analyze_v2.py --runs work/moe_phase2_results/runs/independent --out outputs/moe_phase2/independent
python work/moe_study/analyze_v2.py --runs work/moe_phase2_results/runs/shortcut --out outputs/moe_phase2/shortcut
python work/moe_study/paired_missingness.py --runs work/moe_phase2_results/runs --out work/moe_paired_controls
python work/moe_study/analyze_missingness.py --runs work/moe_paired_controls --out outputs/moe_phase2/paired_controls
python work/moe_study/plot_v2.py --analysis outputs/moe_phase2
```

Average perturbations within source before uncertainty analysis. Crossed bootstrap
resamples training seeds and shared source IDs, preserving pairing. Pointwise 95%
intervals use 1,000 draws; exploratory, no multiplicity correction. Five seeds give
limited tail precision. Test labels do not tune training or hyperparameters.

The local paired-mask control shares identical masks between marker conditions.
The original v2 families used different draws, so their original differences are
not a strictly paired marker ablation. The local control also evaluates the same
shortcut-trained model with and without nuisance/regime correlations. THEORY.md
derives the Gaussian conditional expectation and population irreducible risk.
Oracle predictions know regime and missing locations. Missingness indicators were
unseen during clean training; marker penalties are a distribution-shift diagnostic.

## Server policy and scientific limits

Only yangjian's own account and existing echo-fine environment; no installations,
existing project changes, services or daemons. One explicit temporary directory,
including caches, is removed only after complete download and checksum/manifest
verification. Normal system audit and scheduler records are preserved.

Synthetic regression conclusions do not automatically transfer to language MoE
or clinical systems. Sparse activation alone does not establish semantic expertise.
Dominant-expert removal is exploratory and selects dominant IDs separately for
each task/seed. E4, full literature review, final report and slides remain outside
this completed implementation batch.
