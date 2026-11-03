# E4 fixed-weight extension, 2026-10-19

Approved comparison: clean training, nuisance augmentation, augmentation plus
router consistency. Both profiles, seeds 0-4. Reuse v2 datasets, 100 epochs,
batch 1024, width 128, optimizer and balance coefficient. All arms are freshly
trained from identical per-seed initialization with identical minibatch order.

Each step averages two supervised views. Clean arm uses clean/clean; augmentation
and consistency use clean/noisy. Noise is resampled per minibatch at SD 0.5 on
nuisance coordinates only. Both views contribute equally to balance. Consistency
adds 0.1 KL(p_clean.detach || p_noisy), mean over inputs. No coefficient search
or test-driven changes. Same two forward calls/step and optimizer steps; actual
backward work/runtime can differ. Each arm selects lowest clean validation MSE.
This clean-focused selection is deliberate and may miss a robustness optimum.

Primary endpoint: consistency minus augmentation MSE at nuisance noise SD 1,
separately for independent and shortcut profiles. Clean MSE is the safety
endpoint. Other v2 corruptions, route switching, router loads and interventions
are exploratory. Test uses original three fixed perturbation draws; models and
seeds share test examples and targets. Paired crossed bootstrap over five seeds
and 20,000 source IDs, averaging perturbation draws first; 1000 draws, pointwise
95% intervals. Compare augmentation minus clean as a necessary context control.

This is an exploratory follow-up after E1-E3, not a preregistered confirmation.
One coefficient and one augmentation strength do not establish all consistency
methods ineffective if no benefit occurs. Expert collapse assessed by clean-test
expert loads. No clinical/LLM transfer claim. Raw checkpoints/predictions retained.
