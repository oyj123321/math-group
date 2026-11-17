# Speaker notes

Fifteen minutes. English, matching the slides. About 550 spoken words each, so leave time to point. Numbers are the paper's paired results. An interval that contains 0 means the direction is not established. It does not mean the effect is zero.

---

## Do Anh Dung — 0:00 to 5:00

Slides: title, 15 minutes, "A changed route is not an explanation", "Same model, four cells", "Same u, four tasks".

### Title — about 30 seconds

Good morning. We are Do Anh Dung, Xiangli Qianhui, and Yang Jian. The title is routing robustness in mixture-of-experts, and the subtitle is the whole project: separating expert-input effects from routing effects. A perturbation can change the prediction and, at the same time, change which expert is selected. Those two things move together. That does not tell us which one produced the error.

### 15 minutes — about 40 seconds

We split the talk into three questions. For the first five minutes I ask whether the route caused the error, and I define the contrast we actually compute. Qianhui then asks, for each kind of damage, whether the error is the input, the route, or both. Jian asks whether a router that switches less is a router that predicts better, first on the synthetic tasks and then on SARCOS.

### A changed route is not an explanation — about 70 seconds

Keep these two objects separate. The expert input is the features the selected expert reads. The route is which expert is chosen, together with its gate weight. The switch rate is only a count: the fraction of paired items whose top-1 expert changes. It does not price that change. The new route can be harmless, it can add error, or it can offset error. A single switch percentage cannot tell those apart. That is why a plot of switching is not yet an explanation.

### Same model, four cells — about 80 seconds

We do not train a new model for each cell. We take one trained mixture-of-experts model and evaluate four combinations. The clean cell uses the clean input and the clean route. The ordinary perturbed cell uses the perturbed input and the route that perturbed input induces. The two crossed cells are the contrast. One feeds the perturbed features to the expert selected on the clean input, with the clean gate weight. The other feeds the clean features through the perturbed route. On the forward path we change the expert input first and the route second. The first difference is the input effect. The second difference is the routing effect. They add up to the total change in error. A positive routing term means the new route adds error on that damaged input. A negative term means the new route offsets the damage. The route here is both the expert index and the gate weight, and only the selected expert runs. This figure is that design.

### Same u, four tasks — about 80 seconds

The synthetic data is built so this contrast is readable. Every example carries the same predictive vector u, eight coordinates, and a cue that selects one of four maps. f1 is a normalized sum. f2 contrasts the first half of u with the second half. f3 is a product, so the expert has to use an interaction. f4 is the sine of f1. The target is that map, standardized on the training set. Next to u is a nuisance vector v. In the independent profile, v has no regime offset. In the shortcut profile, training adds twice the cue onto the first four nuisance coordinates, and the test set takes that offset away. The explicit cue is still visible in both profiles. Nuisance noise and nuisance shifts leave u and the cue intact. Masks and cue deletion remove information. That split matters: error can rise even when the task information is still there. Qianhui will put a price on these cases.

If time is short, cut the four formulas down to: "sum, contrast, product, and a sine of the sum."

---

## Xiangli Qianhui — 5:00 to 10:00

Slides: "Experts specialize", "Task kept: the error is the input", "A near-zero total hides two effects", "Deleting information is not one case", "Dense nets: the gap depends on the test".

### Experts specialize — about 50 seconds

Before the error bars, the experts do specialize. This heatmap is the independent profile: the share of each regime's routes that land on each expert. The dominant expert receives 0.579 of a regime's routes. Removing it raises MSE by 0.326, so the specialization is real. That removal cost correlates only 0.11 with the scale-1 noise error. An expert that the clean task depends on is not automatically the explanation of the noise failure.

### Task kept: the error is the input — about 70 seconds

Now hold the task fixed and perturb only the nuisance coordinates. Every number on this slide is a change in MSE multiplied by 10,000, at scale 1. Independent noise: input 1.49, routing −0.10. Independent shift: input 1.60, routing −0.10. Shortcut noise: input 5.31, routing −0.17. Please read 1.49 as a change of 0.000149, not as an error of 1.49. On all three rows, the routing interval includes 0. The detected increase is the input term: the expert is reading damaged nuisance features while the clean route is held fixed. Experts may still switch. That switch is not a detected cause of the extra error. Shortcut shift does not belong on this slide. It is the next one.

### A near-zero total hides two effects — about 55 seconds

Shortcut mean shift, same scale, same units. The input term is +2.33, interval 0.51 to 5.13. The routing term is −2.29, interval −4.88 to −0.61. The total is 0.04, interval −0.47 to 0.53. The headline change is compatible with zero. Inside it, the damaged input raises error, and the new route offsets that damage. Both component intervals exclude 0. So a near-zero total is not evidence that routing did nothing. It is two effects of opposite sign. The new route offsets the damaged input.

### Deleting information is not one case — about 70 seconds

Taking information away is not the nuisance case, and it is not one case. These are raw MSE on the independent profile, not multiplied by 10,000. As the mask grows, the input term goes from 0.16 to 0.59. The routing term stays below 0.01. The switch rate still rises from 0.12 to 0.27. Substantial switching can sit beside an input-dominated decomposition. Cue deletion is different. With the cue off on 50 percent of items, the input term is 0.28 and the routing term is 0.15. Both are harmful. An identity-only control, which restores the clean expert index and keeps the perturbed gate weight, accounts for about 0.18. Here the route is part of the cost, because the router was using the cue to choose the expert.

### Dense nets — about 60 seconds

The dense baselines stop us from turning that into one ranking of mixture-of-experts against dense nets. At a 25 percent mask, both profiles, the mixture-of-experts model and both dense models sit near 0.323. Under 50 percent cue deletion on the independent profile, the mixture-of-experts model is worse: 0.43, against 0.40 for the compute-matched net and 0.35 for the parameter-matched net. Under nuisance noise, the paired gaps between the mixture-of-experts model and the dense nets have intervals that contain 0, so we do not name a winner there. Latency is a separate fact, not an accuracy result. Dispatch takes about 1.19 milliseconds; a dense net takes about 0.14. Matching multiply-accumulates does not match latency. Jian will test whether training the router to switch less improves the prediction.

If time is short, skip the latency sentence. Do not skip the sentence that the noise comparison is not detected.

---

## Yang Jian — 10:00 to 15:00

Slides: "A lower curve is not a detected gain", "Where the switch rate actually drops", "SARCOS", "What these fifteen minutes support", "Where it stops", closing slide.

### A lower curve is not a detected gain — about 50 seconds

Thank you. My question is: if we train the router to switch less, is the prediction better? We have three training ways. First, clean training. Second, we add noise in training. Third, we add noise and a penalty. The penalty makes the noisy route stay close to the clean route. The dots are the average of five models. Green is the lowest. The bars: we pick the five models and the test items again, 1000 times. Each time we compute the average again. The green bar and the orange bar cover each other. So we cannot say that switching less makes the prediction better.

### Where the switch rate actually drops — about 55 seconds

Now we look at how often experts switch. Each row is one test after training: independent noise or shift, shortcut noise or shift. The three columns are the same three training ways: clean, noise in training, and noise plus the penalty. The number is switch rate times 100. A smaller number means fewer switches. In every row, the gray bar is the longest, and the green bar is the shortest. So with noise plus the penalty, the model really switches less.

### SARCOS — about 70 seconds

To make this more sure, we use the SARCOS data. It is a robot arm. The left plot looks like our made-up data. Training with noise makes the error smaller. The extra penalty does not change much. The middle plot: the penalty also makes it switch a little less. The right plot: at noise 0.25, input is about 0.098. The routing part is about 0.0004. Switching experts almost does not help. The special case: we hide 6 of 21 sensors. Routing is minus 0.032. Switching experts helps a bit.

### What these fifteen minutes support — about 40 seconds

First. When noise makes the error larger, it is mainly because experts read dirty input. Second. Switching experts sometimes does almost nothing. Sometimes it reduces the error. Sometimes it makes the error worse. It depends on the test. Do not only look at how many times it switches. Third. We train it to switch less, but we still cannot say the prediction is more accurate.

### Where it stops — about 35 seconds

Our experiment has limits. One: only four small MLP experts. We use one expert at a time. This is not a big MoE like GPT. Two: when we split the error, we only use one order. If we change the order, the two parts can look different. Three: for real data, we only test one SARCOS robot arm. It may not work the same on other data.

### Close — about 15 seconds

This is the end of our talk. Thank you for watching.

If time is short, give the three claims and the SARCOS mask exception, and skip the primary-endpoint digits until a question.

---

## Questions the instructor is likely to ask

Answer the question that was asked, then stop. Do not add the latency number unless the question is about cost.

### 1. You say a changed route is not an explanation. On the shortcut shift, the route is exactly what cancels the error. Which is it?

Both, on different rows. The claim is not that routing never matters. The claim is that a switch count does not tell you the sign. On nuisance noise and nuisance shift, the routing interval includes 0. On the shortcut shift, the routing term is −2.29 times 10 to the −4, and that interval excludes 0: the new route offsets the input damage. The switch rate cannot distinguish those rows. The four-cell contrast can.

### 2. What is the intervention, in one sentence? Is the clean route "optimal"?

Take one trained model. Hold the target fixed. The input effect replaces the expert's features and keeps the clean expert index and the clean gate weight. The routing effect then allows the expert index and the gate weight to become the ones chosen on the perturbed input. We do not claim the clean route is optimal. When the routing term is negative, putting the clean route back would make the error worse.

### 3. You always change the input first. What if you change the route first?

The total is the same. The split need not be. On the shortcut shift, putting the shifted route onto the clean input raises MSE by about 0.000164. Changing the expert input after that lowers it by about 0.000160. So the shifted route is helpful when it is paired with the shifted input, and harmful when it is transplanted onto the clean input. "Input" and "Routing" in the talk are the forward path. We do not claim an order-independent ranking. The interaction is why the limitations slide says the order matters.

### 4. Is +2.33 a large error?

No. That slide is a change in MSE times 10,000. +2.33 means about 0.000233 MSE. The nuisance effects are small absolute changes with a detected direction. The mask and cue-deletion numbers later in the talk are raw MSE, on the order of tenths. If those two scales get mixed, the nuisance result will sound larger than the information-loss result. It is the other way around.

### 5. Your intervals are very wide. Does the cancellation actually happen?

The magnitudes are uncertain. The signs we use are not. For the shortcut shift, the input interval lies entirely above 0, the routing interval lies entirely below 0, and the total interval contains 0. We can say the two effects oppose each other. We cannot say they cancel to exactly zero, and we should not quote 2.33 and 2.29 as precise equals. Five seeds and a crossed bootstrap are why those intervals are wide.

### 6. An interval containing 0 means there is no effect, right?

No. It means this comparison does not establish a direction. It is not a proof that the effect is zero, and it is not a proof that two methods are equivalent. That sentence covers the nuisance routing terms, the mixture-of-experts versus dense noise comparison, and every consistency-minus-augmentation accuracy gap.

### 7. Why is the mixture-of-experts model worse when the cue is removed, if masks tie?

Because the cue is an input to the router, not only a feature inside the expert. At a 25 percent mask the three models tie near 0.323, and the synthetic routing term stays below 0.01 even while the switch rate rises from 0.12 to 0.27. At 50 percent cue deletion on the independent profile, both terms are harmful, input 0.28 and routing 0.15, and the model errors separate: mixture-of-experts 0.43, compute-matched dense 0.40, parameter-matched dense 0.35. Restoring only the clean expert identity, while keeping the perturbed gate weight, accounts for about 0.18 of that. Dense nets have no expert identity to restore.

### 8. The consistency curve is lowest and the switch rate falls. Why won't you call that a gain?

The means go the right way. The paired test does not. The loss is 0.1 times KL of the stopped clean router probabilities against the router on the augmented view. On the synthetic scale-1 tests, consistency has the lowest mean error and the lowest switch rate in all four comparisons, and every consistency-minus-augmentation error interval includes 0. On SARCOS, at the primary endpoint, noise standard deviation 0.25, augmentation lowers macro-NMSE from 0.114 to 0.075. Adding consistency leaves it at 0.075. The paired difference is 0.000056, interval −0.000928 to 0.001403. Switching does fall, from 0.0660 to 0.0609, interval about −0.009 to −0.001. So the stability change is detected, and the extra accuracy is not. We also do not claim the two training methods are equivalent.

### 9. Why does rerouting help on SARCOS masks and not on the synthetic masks?

The perturbations are not the same kind of damage. Synthetic masks zero predictive coordinates of a hand-built task, and the forward routing term stays below 0.01. SARCOS masks zero 6 of 21 real sensors. There the input term is about 0.425 and the routing term is −0.032, interval −0.044 to −0.020. Rerouting offsets part of that sensor damage. It is the exception named on the slide, not a second general law. Synthetic nuisance noise and SARCOS sensor noise also should not be compared by the size of the error: one spares the predictive coordinates, the other corrupts them.

### 10. Is this a result about language-model mixture-of-experts?

No. The router is a 28 to 128 to 4 network. Each expert is a 28 to 128 to 128 to 1 MLP. The prediction is the selected expert's output times its softmax weight. The weight is not reset to 1, and the other experts do not run. There is no attention block and no residual stream. The task cue is given, on the synthetic data and on SARCOS, so this is not a test of discovering tasks. Specialization is real under that cue: dominant share 0.579, removal cost +0.326. That cost correlates only 0.11 with scale-1 noise error, so clean-task specialization is a weak guide to noise robustness.

### 11. If someone asks about compute

The mixture-of-experts model has 85,640 parameters and 24,192 active linear multiply-accumulates. The compute-matched dense net matches the multiply-accumulates and has fewer parameters. The parameter-matched net matches the parameter count and does more compute. Measured dispatch is still about 1.19 milliseconds against about 0.14. That comparison is about cost. It is not an MSE result, and it does not explain the cue-deletion gap.
