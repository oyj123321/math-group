# Gaussian conditional-expectation reference

This reference applies only to the declared synthetic regression tasks. Let
u_1,...,u_8 be independent standard normal variables, with an intact observable
regime cue. Uniformly hide exactly k relevant coordinates and replace them with
zero. The oracle knows which coordinates are missing. Put q=k/8.

For squared error, the Bayes prediction is E[Y | observed coordinates, regime],
and its risk is E[Var(Y | observed coordinates, regime)]. For any prediction h,

E[(Y-h)^2] = E[Var(Y | observations)]
             + E[(E[Y | observations]-h)^2].

Thus information loss is not itself a model failure. The oracle is a mathematical
diagnostic; it is not a learned method or an inference-time routing repair.

## The four raw task functions

1. Sum: Y=sum(u_i)/sqrt(8). Conditional mean is the sum of observed coordinates
   divided by sqrt(8); the irreducible raw MSE is q.
2. Signed sum: Y=(sum(u_1,...,u_4)-sum(u_5,...,u_8))/sqrt(8). Conditional mean
   drops missing coordinates and raw irreducible MSE is again q.
3. Product interactions: Y=(u_1*u_2+u_3*u_4)/sqrt(2). A term has conditional mean
   zero if either factor is missing; otherwise retain the product. A specified
   pair is fully observed with probability (8-k)(7-k)/(8*7), giving population
   raw irreducible MSE 1-(8-k)(7-k)/56.
4. Sine: Y=sin(sum(u_i)/sqrt(8)). Write S_o for the observed normalized sum.
   Missing sum S_m is independent N(0,q). The Gaussian characteristic function
   gives E[sin(S_o+S_m)|S_o]=exp(-q/2)*sin(S_o).
   Its raw risk is (1-exp(-2)-exp(-q)+exp(-2+q))/2.

The runner normalizes each regime using its training mean mu_j and standard
deviation s_j. Normalize oracle predictions as (E[Y|obs]-mu_j)/s_j and divide
the raw theoretical risk by s_j^2. Do not substitute a population variance for
the measured training variance in the implemented comparison.

Population risk formulas are expectations over fresh Gaussian samples and mask
locations. A finite test set may fall below the population expectation; this is
not a violation. Compare finite-sample empirical oracle losses and their paired
bootstrap uncertainty as well as the formulas.

## What this reference does not establish

- Task-cue removal introduces regime ambiguity and requires a different oracle.
- Unobserved-at-training missingness indicators are an additional distribution
  shift. Compare identical masks with and without these indicators.
- Sparse routing and gate-weight interventions can interact; the decomposition
  is path-dependent and its signed terms are not unique causal percentages.
- Five training seeds give limited uncertainty precision. Bootstrap intervals
  are exploratory and pointwise, not multiplicity-adjusted evidence.
