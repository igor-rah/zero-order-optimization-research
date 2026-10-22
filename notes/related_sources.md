# Related sources

Papers and books the project builds on, with one line on why each matters. Updated as the work goes.

## Gradient-free methods and noise

| Source | Why it matters here |
| --- | --- |
| A. Gasnikov, D. Dvinskikh, P. Dvurechensky, E. Gorbunov, A. Beznosikov, A. Lobanov. *Randomized gradient-free methods in convex optimization.* Encyclopedia of Optimization; [arXiv:2211.13566](https://arxiv.org/abs/2211.13566) | the three criteria (calls, rounds, admissible noise); standard noise analysis; §4.4 states the condition E[δe] = 0 that this project generalizes |
| D. Pasechnyuk-Vilensky, I. Pavlov, M. Takáč, A. Gasnikov. *Information-theoretic upper bounds for deterministic noise in zeroth-order convex optimization.* [arXiv:2306.16371](https://arxiv.org/abs/2306.16371) (v5, 2026) | tight thresholds for adversarial deterministic noise; the hidden-valley construction is the template for our hard instances |
| A. Risteski, Y. Li. *Algorithms and matching lower bounds for approximately-convex optimization.* NeurIPS 2016 | original hard construction for non-smooth problems with bounded noise |
| T. Loginov, D. Dvinskikh, N. Tupitsa, O. Osmanov, Yu. Dorn, A. Gasnikov. *Minimax call complexity, rounds and admissible noise in zeroth-order optimization of highly smooth functions.* [arXiv:2610.05517](https://arxiv.org/abs/2610.05517) | decomposition of the oracle error into systematic, stochastic and adversarial parts; rounding falls into the adversarial part there |
| A. Suvorikova et al. *Wall-clock complexity for zeroth-order optimization with tunable oracle fidelity.* [arXiv:2605.31346](https://arxiv.org/abs/2605.31346) | precision of the oracle as a resource; an MCMC oracle as another natural Markov state |
| O. Shamir. *An optimal algorithm for bandit and zero-order convex optimization with two-point feedback.* JMLR 2017 | second moment of the two-point estimator via concentration on the sphere |
| J. Duchi, M. Jordan, M. Wainwright, A. Wibisono. *Optimal rates for zero-order convex optimization: the power of two function evaluations.* IEEE Trans. Inf. Theory 2015 | lower-bound technique for two-point feedback |

## Markovian noise

| Source | Why it matters here |
| --- | --- |
| B. Prokhorov, S. Chebykin, A. Gasnikov, A. Beznosikov. *Gradient-free approaches is a key to an efficient interaction with Markovian stochasticity.* [arXiv:2601.01160](https://arxiv.org/abs/2601.01160) | Markov noise model, d + τ instead of dτ for ZO methods, batching and MLMC; the base of the noise model in this project |
| A. Beznosikov, S. Samsonov, M. Sheshukova, A. Gasnikov, A. Naumov, E. Moulines. *First order methods with Markovian noise: from acceleration to variational inequalities.* NeurIPS 2023 | MLMC estimator for Markovian noise |
| J. Duchi, A. Agarwal, M. Johansson, M. Jordan. *Ergodic mirror descent.* SIAM J. Optim. 2012 | classical analysis of optimization with ergodic noise |
| B. Karimi, B. Miasojedow, E. Moulines, H.-T. Wai. *Non-asymptotic analysis of biased stochastic approximation scheme.* COLT 2019 | state-dependent Markov noise; relevant when the chain depends on the iterate (rounding) |
| R. Dorfman, K. Levy. *Adapting to mixing time in stochastic optimization with Markovian data.* ICML 2022 | adaptation to an unknown mixing time |
| D. Levin, Y. Peres. *Markov Chains and Mixing Times.* AMS | total variation, mixing times, lazy chains |
| R. Douc, E. Moulines, P. Priouret, P. Soulier. *Markov Chains.* Springer 2018 | Poisson equation; weaker forms of ergodicity (drift conditions, polynomial ergodicity) |

## Applications

| Source | Why it matters here |
| --- | --- |
| S. Malladi et al. *Fine-tuning language models with just forward passes* (MeZO). NeurIPS 2023 | memory-efficient ZO fine-tuning; the setting of our rounding experiments |
| B. Widrow, I. Kollár. *Quantization Noise.* Cambridge University Press 2008 | the uniform quantization-noise model that the offset chain makes precise |
