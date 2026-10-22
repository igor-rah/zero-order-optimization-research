# Experiments

Each folder is self-contained: code, raw results (JSON), the script that turns them into figures, and a README with the purpose, how to run it, how to read the data and the main result. Raw results are committed, so every figure can be rebuilt without re-running the experiments.

| Folder | Question | Runtime |
| --- | --- | --- |
| [`rounded_oracle_mezo`](rounded_oracle_mezo/) | Rounding the loss value to bf16 in MeZO fine-tuning: ties, harm, and the tie law | ~4.5 min per run on 2 CPU cores; 16 runs + 5 short sweep runs |
| [`rounded_oracle_convex`](rounded_oracle_convex/) | Rounded two-point ZO on convex problems with known optimum: accuracy floor and window of good radii | minutes on a laptop CPU |
| [`markov_offset_chain`](markov_offset_chain/) | Rounding error as a lazy Markov chain under correlated (lazy) data, convex and MeZO | ~5 min convex; ~8 min per MeZO run |
| [`verification`](verification/) | Numerical checks of every closed-form law | a few minutes |

Dependency: `markov_offset_chain` reuses the model and data loader of `rounded_oracle_mezo` and the problem generator of `rounded_oracle_convex`.

Common notation: μ — smoothing radius (stencil half-width); u — grid unit of the rounding (bf16 ulp); RTN — round-to-nearest; SR — stochastic rounding; a *tie* — the two rounded values of a step are equal, so the step is zero; *span* — |f(x+μz) − f(x−μz)| / u, the number of grid cells the stencil covers; τ_data — how many consecutive steps reuse one minibatch.
