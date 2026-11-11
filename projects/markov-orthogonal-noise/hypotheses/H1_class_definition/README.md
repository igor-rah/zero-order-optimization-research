# H1 — structure of the class N(Δ, τ, γ) and a correct definition of Δ_max

**Purpose.** Test hypothesis H1 of the project *markov-orthogonal-noise*: condition (2) restricts only m = E_π δ; for a fixed radius the class contains "Bessel" m; for all radii ≤ γ₀ it forces m ≡ const; without a fixed budget Δ_max = ∞, so the budget T and the query protocol must be part of the definition. The report (`report.pdf`) contains the proofs. This folder holds the numerical checks and experiments E1.1–E1.4.

## What is run

| Script | What it does | Output |
| --- | --- | --- |
| `code/verify_identities.py` | V1–V10: ball multiplier, ball-Stokes identity (smooth and rough noise), bias bound, Poincaré second moment, mixing sums, pointwise counterexample, two-radius exceptional set, budget formula, unbiasedness for q = 1, 2 | `data/verify.json` |
| `code/verify_symbolic.py` | S1–S7 (sympy/mpmath): Bessel identities, ball Fourier transform, 40-digit plane-wave check at d = 100, simple zeros, constants | `data/verify_symbolic.json` |
| `code/e1_1_bessel_mc.py` | E1.1: Monte Carlo of E_e[m(x+γe)e] for plane waves, d ∈ {5, 20, 100}, sweep over γ\|ω\|, first three zeros, superpositions of 10 waves (+ detuned control) | `data/e1_1.json` |
| `code/e1_2_logistic.py` | E1.2: logistic regression (d = 10, n = 50) with rough hash noise, τ = 1, Δ ∈ {0, 1, 10, 100}·Mγ/√d, batch grid 4⁰…4⁹; frozen-state control. `collapse` mode: E1.2b, matched batches B ∝ Δ² | `data/e1_2.json`, `data/e1_2_collapse.json` |
| `code/e1_3_two_radii.py` | E1.3: wave tuned to a zero at γ₁, bias at a second radius γ₂ ∈ [0.05, 3.2]·γ₁ | `data/e1_3.json` |
| `code/e1_4_batch_variance.py` | E1.4: second moment of the batched estimator under a lazy chain, d ∈ {10, 40, 160}, τ ∈ {1…256}, against an exact formula | `data/e1_4.json` |
| `code/make_figs.py` | all figures and every number quoted in the report, from `data/` only | `figs/*.pdf, *.png`, `data/summary_stats.json` |
| `code/common.py` | Ψ_d, Bessel zeros, sphere sampling, CIs, 64-bit hash noise | — |

Seeds are fixed in each script (5 seeds per experiment).

## How to reproduce

```bash
bash code/run_all.sh          # everything, ~35 min on 2 CPU cores (E1.2 takes ~26 min)
python3 code/make_figs.py     # figures only, from the committed data/
latexmk -xelatex report.tex   # the report (XeLaTeX; Latin Modern fonts)
```

Requirements: Python ≥ 3.10, numpy, scipy, matplotlib; sympy + mpmath for `verify_symbolic.py` (`run_all.sh` keeps the committed JSON if sympy is missing). Re-running E1.3, E1.4 and a sample E1.2 configuration reproduced the committed JSON bit for bit.

## How to read the data

| File | Keys |
| --- | --- |
| `e1_1.json` | per d: `t` (γ\|ω\|), `theory` (−tΨ_d(t)/d), `along`/`se_along` (per seed, along-ω component and its s.e.), `chi2_orth`; `zero_vec` (norm, noise norm, χ²_d and p at the three zeros); `sup`, `sup_detuned` (χ² tests at 20 points per seed, against 0 and against theory) |
| `e1_2.json` | `meta` (d, n, λ, γ, N, B grid, Δ multipliers); per seed: `R`, `M`, `L`, `Dref` = Mγ/√d, `gap0` = f(0) − f*, `err[k]` = f(x̄) − f* for Δ = k·Dref over the B grid, `control["10"]` (frozen state) |
| `e1_2_collapse.json` | per seed: `B` (batches at 10·Dref), `err10`, `err100` (matched 100× larger batches at 100·Dref) |
| `e1_3.json` | per d: `ratios` (γ₂/γ₁), `w` (\|ω\| for k = 1, 2, 3), `theory`, `est`, `se` (flattened 3 × len(ratios), per seed) |
| `e1_4.json` | per d: exact `A`, `S`; `rows` per τ: `alpha`, `tmix`, `B`, `pred` (exact B·E‖ḡ‖²), `bound` (d + 8τ − 3)·d, `per_seed` mean and s.e. (units Δ = γ = 1) |
| `verify*.json` | `ok` flag and diagnostics per check |
| `summary_stats.json` | all derived numbers quoted in the report (z-scores, slopes with CIs, T_ε, collapse ratios) |

## Main result

- (2) ⇔ the ball average of m = E_π δ is constant; the centred Markov part ξ is unconstrained. For a fixed radius, admissible bounded m are exactly those with Fourier transform a single layer on the spheres \|ξ\| = j_{d/2,k}/γ (plane waves iff J_{d/2}(γ\|ω\|) = 0); radii accumulating at 0, or two radii with a ratio outside {j_k/j_l}, force m ≡ const.
- Batched ZO-SGD with burn-in reaches E f(x̄) − f* ≤ ε for **every** Δ after T(Δ) = ⌈36.25 M²R²/ε²⌉·(⌈2d²/(d−1) + (2d + 8τ − 3)dΔ²/(M²γ²)⌉ + τ⌈log₂(16R√dΔ/(γε))⌉₊) rounds = O(Δ²), provided γ ≤ ε/(4M) (expectation variant; the probability variant follows with ε → βε by Markov's inequality). Hence Δ_max = ∞ without a budget for the all-radii class (unconditionally) and for the fixed-radius class with γ ≤ ε/(4M), and Δ_max(T) ≳ (εγ/R)√(T/(d(d+τ))) for T ≥ 2T₁ (so T ≳ τ(MR/ε)² log) — the first branch of the lower half of H2, not covering the regime τ ≳ T.
- Numerics (5 seeds each): Bessel zeros give bias at MC level (χ² rejection 0–7% vs nominal 5%); off-zero bias matches ΔtΨ_d(t)/d; the second-radius bias follows Δ\|ω\|\|Ψ_d(γ₂\|ω\|)\|; the batched second moment matches the exact formula within 1.4% and is additive in d and τ; on logistic regression ε = 0.03 is reached for every Δ, with budget slope 2.31 ± 0.34 in Δ (0.01 above the pre-registered band 2 ± 0.3; consistent with Δ²) (other targets 2.03–2.16; matched-batch collapse ratio 1.04, CI [0.86, 1.25]); the frozen-state control shows the floor that the class excludes.

## Limitations

- Two-radius theorem proven on ℝ^d only; the local version on a ball is cited (Berenstein–Gay for d = 2, Volchkov for general d), not re-checked.
- The algorithmic bound pays a τ·log burn-in per iteration and has conservative constants (E1.4: the constant 8 in front of τ is ≈ 20× pessimistic for a lazy chain).
- E1.2 uses d = 10 and a smooth strongly convex problem (cost), tail averaging and a tuned step; it tests the Δ² scaling, not the constants of the theorem. No optimisation experiment with τ > 1 (E1.4 tests the estimator only).
- In d = 100 Monte Carlo cannot resolve the Bessel structure beyond γ\|ω\| ≈ 40; that regime is covered by 40-digit quadrature (S3).
- The upper side of the complexity-preserving level (Δ_cp ≲ Mγ/√d) is not proven here (H3).
