# H3 — τ = 1: two-sided bound and the √T law for two-point methods

**Purpose.** Local study of hypothesis H3 of the project *markov-orthogonal-noise*: for independent Markov states (τ = 1) and smoothing radius γ ≤ γ_ε, is the noise threshold of two-point methods of radius γ equal to Θ(εγ√T/(dR)), and is the oscillation of δ on scale γ (not Δ) the right parameter? Report: `report.pdf` (source `report.tex`, pdfLaTeX).

## What runs
| script | what | output |
|---|---|---|
| `code/verify.py` | checks V1–V17: Fisher informations, Hellinger bounds, sequential BC bound, separation lemma, hash noise, sphere multiplier, rounding law, constants (sympy / scipy / MC); V15–V17: idle-round constant, range-normalised bandit corollary, round-1 bias | `data/verify.json` |
| `code/e31_collapse.py 1e6` | E3.1: ZO-SGD with rough hash noise, logistic and ℓ1 regression, d ∈ {20, 80}, Δ/Δ_ref ∈ {0,…,300}, T ≤ 1e6 | `data/e31_<problem>_d<d>.json` |
| `code/e31_sensitivity.py` | E3.1 sensitivity (other instance, ±1 noise, γ×2, ℓ1 with γ×5; ±1 noise at levels ×0.378 with the ω̄ of the Beta runs, `pm1_eqomega`, 24 s), d = 20, T ≤ 2e5 | `data/e31s_<variant>.json` |
| `code/e32_smooth_noise.py` | E3.2: smooth noise Δ sin(⟨v,x⟩/ℓ+φ) — (a) estimator second moment, (b) optimisation cost | `data/e32a.json`, `data/e32b_d<d>.json` |
| `code/e34_hard_instance.py` | E3.4: the hard instance of the upper-bound theorem vs ZO-SGD | `data/e34.json` |
| `code/e35_pm1_attack.py` | E3.5: decoding of the ±1 cell noise | `data/e35.json` |
| `code/make_figs.py` | all figures and `data/summary_stats.json`, from `data/` only | `figs/*.pdf` |
| `code/common.py` | problems, hash noise, resumable ZO-SGD (`ZOSGD`, `Job`) | — |

## How to reproduce
```bash
bash code/run_all.sh          # ~50 min on one CPU core; fixed seeds (see Limitations on bitwise reproducibility)
python3 code/make_figs.py     # figures only, from data/
latexmk -pdf report.tex       # the report
```
The simulation scripts stop after ~4 minutes with exit code 3 and resume from a pickled state (`data/.job_*.pkl`) on the next call; `code/driver.sh` loops them. This makes the pipeline usable under a per-command time limit. Requirements: Python 3 with numpy, scipy, matplotlib, sympy.

## How to read the data
* `e31_*.json`: `err[c][k]` = f(x̄_T) − f* at checkpoint `checkpoints[c]` for run `k`; runs are grouped by noise level (`levels`, in units of `Delta_ref` = Mγ/√d), `nrep` runs per level; `G[k]` is the constant of the step and of the theorem bound 3RG/√T + 2Mγ. `e31s_pm1_eqomega.json` has `level_scale` = 0.378: the actual level is `levels × level_scale × Delta_ref` (also in `Delta`).
* `e32a.json`: for each d and ℓ/γ, the MC mean and standard error of E‖n‖² per seed, and the exact value `theory`.
* `e32b_d<d>.json`: errors for each ℓ (`ells`, grouped by `nrep`), noiseless (`err_none`) and Beta-hash (`err_beta`) arms, at checkpoints 1e4, 3e4, 1e5.
* `e34.json`: per (d, T): slope `a`, step scale, final error and the lower bound `LB` for each run.
* `e35.json`: errors vs rounds for the noiseless, decoded-±1, raw-±1 and raw-Beta arms.
* `summary_stats.json`: every number quoted in the report (fits with bootstrap CIs, ratios, maxima).

## Main result
For i.i.d. states with an arbitrary initial law, convex M-Lipschitz f on B_R, γ ≤ ε/(4M), ε < MR/8, T ≥ 1 + 16d²M²R²/((d−1)ε²), in expectation:

  εγ√(T−1)/(4dR) ≤ Δ_max^{E,2pt,γ}(ε; iid; T) ≤ 32π·εγ√T/(dR)

for the class of methods that see only pair differences at radius ≤ γ and draw directions from laws fixed in advance. The lower side is ZO-SGD with batch 1 that idles in round 1 (with a stationary start, √T instead of √(T−1)); the upper side is a linear hypercube with rough cell noise of density cos²(πt/2), analysed with Hellinger distance (bounded noise has infinite KL). The ±1 cell noise is decodable for Δ > 2Mγ (any f) and at every Δ on the linear hypercube, so it gives no upper bound there. For arbitrary algorithms Δ_max^E ≤ 71·ε√T/√d. All bounds hold with ω_δ(2γ)/2 in place of Δ.

## Limitations
* Upper bound proven only for non-adaptive direction laws; adaptive ones remain a hypothesis.
* Constants differ by a factor 128π between the sides.
* Lower side only for convex M-Lipschitz f with γ ≤ ε/(4M); the smooth class with ε/(4M) < γ ≤ √(ε/L) is not covered.
* Bandit comparison (H3.3b) relies on cited rates for losses in [0,1], not re-checked; after rescaling to the range of f they help only at budgets T ≳ C_b d^{k−1}T0, not at complexity-preserving budgets; E3.3 (one-point vs two-point, numerically) was not run.
* E3.2 optimisation-level test is compromised by saturation at d = 80; the estimator-level test is clean.
* E3.1: collapse confirmed on the progress subsets only; the all-points fits for d = 80 exclude ½ by ≤ 0.03; which E3.1 selection was pre-registered cannot be established from the archive. The ±1-noise variant meets the refutation criterion (a = 0.67); at equal ω̄ the all-points fit gives a = 0.52 but the progress subset a = 0.63 [0.52, 0.77] (unresolved).
* E3.4 is a sanity check only (a method that learns nothing passes it).
* Reproducibility: with fixed seeds the logistic runs reproduce to ~1e-10, but the ℓ1 runs are statistically, not bitwise, reproducible (individual ℓ1, d = 80 runs at low noise levels differed by up to 42% on another run, per-level geometric means within ~10%; likely floating-point non-determinism in the LP solution or BLAS).
* Experiments use d ≤ 80 and a single problem instance per main configuration (5 runs = algorithm/noise randomness).
