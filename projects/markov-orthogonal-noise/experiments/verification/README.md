# Numerical verification of the closed-form laws

**Purpose.** Every closed-form statement about the rounded oracle and the offset chain is checked against an independent computation (Monte Carlo, quadrature or symbolic integration) before it is used anywhere else.

## How to reproduce

```bash
python verify_theory.py  > verify_log.txt          # rounded-oracle laws, 84 checks
python verify_markov.py  > verify_markov_log.txt   # offset-chain facts, 27 checks
```

Both scripts print one line per check, `PASS`/`FAIL`, the error and the tolerance, and end with a summary line; `verify_theory.py` exits non-zero if a check fails.

## What is checked

| Script | Claims |
| --- | --- |
| `verify_theory.py` | tie probabilities for round-to-nearest and stochastic rounding under a uniform offset; mean, variance and second moment of the rounded difference; the Gaussian tie laws τ_RTN(s) (closed form) and τ_SR(s) (quadrature); the exact mean E ĝ = κ(c, r) ∇f (parallel and orthogonal parts) and the Poisson-summation identity for κ; the offset average of κ equals 1; the bound \|κ − 1\| ≤ 2q/(1 − q³); Gaussian and sphere moments used in the window theorem |
| `verify_markov.py` | the conditional tie kernel P_tie(ω, r) = 2Φ((1/2 − \|ω\|)/r) − 1; its uniform average equals the tie law; the absorption probability with frozen data; the data-dither bound TV ≤ q/(1 − q³) and \|E e^{2πiω}\| = q for a Gaussian loss spread |

## How to read the data

`verify_log.txt` (from `verify_theory.py`): one line per check, `[PASS] <claim and parameters>: err=<|closed form − independent value|> (tol <tolerance>)`; last line `ALL CHECKS PASSED`.

`verify_markov_log.txt` (from `verify_markov.py`): one line per check, `PASS <claim and parameters>: closed=<closed form> mc=<Monte Carlo / quadrature value> tol=<tolerance>`; last line `ALL PASS`.

## Main result

All 84 + 27 checks pass (logs committed: `verify_log.txt`, `verify_markov_log.txt`).

## Limitations

Each claim is checked at a finite set of parameter values, not proved; Monte Carlo checks hold up to the stated tolerance. `verify_theory.py` needs `sympy`.
