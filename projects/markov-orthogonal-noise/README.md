# Markov-orthogonal deterministic noise

Admissible level of deterministic noise in zeroth-order convex optimization when the noise is orthogonal to the randomization only on average over a Markov state; motivating case: floating-point rounding in low-precision zeroth-order (MeZO-style) fine-tuning.

Started: September 2026 · Status: in progress

## Structure

| Path | Content |
| --- | --- |
| [`notes/problem_statement/`](notes/problem_statement/) | formal problem statement: [English (PDF)](notes/problem_statement/markov_orthogonal_noise_problem_en.pdf), [Russian (PDF)](notes/problem_statement/markov_orthogonal_noise_problem.pdf); LaTeX sources alongside |
| [`notes/related_sources.md`](notes/related_sources.md) | related papers and why each one matters; updated as the work goes |
| [`experiments/`](experiments/) | numerical experiments; each folder has its own README (purpose, how to run, how to read the data, main result) |
| `paper/` | LaTeX sources of the paper (empty until the draft is approved) |

## Experiments at a glance

| Experiment | Question | Main result (small scale) |
| --- | --- | --- |
| [`rounded_oracle_mezo`](experiments/rounded_oracle_mezo/) | What does bf16 rounding of the loss value do to MeZO? | 91% of steps are ties; at the default radius the loss rises instead of falling; a 10× larger radius removes the harm |
| [`rounded_oracle_convex`](experiments/rounded_oracle_convex/) | How accurate can rounded ZO be on convex problems with known optimum? | U-shaped error in the radius; the window of good radii closes; accuracy far below the rounding level |
| [`markov_offset_chain`](experiments/markov_offset_chain/) | Is the rounding error a Markov chain, and what do correlated data do to it? | the offset's memory follows the data chain; with lazy data the method drives its own offset into a frozen region |
| [`verification`](experiments/verification/) | Do the closed-form laws hold numerically? | 84 + 27 Monte Carlo / quadrature checks, all pass |

## Requirements

Python 3.10+, `numpy`, `scipy`, `matplotlib`, `torch` (CPU is enough), `sympy` (verification only). LaTeX: XeLaTeX for the Russian problem statement, pdfLaTeX for the English one.
