# Hypotheses H1–H10

Each hypothesis about zeroth-order optimization with Markov noise orthogonal to the randomization is studied separately, and each report is checked by an independent review. Only reports whose review was accepted are kept here.

## Contents

- `Hk_<slug>/` — one folder per hypothesis: `report.tex`, `report.pdf`, `code/`, `data/`, `figs/`, `README.md`, `summary.md` (a short card ending with a status line); `report_ru.tex` and `report_ru.pdf` when a Russian translation exists.

Statuses: not started, draft (wip), in review, needs revision, verified, published. Verdict is the author's verdict from `summary.md`: confirmed, confirmed in a refined form, refuted, unresolved.

## Run order

| Order | Hypothesis | Depends on |
| --- | --- | --- |
| 1 | H1 | — |
| 2 | H3 | H1 |
| 3 | H5 | H1 |
| 4 | H9 | H1 |
| 5 | H7 | — |
| 6 | H8 | H7 |
| 7 | H10 | H7, H8 |
| 8 | H4 | H1, H3 |
| 9 | H6 | H1 |
| 10 | H2 | H1, H3, H4, H5 |

The rounding branch (H7, H8, H10) runs in parallel with the rest; H2 runs last as the synthesis.

## Status

| Hypothesis | Title | Status | Verdict | Report | Last change |
| --- | --- | --- | --- | --- | --- |
| H1 | Structure of the class and the definition of Δ_max | verified | confirmed in a refined form | [H1_class_definition](H1_class_definition/) | 2026-11-10 |
| H2 | The main formula: threshold as a function of budget and τ | not started | — | — | 2026-11-10 |
| H3 | τ = 1: two-sided bound and the √T law | verified | confirmed in a refined form | [H3_tau1_two_sided](H3_tau1_two_sided/) | 2026-11-11 |
| H4 | τ > 1: reduction to Prokhorov | not started | — | — | 2026-11-10 |
| H5 | Hard constructions | not started | — | — | 2026-11-10 |
| H6 | The radius in the class definition: the γ-transition | not started | — | — | 2026-11-10 |
| H7 | Rounding: self-dithering by curvature | not started | — | — | 2026-11-10 |
| H8 | Method-induced laziness | not started | — | — | 2026-11-10 |
| H9 | Weaker than uniform geometric ergodicity | not started | — | — | 2026-11-10 |
| H10 | Algorithms that exploit the structure | not started | — | — | 2026-11-10 |
