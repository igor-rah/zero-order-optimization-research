#!/usr/bin/env bash
# Reproduce every number and figure of the H1 report.  Run from anywhere:  bash code/run_all.sh
# Total ~35 min on 2 CPU cores (E1.2 ~26 min).  Requires python3 with numpy, scipy, matplotlib;
# the symbolic checks additionally need sympy + mpmath (skipped with a note if absent).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data figs
python3 code/verify_identities.py                         # V1-V10  -> data/verify.json
if python3 -c "import sympy, mpmath" 2>/dev/null; then
  python3 code/verify_symbolic.py                         # S1-S7   -> data/verify_symbolic.json
else
  echo "sympy/mpmath not found: keeping the committed data/verify_symbolic.json"
fi
python3 code/e1_1_bessel_mc.py                            # E1.1    -> data/e1_1.json
python3 code/e1_3_two_radii.py                            # E1.3    -> data/e1_3.json
python3 code/e1_4_batch_variance.py                       # E1.4    -> data/e1_4.json
python3 code/e1_2_logistic.py                             # E1.2    -> data/e1_2.json
python3 code/e1_2_logistic.py collapse                    # E1.2b   -> data/e1_2_collapse.json
python3 code/make_figs.py                                 # figs/*  and data/summary_stats.json
if command -v latexmk >/dev/null; then latexmk -xelatex -interaction=nonstopmode report.tex >/dev/null && echo "report.pdf built"; fi
