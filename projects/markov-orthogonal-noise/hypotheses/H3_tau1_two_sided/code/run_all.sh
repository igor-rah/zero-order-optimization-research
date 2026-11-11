#!/usr/bin/env bash
# Reproduce everything in the H3 archive:  bash code/run_all.sh   (about 50 min on one CPU core)
# Simulation scripts pause every ~4 min (exit code 3) and resume from data/.job_*.pkl; driver.sh loops them.
set -e
cd "$(dirname "$0")"
python3 verify.py > ../data/verify_log.txt
bash driver.sh
grep -q FAILED ../data/driver_log.txt && { echo "a simulation failed, see data/driver_log.txt"; exit 1; }
python3 make_figs.py > ../data/make_figs_log.txt
rm -f ../data/.job_*.pkl
echo "done: data/*.json, figs/*.pdf"
