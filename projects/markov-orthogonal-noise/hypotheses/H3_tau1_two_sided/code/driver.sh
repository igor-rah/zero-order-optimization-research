#!/usr/bin/env bash
# Resumable driver: re-invokes each script until it exits 0 (scripts pause with exit code 3 every ~4 min).
cd "$(dirname "$0")"
for s in "e31_collapse.py 1e6" e32_smooth_noise.py e35_pm1_attack.py e34_hard_instance.py e31_sensitivity.py; do
  while true; do
    python3 $s >> ../data/driver_log.txt 2>&1; rc=$?
    [ $rc -eq 0 ] && break
    [ $rc -ne 3 ] && { echo "FAILED $s rc=$rc" >> ../data/driver_log.txt; break; }
  done
done
echo ALLDONE >> ../data/driver_log.txt
