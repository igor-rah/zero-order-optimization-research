#!/bin/bash
# two parallel lanes, 1 thread each
cd "$(dirname "$0")"
run(){ NT=1 python3 mezo.py --steps 2000 --eval_every 100 --lr 5e-4 "$@" 2>&1 | grep -v Warn; }
( for s in 0 1; do for md in fp32 rtn; do run --mode $md --seed $s; done; done
  run --mode rtn --mu 1e-2; run --mode rtn --mu 1e-1 ) > lane1.log &
( for s in 0 1; do for md in sr rtn_adapt; do run --mode $md --seed $s; done; done
  run --mode fp32 --mu 1e-2; run --mode fp32 --mu 1e-1 ) > lane2.log &
wait; echo ALLDONE >> lane1.log
