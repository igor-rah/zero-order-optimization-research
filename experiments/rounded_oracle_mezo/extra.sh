#!/bin/bash
cd "$(dirname "$0")"
while ! grep -q ALLDONE lane1.log; do sleep 10; done
run(){ NT=1 python3 mezo.py --steps 2000 --eval_every 100 "$@" 2>&1 | grep -v Warn; }
( run --mode sr --lr 5e-5 --tag sr_lr5e-5; run --mode rtn --lr 5e-5 --tag rtn_lr5e-5 ) > lane3.log &
( run --mode sr --lr 5e-6 --tag sr_lr5e-6; run --mode fp32 --lr 5e-5 --tag fp32_lr5e-5 ) > lane4.log &
wait; echo ALLDONE >> lane3.log
