#!/bin/bash
cd "$(dirname "$0")"
export NT=1
python run.py --mode rtn --tau 1
python run.py --mode rtn --tau 16
python run.py --mode rtn --tau 256
python run.py --mode rtn --tau 2000
python run.py --mode rtn --tau 1 --mu 1e-2
python run.py --mode rtn --tau 256 --mu 1e-2
python run.py --mode sgd --tau 256 --lr 0.1 --steps 1000 --eval_every 25
echo LANEDONE
