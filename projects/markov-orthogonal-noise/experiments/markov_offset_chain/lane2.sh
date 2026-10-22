#!/bin/bash
cd "$(dirname "$0")"
export NT=1
python run.py --mode fp32 --tau 1
python run.py --mode fp32 --tau 16
python run.py --mode fp32 --tau 256
python run.py --mode fp32 --tau 1 --mu 1e-2
python run.py --mode fp32 --tau 256 --mu 1e-2
python run.py --mode sgd --tau 1 --lr 0.1 --steps 1000 --eval_every 25
python run.py --mode sgd --tau 16 --lr 0.1 --steps 1000 --eval_every 25
echo LANEDONE
