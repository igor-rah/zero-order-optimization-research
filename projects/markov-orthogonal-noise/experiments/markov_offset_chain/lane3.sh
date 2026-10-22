#!/bin/bash
cd "$(dirname "$0")"
export NT=1
python run.py --mode sgd --tau 1 --lr 7e-4 --steps 2000
python run.py --mode sgd --tau 256 --lr 7e-4 --steps 2000
echo LANEDONE
