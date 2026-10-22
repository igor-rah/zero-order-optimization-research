#!/bin/bash
cd "$(dirname "$0")"
export NT=1
python run.py --mode sgd --tau 16 --lr 7e-4 --steps 2000
echo LANEDONE
