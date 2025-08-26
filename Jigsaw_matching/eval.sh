#!/bin/bash

# Limit CPU threads
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

/home/cederic/anaconda3/envs/jigsaw/bin/python eval_matching.py \
    --cfg /home/cederic/dev/puzzlefusion-plusplus/Jigsaw_matching/experiments/jigsaw_4x4_128_512_250e_cosine_everyday.yaml