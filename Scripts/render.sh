#!/bin/bash

# Limit CPU threads
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

python renderer/render_results.py \
    experiment_name=color_bonesReal_PF++_noNoise_turn_video \
    inference_dir=/home/cederic/dev/puzzlefusion-plusplus/output/denoiser/final_realBones_PFbase/inference/no_noise\
    renderer.output_path=color_bonesReal_PF++_noNoise_turn_video \
    renderer.jigsaw=False \
    renderer.num_samples=1 \
    renderer.only_end=False \
    idx=0 