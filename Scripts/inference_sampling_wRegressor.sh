#!/bin/bash

# Limit CPU threads
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

# Optionally limit CUDA devices
export CUDA_VISIBLE_DEVICES=0

python test.py \
    experiment_name=test_everyday_wRegressor \
    inference_dir=everyday \
    +logger.group="test_inference" \
    +trainer.devices="[0]" \
    denoiser.model.se3=False \
    denoiser.maximizeMetric=perc_mp \
    denoiser.data.overfit=-1 \
    denoiser.data.val_batch_size=20 \
    denoiser.samples=20 \
    denoiser.sampling=True \
    denoiser.discriminator_sampling=True \
    denoiser.sample_gt=False \
    denoiser.model.num_inference_steps=20 \
    denoiser.ae.enc_locdistsnormal=True \
    denoiser.discriminator_data_creation=False \
    denoiser.data.data_val_dir=/home/ssdArray/datasets/Breaking-Bad-Dataset.github.io/data/pc_data_feats/everyday/val/ \
    denoiser.data.matching_data_path_val=/home/ssdArray/datasets/Breaking-Bad-Dataset.github.io/data/matching_data/ \
    denoiser.ckpt_path=/home/cederic/dev/puzzlefusion-plusplus/output/denoiser/final_ablEnc100_all_6l_34k/training/denoiser_final/last.ckpt \
    discriminator.ae_disc.ckpt_path=/home/cederic/dev/puzzlefusion-plusplus/output/discriminator/v3_final_discriminator/training/discriminator/epochFinal179.ckpt \
    denoiser.max_iters=2 \
    denoiser.real_bones=False
