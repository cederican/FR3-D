#!/bin/bash

# Limit CPU threads
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

# Optionally limit CUDA devices
export CUDA_VISIBLE_DEVICES=0,1,2,3

python train_vqvae.py \
    experiment_name=PNPP_100_LocDistNormal \
    data.batch_size=20 \
    data.val_batch_size=20 \
    ae.enc_locdistsnormal=True \
    ckpt_path=/home/cederic/dev/puzzlefusion-plusplus/output/autoencoder/PNPP_100_LocDistNormal/training/last.ckpt \
    +trainer.devices="[0,1,2,3]" \
    +trainer.strategy=ddp \
    trainer.max_epochs=2000 \
    data.overfit=-1 \
    +logger.group="vqvae"
