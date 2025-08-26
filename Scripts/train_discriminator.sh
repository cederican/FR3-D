#!/bin/bash

# Limit CPU threads
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

# Optionally limit CUDA devices
export CUDA_VISIBLE_DEVICES=2,3,5,6

python train_discriminator.py \
    experiment_name=v3_final_discriminator \
    data.batch_size=100 \
    data.val_batch_size=20 \
    ckpt_path=/home/cederic/dev/puzzlefusion-plusplus/output/discriminator/v3_final_discriminator/training/discriminator/last.ckpt \
    +trainer.devices="[0,1,2,3]" \
    +trainer.strategy=ddp \
    trainer.max_epochs=2000 \
    data.overfit=20000 \
    +logger.group="discriminator"
