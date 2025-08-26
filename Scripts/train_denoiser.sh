#!/bin/bash

# Limit CPU threads
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

# Optionally limit CUDA devices
export CUDA_VISIBLE_DEVICES=1,2,3,4,5

python train_denoiser.py \
    experiment_name=final_everyday_Enc100_no_SE3_6l_34k \
    discriminator_sampling=False \
    data.batch_size=64 \
    data.val_batch_size=64 \
    data.overfit=-1 \
    model.encoder_weights_path="/home/cederic/dev/puzzlefusion-plusplus/output/autoencoder/PNPP_100_LocDistNormal/training/last.ckpt"\
    model.se3=True \
    ae.enc_locdistsnormal=False\
    ckpt_path=null \
    +trainer.devices="[0,1,2,3,4]" \
    +trainer.strategy=ddp \
    trainer.max_epochs=2000 \
    +logger.group="final_denoiser"