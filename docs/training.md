## Training

The training consists of three modules as detailed in the thesis. We train the VQ-PointNet++ and the SE(3)-Equivariant conditioned Diffusion Model on 4 Nvidia A40 GPUs. The Reassembly Quality Estimation Regressor is trained on four NVIDIA GeForce RTX 3090 GPUs.

**Stage 1**: VQ-PointNet++:
```
sh ./Scripts/train_vqvae.sh
```

**Stage 2**: SE3 denoiser:
```
sh ./Sripts/train_denoiser.sh
```
You need modify the checkpoint path for the pre-trained VQ-PointNet++ in the script.

**Stage 3**: Reassembly Quality Estimation Regressor:
```
sh ./Sripts/train_discriminator.sh
```
