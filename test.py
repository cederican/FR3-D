import os
import hydra
import lightning.pytorch as pl
from FR3D.denoiser.dataset.dataset import build_test_dataloader
from FR3D.denoiser.dataset.dataset_real import build_test_dataloader_real
import torch
from FR3D.eval import InferenceModel


@hydra.main(version_base=None, config_path="config", config_name="eval")
def main(cfg):

    pl.seed_everything(cfg.test_seed, workers=True)

    inference_dir = os.path.join(cfg.experiment_output_path, "inference", cfg.inference_dir)
    os.makedirs(inference_dir, exist_ok=True)

    denoiser_only_flag = cfg.denoiser.max_iters == 1

    if not cfg.denoiser.real_bones:
        test_loader = build_test_dataloader(cfg.denoiser, denoiser_only_flag)
    else:
        test_loader = build_test_dataloader_real(cfg.denoiser, denoiser_only_flag)

    model = InferenceModel(cfg)

    denoiser_weights = torch.load(cfg.denoiser.ckpt_path, weights_only=False)['state_dict']
    
    if not denoiser_only_flag:
        if cfg.discriminator.ae_disc.ckpt_path is not None:
            discriminator_weights = torch.load(cfg.discriminator.ae_disc.ckpt_path, weights_only=False)['state_dict']
            model.discriminator.load_state_dict(
                {k.replace('discriminator.', ''): v for k, v in discriminator_weights.items() 
                 if k.startswith('discriminator.')}
            )

    model.denoiser.load_state_dict(
        {k.replace('denoiser.', ''): v for k, v in denoiser_weights.items() 
         if k.startswith('denoiser.')}
    )

    model.encoder.load_state_dict(
        {k.replace('encoder.', ''): v for k, v in denoiser_weights.items() 
         if k.startswith('encoder.')}
    )

    logger = hydra.utils.instantiate(cfg.logger)
    
    trainer = pl.Trainer(accelerator=cfg.accelerator, devices=cfg.devices, strategy=cfg.strategy, max_epochs=1, logger=logger, precision=cfg.precision)
    
    trainer.test(model=model, dataloaders=test_loader)


if __name__ == '__main__':
    main()
