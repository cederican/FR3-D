import os
import torch
import hydra
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor
from FR3D.denoiser.dataset.dataset import build_geometry_dataloader
from omegaconf import OmegaConf
from torchsummary import summary


def init_callbacks(cfg):
    checkpoint_monitor = hydra.utils.instantiate(cfg.checkpoint_monitor)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    return [checkpoint_monitor, lr_monitor]


@hydra.main(version_base=None, config_path="config/denoiser", config_name="global_config")
def main(cfg):
    # fix the seed
    pl.seed_everything(cfg.train_seed, workers=True)

    # create directories for training outputs
    os.makedirs(os.path.join(cfg.experiment_output_path, "training"), exist_ok=True)

    # initialize data
    train_loader, val_loader = build_geometry_dataloader(cfg)
    
    # initialize model
    model = hydra.utils.instantiate(cfg.model.model_name, cfg)

    # sample_batch = next(iter(train_loader))
    # summary(model, input_size=sample_batch["part_pcs"].shape[1:], device="cpu")

    if cfg.model.encoder_weights_path is not None:
        encoder_weights = torch.load(cfg.model.encoder_weights_path)['state_dict']
        model.encoder.load_state_dict({k.replace('ae.', ''): v for k, v in encoder_weights.items()})
        # freeze the encoder
        for param in model.encoder.parameters():
            param.requires_grad = False

    # initialize logger
    logger = hydra.utils.instantiate(cfg.logger)
    
    # Log the config to wandb
    if isinstance(logger, pl.loggers.WandbLogger):
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    # initialize callbacks
    callbacks = init_callbacks(cfg)
    
    # initialize trainer
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        **cfg.trainer
    )

    # check the checkpoint
    if cfg.ckpt_path is not None:
        assert os.path.exists(cfg.ckpt_path), "Error: Checkpoint path does not exist."

    # start training
    trainer.fit(
        model=model, 
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=cfg.ckpt_path
    )
  
if __name__ == '__main__':
    main()