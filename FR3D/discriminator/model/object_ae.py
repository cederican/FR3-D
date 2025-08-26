import torch
import hydra
import lightning.pytorch as pl
import copy
import numpy as np
import wandb

class ObjectAE(pl.LightningModule):
    def __init__(self, cfg):
        super(ObjectAE, self).__init__()

        self.discriminator = hydra.utils.instantiate(cfg.ae_disc.ae_disc_name, cfg)
        self.cfg = cfg

    def forward(self, data_dict):
        original_data_dict = copy.deepcopy(data_dict)

        num_points = data_dict['num_points'] # [B, 1] or [B]
        num_frags = num_points // 1000 # [B, 1] or [B]

        pred_pcs = data_dict['pred_pcs']
        gt_locdists = data_dict['gt_locdists'] # [BN, 1]
        num_points = data_dict['num_points']

        B, N, _ = pred_pcs.shape

        range_tensor = torch.arange(N).unsqueeze(0).expand(B, -1).to(pred_pcs.device)
        mask = range_tensor < num_points.unsqueeze(1)
        pred_pcs = pred_pcs[mask]
        gt_locdists = gt_locdists[mask]
        
        data_dict['pred_pcs'] = pred_pcs
        data_dict['gt_locdists'] = gt_locdists

        data_dict['num_frags'] = num_frags
        data_dict["iters"] = self.trainer.global_step   
        
        output_dict = self.discriminator(data_dict)

        return output_dict, original_data_dict

    def _loss(self, data_dict, output_dict):
        loss_dict = self.discriminator.loss(data_dict, output_dict)

        return loss_dict
    
    def on_train_epoch_start(self):
        self.trainer.train_dataloader.sampler.set_epoch(self.current_epoch)
 
    def training_step(self, data_dict, idx):
        output_dict, _ = self(data_dict)

        loss_dict = self._loss(data_dict, output_dict)
        loss_weights = self.cfg.model.loss_weights

        total_loss = 0
        for loss_name, loss_value in loss_dict.items():
            if "loss" in loss_name:   
                weighted_loss = loss_value * loss_weights.get(loss_name, 1.0)
                total_loss += weighted_loss
            self.log(f"train_loss/{loss_name}", loss_value, on_step=True, on_epoch=False)
        self.log(f"train_loss/total_loss", total_loss, on_step=True, on_epoch=False)
    
        return total_loss
    
    def validation_step(self, data_dict, idx):
        output_dict, _ = self(data_dict)
        loss_dict = self._loss(data_dict, output_dict)

        loss_weights = self.cfg.model.loss_weights
        
        total_loss = 0
        for loss_name, loss_value in loss_dict.items():
            if "valid_positions" in loss_name:
                continue
            if "loss" in loss_name:
                weighted_loss = loss_value * loss_weights.get(loss_name, 1.0)
                total_loss += weighted_loss
            self.log(f"val_loss/{loss_name}", loss_value, on_step=False, on_epoch=True)
        self.log(f"val_loss/total_loss", total_loss, on_step=False, on_epoch=True)

        top3_valid_positions = loss_dict['top3_valid_positions']
        if not hasattr(self, "top3_val_match_positions"):
            self.top3_val_match_positions = []
        self.top3_val_match_positions.append(top3_valid_positions)

        top1_valid_positions = loss_dict['top1_valid_positions']
        if not hasattr(self, "top1_val_match_positions"):
            self.top1_val_match_positions = []
        self.top1_val_match_positions.append(top1_valid_positions)

        
    def on_validation_epoch_end(self) -> None:
        def gather_and_log(tensor_list, name, title):
            if not tensor_list:
                return

            local_tensor = torch.cat(tensor_list, dim=0)

            gathered_tensor = self.all_gather(local_tensor)

            if self.trainer.is_global_zero:
                full_tensor = gathered_tensor.reshape(-1).cpu().numpy()

                counts, bin_edges = np.histogram(full_tensor, bins=20, range=(0, 20))
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                bin_labels = [int(c) for c in bin_centers]

                table = wandb.Table(data=[[label, int(counts[i])] for i, label in enumerate(bin_labels)],
                                    columns=["rank_index", "count"])
                bar_chart = wandb.plot.bar(table, "rank_index", "count", title=title)

                self.logger.experiment.log({
                    f"{name}_histogram": wandb.Histogram(np_histogram=(counts, bin_edges)),
                    f"{name}_barchart": bar_chart,
                    "global_step": self.global_step
                })

        if hasattr(self, "top3_val_match_positions"):
            gather_and_log(self.top3_val_match_positions, "val_top3_match_index", "Top-3 Rank Histogram")
            del self.top3_val_match_positions

        if hasattr(self, "top1_val_match_positions"):
            gather_and_log(self.top1_val_match_positions, "val_top1_match_index", "Top-1 Rank Histogram")
            del self.top1_val_match_positions

    def test_step(self, data_dict, idx):
        pass
   
    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=1e-4,
            betas=(0.95, 0.999),
            weight_decay=1e-4,
            eps=1e-08,
        )
        lr_scheduler = hydra.utils.instantiate(self.cfg.model.lr_scheduler, optimizer=optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
