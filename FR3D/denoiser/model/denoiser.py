"""
Adapted from puzzlefusion++
https://github.com/eric-zqwang/puzzlefusion-plusplus
"""

import torch
from torch.nn import functional as F
import lightning.pytorch as pl
import hydra
from FR3D.denoiser.model.modules.denoiser_transformer import DenoiserTransformer
from chamferdist import ChamferDistance
from FR3D.denoiser.evaluation.evaluator import (
    calc_part_acc,
    trans_metrics,
    rot_metrics,
    calc_shape_cd,
)
from FR3D.denoiser.model.modules.custom_diffusers import PiecewiseScheduler
from pytorch3d import transforms
from FR3D.denoiser.utils import build_batched_edge_index, build_batched_fc_edge_index, build_fully_connected_intra_part_edge_index

class Denoiser(pl.LightningModule):
    def __init__(self, cfg):
        super(Denoiser, self).__init__()
        self.cfg = cfg
        self.denoiser = DenoiserTransformer(cfg)

        self.save_hyperparameters()

        self.noise_scheduler = PiecewiseScheduler(
            num_train_timesteps=cfg.model.DDPM_TRAIN_STEPS,
            beta_schedule=cfg.model.DDPM_BETA_SCHEDULE,
            prediction_type=cfg.model.PREDICT_TYPE, 
            beta_start=cfg.model.BETA_START,
            beta_end=cfg.model.BETA_END,
            clip_sample=False,
            timestep_spacing=self.cfg.model.timestep_spacing
        )

        self.encoder = hydra.utils.instantiate(cfg.ae.ae_name, cfg)

        self.cd_loss = ChamferDistance()
        self.num_points = cfg.model.num_point
        self.num_channels = cfg.model.num_dim

        self.noise_scheduler.set_timesteps(
            num_inference_steps=cfg.model.num_inference_steps
        )

        self.rmse_r_list = []
        self.rmse_t_list = []
        self.acc_list = []
        self.cd_list = []

        self.metric = ChamferDistance()
        self.npz_save_counter = 0


    def _apply_rots(self, part_pcs, noise_params):
        """
        Apply Noisy rotations to all points
        """
        noise_quat = noise_params[..., 3:]
        noise_quat = noise_quat / noise_quat.norm(dim=-1, keepdim=True)
        part_pcs = transforms.quaternion_apply(noise_quat.unsqueeze(1), part_pcs)
        
        return part_pcs
    
    def _apply_trans_get_part_centers(self, part_pcs, noise_params):
        """
        Apply Noisy translations to all points
        """
        noise_trans = noise_params[..., :3]
        part_pcs = part_pcs + noise_trans.unsqueeze(1)
        centers = part_pcs.mean(dim=1, keepdim=False)
        
        return centers

    def _reduce_normals_and_locdists(self, part_pcs_gt, part_pcs_reduced, part_normals, part_locdists, radius=0.075):
        """
        PyTorch version: For each reduced point, find gt points within `radius` and aggregate their normals and locdists.
        Args:
            part_pcs_gt: (B, N_gt, 3) ground truth point coordinates (torch.Tensor)
            part_pcs_reduced: (B, N_reduced, 3) reduced point coordinates (torch.Tensor)
            part_normals: (B, N_gt, 3) normals for gt points (torch.Tensor)
            part_locdists: (B, N_gt) local dists for gt points (torch.Tensor)
            radius: float, search radius
            agg: str, aggregation method ('mean' or 'weighted')
        Returns:
            reduced_normals: (B, N_reduced, 3)
            reduced_locdists: (B, N_reduced)
        """
        B, N_reduced, _ = part_pcs_reduced.shape
        device = part_pcs_reduced.device

        dists = torch.cdist(part_pcs_reduced, part_pcs_gt)  # (B, N_reduced, N_gt)
        within_radius = dists <= radius  # (B, N_reduced, N_gt)

        neighbor_counts = within_radius.sum(dim=2)  # (B, N_reduced)

        neighbor_counts_clamped = neighbor_counts.clamp(min=1).unsqueeze(-1)

        part_normals_exp = part_normals.unsqueeze(1).expand(-1, N_reduced, -1, -1)  # (B, N_reduced, N_gt, 3)
        part_locdists_exp = part_locdists.unsqueeze(1).expand(-1, N_reduced, -1)    # (B, N_reduced, N_gt)

        mask = within_radius.unsqueeze(-1)  # (B, N_reduced, N_gt, 1)
        normals_sum = (part_normals_exp * mask).sum(dim=2)  # (B, N_reduced, 3)
        locdists_sum = (part_locdists_exp * within_radius).sum(dim=2)  # (B, N_reduced)

        reduced_normals = normals_sum / neighbor_counts_clamped
        reduced_locdists = locdists_sum / neighbor_counts_clamped.squeeze(-1)

        # Fallback for points with zero neighbors: use nearest neighbor
        no_neighbors = neighbor_counts == 0  # (B, N_reduced)
        if no_neighbors.any():
            nn_indices = dists.argmin(dim=2)  # (B, N_reduced)
            batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, N_reduced)
            reduced_normals[no_neighbors] = part_normals[batch_indices[no_neighbors], nn_indices[no_neighbors]]
            reduced_locdists[no_neighbors] = part_locdists[batch_indices[no_neighbors], nn_indices[no_neighbors]]
        
        return reduced_normals, reduced_locdists


    def _extract_features(self, data_dict, part_valids, noisy_trans_and_rots):

        part_pcs = data_dict['part_pcs'][part_valids]  # (valid_P, num_points, 3)
        part_normals = data_dict['part_normals'][part_valids]  # (valid_P, num_points, 3)
        part_locdists = data_dict['part_locdists'][part_valids]  # (valid_P, num_points, 1)

        part_pcs = self._apply_rots(part_pcs, noisy_trans_and_rots)
        part_normals = self._apply_rots(part_normals, noisy_trans_and_rots)
        noisy_part_centers = self._apply_trans_get_part_centers(part_pcs, noisy_trans_and_rots)

        if self.cfg.ae.enc_locdistsnormal:
            encoder_out_dict = self.encoder.encode(part_pcs, part_normals, part_locdists)
        else:
            encoder_out_dict = self.encoder.encode(part_pcs)

        encoder_out_dict["normals"], encoder_out_dict["locdists"] = self._reduce_normals_and_locdists(part_pcs, encoder_out_dict["coord"], part_normals, part_locdists)

        return encoder_out_dict, noisy_part_centers


    def forward(self, data_dict):

        B, P, N, C = data_dict["part_pcs"].shape
        part_valids = data_dict['part_valids'].bool()

        gt_trans = data_dict['part_trans'][part_valids] # (valid_P, 3)
        gt_rots = data_dict['part_rots'][part_valids] # (valid_P, 4)
        gt_trans_and_rots = torch.cat([gt_trans, gt_rots], dim=-1) # (valid_P, 7)

        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,),
                                  device=self.device).long()
        timesteps = timesteps.repeat(P, 1).T  # (B, P)
        timesteps = timesteps[part_valids]  # (valid_P,)

        noise = torch.randn(gt_trans_and_rots.shape, device=self.device)
        noisy_trans_and_rots = self.noise_scheduler.add_noise(
            gt_trans_and_rots, noise, timesteps
        )  # (valid_P, 7)
        noisy_trans_and_rots[data_dict["ref_part"][part_valids]] = gt_trans_and_rots[
            data_dict["ref_part"][part_valids]
        ]  # (valid_P, 7)

        reference_gt_and_rots = torch.zeros_like(gt_trans_and_rots, device=self.device)
        reference_gt_and_rots[data_dict["ref_part"][part_valids]] = gt_trans_and_rots[data_dict["ref_part"][part_valids]]

        encoder_out_dict, noisy_part_centers = self._extract_features(data_dict, part_valids, noisy_trans_and_rots)

        pred_noise = self.denoiser(
            noisy_trans_and_rots, 
            timesteps,
            encoder_out_dict,
            part_valids,
            data_dict["part_scale"][part_valids],
            data_dict["ref_part"][part_valids],
            build_fully_connected_intra_part_edge_index(data_dict['graph'], num_points_per_part=self.num_points) if self.cfg.model.se3 else None,
            build_batched_edge_index(data_dict['graph']) if self.cfg.model.se3 else None,
            build_batched_fc_edge_index(data_dict['part_valids']) if self.cfg.model.se3 else None,
            noisy_part_centers if self.cfg.model.se3 else None
        )

        output_dict = {
            'pred_noise': pred_noise,
            'gt_noise': noise,
            'timesteps': timesteps,
            'gt_trans_and_rots': gt_trans_and_rots
        }

        return output_dict


    def _loss(self, data_dict, output_dict):
        pred_noise = output_dict['pred_noise']
        part_valids = data_dict['part_valids'].bool()
        noise = output_dict['gt_noise']
        ref_part_mask = ~data_dict["ref_part"][part_valids]
        mse_loss = F.mse_loss(pred_noise[ref_part_mask], noise[ref_part_mask])

        return {'diff_mse_loss': mse_loss}

    def training_step(self, data_dict, idx):
        #torch.autograd.set_detect_anomaly(True)
        output_dict = self(data_dict)
        loss_dict = self._loss(data_dict, output_dict)
        loss_weights = self.cfg.model.loss_weights
        total_loss = 0
        for loss_name, loss_value in loss_dict.items():
            weighted_loss = loss_value * loss_weights.get(loss_name, 1.0)
            total_loss += weighted_loss
            self.log(f"train_loss/{loss_name}", loss_value, on_step=True, on_epoch=False)
        self.log(f"train_loss/total_loss", total_loss, on_step=True, on_epoch=False)
        return total_loss


    def _calc_val_loss(self, data_dict):
        output_dict = self(data_dict)
        loss_dict = self._loss(data_dict, output_dict)
        loss_weights = self.cfg.model.loss_weights
        total_loss = 0
        for loss_name, loss_value in loss_dict.items():
            weighted_loss = loss_value * loss_weights.get(loss_name, 1.0)
            total_loss += weighted_loss
            self.log(f"val_loss/{loss_name}", loss_value, on_step=False, on_epoch=True)
        self.log(f"val_loss/total_loss", total_loss, on_step=False, on_epoch=True)


    def validation_step(self, data_dict, idx):
        self._calc_val_loss(data_dict)

        part_valids = data_dict['part_valids'].bool()
        part_scale = data_dict["part_scale"][part_valids]  # (valid_P, 1)
        ref_part = data_dict["ref_part"][part_valids]  # (valid_P,)
        pts = data_dict["part_pcs"]
        B, P = part_valids.shape
        
        gt_trans = data_dict['part_trans'][part_valids] 
        gt_rots = data_dict['part_rots'][part_valids] 
        gt_trans_and_rots = torch.cat([gt_trans, gt_rots], dim=-1)

        noisy_trans_and_rots = torch.randn(gt_trans_and_rots.shape, device=self.device)  # (valid_P, 7)    
        reference_gt_and_rots = torch.zeros_like(gt_trans_and_rots, device=self.device)
        reference_gt_and_rots[ref_part] = gt_trans_and_rots[ref_part]
        noisy_trans_and_rots[ref_part] = reference_gt_and_rots[ref_part]

        intra_part_edge_index = build_fully_connected_intra_part_edge_index(data_dict['graph'], num_points_per_part=self.num_points) if self.cfg.model.se3 else None
        n_inter_part_edge_index = build_batched_edge_index(data_dict['graph']) if self.cfg.model.se3 else None
        fc_inter_edge_index = build_batched_fc_edge_index(data_dict['part_valids']) if self.cfg.model.se3 else None

        for t in self.noise_scheduler.timesteps:
            timesteps = t.reshape(-1).repeat(len(noisy_trans_and_rots)).to(self.device)
            encoder_out_dict, noisy_part_centers = self._extract_features(data_dict, part_valids, noisy_trans_and_rots)
            
            pred_noise = self.denoiser(
                noisy_trans_and_rots, 
                timesteps,
                encoder_out_dict,
                part_valids,
                part_scale,
                ref_part,
                intra_part_edge_index,
                n_inter_part_edge_index,
                fc_inter_edge_index,
                noisy_part_centers if self.cfg.model.se3 else None
            )

            noisy_trans_and_rots = self.noise_scheduler.step(
                pred_noise, t, noisy_trans_and_rots
            ).prev_sample
            noisy_trans_and_rots[ref_part] = reference_gt_and_rots[ref_part].to(
                dtype=noisy_trans_and_rots.dtype
            ) 

        pred_trans = noisy_trans_and_rots[..., :3].detach()
        pred_rots = noisy_trans_and_rots[..., 3:].detach()

        # Recover SE3 back to padded mode
        pred_trans_padded = torch.zeros(
            (B, P, 3), device=pred_trans.device, dtype=pred_trans.dtype
        )
        pred_rots_padded = torch.zeros(
            (B, P, 4), device=pred_rots.device, dtype=pred_rots.dtype
        )
        gt_trans_padded = torch.zeros(
            (B, P, 3), device=gt_trans.device, dtype=pred_trans.dtype
        )
        gt_rots_padded = torch.zeros(
            (B, P, 4), device=gt_rots.device, dtype=pred_rots.dtype
        )
        pred_trans_padded[part_valids] = pred_trans
        pred_rots_padded[part_valids] = pred_rots
        gt_trans_padded[part_valids] = gt_trans.to(dtype=gt_trans_padded.dtype)
        gt_rots_padded[part_valids] = gt_rots.to(dtype=gt_rots_padded.dtype)

        B, P, N, C = pts.shape

        expanded_part_scale = data_dict["part_scale"].unsqueeze(-1).expand(-1, -1, N, -1)
        pts = pts * expanded_part_scale

        acc, _, _ = calc_part_acc(pts, trans1=pred_trans_padded, trans2=gt_trans_padded,
                            rot1=pred_rots_padded, rot2=gt_rots_padded, valids=data_dict['part_valids'], 
                            chamfer_distance=self.metric)
        
        shape_cd = calc_shape_cd(pts, trans1=pred_trans_padded, trans2=gt_trans_padded,
                            rot1=pred_rots_padded, rot2=gt_rots_padded, valids=data_dict['part_valids'], 
                            chamfer_distance=self.metric)
        
        rmse_r = rot_metrics(pred_rots_padded, gt_rots_padded, data_dict['part_valids'], 'rmse')
        rmse_t = trans_metrics(pred_trans_padded, gt_trans_padded,  data_dict['part_valids'], 'rmse')
        
        self.acc_list.append(acc)
        self.rmse_r_list.append(rmse_r)
        self.rmse_t_list.append(rmse_t)
        self.cd_list.append(shape_cd)


    def on_validation_epoch_end(self):
        total_acc = torch.mean(torch.cat(self.acc_list))
        total_rmse_t = torch.mean(torch.cat(self.rmse_t_list))
        total_rmse_r = torch.mean(torch.cat(self.rmse_r_list))
        total_shape_cd = torch.mean(torch.cat(self.cd_list))
        
        self.log(f"eval/part_acc", total_acc, sync_dist=True)
        self.log(f"eval/rmse_t", total_rmse_t, sync_dist=True)
        self.log(f"eval/rmse_r", total_rmse_r, sync_dist=True)
        self.log(f"eval/shape_cd", total_shape_cd, sync_dist=True)
        self.acc_list = []
        self.rmse_t_list = []
        self.rmse_r_list = []
        self.cd_list = []
        return total_acc, total_rmse_t, total_rmse_r, total_shape_cd
    

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=2e-4,
            betas=(0.95, 0.999),
            weight_decay=1e-6,
            eps=1e-08,
        )
        lr_scheduler = hydra.utils.instantiate(self.cfg.model.lr_scheduler, optimizer=optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
