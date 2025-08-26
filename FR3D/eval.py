"""
Adapted from puzzlefusion++
https://github.com/eric-zqwang/puzzlefusion-plusplus
"""

import torch
import lightning.pytorch as pl
import hydra
import time
from FR3D.denoiser.model.modules.denoiser_transformer import DenoiserTransformer
from FR3D.discriminator.model.modules.discriminator import Discriminator
from FR3D.denoiser.utils import recenter_pc, pad_data
from FR3D.denoiser.model.modules.encoder import VQVAE
from FR3D.denoiser.model.denoiser import (
    build_batched_edge_index,
    build_batched_fc_edge_index,
    build_fully_connected_intra_part_edge_index
)
from chamferdist import ChamferDistance
from FR3D.denoiser.evaluation.evaluator import (
    calc_part_acc,
    trans_metrics,
    rot_metrics,
    calc_shape_cd,
    search_top_k,
    discriminator_topK,
    calculate_pred_locdists,
    
)
import numpy as np
from FR3D.denoiser.model.modules.custom_diffusers import PiecewiseScheduler
from pytorch3d import transforms
import itertools
import os
from utils.node_merge_utils import (
    get_final_pose_pts_dynamic,
    get_distance_for_matching_pts,
)
from FR3D.denoiser.evaluation.transform import (
    transform_pc,
    transform_normal,
    transform_normalize,
)


class InferenceModel(pl.LightningModule):
    def __init__(self, cfg):
        super(InferenceModel, self).__init__()
        self.cfg = cfg
        self.encoder = VQVAE(cfg.denoiser)
        self.denoiser = DenoiserTransformer(cfg.denoiser)
        if cfg.denoiser.discriminator_sampling:
            if not self.cfg.denoiser.sample_gt:
                self.discriminator = Discriminator(cfg.discriminator)
        
        self.save_hyperparameters()

        self.noise_scheduler = PiecewiseScheduler(
            num_train_timesteps=cfg.denoiser.model.DDPM_TRAIN_STEPS,
            beta_schedule=cfg.denoiser.model.DDPM_BETA_SCHEDULE,
            prediction_type=cfg.denoiser.model.PREDICT_TYPE,
            beta_start=cfg.denoiser.model.BETA_START,
            beta_end=cfg.denoiser.model.BETA_END,
            clip_sample=False,
            timestep_spacing=self.cfg.denoiser.model.timestep_spacing
        )

        self.num_points = cfg.denoiser.model.num_point
        self.num_channels = cfg.denoiser.model.num_dim

        self.noise_scheduler.set_timesteps(
            num_inference_steps=cfg.denoiser.model.num_inference_steps
        )

        self.rmse_r_list = []
        self.rmse_t_list = []
        self.acc_list = []
        self.cd_list = []
        self.perc_mp_list = []
        self.correct_topk = []

        self.metric = ChamferDistance()

        self.noisy_trans_and_rots = None
        self.random_array = None

        self.mesh_path = cfg.renderer.mesh_path

        self.npz_save_counter = 0
    
    def generate_noisy_trans_and_rots(self, gt_trans_and_rots, num_valids, idx):
        if self.random_array is None or idx % self.cfg.denoiser.samples == 0:
            self.random_array = torch.randn(num_valids, gt_trans_and_rots.shape[1], device=self.device)
        return self.random_array

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

        # Avoid division by zero
        neighbor_counts_clamped = neighbor_counts.clamp(min=1).unsqueeze(-1)

        # Expand for broadcasting
        part_normals_exp = part_normals.unsqueeze(1).expand(-1, N_reduced, -1, -1)  # (B, N_reduced, N_gt, 3)
        part_locdists_exp = part_locdists.unsqueeze(1).expand(-1, N_reduced, -1)    # (B, N_reduced, N_gt)

        # Masked sum
        mask = within_radius.unsqueeze(-1)  # (B, N_reduced, N_gt, 1)
        normals_sum = (part_normals_exp * mask).sum(dim=2)  # (B, N_reduced, 3)
        locdists_sum = (part_locdists_exp * within_radius).sum(dim=2)  # (B, N_reduced)

        # Mean aggregation
        reduced_normals = normals_sum / neighbor_counts_clamped
        reduced_locdists = locdists_sum / neighbor_counts_clamped.squeeze(-1)

        # Fallback for points with zero neighbors: use nearest neighbor
        no_neighbors = neighbor_counts == 0  # (B, N_reduced)
        if no_neighbors.any():
            # Get indices of nearest gt point for each reduced point
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

        if self.cfg.denoiser.ae.enc_locdistsnormal:
            encoder_out_dict = self.encoder.encode(part_pcs, part_normals, part_locdists)
        else:
            encoder_out_dict = self.encoder.encode(part_pcs)

        encoder_out_dict["normals"], encoder_out_dict["locdists"] = self._reduce_normals_and_locdists(part_pcs, encoder_out_dict["coord"], part_normals, part_locdists)

        return encoder_out_dict, noisy_part_centers

    
    def test_denoiser_only(self, data_dict):

        B, P, N, C = data_dict["part_pcs"].shape
        part_valids = data_dict['part_valids'].bool()
        num_valids = part_valids[0].sum().item()
        part_scale = data_dict["part_scale"][part_valids]  # (valid_P, 1)
        ref_part = data_dict["ref_part"][part_valids]  # (valid_P,)
        pts = data_dict["part_pcs"]

        gt_trans = data_dict['part_trans'][part_valids] # (valid_P, 3)
        gt_rots = data_dict['part_rots'][part_valids] # (valid_P, 4)
        gt_trans_and_rots = torch.cat([gt_trans, gt_rots], dim=-1) # (valid_P, 7)
        
        noisy_trans_and_rots = torch.randn(gt_trans_and_rots.shape, device=self.device)       
        reference_gt_and_rots = torch.zeros_like(gt_trans_and_rots, device=self.device)
        reference_gt_and_rots[ref_part] = gt_trans_and_rots[ref_part]
        noisy_trans_and_rots[ref_part] = reference_gt_and_rots[ref_part]
        
        intra_part_edge_index = build_fully_connected_intra_part_edge_index(data_dict['graph'], num_points_per_part=self.num_points) if self.cfg.denoiser.model.se3 else None
        n_inter_part_edge_index = build_batched_edge_index(data_dict['graph']) if self.cfg.denoiser.model.se3 else None
        fc_inter_edge_index = build_batched_fc_edge_index(data_dict['part_valids']) if self.cfg.denoiser.model.se3 else None

        all_pred_trans_rots = torch.zeros((B, len(self.noise_scheduler.timesteps)+1, num_valids, gt_trans_and_rots.shape[1]), device='cpu') # (B, T, num_valids, 7)
        all_pred_trans_rots[:, 0, :, :] = noisy_trans_and_rots.reshape(B, num_valids, -1).detach().cpu()
        i = 0

        for t in self.noise_scheduler.timesteps:
            timesteps = t.reshape(-1).repeat(len(noisy_trans_and_rots)).cuda()
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
                noisy_part_centers if self.cfg.denoiser.model.se3 else None
            )
            noisy_trans_and_rots = self.noise_scheduler.step(pred_noise, t, noisy_trans_and_rots).prev_sample
            noisy_trans_and_rots[ref_part] = reference_gt_and_rots[ref_part].to(dtype=noisy_trans_and_rots.dtype)   
            all_pred_trans_rots[:, i+1, :, :] = noisy_trans_and_rots.reshape(B, num_valids, -1).detach().cpu()
            i += 1
            
        pred_trans = noisy_trans_and_rots[..., :3]
        pred_rots = noisy_trans_and_rots[..., 3:]

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

        self._save_inference_data(
                                data_dict=data_dict,
                                pred_trans_rots=all_pred_trans_rots[0].unsqueeze(1),
                                acc=acc,
                            )
    

    def test_discriminator_sampling(self, data_dict, idx):

        print(f"Sampling with {self.cfg.denoiser.samples} samples per inference step")

        B, P, N, C = data_dict["part_pcs"].shape
        part_valids = data_dict['part_valids'].bool()
        num_valids = part_valids[0].sum().item()
        part_scale = data_dict["part_scale"][part_valids]  # (valid_P, 1)
        ref_part = data_dict["ref_part"][part_valids]  # (valid_P,)
        pts = data_dict["part_pcs"]

        gt_trans = data_dict['part_trans'][part_valids] # (valid_P, 3)
        gt_rots = data_dict['part_rots'][part_valids] # (valid_P, 4)
        gt_trans_and_rots = torch.cat([gt_trans, gt_rots], dim=-1) # (valid_P, 7)

        # if self.random_array is None:
        random_array = torch.randn(num_valids, gt_trans_and_rots.shape[1], device=self.device)
        #random_array = self.generate_noisy_trans_and_rots(gt_trans_and_rots, num_valids, idx)
        noisy_trans_and_rots = random_array.unsqueeze(0).expand(B, -1, -1)
        noisy_trans_and_rots = noisy_trans_and_rots.reshape(gt_trans_and_rots.shape) 

        reference_gt_and_rots = torch.zeros_like(gt_trans_and_rots, device=self.device)
        reference_gt_and_rots[ref_part] = gt_trans_and_rots[ref_part]
        noisy_trans_and_rots[ref_part] = reference_gt_and_rots[ref_part]

        all_pred_trans_rots = torch.zeros((B, len(self.noise_scheduler.timesteps)+1, num_valids, gt_trans_and_rots.shape[1]), device='cpu') # (B, T, num_valids, 7)
        all_pred_trans_rots[:, 0, :, :] = noisy_trans_and_rots.reshape(B, num_valids, -1).detach().cpu()
        i = 0

        intra_part_edge_index = build_fully_connected_intra_part_edge_index(data_dict['graph'], num_points_per_part=self.num_points) if self.cfg.denoiser.model.se3 else None
        n_inter_part_edge_index = build_batched_edge_index(data_dict['graph']) if self.cfg.denoiser.model.se3 else None
        fc_inter_edge_index = build_batched_fc_edge_index(data_dict['part_valids']) if self.cfg.denoiser.model.se3 else None

        for t in self.noise_scheduler.timesteps:
                timesteps = t.reshape(-1).repeat(len(noisy_trans_and_rots)).cuda()
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
                    noisy_part_centers if self.cfg.denoiser.model.se3 else None
                )
                noisy_trans_and_rots = self.noise_scheduler.step(pred_noise, t, noisy_trans_and_rots).prev_sample
                noisy_trans_and_rots[ref_part] = reference_gt_and_rots[ref_part]
                
                all_pred_trans_rots[:, i+1, :, :] = noisy_trans_and_rots.reshape(B, num_valids, -1).detach().cpu()
                i += 1
        
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

        expanded_part_scale = data_dict["part_scale"].unsqueeze(-1).expand(-1, -1, N, -1)
        pts = pts * expanded_part_scale

        batch_acc = []
        batch_cd = []
        batch_rmse_r = []
        batch_rmse_t = []

        batch_pred_pcs = []
        batch_num_points = []

        # visualize export
        # visualize_save(pts, all_pred_trans_rots, part_valids, data_dict['part_locdists'], idx)

        for b in range(B):
            acc, _, _ = calc_part_acc(pts[b].unsqueeze(0), trans1=pred_trans_padded[b].unsqueeze(0), trans2=gt_trans_padded[b].unsqueeze(0),
                                rot1=pred_rots_padded[b].unsqueeze(0), rot2=gt_rots_padded[b].unsqueeze(0), valids=data_dict['part_valids'][b].unsqueeze(0), 
                                chamfer_distance=self.metric)

            shape_cd = calc_shape_cd(pts[b].unsqueeze(0), trans1=pred_trans_padded[b].unsqueeze(0), trans2=gt_trans_padded[b].unsqueeze(0),
                                rot1=pred_rots_padded[b].unsqueeze(0), rot2=gt_rots_padded[b].unsqueeze(0), valids=data_dict['part_valids'][b].unsqueeze(0), 
                                chamfer_distance=self.metric)

            rmse_r = rot_metrics(pred_rots_padded[b].unsqueeze(0), gt_rots_padded[b].unsqueeze(0), data_dict['part_valids'][b].unsqueeze(0), 'rmse')
            rmse_t = trans_metrics(pred_trans_padded[b].unsqueeze(0), gt_trans_padded[b].unsqueeze(0),  data_dict['part_valids'][b].unsqueeze(0), 'rmse')


            pred_pcs = transform_pc(pred_trans_padded[b].unsqueeze(0), pred_rots_padded[b].unsqueeze(0), pts[b].unsqueeze(0))
            pred_pcs = pred_pcs[part_valids[b].unsqueeze(0)]  # (valid_P, N, 3)
            num_parts, num_points, _ = pred_pcs.shape
            pred_pcs = pred_pcs.reshape(-1, 3)
            pred_pcs, _ = recenter_pc(pred_pcs)
            pred_pcs = pad_data(pred_pcs, num_points)

            scale = torch.amax(torch.abs(pred_pcs), dim=0, keepdim=True)  # shape: [1, 3]
            scale[scale == 0] = 1
            pred_pcs = pred_pcs / scale

            batch_acc.append(1 - acc)
            batch_cd.append(shape_cd)
            batch_rmse_r.append(rmse_r)
            batch_rmse_t.append(rmse_t)

            batch_pred_pcs.append(pred_pcs)
            batch_num_points.append(torch.tensor(num_parts * num_points, dtype=torch.int32, device=self.device))
        
        array_pred_pcs = torch.stack(batch_pred_pcs, dim=0)  # (B, valid_P, N, 3)
        array_num_points = torch.stack(batch_num_points, dim=0)  # (B, valid_P)
        array_gt_locdists = torch.stack([data_dict['part_locdists'][b][part_valids[b]].unsqueeze(-1) for b in range(B)], dim=0).reshape(B, -1, 1)  # (B, valid_P, N, 1)

        array_acc = torch.stack(batch_acc, dim=0)  # (B,)
        array_shape_cd = torch.stack(batch_cd, dim=0)  # (B,) 
        array_rmse_r = torch.stack(batch_rmse_r, dim=0)  # (B,)
        array_rmse_t = torch.stack(batch_rmse_t, dim=0)  # (B,)

        acc_norm, acc_scaled = transform_normalize(array_acc)
        shape_cd_norm, shape_cd_scaled = transform_normalize(array_shape_cd)
        rmse_r_norm, rmse_r_scaled = transform_normalize(array_rmse_r)
        rmse_t_norm, rmse_t_scaled = transform_normalize(array_rmse_t)

        agg_metric = (
            acc_norm +
            shape_cd_norm +
            rmse_r_norm +
            rmse_t_norm
        ) / 4.0

        B, X, _ = array_pred_pcs.shape
    
        range_tensor = torch.arange(X).unsqueeze(0).expand(B, -1).to(array_pred_pcs.device)
        mask = range_tensor < array_num_points.unsqueeze(1)

        pred_pcs = array_pred_pcs[mask]
        gt_locdists = array_gt_locdists.reshape(-1,1)
        num_points = array_num_points

        disc_dict = {
            'pred_pcs': pred_pcs,
            'gt_locdists': gt_locdists,
            'num_points': num_points,
        }

        if self.cfg.denoiser.sample_gt:
            disc_output_dict = {'metric_pred': agg_metric}
        else:
            disc_output_dict = self.discriminator(disc_dict)

        best_k_ID_pred, best_k_ID_gt, mask_ratio = discriminator_topK(
            disc_output_dict['metric_pred'], # B, 1
            agg_metric, # B, 1
            k=1,
            architecture= self.cfg.discriminator.ae_disc.architecture,
        )

        self.correct_topk.append(torch.tensor([mask_ratio], dtype=torch.float16 ,device=self.device))

        if self.cfg.denoiser.sample_gt:
            topK = best_k_ID_gt.item()
        else:
            topK = best_k_ID_pred.item()

        print(f"Best k ID pred: {best_k_ID_pred}, Best k ID gt: {best_k_ID_gt}, Mask ratio < 0.2 ranking: {mask_ratio}")
        print(f"Selected top1: {topK}, GTtop1: {best_k_ID_gt[0].item()}")

        pts = data_dict['part_pcs'][topK].unsqueeze(0)
        pred_trans = pred_trans_padded[topK].unsqueeze(0)
        pred_rots = pred_rots_padded[topK].unsqueeze(0)

        expanded_part_scale = data_dict["part_scale"][topK].unsqueeze(0).unsqueeze(-1).expand(-1, -1, N, -1)
        pts = pts * expanded_part_scale

        gt_trans = gt_trans_padded[topK].unsqueeze(0)
        gt_rots = gt_rots_padded[topK].unsqueeze(0)

        acc, _, _ = calc_part_acc(pts, trans1=pred_trans, trans2=gt_trans,
                            rot1=pred_rots, rot2=gt_rots, valids=data_dict['part_valids'][topK].unsqueeze(0), 
                            chamfer_distance=self.metric)
        shape_cd = calc_shape_cd(pts, trans1=pred_trans, trans2=gt_trans,
                            rot1=pred_rots, rot2=gt_rots, valids=data_dict['part_valids'][topK].unsqueeze(0), 
                            chamfer_distance=self.metric)
        rmse_r = rot_metrics(pred_rots, gt_rots, data_dict['part_valids'][topK].unsqueeze(0), 'rmse')
        rmse_t = trans_metrics(pred_trans, gt_trans,  data_dict['part_valids'][topK].unsqueeze(0), 'rmse')
        

        self.acc_list.append(acc)
        self.rmse_r_list.append(rmse_r)
        self.rmse_t_list.append(rmse_t)
        self.cd_list.append(shape_cd)
        
        self._save_inference_data(data_dict, all_pred_trans_rots[topK].unsqueeze(1), acc, idx, topK)
        
    
    def test_discriminator_data_creation(self, data_dict, idx):
        """
        function to sample data from the denoiser and save this data, such that in a next step a distinct discrimintaor can be trained with this data.
        """

        print(f"Sampling with {self.cfg.denoiser.samples} samples per inference step")

        B, P, N, C = data_dict["part_pcs"].shape
        part_valids = data_dict['part_valids'].bool()
        num_valids = part_valids[0].sum().item()
        part_scale = data_dict["part_scale"][part_valids]  # (valid_P, 1)
        ref_part = data_dict["ref_part"][part_valids]  # (valid_P,)
        pts = data_dict["part_pcs"]

        gt_trans = data_dict['part_trans'][part_valids] # (valid_P, 3)
        gt_rots = data_dict['part_rots'][part_valids] # (valid_P, 4)
        gt_trans_and_rots = torch.cat([gt_trans, gt_rots], dim=-1) # (valid_P, 7)
        
        random_array = torch.randn(num_valids, gt_trans_and_rots.shape[1], device=self.device)
        noisy_trans_and_rots = random_array.unsqueeze(0).expand(B, -1, -1)
        noisy_trans_and_rots = noisy_trans_and_rots.reshape(gt_trans_and_rots.shape)   

        reference_gt_and_rots = torch.zeros_like(gt_trans_and_rots, device=self.device)
        reference_gt_and_rots[ref_part] = gt_trans_and_rots[ref_part]
        noisy_trans_and_rots[ref_part] = reference_gt_and_rots[ref_part]

        all_pred_trans_rots = torch.zeros((B, len(self.noise_scheduler.timesteps) + 1, num_valids, gt_trans_and_rots.shape[1]), device='cpu') # (B, T, num_valids, 7)
        all_pred_trans_rots[:, 0, :, :] = noisy_trans_and_rots.reshape(B, num_valids, -1).detach().cpu()
        i = 0

        intra_part_edge_index = build_fully_connected_intra_part_edge_index(data_dict['graph'], num_points_per_part=self.num_points) if self.cfg.denoiser.model.se3 else None
        n_inter_part_edge_index = build_batched_edge_index(data_dict['graph']) if self.cfg.denoiser.model.se3 else None
        fc_inter_edge_index = build_batched_fc_edge_index(data_dict['part_valids']) if self.cfg.denoiser.model.se3 else None

        for t in self.noise_scheduler.timesteps:
                timesteps = t.reshape(-1).repeat(len(noisy_trans_and_rots)).cuda()
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
                    noisy_part_centers if self.cfg.denoiser.model.se3 else None
                )
                noisy_trans_and_rots = self.noise_scheduler.step(pred_noise, t, noisy_trans_and_rots).prev_sample
                noisy_trans_and_rots[ref_part] = reference_gt_and_rots[ref_part]
                
                all_pred_trans_rots[:, i+1, :, :] = noisy_trans_and_rots.reshape(B, num_valids, -1).detach().cpu()
                i += 1
        
        pred_trans = noisy_trans_and_rots[..., :3]
        pred_rots = noisy_trans_and_rots[..., 3:]

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

        expanded_part_scale = data_dict["part_scale"].unsqueeze(-1).expand(-1, -1, N, -1)
        pts = pts * expanded_part_scale

        batch_acc = []
        batch_cd = []
        batch_rmse_r = []
        batch_rmse_t = []

        batch_pred_pcs = []
        batch_pred_normals = []
        batch_pred_locdists = []

        for b in range(B):
            acc, _, _ = calc_part_acc(pts[b].unsqueeze(0), trans1=pred_trans_padded[b].unsqueeze(0), trans2=gt_trans_padded[b].unsqueeze(0),
                                rot1=pred_rots_padded[b].unsqueeze(0), rot2=gt_rots_padded[b].unsqueeze(0), valids=data_dict['part_valids'][b].unsqueeze(0), 
                                chamfer_distance=self.metric)

            shape_cd = calc_shape_cd(pts[b].unsqueeze(0), trans1=pred_trans_padded[b].unsqueeze(0), trans2=gt_trans_padded[b].unsqueeze(0),
                                rot1=pred_rots_padded[b].unsqueeze(0), rot2=gt_rots_padded[b].unsqueeze(0), valids=data_dict['part_valids'][b].unsqueeze(0), 
                                chamfer_distance=self.metric)

            rmse_r = rot_metrics(pred_rots_padded[b].unsqueeze(0), gt_rots_padded[b].unsqueeze(0), data_dict['part_valids'][b].unsqueeze(0), 'rmse')
            rmse_t = trans_metrics(pred_trans_padded[b].unsqueeze(0), gt_trans_padded[b].unsqueeze(0),  data_dict['part_valids'][b].unsqueeze(0), 'rmse')

            self.acc_list.append(acc)
            self.rmse_r_list.append(rmse_r)
            self.rmse_t_list.append(rmse_t)
            self.cd_list.append(shape_cd)

            pred_pcs = transform_pc(pred_trans_padded[b].unsqueeze(0), pred_rots_padded[b].unsqueeze(0), pts[b].unsqueeze(0))

            batch_acc.append(1 - acc)
            batch_cd.append(shape_cd)
            batch_rmse_r.append(rmse_r)
            batch_rmse_t.append(rmse_t)

            batch_pred_pcs.append(pred_pcs[part_valids[b].unsqueeze(0)])
            batch_pred_normals.append(transform_normal(pred_rots_padded[b].unsqueeze(0), data_dict["part_normals"][b].unsqueeze(0))[part_valids[b].unsqueeze(0)])
            batch_pred_locdists.append(calculate_pred_locdists(pred_pcs[part_valids[b].unsqueeze(0)]))
        
        array_pred_pcs = torch.stack(batch_pred_pcs, dim=0)  # (B, valid_P, N, 3)
        array_pred_normals = torch.stack(batch_pred_normals, dim=0)  # (B, valid_P, N, 3)
        array_pred_locdists = torch.stack(batch_pred_locdists, dim=0)  # (B, valid_P, N, 1)
        array_gt_locdists = torch.stack([data_dict['part_locdists'][b][part_valids[b]].unsqueeze(-1) for b in range(B)], dim=0)  # (B, valid_P, N, 1)

        array_acc = torch.stack(batch_acc, dim=0)  # (B,)
        array_shape_cd = torch.stack(batch_cd, dim=0)  # (B,) 
        array_rmse_r = torch.stack(batch_rmse_r, dim=0)  # (B,)
        array_rmse_t = torch.stack(batch_rmse_t, dim=0)  # (B,)

        acc_norm, acc_scaled = transform_normalize(array_acc)
        shape_cd_norm, shape_cd_scaled = transform_normalize(array_shape_cd)
        rmse_r_norm, rmse_r_scaled = transform_normalize(array_rmse_r)
        rmse_t_norm, rmse_t_scaled = transform_normalize(array_rmse_t)

        save_dict = {
            "gt_locdists": array_gt_locdists,
            "pred_pcs": array_pred_pcs,
            "pred_normals": array_pred_normals,
            "pred_locdists": array_pred_locdists,
            "pred_metrics": {
                "part_acc": torch.stack([array_acc, acc_norm, acc_scaled], dim=1).squeeze(2),  # (B, 3)
                "shape_cd": torch.stack([array_shape_cd, shape_cd_norm, shape_cd_scaled], dim=1).squeeze(2),  # (B, 3)
                "rmse_r": torch.stack([array_rmse_r, rmse_r_norm, rmse_r_scaled], dim=1).squeeze(2),  # (B, 3)
                "rmse_t": torch.stack([array_rmse_t, rmse_t_norm, rmse_t_scaled], dim=1).squeeze(2),  # (B, 3)
            }
        }

        self._save_inference_discriminator_data(save_dict)

    
    def test_sampling(self, data_dict, idx):
        """
        Test function to sample many data points and rank them by the perc_mp metric and take the top 1.
        """

        print(f"Sampling with {self.cfg.denoiser.samples} samples per inference step")

        B, P, N, C = data_dict["part_pcs"].shape
        part_valids = data_dict['part_valids'].bool()
        num_valids = part_valids[0].sum().item()
        part_scale = data_dict["part_scale"][part_valids]  # (valid_P, 1)
        ref_part = data_dict["ref_part"][part_valids]  # (valid_P,)
        pts = data_dict["part_pcs"]

        gt_trans = data_dict['part_trans'][part_valids] # (valid_P, 3)
        gt_rots = data_dict['part_rots'][part_valids] # (valid_P, 4)
        gt_trans_and_rots = torch.cat([gt_trans, gt_rots], dim=-1) # (valid_P, 7)
        
        random_array = torch.randn(num_valids, gt_trans_and_rots.shape[1], device=self.device)
        noisy_trans_and_rots = random_array.unsqueeze(0).expand(B, -1, -1)
        noisy_trans_and_rots = noisy_trans_and_rots.reshape(gt_trans_and_rots.shape)    
        reference_gt_and_rots = torch.zeros_like(gt_trans_and_rots, device=self.device)
        reference_gt_and_rots[ref_part] = gt_trans_and_rots[ref_part]
        noisy_trans_and_rots[ref_part] = reference_gt_and_rots[ref_part]

        all_pred_trans_rots = torch.zeros((B, len(self.noise_scheduler.timesteps), num_valids, gt_trans_and_rots.shape[1]), device='cpu') # (B, T, num_valids, 7)
        i = 0

        intra_part_edge_index = build_fully_connected_intra_part_edge_index(data_dict['graph'], num_points_per_part=self.num_points) if self.cfg.denoiser.model.se3 else None
        n_inter_part_edge_index = build_batched_edge_index(data_dict['graph']) if self.cfg.denoiser.model.se3 else None
        fc_inter_edge_index = build_batched_fc_edge_index(data_dict['part_valids']) if self.cfg.denoiser.model.se3 else None

        for t in self.noise_scheduler.timesteps:
                timesteps = t.reshape(-1).repeat(len(noisy_trans_and_rots)).cuda()
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
                    noisy_part_centers if self.cfg.denoiser.model.se3 else None
                )
                noisy_trans_and_rots = self.noise_scheduler.step(pred_noise, t, noisy_trans_and_rots).prev_sample
                noisy_trans_and_rots[ref_part] = reference_gt_and_rots[ref_part]
                
                all_pred_trans_rots[:, i, :, :] = noisy_trans_and_rots.reshape(B, num_valids, -1).detach().cpu()
                i += 1
        
        pred_trans = noisy_trans_and_rots[..., :3]
        pred_rots = noisy_trans_and_rots[..., 3:]

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
        pred_n = torch.cat((pred_trans_padded, pred_rots_padded), dim=-1)  # (B, P, 7)
        gt_t_and_r = torch.cat((gt_trans_padded, gt_rots_padded), dim=-1)  # (B, P, 7)

        pred_trans = pred_n[..., :3]
        pred_rots = pred_n[..., 3:]

        expanded_part_scale = data_dict["part_scale"].unsqueeze(-1).expand(-1, -1, N, -1)
        pts = pts * expanded_part_scale

        # pre-computed edges, corresponding index, dynamic sample points, surface points
        edges = data_dict['edges']
        corr = data_dict['correspondences']
        part_pcs_by_area = data_dict['part_pcs_by_area']
        critical_pcs_idx = data_dict['critical_pcs_idx']
        n_pcs = data_dict['n_pcs']
        n_critical_pcs = data_dict['n_critical_pcs']
        num_parts = data_dict["num_parts"]

        accumulated_bins = {} 
        mean_dist_mpts = []
        perc_mpts = []
            
        for b in range(B):
            part_pcs_by_area_transformed = get_final_pose_pts_dynamic(
                part_pcs_by_area[b].unsqueeze(0),
                n_pcs[b].unsqueeze(0),
                pred_trans[b].unsqueeze(0),
                pred_rots[b].unsqueeze(0),
                num_parts[b].unsqueeze(0),
            )

            for i in range(edges.shape[1]):
                if edges.shape[1] == 0:
                    print(f"edges.shape[1] == 0")
                idx2 = edges[b, i, 0]
                idx1 = edges[b, i, 1]
                cd_per_point = get_distance_for_matching_pts(
                    idx1,
                    idx2, 
                    part_pcs_by_area_transformed,
                    n_pcs[b].unsqueeze(0), 
                    n_critical_pcs[b].unsqueeze(0),
                    critical_pcs_idx[b].unsqueeze(0), 
                    corr[i][b].unsqueeze(0),
                    data_dict["data_id"][b].item(), 
                    self.metric
                )
                mean_dist_mpts.append(cd_per_point.mean().unsqueeze(0).detach().cpu())
                bins = self._make_cd_to_bins(cd_per_point)
                accumulated_bins[f'{b}'] = bins if i == 0 else accumulated_bins[f'{b}'] + bins
            
            if accumulated_bins:
                perc_mp = (1 - torch.sum(accumulated_bins[f'{b}'][1:6]) / torch.sum(accumulated_bins[f'{b}'])).unsqueeze(0)
                perc_mpts.append(perc_mp)
            else:
                perc_mpts.append(torch.zeros(1, device=self.device))
            
        perc_mp_array = torch.cat(perc_mpts)
        top_k_IDs, _ = search_top_k(
            perc_mp=perc_mp_array,
            k=self.cfg.denoiser.top_k,
            maximizeMetric=self.cfg.denoiser.maximizeMetric
        )      
                
        pts = data_dict['part_pcs'][top_k_IDs.item()].unsqueeze(0)
        pred_trans = pred_n[..., :3][top_k_IDs.item()].unsqueeze(0)
        pred_rots = pred_n[..., 3:][top_k_IDs.item()].unsqueeze(0)

        expanded_part_scale = data_dict["part_scale"][top_k_IDs.item()].unsqueeze(0).unsqueeze(-1).expand(-1, -1, N, -1)
        pts = pts * expanded_part_scale

        gt_trans = gt_t_and_r[top_k_IDs.item()].unsqueeze(0)[..., :3]
        gt_rots = gt_t_and_r[top_k_IDs.item()].unsqueeze(0)[..., 3:]

        acc, _, _ = calc_part_acc(pts, trans1=pred_trans, trans2=gt_trans,
                            rot1=pred_rots, rot2=gt_rots, valids=data_dict['part_valids'][top_k_IDs.item()].unsqueeze(0), 
                            chamfer_distance=self.metric)
        shape_cd = calc_shape_cd(pts, trans1=pred_trans, trans2=gt_trans,
                            rot1=pred_rots, rot2=gt_rots, valids=data_dict['part_valids'][top_k_IDs.item()].unsqueeze(0), 
                            chamfer_distance=self.metric)
        rmse_r = rot_metrics(pred_rots, gt_rots, data_dict['part_valids'][top_k_IDs.item()].unsqueeze(0), 'rmse')
        rmse_t = trans_metrics(pred_trans, gt_trans,  data_dict['part_valids'][top_k_IDs.item()].unsqueeze(0), 'rmse')
        
        perc_mp = perc_mp_array[top_k_IDs.item()].unsqueeze(0)

        self.acc_list.append(acc)
        self.rmse_r_list.append(rmse_r)
        self.rmse_t_list.append(rmse_t)
        self.cd_list.append(shape_cd)
        self.perc_mp_list.append(perc_mp)
        
        self._save_inference_data(data_dict, all_pred_trans_rots[top_k_IDs.item()].unsqueeze(1), acc, idx, top_k_IDs.item())


    def test_step(self, data_dict, idx):
        start_time = time.perf_counter()

        if self.cfg.denoiser.max_iters == 1:
            self.test_denoiser_only(data_dict)
            elapsed = time.perf_counter() - start_time
            self.inference_times.extend([elapsed])
            return
        
        if self.cfg.denoiser.sampling:
            if self.cfg.denoiser.discriminator_data_creation:
                self.test_discriminator_data_creation(data_dict, idx)
            elif self.cfg.denoiser.discriminator_sampling:
                self.test_discriminator_sampling(data_dict, idx)
                elapsed = time.perf_counter() - start_time
                self.inference_times.extend([elapsed])
            else:
                self.test_sampling(data_dict, idx)
                elapsed = time.perf_counter() - start_time
                self.inference_times.extend([elapsed])
            return


    def _save_inference_discriminator_data(self, save_dict=None):

        # save the training and val data for the discriminator
        os.makedirs(self.cfg.denoiser.disc_train_path, exist_ok=True)
        os.makedirs(self.cfg.denoiser.disc_val_path, exist_ok=True)
        if save_dict is not None:
            pred_pcs = save_dict['pred_pcs']
            pred_normals = save_dict['pred_normals']
            pred_locdists = save_dict['pred_locdists']
            gt_locdists = save_dict['gt_locdists']
            pred_metrics_part_acc = save_dict['pred_metrics']['part_acc']
            pred_metrics_shape_cd = save_dict['pred_metrics']['shape_cd']
            pred_metrics_rmse_r = save_dict['pred_metrics']['rmse_r']
            pred_metrics_rmse_t = save_dict['pred_metrics']['rmse_t']
            path = self.cfg.denoiser.disc_val_path if self.npz_save_counter <= 6000 else self.cfg.denoiser.disc_train_path
            counter = self.npz_save_counter if self.npz_save_counter <= 6000 else self.npz_save_counter - 6000

            np.savez(
                os.path.join(path, f'{counter:05}.npz'),

                gt_locdists=gt_locdists.cpu().numpy(),
                pred_pcs=pred_pcs.cpu().numpy(),
                pred_normals=pred_normals.cpu().numpy(),
                pred_locdists=pred_locdists.cpu().numpy(),
                pred_metrics_part_acc=pred_metrics_part_acc.cpu().numpy(),
                pred_metrics_shape_cd=pred_metrics_shape_cd.cpu().numpy(),
                pred_metrics_rmse_r=pred_metrics_rmse_r.cpu().numpy(),
                pred_metrics_rmse_t=pred_metrics_rmse_t.cpu().numpy(),
            )

            self.npz_save_counter += 1


    def _save_inference_data(self, data_dict, pred_trans_rots, acc, idx=0, top_k_ID=None):
        T, B, _, _ = pred_trans_rots.shape

        for i in range(B):
            if top_k_ID is not None:
                i = top_k_ID
            save_dir = os.path.join(
                self.cfg.experiment_output_path,
                "inference", 
                self.cfg.inference_dir, 
                str(data_dict['data_id'][i].item()), #f"{data_dict['data_id'][i].item()}_{idx}",
            )

            os.makedirs(save_dir, exist_ok=True)
            c_trans_rots = pred_trans_rots[:, 0, ...]
            mask = data_dict["part_valids"][i] == 1
            np.save(os.path.join(save_dir, f"predict_{acc[0]}.npy"), c_trans_rots)
            gt_transformation = torch.cat(
                [data_dict["part_trans"][i],
                    data_dict["part_rots"][i]], dim=-1
            )[mask]

            np.save(os.path.join(
                save_dir, "gt.npy"),
                gt_transformation.cpu().numpy()
            )

            init_pose_r = data_dict["init_pose_r"][i]
            init_pose_t = data_dict["init_pose_t"][i]
            init_pose = torch.cat([init_pose_t, init_pose_r], dim=-1)
            np.save(os.path.join(
                save_dir, "init_pose.npy"),
                init_pose.cpu().numpy()
            )

            with open(os.path.join(save_dir, "mesh_file_path.txt"), "w") as f:
                f.write(data_dict["mesh_file_path"][i])

    
    def on_test_start(self):
        self.inference_times = []
     
    def on_test_epoch_end(self):
        times = np.array(self.inference_times)
        mean = times.mean()
        std = times.std()
        sem = std / np.sqrt(len(times)) if len(times) > 1 else 0.0

        print(f"\n🔍 Inference Timing Stats:")
        print(f"   Mean time per sample: {mean:.6f} s")
        print(f"   Std dev:              {std:.6f} s")
        print(f"   SEM (standard error): {sem:.6f} s\n")
        self.log(f"eval/mean_speed", mean, sync_dist=True)
        self.log(f"eval/std_speed", std, sync_dist=True)
        self.log(f"eval/sem_speed", sem, sync_dist=True)

        acc_vals = torch.cat(self.acc_list)
        rmse_t_vals = torch.cat(self.rmse_t_list)
        rmse_r_vals = torch.cat(self.rmse_r_list)
        shape_cd_vals = torch.cat(self.cd_list)
        if not len(self.perc_mp_list) == 0:
            perc_mp_vals = torch.cat(self.perc_mp_list)
        if self.cfg.denoiser.discriminator_sampling:
            coorect_topk = torch.cat(self.correct_topk)

        total_acc = torch.mean(acc_vals)
        total_rmse_t = torch.mean(rmse_t_vals)
        total_rmse_r = torch.mean(rmse_r_vals)
        total_shape_cd = torch.mean(shape_cd_vals)
        if not len(self.perc_mp_list) == 0:
            total_perc_mp = torch.mean(perc_mp_vals)
        if self.cfg.denoiser.discriminator_sampling:
            total_correct_topk = torch.mean(coorect_topk)

        std_acc = torch.std(acc_vals, unbiased=True)
        std_rmse_t = torch.std(rmse_t_vals, unbiased=True)
        std_rmse_r = torch.std(rmse_r_vals, unbiased=True)
        std_shape_cd = torch.std(shape_cd_vals, unbiased=True)
        if not len(self.perc_mp_list) == 0:
            std_perc_mp = torch.std(perc_mp_vals, unbiased=True)
        if self.cfg.denoiser.discriminator_sampling:
            std_correct_topk = torch.std(coorect_topk, unbiased=True)

        sem_acc = std_acc / np.sqrt(len(self.acc_list))
        sem_rmse_t = std_rmse_t / np.sqrt(len(self.rmse_t_list))
        sem_rmse_r = std_rmse_r / np.sqrt(len(self.rmse_r_list))
        sem_shape_cd = std_shape_cd / np.sqrt(len(self.cd_list))
        if not len(self.perc_mp_list) == 0:
            sem_perc_mp = std_perc_mp / np.sqrt(len(self.perc_mp_list))
        if self.cfg.denoiser.discriminator_sampling:
            sem_correct_topk = std_correct_topk / np.sqrt(len(self.correct_topk))
       
        self.log(f"eval/part_acc", total_acc, sync_dist=True)
        self.log(f"eval/rmse_t", total_rmse_t, sync_dist=True)
        self.log(f"eval/rmse_r", total_rmse_r, sync_dist=True)
        self.log(f"eval/shape_cd", total_shape_cd, sync_dist=True)
        if not len(self.perc_mp_list) == 0:
            self.log(f"eval/perc_mp", total_perc_mp, sync_dist=True)
        if self.cfg.denoiser.discriminator_sampling:
            self.log(f"eval/correct_topk", total_correct_topk, sync_dist=True)

        self.log(f"eval/std_part_acc", std_acc, sync_dist=True)
        self.log(f"eval/std_rmse_t", std_rmse_t, sync_dist=True)
        self.log(f"eval/std_rmse_r", std_rmse_r, sync_dist=True)
        self.log(f"eval/std_shape_cd", std_shape_cd, sync_dist=True)
        if not len(self.perc_mp_list) == 0:
            self.log(f"eval/std_perc_mp", std_perc_mp, sync_dist=True)
        if self.cfg.denoiser.discriminator_sampling:
            self.log(f"eval/std_correct_topk", std_correct_topk, sync_dist=True)

        self.log(f"eval/sem_part_acc", sem_acc, sync_dist=True)
        self.log(f"eval/sem_rmse_t", sem_rmse_t, sync_dist=True)
        self.log(f"eval/sem_rmse_r", sem_rmse_r, sync_dist=True)
        self.log(f"eval/sem_shape_cd", sem_shape_cd, sync_dist=True)
        if not len(self.perc_mp_list) == 0:
            self.log(f"eval/sem_perc_mp", sem_perc_mp, sync_dist=True)
        if self.cfg.denoiser.discriminator_sampling:
            self.log(f"eval/sem_correct_topk", sem_correct_topk, sync_dist=True)
        
        self.acc_list = []
        self.rmse_t_list = []
        self.rmse_r_list = []
        self.cd_list = []
        if not len(self.perc_mp_list) == 0:
            self.perc_mp_list = []
        if self.cfg.denoiser.discriminator_sampling:
            self.correct_topk = []


        if not len(self.perc_mp_list) == 0:
            return total_acc, total_rmse_t, total_rmse_r, total_shape_cd, total_perc_mp
        if self.cfg.denoiser.discriminator_sampling:
            return total_acc, total_rmse_t, total_rmse_r, total_shape_cd, total_correct_topk
        return total_acc, total_rmse_t, total_rmse_r, total_shape_cd

    def _get_edge_mask(self, num_parts, P):
        B = num_parts.shape[0]
        nodes = range(P)
        edges = list(itertools.combinations(nodes, 2))
        edges = torch.tensor(edges, dtype=torch.int32, device=self.device)
        edges = edges.unsqueeze(0).expand(B, -1, -1)
        mask = (edges[:, :, 0] < num_parts.unsqueeze(1)) & (edges[:, :, 1] < num_parts.unsqueeze(1))
        return mask
    
    def _make_cd_to_bins(self, cd):
        bins = torch.tensor([0.0, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 100], device=self.device)
        bin_indices = torch.bucketize(cd.squeeze(0), bins, right=True)
        counts = torch.bincount(bin_indices, minlength=bins.numel())
        return counts[1:7]

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
