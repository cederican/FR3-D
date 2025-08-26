import torch
import torch.nn.functional as F
import numpy as np
from FR3D.denoiser.evaluation.transform import (
    transform_pc,
    quaternion_to_euler,
)
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree


def _valid_mean(loss_per_part, valids):
    """Average loss values according to the valid parts.

    Args:
        loss_per_part: [B, P]
        valids: [B, P], 1 for input parts, 0 for padded parts

    Returns:
        [B], loss per data in the batch, averaged over valid parts
    """
    nan_mask = torch.isnan(loss_per_part)
    loss_per_part[nan_mask] = 0.
    valids = valids.float().detach()
    loss_per_data = (loss_per_part * valids).sum(1) / valids.sum(1)
    return loss_per_data


def trans_metrics(trans1, trans2, valids, metric):
    """Evaluation metrics for translation.

    Metrics used in the NSM paper.

    Args:
        trans1: [B, P, 3], pred translation
        trans2: [B, P, 3], gt translation
        valids: [B, P], 1 for input parts, 0 for padded parts
        metric: str, 'mse', 'rmse' or 'mae'

    Returns:
        [B], metric per data in the batch
    """
    assert metric in ['mse', 'rmse', 'mae']
    if metric == 'mse':
        metric_per_data = (trans1 - trans2).pow(2).mean(dim=-1)  # [B, P]
    elif metric == 'rmse':
        metric_per_data = (trans1 - trans2).pow(2).mean(dim=-1)**0.5
    else:
        metric_per_data = (trans1 - trans2).abs().mean(dim=-1)
    metric_per_data = _valid_mean(metric_per_data, valids)
    return metric_per_data


@torch.no_grad()
def rot_metrics(rot1, rot2, valids, metric):
    """Evaluation metrics for rotation in euler angle (degree) space.

    Metrics used in the NSM paper.

    Args:
        rot1: [B, P, 4], pred quat
        rot2: [B, P, 4], gt quat
        valids: [B, P], 1 for input parts, 0 for padded parts
        metric: str, 'mse', 'rmse' or 'mae'

    Returns:
        [B], metric per data in the batch
    """
    assert metric in ['mse', 'rmse', 'mae']
    deg1 = quaternion_to_euler(rot1, to_degree=True)  # [B, P, 3]
    deg2 = quaternion_to_euler(rot2, to_degree=True)

    diff1 = (deg1 - deg2).abs()
    diff2 = 360. - (deg1 - deg2).abs()
    # since euler angle has the discontinuity at 180
    diff = torch.minimum(diff1, diff2)
    if metric == 'mse':
        metric_per_data = diff.pow(2).mean(dim=-1)  # [B, P]
    elif metric == 'rmse':
        metric_per_data = diff.pow(2).mean(dim=-1)**0.5
    else:
        metric_per_data = diff.abs().mean(dim=-1)
    metric_per_data = _valid_mean(metric_per_data, valids)
    return metric_per_data


@torch.no_grad()
def calc_part_acc(pts, trans1, trans2, rot1, rot2, valids, chamfer_distance):
    """Compute the `Part Accuracy` in the paper.

    We compute the per-part chamfer distance, and the distance lower than a
        threshold will be considered as correct.

    Args:
        pts: [B, P, N, 3], model input point cloud to be transformed
        trans1: [B, P, 3], pred_translation
        trans2: [B, P, 3], gt_translation
        rot1: [B, P, 4], Rotation3D, quat or rmat
        rot2: [B, P, 4], Rotation3D, quat or rmat
        valids: [B, P], 1 for input parts, 0 for padded parts

    Returns:
        [B], accuracy per data in the batch
    """
    B, P = pts.shape[:2]

    pts1 = transform_pc(trans1, rot1, pts)  # [B, P, N, 3]
    pts2 = transform_pc(trans2, rot2, pts)

    pts1 = pts1.flatten(0, 1)  # [B*P, N, 3]
    pts2 = pts2.flatten(0, 1)
    loss_per_data = chamfer_distance(pts1, pts2, bidirectional=True, 
                                    point_reduction="mean", batch_reduction=None,)  # [B*P, N]
    loss_per_data = loss_per_data.view(B, P).type_as(pts)

    # part with CD < `thre` is considered correct
    thre = 0.01
    acc_per_part = (loss_per_data < thre) & (valids == 1)
    # the official code is doing avg per-shape acc (not per-part)
    acc = acc_per_part.sum(-1) / (valids == 1).sum(-1)
    return acc, acc_per_part, loss_per_data


@torch.no_grad()
def calc_shape_cd(pts, trans1, trans2, rot1, rot2, valids, chamfer_distance):
    
    B, P, N, _ = pts.shape
    
    valid_mask = valids[..., None, None]  # [B, P, 1, 1]
    
    pts = pts.detach().clone()
    
    pts = pts.masked_fill(valid_mask == 0, 1e3)
    
    pts1 = transform_pc(trans1, rot1, pts)  # [B, P, N, 3]
    pts2 = transform_pc(trans2, rot2, pts)
    
    shape1 = pts1.flatten(1, 2)
    shape2 = pts2.flatten(1, 2)
    
    shape_cd = chamfer_distance(
        shape1, 
        shape2, 
        bidirectional=True, 
        point_reduction=None, 
        batch_reduction=None
    )
    
    shape_cd = shape_cd.view(B, P, N).mean(-1)
    shape_cd = _valid_mean(shape_cd, valids)
    
    return shape_cd


@torch.no_grad()
def discriminator_topK(
        metric_pred, # B, 1
        metric_gt, # B, 1
        k=1,
        architecture="ptv3", # "ptv3", "pn2"
):      
        pred = metric_pred.squeeze(-1)
        gt = metric_gt.squeeze(-1)

        pred_topk_ids = torch.topk(pred, k=k, largest=False).indices
        gt_topk_ids = torch.topk(gt, k=k, largest=False).indices

        mask_topk = gt < 0.2
        mask_topk_ids = torch.nonzero(mask_topk, as_tuple=False)
        mask_topk_matches = torch.isin(pred_topk_ids, mask_topk_ids).sum().item()
        mask_ratio = mask_topk_matches / k
        
        return pred_topk_ids, gt_topk_ids, mask_ratio
    
  
@torch.no_grad()
def search_top_k(acc = None, shape_cd = None, rmse_r = None, rmse_t = None, perc_mp = None, facs = None, k = None, maximizeMetric=None):
    if maximizeMetric == 'facs':
        top_k_index = torch.zeros_like(facs, dtype=torch.bool)
        min_val = facs.min()
        min_indices = (facs == min_val).nonzero(as_tuple=False)
        if len(min_indices) > 0:
            chosen_idx = min_indices[torch.randint(len(min_indices), (1,))]
            top_k_index[chosen_idx[0]] = True
            best_k_IDs = chosen_idx[0]
        return best_k_IDs, None

    if maximizeMetric == 'perc_mp':
        top_k_index = torch.zeros_like(perc_mp, dtype=torch.bool)
        max_val = perc_mp.max()
        max_indices = (perc_mp == max_val).nonzero(as_tuple=False)

        if len(max_indices) > 0:
            chosen_idx = max_indices[torch.randint(len(max_indices), (1,))]
            top_k_index[chosen_idx[0]] = True
            best_k_IDs = chosen_idx[0]

        return best_k_IDs, None

    else:
        if maximizeMetric is not None:
            if maximizeMetric == 'part_acc':
                top_k_index = torch.zeros_like(acc, dtype=torch.bool)
                top_k_index[acc == acc.max()] = True
                metrics = torch.stack([rmse_t, rmse_r, shape_cd], dim=1)
            elif maximizeMetric == 'shape_cd':
                top_k_index = torch.zeros_like(shape_cd, dtype=torch.bool)
                top_k_index[shape_cd == shape_cd.min()] = True
                metrics = torch.stack([1 - acc, rmse_t, rmse_r], dim=1)
            elif maximizeMetric == 'rmse_r':
                top_k_index = torch.zeros_like(rmse_r, dtype=torch.bool)
                top_k_index[rmse_r == rmse_r.min()] = True
                metrics = torch.stack([1 - acc, rmse_t, shape_cd], dim=1)
            elif maximizeMetric == 'rmse_t':
                top_k_index = torch.zeros_like(rmse_t, dtype=torch.bool)
                top_k_index[rmse_t == rmse_t.min()] = True
                metrics = torch.stack([1 - acc, rmse_r, shape_cd], dim=1)
            elif maximizeMetric == 'all':
                metrics = torch.stack([1 - acc, rmse_t, rmse_r, shape_cd], dim=1) # B,4

        norm_metrics = torch.zeros_like(metrics)

        for col in range(metrics.shape[1]):
            min_val = metrics[:, col].min() if maximizeMetric == 'all' else metrics[top_k_index, col].min()
            max_val = metrics[:, col].max() if maximizeMetric == 'all' else metrics[top_k_index, col].max()

            norm_metrics[:, col] = (metrics[:, col] - min_val) / (max_val - min_val + 1e-8)

        norm_score = norm_metrics.mean(dim=1)
        #norm_score[~top_k_index] = 1e10
        best_k_IDs = torch.topk(norm_score, k=k, largest=False).indices
        bad_mask = torch.ones_like(norm_score, dtype=torch.bool)
        bad_mask[best_k_IDs] = False
        bad_IDs = bad_mask.nonzero(as_tuple=True)[0]

        return best_k_IDs, bad_IDs

def calculate_pred_locdists(
        pred_pcs, # [P, N, 3]
        pred_normals=None, # [P, N, 3]
    ):

    INTERFACE_THRESHOLD = 0.05
    device = pred_pcs.device
    pred_pcs = pred_pcs.cpu().numpy()
    pred_normals = pred_normals.cpu().numpy() if pred_normals is not None else None
    P, N, _ = pred_pcs.shape

    def compute_interface_values(points, all_other_points_tree):
        dists, indices = all_other_points_tree.query(points, k=1)
        values = np.clip(1.0 - (dists / INTERFACE_THRESHOLD), 0.0, 1.0)
        return values, indices

    interface_scores = np.zeros((P, N), dtype=np.float32)
    alignment_scores = []

    for p in range(P):
        points = pred_pcs[p]  # (N, 3)
        normals = pred_normals[p] if pred_normals is not None else None

        all_other_points = np.vstack([pred_pcs[q] for q in range(P) if q != p])
        all_other_normals = np.vstack([pred_normals[q] for q in range(P) if q != p]) if pred_normals is not None else None
        tree = cKDTree(all_other_points)

        interface_vals, nn_inidices = compute_interface_values(points, tree)

        matched_normals = all_other_normals[nn_inidices] if all_other_normals is not None else None
        cos_sims = np.einsum('ij,ij->i', normals, matched_normals) if all_other_normals is not None else None
        normal_alignment = 1.0 - np.abs(cos_sims) if all_other_normals is not None else None

        pointwise_facs = interface_scores * normal_alignment if all_other_normals is not None else None
        alignment_scores.append(pointwise_facs) if all_other_normals is not None else None

        interface_scores[p] = interface_vals

    interface_scores = interface_scores[..., None]

    return torch.tensor(interface_scores, dtype=torch.float32, device=device)
      


        