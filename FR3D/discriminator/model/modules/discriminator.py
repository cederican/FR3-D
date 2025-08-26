import torch
import torch.nn as nn
from PointTransformerV3.model import PointTransformerV3
from torchmetrics.retrieval import RetrievalNormalizedDCG, RetrievalMAP
from collections import defaultdict


class Discriminator(nn.Module):
    def __init__(self, cfg):
        super(Discriminator, self).__init__()

        self.ptv3 = PointTransformerV3(cfg)
        self.encoder = self.ptv3.encode
        self.decoder = self.ptv3.decode

        self.cfg = cfg

        self.mae_loss = nn.L1Loss()
        self.ranking_loss = grouped_pairwise_ranking_loss

        self.normalizedDCG = RetrievalNormalizedDCG(top_k=20)
        self.map = RetrievalMAP(top_k=20)

    def forward(self, data_dict):
        
        coords = data_dict['pred_pcs'] # BN, 3
        gt_locdists = data_dict['gt_locdists'] # BN, 1
        offset = torch.cumsum(data_dict['num_points'], dim=0).to(coords.device)
        ptv3_data_dict = {
            "feat": torch.cat([coords, gt_locdists], dim=-1),
            "coord": coords,
            "grid_size": self.cfg.ae_disc.grid_size, 
            "offset": offset
        }
        ptv3_data_dict = self.encoder(ptv3_data_dict)
        output_dict = {
            "z_e": ptv3_data_dict.feat,
            "metric_pred": ptv3_data_dict.metric_pred,
        }

        return output_dict

    def loss(self, data_dict, output_dict):
        loss_dict = {}        

        metric_pred = output_dict["metric_pred"].squeeze(-1)
        metric_gt = data_dict["agg_target_metric"].squeeze(-1)

        embeds = output_dict["z_e"]

        reg_loss = self.mae_loss(metric_pred, metric_gt)
        ranking_loss, _ = self.ranking_loss(metric_pred, metric_gt, data_dict['file_name'], margin=0.2)

        # logging 
        if not self.training:
            with torch.no_grad():
                embeds_flat = embeds.reshape(-1, embeds.shape[-1])  # [B*N, D]
                dists = torch.cdist(embeds_flat, embeds_flat)  # [B*N, B*N]
                mean_dist = dists[~torch.eye(dists.size(0), dtype=bool, device=dists.device)].mean().item()
                loss_dict['z_mean_embed_dist'] = torch.tensor(mean_dist, device=metric_pred.device)

                pred_topk_ids, gt_topk_ids = s_topk(metric_pred, metric_gt, k=1)
                topk_matches = torch.isin(pred_topk_ids, gt_topk_ids).sum().item()
                loss_dict['agg_top1_matches'] = torch.tensor(topk_matches, device=metric_pred.device)

                pred_topk_ids, gt_topk_ids = s_topk(metric_pred, metric_gt, k=3)
                topk_matches = torch.isin(pred_topk_ids, gt_topk_ids).sum().item()
                loss_dict['agg_top3_matches'] = torch.tensor(topk_matches / 3, device=metric_pred.device)

                pred_topk_ids, gt_topk_ids = s_topk(metric_pred, metric_gt, k=3, k_gt=1)
                topk_matches = torch.isin(pred_topk_ids, gt_topk_ids).sum().item()
                loss_dict['agg_top3_intop1gt_acc'] = torch.tensor(topk_matches, device=metric_pred.device)

                pred_topk_ids, gt_topk_ids = s_topk(metric_pred, metric_gt, k=3, k_gt=5)
                topk_matches = torch.isin(pred_topk_ids, gt_topk_ids).sum().item()
                loss_dict['agg_top3_intop5gt_acc'] = torch.tensor(topk_matches / 3, device=metric_pred.device)

                pred_topk_ids, gt_topk_ids = s_topk(metric_pred, metric_gt, k=3, k_gt=10)
                topk_matches = torch.isin(pred_topk_ids, gt_topk_ids).sum().item()
                loss_dict['agg_top3_intop10gt_acc'] = torch.tensor(topk_matches / 3, device=metric_pred.device)

                # create histogram of pred topk at which position are they in gt data
                pred_topk_ids, gt_topk_ids = s_topk(metric_pred, metric_gt, k=3, k_gt=20)
                pred_exp = pred_topk_ids[:, None]  
                gt_exp = gt_topk_ids[None, :]     
                matches = pred_exp == gt_exp
                positions = matches.float().argmax(dim=1)
                has_match = matches.any(dim=1)
                positions[~has_match] = -1

                loss_dict['top3_valid_positions'] = positions.detach()
                loss_dict['top1_valid_positions'] = positions[0].unsqueeze(0).detach()

                indexes = torch.zeros_like(metric_pred, dtype=torch.int64, device=metric_pred.device)
                ndcg_score = self.normalizedDCG(1 - metric_pred, 1 - metric_gt, indexes)
                map_score = self.map(1 - metric_pred, metric_gt < 0.2, indexes)
                loss_dict['agg_ndcg'] = torch.tensor(ndcg_score, device=metric_pred.device)
                loss_dict['agg_map'] = torch.tensor(map_score, device=metric_pred.device)

        loss_dict['reg_loss'] = reg_loss
        loss_dict['ranking_loss'] = ranking_loss

        return loss_dict

def s_topk(pred, gt, k=1, k_gt=None):
    if k_gt is None:
        k_gt = k
    pred_topk_ids = torch.topk(pred, k=k, largest=False).indices
    gt_topk_ids = torch.topk(gt, k=k_gt, largest=False).indices

    return pred_topk_ids, gt_topk_ids


def multi_metric_pairwise_ranking_loss(preds, targets, margin: float = 0.1):
    # Create all pairwise differences
    diff_pred = preds.unsqueeze(1) - preds.unsqueeze(0)     # shape [B, B]
    diff_target = targets.unsqueeze(1) - targets.unsqueeze(0)  # shape [B, B]

    # Mask: where target[i] < target[j] → i should be ranked better than j
    mask = diff_target < 0  # boolean mask [B, B]

    # Apply hinge loss only on valid pairs
    loss_matrix = torch.clamp(diff_pred + margin, min=0.0)

    # Mask the irrelevant elements
    masked_loss = loss_matrix * mask.float()

    # Average over number of valid pairs
    num_pairs = mask.sum().clamp(min=1.0)
    loss = masked_loss.sum() / num_pairs

    return loss


def grouped_pairwise_ranking_loss(preds, targets, file_names, margin: float = 0.1):
   
    file_to_indices = defaultdict(list)
    for i, fname in enumerate(file_names):
        file_to_indices[fname].append(i)

    group_losses = []
    for fname, idxs in file_to_indices.items():
        idxs_tensor = torch.tensor(idxs, device=preds.device)
        group_preds = preds[idxs_tensor]
        group_targets = targets[idxs_tensor]

        loss = multi_metric_pairwise_ranking_loss(group_preds.unsqueeze(-1), group_targets.unsqueeze(-1), margin)
        group_losses.append(loss)

    return torch.stack(group_losses).mean(), None

