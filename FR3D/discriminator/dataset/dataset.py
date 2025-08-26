import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from tqdm import tqdm
import copy
import random


class GeometryPartDataset(Dataset):

    def __init__(
        self,
        cfg,
        data_dir,
        overfit=-1,
        train=True,
    ):
        self.cfg = cfg
        self.data_dir = data_dir
        self.data_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.npz')])

        self.max_num_part = cfg.data.max_num_part
        self.min_num_part = cfg.data.min_num_part

        if overfit != -1: 
            #self.data_files = self.data_files[:overfit]
            overfit = min(overfit, len(self.data_files))
            random_idx = random.sample(range(len(self.data_files)), overfit)
            self.data_files = [self.data_files[i] for i in random_idx]
        
        self.data_list = []

        for file_name in tqdm(self.data_files):
            data_dict = np.load(os.path.join(self.data_dir, file_name))

            gt_locdists = data_dict['gt_locdists']  # [B_inst, P, N, 1]
            pred_pcs = data_dict['pred_pcs'] # [B_inst, P, N, 3]

            pred_metrics_part_acc = data_dict['pred_metrics_part_acc'] # [B_inst, 3] raw; norm 0,1 ; scaled -1,1, is already 1 - acc
            pred_metrics_shape_cd = data_dict["pred_metrics_shape_cd"]
            pred_metrics_rmse_r = data_dict['pred_metrics_rmse_r']
            pred_metrics_rmse_t = data_dict['pred_metrics_rmse_t']

            # include just the harder samples where raw metrics really differ largely
            targets = np.concatenate([
                pred_metrics_part_acc[..., 0:1], 
                pred_metrics_shape_cd[..., 0:1], 
                pred_metrics_rmse_r[..., 0:1], 
                pred_metrics_rmse_t[..., 0:1]
            ], axis=-1)

            if train and not np.any(targets[:,0] != 0):
                continue

            agg_metric = (
                pred_metrics_part_acc[..., 1:2] +
                pred_metrics_shape_cd[..., 1:2] +
                pred_metrics_rmse_r[..., 1:2] +
                pred_metrics_rmse_t[..., 1:2]
            )
            sorted_indices = np.argsort(agg_metric.flatten())

            labels = np.empty_like(agg_metric, dtype=int)
            num_classes = 4
            for i in range(0, len(agg_metric), num_classes):
                group_label = i // num_classes
                group_indices = sorted_indices[i:i + num_classes]
                labels[group_indices] = group_label
            
            for b in range(pred_pcs.shape[0]):
                gt_locdists_b = gt_locdists[b]  # [P, N, 1]
                pred_pcs_b = pred_pcs[b]  # [P, N, 3]

                pred_metrics_part_acc_b = pred_metrics_part_acc[b]  # [3]
                pred_metrics_shape_cd_b = pred_metrics_shape_cd[b]  # [1]
                pred_metrics_rmse_r_b = pred_metrics_rmse_r[b]  # [1]
                pred_metrics_rmse_t_b = pred_metrics_rmse_t[b]  # [1]

                labels_b = labels[b]  # [1]

                sample = {
                    'gt_locdists': gt_locdists_b,
                    'pred_pcs': pred_pcs_b,
                    'pred_metrics_part_acc': pred_metrics_part_acc_b,
                    'pred_metrics_shape_cd': pred_metrics_shape_cd_b,
                    'pred_metrics_rmse_r': pred_metrics_rmse_r_b,
                    'pred_metrics_rmse_t': pred_metrics_rmse_t_b,
                    'file_name': file_name,
                    'pred_labels': labels_b,
                }

                self.data_list.append(sample)

    @staticmethod
    def _recenter_pc(pc):
        """pc: [N, 3]"""
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid[None]
        return pc, centroid

    @staticmethod
    def _rotate_pc(pc, normal=None):
        """pc: [N, 3]"""
        rot_mat = R.random().as_matrix()
        rotated_pc = (rot_mat @ pc.T).T
        quat_gt = R.from_matrix(rot_mat.T).as_quat()
        # we use scalar-first quaternion
        quat_gt = quat_gt[[3, 0, 1, 2]]
        if normal is None:
            return rotated_pc, None, quat_gt
        rotated_normal = (rot_mat @ normal.T).T
        return rotated_pc, rotated_normal, quat_gt
    
    @staticmethod
    def shuffle_pc(pc, normal=None, locdists=None):
        """pc: [N, 3]"""
        order = np.arange(pc.shape[0])
        random.shuffle(order)
        shuffled_pc = pc[order]
        if normal is None and locdists is None:
            return shuffled_pc, None, None, order
        if normal is not None and locdists is None:
            shuffled_normal = normal[order]
            return shuffled_pc, shuffled_normal, None, order
        if normal is None and locdists is not None:
            shuffled_locdists = locdists[order]
            return shuffled_pc, None, shuffled_locdists, None
        shuffled_normal = normal[order]
        shuffled_locdists = locdists[order]
        return shuffled_pc, shuffled_normal, shuffled_locdists, order


    def _pad_data(self, data, num_points):
        """Pad data to shape [`self.max_num_part * num_points`, ...]."""
        data = np.array(data)
        pad_shape = (self.max_num_part * num_points, ) + tuple(data.shape[1:])
        pad_data = np.zeros(pad_shape, dtype=np.float32)
        pad_data[:data.shape[0]] = data
        return pad_data


    def __getitem__(self, idx):
        data_dict = copy.deepcopy(self.data_list[idx])

        gt_locdists = data_dict['gt_locdists']  # [P, N, 1]
        pred_pcs = data_dict['pred_pcs'] # [P, N, 3]

        pred_metrics_part_acc = data_dict['pred_metrics_part_acc'][...,1:2]
        pred_metrics_shape_cd = data_dict['pred_metrics_shape_cd'][...,1:2]
        pred_metrics_rmse_r = data_dict['pred_metrics_rmse_r'][...,1:2]
        pred_metrics_rmse_t = data_dict['pred_metrics_rmse_t'][...,1:2]

        agg_metric = (
            pred_metrics_part_acc +
            pred_metrics_shape_cd +
            pred_metrics_rmse_r +
            pred_metrics_rmse_t
        ) / 4.0

        data_dict['agg_target_metric'] = agg_metric

        num_parts, num_points, _ = pred_pcs.shape

        pcs = pred_pcs.reshape(-1, 3) # [P*N, 3]
        gt_locdists = gt_locdists.reshape(-1, 1) # [P*N, 1]

        pcs, _ = self._recenter_pc(pcs)
        pcs, _, _ = self._rotate_pc(pcs, None)
        pcs, _, gt_locdists, _ = self.shuffle_pc(pcs, None, gt_locdists)

        pcs = self._pad_data(pcs, 1000)  # [20000, 3]
        gt_locdists = self._pad_data(gt_locdists, 1000)

        scale = np.max(np.abs(pcs), axis=(0,1), keepdims=True)
        scale[scale == 0] = 1
        pcs /= scale
        
        data_dict['pred_pcs'] = pcs
        data_dict['gt_locdists'] = gt_locdists
        data_dict['num_points'] = num_parts * num_points

        return data_dict
        

    def __len__(self):
        return len(self.data_list)

def build_geometry_dataloader(cfg):
    data_dict = dict(
        cfg=cfg,
        data_dir=cfg.data.disc_train_dir,
        overfit=cfg.data.overfit,
        train=True,
    )
    train_set = GeometryPartDataset(**data_dict)
    data_dict['data_dir'] = cfg.data.disc_val_dir
    data_dict['train'] = False
    val_set = GeometryPartDataset(**data_dict)

    return train_set, val_set
