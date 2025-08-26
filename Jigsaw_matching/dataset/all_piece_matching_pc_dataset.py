""" Use pre-processed point cloud data for training. """

import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import copy
import random

class AllPieceMatchingPCDataset(Dataset):

    def __init__(
            self,
            cfg,
            data_dir,
            data_fn,
            category='',
            rot_range=-1,
            overfit=-1,
            min_num_part=2,
            max_num_part=20,
    ):
        self.cfg = cfg
        self.category = category if category.lower() != 'all' else ''
        self.data_dir = data_dir
        self.data_fn = data_fn

        self.data_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.npz')])

        self.max_num_part = max_num_part
        self.min_num_part = min_num_part

        if overfit != -1:
            self.data_files = self.data_files[:overfit]

        self.data_list = []
        self.rot_range = rot_range

        for file_name in tqdm(self.data_files):
            data_dict = np.load(os.path.join(self.data_dir, file_name))

            data_id = data_dict['data_id'].item()
            part_valids = data_dict['part_valids']
            num_parts = data_dict["num_parts"].item()
            mesh_file_path = data_dict['mesh_file_path'].item()
            #part_pcs = data_dict['part_pcs']
            gt_pcs = data_dict['gt_pcs']
            #part_quat = data_dict['part_quat']
            #part_trans = data_dict['part_trans']
            n_pcs = data_dict['n_pcs']
            critical_label_thresholds = data_dict['critical_label_thresholds']

            sample = {
                'data_id': data_id,
                'part_valids': part_valids,
                'num_parts': num_parts,
                'mesh_file_path': mesh_file_path,
                #'part_pcs': part_pcs,
                'gt_pcs': gt_pcs,
                #'part_quat': part_quat,
                #'part_trans': part_trans,
                'n_pcs': n_pcs,
                'critical_label_thresholds': critical_label_thresholds,
            }

            if num_parts > self.max_num_part or num_parts < self.min_num_part:
                continue

            self.data_list.append(sample)
    
    @staticmethod
    def _shuffle_pc(pc, pc_gt):
        """pc: [N, 3]"""
        order = np.arange(pc.shape[0])
        random.shuffle(order)
        pc = pc[order]
        pc_gt = pc_gt[order]
        return pc, pc_gt
    
    def _pad_data(self, data, pad_size=None):
        """Pad data to shape [`self.max_num_part`, data.shape[1], ...]."""
        if pad_size is None:
            pad_size = self.max_num_part
        data = np.array(data)
        if len(data.shape) > 1:
            pad_shape = (pad_size,) + tuple(data.shape[1:])
        else:
            pad_shape = (pad_size,)
        pad_data = np.zeros(pad_shape, dtype=data.dtype)
        pad_data[: data.shape[0]] = data
        return pad_data
    
    @staticmethod
    def _recenter_pc(pc):
        """pc: [N, 3]"""
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid[None]
        return pc, centroid

    @staticmethod
    def _rotate_pc(pc):
        """pc: [N, 3]"""
        rot_mat = R.random().as_matrix()
        pc = (rot_mat @ pc.T).T
        quat_gt = R.from_matrix(rot_mat.T).as_quat()
        # we use scalar-first quaternion
        quat_gt = quat_gt[[3, 0, 1, 2]]
        return pc, quat_gt

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        data_dict = copy.deepcopy(self.data_list[index])
        num_parts = data_dict['num_parts']
        n_pcs = data_dict['n_pcs']
        pcs = data_dict['gt_pcs']
        pcs = np.split(pcs, np.cumsum(n_pcs)[:num_parts-1])
        cur_pts, cur_quat, cur_trans, cur_pts_gt = [], [], [], []
        for i in range(num_parts):
            pc = pcs[i]
            pc_gt = pc.copy()
            pc, gt_trans = self._recenter_pc(pc)
            pc, gt_quat = self._rotate_pc(pc)
            pc_shuffle, pc_gt_shuffle = self._shuffle_pc(pc, pc_gt)

            cur_pts.append(pc_shuffle)
            cur_quat.append(gt_quat)
            cur_trans.append(gt_trans)
            cur_pts_gt.append(pc_gt_shuffle)
        
        cur_pts = np.concatenate(cur_pts).astype(np.float32) 
        cur_pts_gt = np.concatenate(cur_pts_gt).astype(np.float32)
        cur_quat = self._pad_data(np.stack(cur_quat, axis=0), self.max_num_part).astype(np.float32)  # [P, 4]
        cur_trans = self._pad_data(np.stack(cur_trans, axis=0), self.max_num_part).astype(np.float32)  # [P, 3]
        
        data_dict['gt_pcs'] = cur_pts_gt
        data_dict['part_pcs'] = cur_pts
        data_dict['part_quat'] = cur_quat
        data_dict['part_trans'] = cur_trans

        return data_dict

def build_all_piece_matching_pc_dataloader(cfg):
    data_dict = dict(
        cfg=cfg,
        data_dir=cfg.DATA.PC_DATA_DIR_TRAIN,
        data_fn=cfg.DATA.DATA_FN.format("train"),
        category=cfg.DATA.CATEGORY,
        rot_range=cfg.DATA.ROT_RANGE,
        overfit=cfg.DATA.OVERFIT,
        min_num_part=cfg.DATA.MIN_NUM_PART,
        max_num_part=cfg.DATA.MAX_NUM_PART,
    )
    train_set = AllPieceMatchingPCDataset(**data_dict)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )

    data_dict['data_dir'] = cfg.DATA.PC_DATA_DIR_VAL
    data_dict['data_fn'] = cfg.DATA.DATA_FN.format('val')

    val_set = AllPieceMatchingPCDataset(**data_dict)
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=cfg.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )
    return train_loader, val_loader