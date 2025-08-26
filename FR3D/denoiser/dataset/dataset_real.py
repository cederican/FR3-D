import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import copy
from FR3D.denoiser.model.modules.custom_diffusers import PiecewiseScheduler
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
import random

class GeometryLatentDataset(Dataset):
    def __init__(
            self,
            cfg,
            data_dir,
            overfit,
            data_fn,
            denoiser_only_flag = False,
            sampling = False,
    ):
        self.cfg = cfg
        self.mode = data_fn
        self.data_dir = data_dir
        self.data_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.npz')])
        self.noise_scheduler = PiecewiseScheduler()
        self.max_num_part = self.cfg.data.max_num_part
        self.denoiser_only_flag = denoiser_only_flag
        self.sampling = sampling
        self.real_bones = cfg.real_bones
        if self.sampling:
            self.fixed_rot_mat = R.random(random_state=42).as_matrix()
            self.samples = self.cfg.samples

        if overfit != -1:
            #self.data_files = self.data_files[:overfit]
            self.data_files = [self.data_files[i] for i in overfit]

        if self.mode == "test" or self.mode == "val":
            self.matching_data_path = self.cfg.data.matching_data_path_val
        else:
            self.matching_data_path = self.cfg.data.matching_data_path_train
        
        self.data_list = []

        for file_name in tqdm(self.data_files):
            data_dict = np.load(os.path.join(self.data_dir, file_name))
            num_parts = data_dict["num_parts"].item()
            data_id = data_dict['data_id'].item()
            part_valids = data_dict['part_valids']
            part_pcs_gt = data_dict['part_pcs_gt']
            mesh_file_path = data_dict['mesh_file_path'].item()
                
            graph = data_dict['graph']
            ref_part = data_dict['ref_part']

            part_normals = data_dict['normals']
            part_locdists = np.zeros((part_normals.shape[0], part_normals.shape[1]))


            sample = {
                'data_id': data_id,
                'part_valids': part_valids,
                'mesh_file_path': mesh_file_path,
                'num_parts': num_parts,
                'ref_part': ref_part,
                'part_pcs_gt': part_pcs_gt,
                'graph': graph,
                'part_normals': part_normals,
                'part_locdists': part_locdists,
            }

            if self.mode == "test" and denoiser_only_flag is False:
                matching_data_path = os.path.join(self.matching_data_path, str(data_id) + '.npz')
                if not os.path.exists(matching_data_path):
                    continue
                matching_data = np.load(matching_data_path, allow_pickle=True)
                ### here change for gt instead of jigsaw!!!!!
                edges = matching_data['edges']
                correspondences = matching_data['correspondence']
                gt_pc_by_area = matching_data['gt_pcs']
                critical_pcs_idx = matching_data['critical_pcs_idx']
                n_pcs = matching_data['n_pcs']
                n_critical_pcs = matching_data['n_critical_pcs']
                    
                if correspondences.shape[0] != 1:
                    if correspondences.dtype == "O":
                        if len(correspondences.shape) == 1:
                            sample['correspondences'] = correspondences.tolist()
                        else:
                            sample['correspondences'] = [np.array(correspondences[i].tolist()) for i in range(correspondences.shape[0])]
                    else:
                        sample['correspondences'] = [correspondences[i] for i in range(correspondences.shape[0])]
                else:
                    sample['correspondences'] = [np.array(correspondences.squeeze().tolist())]
                    
                sample['gt_pc_by_area'] = gt_pc_by_area
                sample['critical_pcs_idx'] = critical_pcs_idx
                sample['edges'] = edges
                sample['n_pcs'] = n_pcs
                sample['n_critical_pcs'] = n_critical_pcs

            if self.sampling:
                for _ in range(self.samples):
                    self.data_list.append(copy.deepcopy(sample))    
            else:
                self.data_list.append(sample)

    
    def _anchor_coords(self, pcs, global_t, global_r):
        global_r = R.from_quat(global_r[[1, 2, 3, 0]]).inv()
        
        pcs = global_r.apply(pcs)
        pcs = pcs - global_t
        return pcs


    def _move_to_init_pose(self, pcs, n_pcs, num_parts, trans, rots):
        final_pose_pts = []
        
        pcs_count = 0
        for i in range(num_parts):
            c_pcs = pcs[pcs_count:pcs_count+n_pcs[i]]
            
            c_pcs = c_pcs - trans[i]
            c_pcs = R.from_quat(rots[i][[1, 2, 3, 0]]).inv().apply(c_pcs) # [w, x, y, z] -> [x, y, z, w]
            
            final_pose_pts.append(c_pcs)
            pcs_count += n_pcs[i]
        
        final_pose_pts = np.concatenate(final_pose_pts, axis=0)
                
        return final_pose_pts
    
    
    def __len__(self):
        return len(self.data_list)

    def generate_fixed_rot_mat(self):
        self.fixed_rot_mat = R.random(random_state=42).as_matrix()

    def _recenter_pc(self, pc):
        """pc: [N, 3]"""
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid[None]
        return pc, centroid
    
    def _rotate_pc(self, pc, normal=None):
        """
        pc: [N, 3]
        """

        if self.sampling:
            rot_mat = self.fixed_rot_mat
        else:
            rot_mat = R.random().as_matrix()
        rotated_pc = (rot_mat @ pc.T).T
        quat_gt = R.from_matrix(rot_mat.T).as_quat()
        # we use scalar-first quaternion
        quat_gt = quat_gt[[3, 0, 1, 2]]
        if normal is None:
            return rotated_pc, None, quat_gt
        rotated_normal = (rot_mat @ normal.T).T
        return rotated_pc, rotated_normal, quat_gt
    
    def shuffle_pc(self, pc, normal=None, locdists=None):
        """pc: [N, 3]"""
        order = np.arange(pc.shape[0])
        random.shuffle(order)
        shuffled_pc = pc[order]
        if normal is None and locdists is None:
            return shuffled_pc, None, None, order
        
        shuffled_normal = normal[order]
        shuffled_locdists = locdists[order]
        return shuffled_pc, shuffled_normal, shuffled_locdists, order
    
    def _rotate_whole_part(self, pc, normal=None):
        """
        pc: [P, N, 3]
        """
        P, N, _ = pc.shape
        pc = pc.reshape(-1, 3)
        if self.sampling:
            rot_mat = self.fixed_rot_mat
        else:
            rot_mat = R.random().as_matrix()
        pc = (rot_mat @ pc.T).T
        quat_gt = R.from_matrix(rot_mat.T).as_quat()
        # we use scalar-first quaternion
        quat_gt = quat_gt[[3, 0, 1, 2]]
        if normal is None:
            return pc.reshape(P, N, 3), None, quat_gt
        rotated_normal = (rot_mat @ normal.reshape(-1, 3).T).T.reshape(P, N, 3)
        return pc.reshape(P, N, 3), rotated_normal, quat_gt
    
    def _recenter_ref(self, pc, ref_part):
        """
        pc: [P, N, 3]
        """
        P, N, _ = pc.shape
        ref_idx = np.where(ref_part)[0]
        centroid = np.mean(pc[ref_idx.item()], axis=0)
        pc = pc - centroid
        return pc, centroid
    
    def _pad_data(self, data):
        """Pad data to shape [`self.max_num_part`, data.shape[1], ...]."""
        data = np.array(data)
        pad_shape = (self.max_num_part, ) + tuple(data.shape[1:])
        pad_data = np.zeros(pad_shape, dtype=np.float32)
        pad_data[:data.shape[0]] = data
        return pad_data
    
    def _pad_lst(self, lst):
        """add zero entries such that lst is self.max_num_parts long"""
        if isinstance(lst[0], Meshes) if lst else False:
            padded = [None] * self.max_num_part
            for i in range(min(len(lst), self.max_num_part)):
                padded[i] = lst[i]
            return padded
              
    def __getitem__(self, idx):
        
        data_dict = copy.deepcopy(self.data_list[idx])
        num_parts = data_dict['num_parts']
        part_pcs_gt = data_dict['part_pcs_gt']
        normals_gt = data_dict.get('part_normals', None)
        part_locdists_gt = data_dict.get('part_locdists', None)
        
        ref_part = data_dict['ref_part']

        if self.sampling:
            if idx % self.samples == 0:
                self.generate_fixed_rot_mat()
        
        part_pcs_final, part_normals_final, pose_gt_r = self._rotate_whole_part(part_pcs_gt, normals_gt)
        part_pcs_final, pose_gt_t = self._recenter_ref(part_pcs_final, ref_part)
        
        cur_pts, cur_normals, cur_locdists, cur_quat, cur_trans = [], [], [], [], []
        
        for i in range(num_parts):
            pc = part_pcs_final[i]
            normal = part_normals_final[i] if normals_gt is not None else None
            loc_dist = part_locdists_gt[i] if part_locdists_gt is not None else None
            pc, gt_trans = self._recenter_pc(pc)
            pc, normal, gt_quat = self._rotate_pc(pc, normal)
            pc, normal, loc_dist, order = self.shuffle_pc(pc, normal, loc_dist)
            # Shuffle gt as well
            part_pcs_gt[i] = part_pcs_gt[i][order]
            if normals_gt is not None and part_locdists_gt is not None:
                normals_gt[i] = normals_gt[i][order]
                part_locdists_gt[i] = part_locdists_gt[i][order]
            
            cur_quat.append(gt_quat)
            cur_trans.append(gt_trans)
            cur_pts.append(pc)
            if normals_gt is not None and part_locdists_gt is not None:
                cur_normals.append(normal)
                cur_locdists.append(loc_dist)
                        
        cur_pts = self._pad_data(np.stack(cur_pts, axis=0)).astype(np.float32)  # [P, N, 3]
        cur_normals = self._pad_data(np.stack(cur_normals, axis=0)).astype(np.float32) if cur_normals else None  # [P, N, 3]
        cur_locdists = self._pad_data(np.stack(cur_locdists, axis=0)).astype(np.float32) if cur_locdists else None  # [P, N, 1]
        cur_quat = self._pad_data(np.stack(cur_quat, axis=0)).astype(np.float32)  # [P, 4]
        cur_trans = self._pad_data(np.stack(cur_trans, axis=0)).astype(np.float32)  # [P, 3]
        part_pcs_gt = self._pad_data(np.stack(part_pcs_gt, axis=0)).astype(np.float32) # [P, N, 3]
        normals_gt = self._pad_data(np.stack(normals_gt, axis=0)).astype(np.float32) if normals_gt is not None else None  # [P, N, 3]
        part_locdists_gt = self._pad_data(np.stack(part_locdists_gt, axis=0)).astype(np.float32) if part_locdists_gt is not None else None  # [P, N, 1]

        
        if self.mode == 'test' and self.denoiser_only_flag is False:        
            gt_pc_by_area = self._anchor_coords(
                data_dict['gt_pc_by_area'], 
                pose_gt_t, 
                pose_gt_r
            )

            part_pcs_by_area = self._move_to_init_pose(
                gt_pc_by_area,
                data_dict['n_pcs'],
                num_parts,
                cur_trans,
                cur_quat
            )
            
            data_dict['part_pcs_by_area'] = part_pcs_by_area.astype(np.float32)
        
        
        # Normalize the part pcs
        scale = np.max(np.abs(cur_pts), axis=(1,2), keepdims=True)
        scale[scale == 0] = 1
        cur_pts = cur_pts / scale
        
        data_dict['part_pcs'] = cur_pts
        data_dict['part_pcs_gt'] = part_pcs_gt
        data_dict['part_normals'] = cur_normals
        data_dict['part_normals_gt'] = normals_gt
        data_dict['part_locdists'] = cur_locdists
        data_dict['part_locdists_gt'] = part_locdists_gt

        data_dict['part_rots'] = cur_quat
        data_dict['part_trans'] = cur_trans

        data_dict['part_scale'] = scale.squeeze(-1)

        data_dict['init_pose_r'] = pose_gt_r
        data_dict['init_pose_t'] = pose_gt_t

        
        # Only one reference part
        if self.cfg.model.multiple_ref_parts is False:
            return data_dict
    
        if self.mode != 'train':
            return data_dict

        num_parts = data_dict['num_parts']
        if num_parts == 2:
            return data_dict
        
        # half of the time, only one reference part
        if np.random.rand() < 0.5:
            return data_dict
        
        # Randomly sample more reference parts which connected to the original reference part
        ref_part = data_dict['ref_part']
        graph = data_dict['graph']
        scale = data_dict['part_scale']

        ref_part_idx = np.where(ref_part)[0]
        connect_parts = np.where(graph[ref_part_idx, :])[1]

        larger_connect_parts = [part for part in connect_parts if scale[part] > 0.05]
        if not larger_connect_parts:
            return data_dict

        num_connect_parts = len(larger_connect_parts)

        sample_num = np.random.randint(0, num_connect_parts)
        
        sample_ref_parts = np.random.choice(connect_parts, sample_num, replace=False)
        ref_part[sample_ref_parts] = True

        
        data_dict['ref_part'] = ref_part
        part_trans_ref = data_dict['part_trans'][sample_ref_parts]
        part_rots_ref = data_dict['part_rots'][sample_ref_parts]
        # random perturb the reference part
        noise_trans = torch.randn(part_trans_ref.shape)
        noise_rots = torch.randn(part_rots_ref.shape)
        timesteps = torch.randint(0, 50, (1,)).long()
        
        part_trans_ref = self.noise_scheduler.add_noise(torch.tensor(part_trans_ref), noise_trans, timesteps).numpy()
        part_rots_ref = self.noise_scheduler.add_noise(torch.tensor(part_rots_ref), noise_rots, timesteps).numpy()

        data_dict['part_trans'][sample_ref_parts] = part_trans_ref
        data_dict['part_rots'][sample_ref_parts] = part_rots_ref

        
        return data_dict

def custom_collate_fn(batch):
    collated = {}

    for key in batch[0]:
        values = [d[key] for d in batch]

        if isinstance(values[0], list) and all(
            isinstance(v, Meshes) or v is None for v in values[0]
        ):
            collated[key] = values
        elif isinstance(values[0], np.ndarray):
            collated[key] = torch.stack([torch.tensor(v) for v in values])
        elif isinstance(values[0], str):
            collated[key] = values
        else:
            collated[key] = torch.tensor(values)

    return collated


def build_geometry_dataloader(cfg):
    data_dict = dict(
        cfg=cfg,
        data_dir=cfg.data.data_dir,
        overfit=cfg.data.overfit,
        data_fn="train",
        sampling=cfg.sampling,
    )
    train_set = GeometryLatentDataset(**data_dict)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(cfg.data.num_workers > 0),
    )


    data_dict['data_fn'] = "val"
    data_dict['data_dir'] = cfg.data.data_val_dir
    val_set = GeometryLatentDataset(**data_dict)
    val_loader = DataLoader(
            dataset=val_set,
            batch_size=cfg.data.val_batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=(cfg.data.num_workers > 0),
    )
    return train_loader, val_loader


def build_test_dataloader(cfg, denoiser_only_flag):
    data_dict = dict(
        cfg=cfg,
        data_dir=cfg.data.data_val_dir,
        overfit=cfg.data.overfit,
        data_fn="test",
        denoiser_only_flag=denoiser_only_flag,
        sampling=cfg.sampling,
    )

    val_set = GeometryLatentDataset(**data_dict)
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=cfg.data.val_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.data.num_workers > 0),
    )
    return val_loader

def build_test_dataloader_real(cfg, denoiser_only_flag):
    data_dict = dict(
        cfg=cfg,
        data_dir=cfg.data.data_val_dir,
        overfit=cfg.data.overfit,
        data_fn="test",
        denoiser_only_flag=denoiser_only_flag,
        sampling=cfg.sampling,
    )

    val_set = GeometryLatentDataset(**data_dict)
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=cfg.data.val_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.data.num_workers > 0),
    )
    return val_loader

