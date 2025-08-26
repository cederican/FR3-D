import os
import random

import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler

INTERFACE_THRESHOLD = 0.05

class GeometryPartDataset(Dataset):
    """
    Geometry part assembly dataset.

    We follow the data prepared by Breaking Bad dataset:
        https://breaking-bad-dataset.github.io/
    """

    def __init__(
        self,
        data_dir,
        data_fn,
        data_keys,
        cfg,
        category='',
        num_points=1000,
        min_num_part=2,
        max_num_part=20,
        shuffle_parts=False,
        rot_range=-1,
        overfit=-1,
    ):
        self.cfg = cfg
        # store parameters
        self.category = category if category.lower() != 'all' else ''
        self.data_dir = data_dir
        self.num_points = num_points
        self.min_num_part = min_num_part
        self.max_num_part = max_num_part  # ignore shapes with more parts
        self.shuffle_parts = shuffle_parts  # shuffle part orders
        self.rot_range = rot_range  # rotation range in degree
        self.type = 'real' if 'real' in data_fn else 'synthetic'

        # list of fracture folder path
        self.data_list = self._read_data(data_fn)
        if overfit > 0:
            self.data_list = self.data_list[:overfit]

        # additional data to load, e.g. ('part_ids', 'instance_label')
        self.data_keys = data_keys

    def _read_data(self, data_fn):
        """Filter out invalid number of parts."""
        with open(os.path.join(self.data_dir, data_fn), 'r') as f:
            mesh_list = [line.strip() for line in f.readlines()]
            if self.category:
                mesh_list = [
                    line for line in mesh_list
                    if self.category in line.split('/')
                ]
        data_list = []

        for mesh in mesh_list:
            mesh_dir = os.path.join(self.data_dir, mesh)
            if not os.path.isdir(mesh_dir):
                print(f'{mesh} does not exist')
                continue
            fracs = os.listdir(mesh_dir)
            fracs.sort()
            for frac in fracs:
                # we take both fractures and modes for training
                if 'fractured' not in frac and 'mode' not in frac:
                    continue
                frac = os.path.join(mesh, frac)
                num_parts = len(os.listdir(os.path.join(self.data_dir, frac)))
                if self.min_num_part <= num_parts <= self.max_num_part:
                    data_list.append(frac)
        return data_list
    
    def _are_meshes_connected(self, mesh_a, mesh_b):
        """
        Check if two meshes share any vertices.
        
        Args:
            mesh_a, mesh_b (trimesh.Trimesh): The two mesh objects to compare.
            
        Returns:
            bool: True if the meshes share at least one vertex, False otherwise.
        """
        # Round vertices to a precision to mitigate floating-point issues
        precision = 5
        vertices_a = np.round(mesh_a.vertices, decimals=precision)
        vertices_b = np.round(mesh_b.vertices, decimals=precision)

        # Create a set of unique vertices for each mesh
        unique_vertices_a = set(map(tuple, vertices_a))
        unique_vertices_b = set(map(tuple, vertices_b))

        # Check if there is any intersection between the sets of unique vertices
        shared_vertices = unique_vertices_a.intersection(unique_vertices_b)

        return len(shared_vertices) > 0
    
    def _check_connectivity(self, meshes):
        """
        Generate a connectivity matrix for the input meshes.
        args:
            meshes: list of trimesh objects
        returns:
            A numpy array where element (i, j) is True if meshes[i] and meshes[j] are connected.
        """
        num_meshes = len(meshes)
        connectivity_matrix = np.zeros((self.max_num_part, self.max_num_part), dtype=bool)

        if not self.type == 'real':
            for i in range(num_meshes):
                for j in range(i + 1, num_meshes):  # Check each pair once
                    if self._are_meshes_connected(meshes[i], meshes[j]):
                        connectivity_matrix[i, j] = True
                        connectivity_matrix[j, i] = True  # Ensure symmetry for undirected graph

        return connectivity_matrix.astype(bool)

    @staticmethod
    def _recenter_pc(pc):
        """pc: [N, 3]"""
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid[None]
        return pc, centroid
    
    def _rotate_pc(self, pc):
        """pc: [N, 3]"""

        rot_mat = R.random().as_matrix()

        pc = (rot_mat @ pc.T).T
        quat_gt = R.from_matrix(rot_mat.T).as_quat()
        # we use scalar-first quaternion
        quat_gt = quat_gt[[3, 0, 1, 2]]
        return pc, quat_gt
    
    def _pad_data(self, data):
        """Pad data to shape [`self.max_num_part`, data.shape[1], ...]."""
        data = np.array(data)
        pad_shape = (self.max_num_part, ) + tuple(data.shape[1:])
        pad_data = np.zeros(pad_shape, dtype=np.float32)
        pad_data[:data.shape[0]] = data
        return pad_data
    
    def sample_points_from_mesh(self, mesh, n_samples):
        points, face_indices = trimesh.sample.sample_surface_even(mesh, n_samples)
        if len(points) < n_samples:
            concat_points, concat_face_indices = trimesh.sample.sample_surface(mesh, n_samples - len(points))     
            points = np.concatenate([points, concat_points], axis=0)
            face_indices = np.concatenate([face_indices, concat_face_indices], axis=0)
        normals = mesh.face_normals[face_indices]

        return {
            'points': points,
            'normals': normals,
            'mesh': mesh
        }
    
    def compute_interface_values(self, points, all_other_points_tree):
        dists, _ = all_other_points_tree.query(points, k=1)
        values = np.clip(1.0 - (dists / INTERFACE_THRESHOLD), 0.0, 1.0)
        return values

    def _get_pcs(self, data_folder):
        """Read mesh and sample point cloud from a folder."""
        # `data_folder`: xxx/plate/1d4093ad2dfad9df24be2e4f911ee4af/fractured_0
        data_folder = os.path.join(self.data_dir, data_folder)
        mesh_files = os.listdir(data_folder)
        mesh_files.sort()
        if not self.min_num_part <= len(mesh_files) <= self.max_num_part:
            raise ValueError

        # shuffle part orders
        if self.shuffle_parts:
            random.shuffle(mesh_files)
        
        # read mesh and sample points
        meshes = [
            trimesh.load(os.path.join(data_folder, mesh_file))
            for mesh_file in mesh_files
        ]

        # Check if the meshes are connected
        graph = self._check_connectivity(meshes)
        
        all_data = [
            self.sample_points_from_mesh(mesh, self.num_points)
            for mesh in meshes
        ]

        pcs = []
        normals = []
        feats = []
        for data in all_data:
            points = data['points']
            other_points = np.vstack([d['points'] for d in all_data if not np.array_equal(d['points'], points)])
            tree = cKDTree(other_points)
            feat_vals = self.compute_interface_values(points, tree)

            for i in range(len(points)):
                pcs.append(points[i])
                normals.append(data['normals'][i])
                feats.append(feat_vals[i])

        return np.stack(pcs, axis=0).reshape(-1, self.num_points, 3), graph, np.stack(normals, axis=0).reshape(-1, self.num_points, 3), np.stack(feats, axis=0).reshape(-1, self.num_points) #, np.stack(evecs, axis=0)

    def __getitem__(self, index):
        """
        data_dict = {
            'part_pcs': MAX_NUM x N x 3
                The points sampled from each part.

            'part_valids': MAX_NUM
                1 for shape parts, 0 for padded zeros.

            'data_id': int
                ID of the data.
        }
        """
        pcs, graph, normals, feat_vals = self._get_pcs(self.data_list[index])
        num_parts = pcs.shape[0]
        
        scale = np.max(pcs, axis=(1,2)) - np.min(pcs, axis=(1, 2))
        
        ref_part = np.zeros((self.max_num_part), dtype=np.float32)
        ref_part_idx = np.argmax(scale[:num_parts])
        ref_part[ref_part_idx] = 1
        ref_part = ref_part.astype(bool)

        data_dict = {
            'ref_part': ref_part,
            'part_pcs_gt': pcs,
            'normals': normals,
            'feat_vals': feat_vals,
        }
        
        # valid part masks
        valids = np.zeros((self.max_num_part), dtype=np.float32)
        valids[:num_parts] = 1.
        data_dict['part_valids'] = valids
        data_dict['mesh_file_path'] = self.data_list[index]
        data_dict['num_parts'] = num_parts
        data_dict['graph'] = graph
        
        data_dict['category'] = self.data_list[index].split('/')[1].lower()

        # data_id
        data_dict['data_id'] = index

        return data_dict

    def __len__(self):
        return len(self.data_list)



class StartFromSampler(Sampler):
    def __init__(self, data_source, start_id):
        self.data_source = data_source
        self.start_id = start_id

    def __iter__(self):
        return iter(range(self.start_id, len(self.data_source)))

    def __len__(self):
        return len(self.data_source) - self.start_id

def build_geometry_dataloader_real(cfg, data_type, start_id):
    data_dict = dict(
        data_dir=cfg.data.mesh_data_dir,
        data_fn=cfg.data.data_fn.format('real'),
        data_keys=cfg.data.data_keys,
        cfg=cfg,
        category=cfg.data.category,
        num_points=cfg.data.num_pc_points,
        min_num_part=cfg.data.min_num_part,
        max_num_part=cfg.data.max_num_part,
        shuffle_parts=cfg.data.shuffle_parts,
        rot_range=cfg.data.rot_range,
        overfit=cfg.data.overfit,
    )
    test_set = GeometryPartDataset(**data_dict)
    test_sampler = StartFromSampler(test_set, start_id)
    test_loader = DataLoader(
        dataset=test_set,
        sampler=test_sampler,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(cfg.data.num_workers > 0),
    )
    return test_loader


def build_geometry_dataloader(cfg, data_type, start_id):
    data_dict = dict(
        data_dir=cfg.data.mesh_data_dir,
        data_fn=cfg.data.data_fn.format('train'),
        data_keys=cfg.data.data_keys,
        cfg=cfg,
        category=cfg.data.category,
        num_points=cfg.data.num_pc_points,
        min_num_part=cfg.data.min_num_part,
        max_num_part=cfg.data.max_num_part,
        shuffle_parts=cfg.data.shuffle_parts,
        rot_range=cfg.data.rot_range,
        overfit=cfg.data.overfit,
    )
    train_set = GeometryPartDataset(**data_dict)
    train_sampler = StartFromSampler(train_set, start_id)
    train_loader = DataLoader(
        dataset=train_set,
        sampler=train_sampler,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(cfg.data.num_workers > 0),
    )

    data_dict['data_fn'] = cfg.data.data_fn.format('val')
    data_dict['shuffle_parts'] = False
    val_set = GeometryPartDataset(**data_dict)
    val_sampler = StartFromSampler(val_set, start_id)
    val_loader = DataLoader(
        dataset=val_set,
        sampler=val_sampler,
        batch_size=cfg.data.val_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.data.num_workers > 0),
    )
    if data_type == 'train':
        return train_loader
    elif data_type == 'val':
        return val_loader
    else:
        raise ValueError(f"Invalid data type: {data_type}")
