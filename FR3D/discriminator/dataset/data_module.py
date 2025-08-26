import lightning.pytorch as pl
import torch
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
import math
import numpy as np
from FR3D.discriminator.dataset.dataset import build_geometry_dataloader

class ContiguousDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, block_multiple=20):
        if num_replicas is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                num_replicas = torch.distributed.get_world_size()
            else:
                num_replicas = 1
        if rank is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            else:
                rank = 0
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.dataset_length = len(self.dataset)
        min_block_size = math.ceil(self.dataset_length / self.num_replicas)
        self.block_size = int(math.ceil(min_block_size / block_multiple) * block_multiple)
        self.total_size = self.block_size * self.num_replicas

    def __iter__(self):
        indices = list(range(self.dataset_length))
        if len(indices) < self.total_size:
            indices += indices[:(self.total_size - len(indices))]
        start = self.rank * self.block_size
        end = start + self.block_size
        return iter(indices[start:end])

    def __len__(self):
        return self.block_size

class TrainContiguousDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, block_multiple=20, shuffle=True, seed=0):
        if num_replicas is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                num_replicas = torch.distributed.get_world_size()
            else:
                num_replicas = 1
        if rank is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            else:
                rank = 0
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.dataset_length = len(self.dataset)
        min_block_size = math.ceil(self.dataset_length / self.num_replicas)
        self.block_size = int(math.ceil(min_block_size / block_multiple) * block_multiple)
        self.total_size = self.block_size * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        indices = list(range(self.dataset_length))
        if len(indices) < self.total_size:
            indices += indices[:(self.total_size - len(indices))]
        start = self.rank * self.block_size
        end = start + self.block_size
        block_indices = indices[start:end]
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
    
            subblock_size = 20
            num_subblocks = len(block_indices) // subblock_size
    
            # Split block_indices into sub-blocks
            subblocks = [block_indices[i * subblock_size : (i + 1) * subblock_size]
                         for i in range(num_subblocks)]
    
            # Shuffle each sub-block internally
            for sb in subblocks:
                rng.shuffle(sb)
    
            # Shuffle the order of sub-blocks themselves
            rng.shuffle(subblocks)
    
            # Flatten shuffled sub-blocks back into one list
            shuffled_block = [item for subblock in subblocks for item in subblock]
    
            # Handle leftover elements (should not happen if block_size multiple of 20)
            leftover = block_indices[num_subblocks * subblock_size :]
            if leftover:
                rng.shuffle(leftover)
                shuffled_block.extend(leftover)
    
            block_indices = shuffled_block
    
        return iter(block_indices)

    def __len__(self):
        return self.block_size
    
    def set_epoch(self, epoch):
        self.epoch = epoch


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_data = None
        self.val_data = None
    
    def setup(self, stage=None):
        self.train_set, self.val_set = build_geometry_dataloader(self.cfg)

    def train_dataloader(self):
        train_sampler = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            train_sampler = TrainContiguousDistributedSampler(self.train_set, shuffle=True)

        return DataLoader(
            dataset=self.train_set,
            batch_size=self.cfg.data.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(self.cfg.data.num_workers > 0),
        )

    def val_dataloader(self):
        val_sampler = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            val_sampler = ContiguousDistributedSampler(self.val_set)

        return DataLoader(
            dataset=self.val_set,
            batch_size=self.cfg.data.val_batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(self.cfg.data.num_workers > 0),
        )