from .all_piece_matching_dataset import build_all_piece_matching_dataloader
from .all_piece_matching_pc_dataset import build_all_piece_matching_pc_dataloader
from .dataset_config import dataset_cfg


def build_dataloader(cfg):
    dataset = cfg.DATASET.lower().split(".")
    if dataset[0] == "breaking_bad" or dataset[0] == "bones":
        if dataset[1] == "all_piece_matching":
            return build_all_piece_matching_dataloader(cfg)
        elif dataset[1] == "all_piece_matching_pc":
            return build_all_piece_matching_pc_dataloader(cfg)
        else:
            raise NotImplementedError(f"Dataset {dataset} not implemented")
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented")
