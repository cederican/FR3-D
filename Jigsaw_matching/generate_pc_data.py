"""
Code to generate point cloud data from the dataset.
"""

from dataset.all_piece_matching_dataset import build_all_piece_matching_dataloader
import os
import numpy as np
from tqdm import tqdm

def main(cfg):
    cfg.BATCH_SIZE = 1
    cfg.VAL_BATCH_SIZE = 1
    train_loader, val_loader = build_all_piece_matching_dataloader(cfg)

    def save_data(loader, data_type):
        save_path = f"{cfg.SAVE_PC_DATA_PATH}/{data_type}/"
        os.makedirs(save_path, exist_ok=True)

        for i, data_dict in tqdm(enumerate(loader), total=len(loader), desc=f"Processing {data_type} data"):
            data_id = data_dict['data_id'][0].item()
            part_valids = data_dict['part_valids'][0]
            num_parts = data_dict['num_parts'][0].item()
            mesh_file_path = data_dict['mesh_file_path'][0]
            part_pcs = data_dict['part_pcs'][0]
            gt_pcs = data_dict['gt_pcs'][0]
            part_quat = data_dict['part_quat'][0]
            part_trans = data_dict['part_trans'][0]
            n_pcs = data_dict['n_pcs'][0]
            critical_label_thresholds = data_dict['critical_label_thresholds'][0]

            np.savez(
                os.path.join(save_path, f'{data_id:05}.npz'),
                data_id=data_id,
                part_valids=part_valids.cpu().numpy(),
                num_parts=num_parts,
                mesh_file_path=mesh_file_path,
                #part_pcs=part_pcs.cpu().numpy(),
                gt_pcs=gt_pcs.cpu().numpy(),
                #part_quat=part_quat.cpu().numpy(),
                #part_trans=part_trans.cpu().numpy(),
                n_pcs=n_pcs.cpu().numpy(),
                critical_label_thresholds=critical_label_thresholds.cpu().numpy(),
            )


    save_data(train_loader, 'train')
    save_data(val_loader, 'val')

# python generate_pc_data.py DATA.SAVE_PC_DATA_PATH=pc_data/everyday
if __name__ == '__main__':
    from utils.config import cfg
    from utils.parse_args import parse_args
    from utils.print_easydict import print_easydict
    from utils.dup_stdout_manager import DupStdoutFileManager

    args = parse_args("Jigsaw")

    main(cfg)