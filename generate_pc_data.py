"""
Code to generate point cloud data from the dataset.
"""

import hydra
from FR3D.vqvae.dataset.dataset import build_geometry_dataloader
import os
import numpy as np
from tqdm import tqdm

@hydra.main(config_path='config/ae', config_name='global_config.yaml')
def main(cfg):
    cfg.data.batch_size = 1
    cfg.data.val_batch_size = 1

    
    def save_data(data_type):
        save_path = f"{cfg.data.save_pc_data_path}/{data_type}/"
        os.makedirs(save_path, exist_ok=True)

        existing_files = [f for f in os.listdir(save_path) if f.endswith('.npz')]
        if existing_files:
            last_file = max(existing_files)
            last_id = int(last_file.split('.')[0])
            print(f"Found existing files. Last processed ID: {last_id}")
        
        start_id = 0

        loader = build_geometry_dataloader(cfg, data_type, start_id)

        for i, data_dict in tqdm(enumerate(loader), total=len(loader), desc=f"Processing {data_type} data"):
            data_id = data_dict['data_id'][0].item()
            part_valids = data_dict['part_valids'][0]
            num_parts = data_dict['num_parts'][0].item()
            mesh_file_path = data_dict['mesh_file_path'][0]
            graph = data_dict['graph'][0]
            category = data_dict['category'][0]
            part_pcs_gt = data_dict['part_pcs_gt'][0]
            ref_part = data_dict['ref_part'][0]
            normals = data_dict['normals'][0]
            feat_vals = data_dict['feat_vals'][0]

            np.savez(
                os.path.join(save_path, f'{data_id:05}.npz'),
                data_id=data_id,
                part_valids=part_valids.cpu().numpy(),
                num_parts=num_parts,
                mesh_file_path=mesh_file_path,
                graph=graph.cpu().numpy(),
                category=category,
                part_pcs_gt=part_pcs_gt.cpu().numpy(),
                ref_part=ref_part.cpu().numpy(),
                normals=normals.cpu().numpy(),
                feat_vals=feat_vals.cpu().numpy(),
            )

    # Save train data
    save_data('train')
    # Save validation data
    save_data('val')

# python generate_pc_data.py +data.save_pc_data_path=pc_data/everyday
if __name__ == '__main__':
    main()