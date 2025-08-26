"""
Adapted from puzzlefusion++
https://github.com/eric-zqwang/puzzlefusion-plusplus
"""

import torch.nn as nn
import torch
from FR3D.vqvae.model.modules.pn2 import PN2
from FR3D.vqvae.model.modules.quantizer import VectorQuantizer
from chamferdist import ChamferDistance


class VQVAE(nn.Module):
    def __init__(self, cfg):
        super(VQVAE, self).__init__()
        self.pn2 = PN2(cfg)
        self.cfg = cfg
        self.encoder = self.pn2.encode
        self.decoder = self.pn2.decode
        self.vector_quantization = VectorQuantizer(
            cfg.ae.n_embeddings,
            cfg.ae.embedding_dim,
            cfg.ae.beta
        )
        self.cd_loss = ChamferDistance()


    def forward(self, data_dict, verbose=False):
        """
        x.shape = (batch, C, L)
        """
        x = data_dict["part_pcs"].permute(0, 2, 1)
        normals = data_dict["part_normals"].permute(0, 2, 1)
        locdists = data_dict["part_locdists"].unsqueeze(2).permute(0, 2, 1)

        feat = torch.cat([normals, locdists], dim=1)  # B, C+3, L

        z_e, xyz = self.encoder(x, feat)

        B, L, C = z_e.shape

        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(
            z_e.reshape(B, 4 * L, -1)
        )

        z_q = z_q.reshape(B, L, -1)

        x_hat = self.decoder(z_q)

        output_dict = {
            'embedding_loss': embedding_loss,
            'pc_offset': x_hat,
            'perplexity': perplexity,
            "xyz": xyz,
            "z_q": z_q
        }

        return output_dict
    
    def encode(self, part_pcs, normals=None, locdists=None):
        """
        x.shape = (batch, C, L)
        """
        x = part_pcs.permute(0, 2, 1)

        if normals is not None and locdists is not None:
            normals = normals.permute(0, 2, 1)
            locdists = locdists.unsqueeze(2).permute(0, 2, 1) if locdists.dim() == 2 else locdists.permute(0, 2, 1)
            feat = torch.cat([normals, locdists], dim=1)  # B, C+3, L
        else:
            feat = None

        z_e, xyz = self.encoder(x, feat)
        
        B, L, C = z_e.shape

        _, z_q, _, _, _ = self.vector_quantization(
            z_e.reshape(B, 4 * L, -1)
        )
        z_q = z_q.reshape(B, L, -1)

        batch_indices = torch.repeat_interleave(torch.arange(B), L).to(xyz.device)
        output_dict = {
            "feat": z_q.reshape(B, L, -1), # B, N, C
            "coord": xyz.reshape(B, L, -1), # B, N, 3
            "batch": batch_indices, # B * N
        }

        return output_dict
    
    def decode(self, z_q):
        pred_offsets = self.decoder(z_q)
        return pred_offsets


    def loss(self, data_dict, output_dict):
        loss_dict = {}

        pred_offset = output_dict["pc_offset"]
        xyz = output_dict["xyz"]

        pc_recon = pred_offset + xyz.unsqueeze(2)
        pc_recon = pc_recon.reshape(-1, 1000, 3)

        cd_loss = self.cd_loss(pc_recon, data_dict['part_pcs'], bidirectional=True)
        loss_dict['cd_loss'] = cd_loss
        loss_dict['embedding_loss'] = output_dict['embedding_loss']

        return loss_dict

