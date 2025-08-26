"""
Adapted from puzzlefusion++
https://github.com/eric-zqwang/puzzlefusion-plusplus
"""

import torch.nn as nn
import torch
import torch_scatter
from utils.model_utils import EmbedderNerf
from FR3D.denoiser.model.modules.attention import EncoderLayer, SE3EncoderLayer

class DenoiserTransformer(nn.Module):

    def __init__(self, cfg):
        super(DenoiserTransformer, self).__init__()
        self.cfg = cfg

        self.embed_dim = cfg.model.embed_dim
        self.out_channels = cfg.model.out_channels
        self.num_layers = cfg.model.num_layers
        self.num_heads = cfg.model.num_heads

        self.ref_part_emb = nn.Embedding(2, cfg.model.embed_dim)
        self.activation = nn.SiLU()

        num_embeds_ada_norm = 6 * self.embed_dim

        self.transformer_layers = nn.ModuleList(
            [
                EncoderLayer(
                    dim=self.embed_dim,
                    num_attention_heads=self.num_heads,
                    attention_head_dim=self.embed_dim // self.num_heads,
                    dropout=0.2,
                    activation_fn='geglu',
                    num_embeds_ada_norm=num_embeds_ada_norm, 
                    attention_bias=False,
                    norm_elementwise_affine=True,
                    final_dropout=False,
                    se3=self.cfg.model.se3,
                    path_drop=self.cfg.model.path_drop[i],
                    irreps_out_dim=8,
                    irreps_in_dim=64
                )
                for i in range(self.num_layers)
            ]
        )

        if self.cfg.model.se3:
            self.se3transformer_layers = SE3EncoderLayer(
                norm_elementwise_affine=True,
                latent_dim=cfg.model.num_dim,
            )

        multires = 10
        embed_kwargs = {
            'include_input': True,
            'input_dims': 7,
            'max_freq_log2': multires - 1,
            'num_freqs': multires,
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos],
        }
        
        embedder_obj = EmbedderNerf(**embed_kwargs)
        self.param_embedding = lambda x, eo=embedder_obj: eo.embed(x)

        embed_pos_kwargs = {
            'include_input': True,
            'input_dims': 3,
            'max_freq_log2': multires - 1,
            'num_freqs': multires,
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos],
        }
        embedder_pos = EmbedderNerf(**embed_pos_kwargs)
        # Pos embedding for positions of points xyz
        self.pos_embedding = lambda x, eo=embedder_pos: eo.embed(x)

        embed_scale_kwargs = {
            'include_input': True,
            'input_dims': 1,
            'max_freq_log2': multires - 1,
            'num_freqs': multires,
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos],
        }
        embedder_scale = EmbedderNerf(**embed_scale_kwargs)
        self.scale_embedding = lambda x, eo=embedder_scale: eo.embed(x)

        embed_normal_kwargs = {
            "include_input": True,
            "input_dims": 3,
            "max_freq_log2": multires - 1,
            "num_freqs": multires,
            "log_sampling": True,
            "periodic_fns": [torch.sin, torch.cos],
        }
        embedder_normal = EmbedderNerf(**embed_normal_kwargs)
        # Normal embedding for normals of points
        self.normal_embedding = lambda x, eo=embedder_normal: eo.embed(x)

        embed_locdist_kwargs = {
            "include_input": True,
            "input_dims": 1,
            "max_freq_log2": multires - 1,
            "num_freqs": multires,
            "log_sampling": True,
            "periodic_fns": [torch.sin, torch.cos],
        }
        embedder_locdist = EmbedderNerf(**embed_locdist_kwargs)
        # Locdist embedding for points
        self.locdist_embedding = lambda x, eo=embedder_locdist: eo.embed(x)
        
        self.shape_embedding = nn.Linear(
            cfg.model.num_dim 
            + embedder_scale.out_dim
            + embedder_pos.out_dim
            + embedder_normal.out_dim
            + embedder_locdist.out_dim,
            self.embed_dim
        )

        self.param_fc = nn.Linear(embedder_obj.out_dim, self.embed_dim)

        # mlp out for translation N, 256 -> N, 3
        self.mlp_out_trans = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.SiLU(),
            nn.Linear(self.embed_dim // 2, 3),
        )

        # mlp out for rotation N, 256 -> N, 4
        self.mlp_out_rot = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.SiLU(),
            nn.Linear(self.embed_dim // 2, 4),
        )

    def _gen_cond(
        self,
        x,  # (valid_P, 7)
        latent, #  (valid_P, n_points, 3) 
        scale,  # (valid_P, 1)
    ):
        # pos encoding for super points' coordinates
        xyz = latent["coord"].flatten(0,1)  # (n_points, 3)
        xyz_pos_emb = self.pos_embedding(xyz)  # (n_points, pos_emb_dim=63)

        normal = latent["normals"].flatten(0,1)  # (n_points, 3)
        normal_emb = self.normal_embedding(normal)  # (n_points, normal_emb_dim=63)

        locdist = latent["locdists"].flatten(0,1).unsqueeze(1)  # (n_points, 1)
        locdist_emb = self.locdist_embedding(locdist)  # (n_points, loc_dist_emb_dim=21)

        scale_emb = self.scale_embedding(scale)  # (valid_P, scale_emb_dim=21)
        scale_emb = scale_emb[latent['batch']]  # (n_points, scale_emb_dim=21)

        concat_emb = torch.cat(
            (latent["feat"].flatten(0,1), xyz_pos_emb, normal_emb, locdist_emb, scale_emb), dim=-1
        )  # (n_points, in_dim + pos_emb_dim + normal_emb_dim + loc_dist_emb_dim + scale_emb_dim)
        shape_emb = self.shape_embedding(concat_emb)  # (n_points, embed_dim)

        x_emb = self.param_fc(self.param_embedding(x))  # (valid_P, embed_dim)
        return x_emb, shape_emb


    def _out(self, data_emb):
        trans = self.mlp_out_trans(data_emb)
        rots = self.mlp_out_rot(data_emb)

        return torch.cat([trans, rots], dim=-1)

    def _add_ref_part_emb(
        self,
        x_emb,  # (valid_P, embed_dim)
        ref_part,  # (valid_P,)
    ):
        valid_P = x_emb.shape[0]
        ref_part_emb = self.ref_part_emb.weight[0].repeat(valid_P, 1)
        ref_part_emb[ref_part.to(torch.bool)] = self.ref_part_emb.weight[1]

        x_emb = x_emb + ref_part_emb
        return x_emb

    def _gen_mask(self, B, N, L, part_valids):
        self_block = torch.ones(L, L, device=part_valids.device)  # Each L points should talk to each other
        self_mask = torch.block_diag(*([self_block] * N))  # Create block diagonal tensor
        self_mask = self_mask.unsqueeze(0).repeat(B, 1, 1)  # Expand dimensions to [B, N*L, N*L]
        self_mask = self_mask.to(torch.bool)
        gen_mask = part_valids.unsqueeze(-1).repeat(1, 1, L).flatten(1, 2)
        gen_mask = gen_mask.to(torch.bool)
        
        return self_mask, gen_mask

    def forward(self,
                x, # (valid_P, 7)
                timesteps, # (valid_P,)
                latent, # enc_out_dict 
                part_valids, # (B, P)
                scale, 
                ref_part,
                point_edge_index=None, 
                part_edge_index=None, 
                fc_part_edge_index=None, 
                noisy_part_centers=None
    ):
        """
        x: noisy_trans_and_rots (valid_P, 7)
        latent: {
            "feat": (valid_P, L, 64), # B, N, L, C
            "coord": (valid_P, L, 3), # B, N, L, 3
            "normals": (valid_P, L, 3), # B, N, L, 3
            "locdists": (valid_P, L, 1), # B, N, L, 1
            "batch": (valid_P * L,), # B * N
        }    
        """
        if self.cfg.model.se3:
            with torch.cuda.amp.autocast(enabled=False):
                se3_scalars, se3_vectors = self.se3transformer_layers(
                    noisy_part_centers.float(),
                    latent["feat"].float(),
                    latent["coord"].float(),
                    point_edge_index, 
                    part_edge_index, 
                    fc_part_edge_index,
                    self.cfg.real_bones
                )
                se3_scalars = se3_scalars.flatten(0, 1)  # (valid_P * L, se3_scalars_dim)
                se3_vectors = se3_vectors.flatten(0, 1)  # (valid_P * L, se3_vectors_dim)
            if torch.is_autocast_enabled():
                se3_scalars = se3_scalars.half()
                se3_vectors = se3_vectors.half() 

        # (valid_P, embed_dim), (n_points, embed_dim)
        x_emb, shape_emb = self._gen_cond(x, latent, scale)
        # (valid_P, embed_dim)
        x_emb = self._add_ref_part_emb(x_emb, ref_part)
        # broadcast x_emb to all points (n_points, embed_dim)
        x_emb = x_emb[latent["batch"]]

        # (n_points, embed_dim)
        data_emb = x_emb + shape_emb

        self_attn_seqlen = torch.bincount(latent["batch"])  # (valid_P,)
        self_attn_max_seqlen = self_attn_seqlen.max()
        self_attn_cu_seqlens = nn.functional.pad(
            torch.cumsum(self_attn_seqlen, 0), (1, 0)
        ).to(torch.int32)

        points_per_part = torch.zeros_like(part_valids, dtype=self_attn_seqlen.dtype)
        points_per_part[part_valids] = self_attn_seqlen
        global_attn_seqlen = points_per_part.sum(1)
        global_attn_max_seqlen = global_attn_seqlen.max()
        global_attn_cu_seqlens = nn.functional.pad(
            torch.cumsum(global_attn_seqlen, 0), (1, 0)
        ).to(torch.int32)
        
        for i, layer in enumerate(self.transformer_layers):
            data_emb = layer(
                hidden_states=data_emb,
                timestep=timesteps,
                batch=latent["batch"],
                self_attn_seqlens=self_attn_seqlen,
                self_attn_cu_seqlens=self_attn_cu_seqlens,
                self_attn_max_seqlen=self_attn_max_seqlen,
                global_attn_seqlens=global_attn_seqlen,
                global_attn_cu_seqlens=global_attn_cu_seqlens,
                global_attn_max_seqlen=global_attn_max_seqlen,
                se3_scalars=se3_scalars if self.cfg.model.se3 else None,
                se3_vectors=se3_vectors if self.cfg.model.se3 else None,
            )
        
        # scatter to each part
        data_emb = torch_scatter.segment_csr(
            data_emb,
            self_attn_cu_seqlens.long(),
            reduce="mean",
        )  # (valid_P, embed_dim)

        # data_emb (B, N*L, C)
        out_trans_rots = self._out(data_emb)

        return out_trans_rots # (valid_P, 7)


