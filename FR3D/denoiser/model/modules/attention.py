"""
Adapted from puzzlefusion++
https://github.com/eric-zqwang/puzzlefusion-plusplus

Adapted from GARF
https://github.com/ai4ce/GARF
"""

import torch
import torch.nn as nn
from typing import Optional
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from e3nn.o3 import Irreps
from FR3D.denoiser.model.modules.se3 import SE3Attention

import flash_attn

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        return x / keep_prob * binary_tensor


class MyAdaLayerNorm(nn.Module):
    """
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__()
        self.timestep_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.timestep_embbedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim
        )
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        emb = self.linear(
            self.silu(self.timestep_embbedder(self.timestep_proj(timestep)))
        )  # (n_points, embedding_dim * 2)
        # (valid_P, embedding_dim), (valid_P, embedding_dim)
        scale, shift = emb.chunk(2, dim=1)
        # broadcast to (n_points, embedding_dim)
        scale = scale[batch]
        shift = shift[batch]

        return self.norm(x) * (1 + scale) + shift


class SE3EncoderLayer(nn.Module):
    def __init__(self,
        norm_elementwise_affine: bool,
        latent_dim: int,
    ):
        super().__init__()

        self.latent_norm = nn.LayerNorm(latent_dim)

        self.intra_se3_attn = SE3Attention(
            irreps_in=Irreps("64x0e + 2x1e"),
            irreps_query=Irreps("128x0e + 8x1e"),
            irreps_key=Irreps("128x0e + 8x1e"),
            irreps_out=Irreps("64x0e + 2x1e"),
            num_radial=10
        )
        self.intra_norm = nn.LayerNorm(self.intra_se3_attn.irreps_out[0].dim, elementwise_affine=True)

        self.inter_se3_attn = SE3Attention(
            irreps_in=Irreps("64x0e + 2x1e"),
            irreps_query=Irreps("128x0e + 8x1e"),
            irreps_key=Irreps("128x0e + 8x1e"),
            irreps_out=Irreps("64x0e + 2x1e"),
            num_radial=10
        )
        self.inter_norm = nn.LayerNorm(self.inter_se3_attn.irreps_out[0].dim, elementwise_affine=True)

        self.fc_inter_se3_attn = SE3Attention(
            irreps_in=Irreps("64x0e + 2x1e"),
            irreps_query=Irreps("128x0e + 8x1e"),
            irreps_key=Irreps("128x0e + 8x1e"),
            irreps_out=Irreps("64x0e + 2x1e"),
            num_radial=10
        )
        self.fc_inter_norm = nn.LayerNorm(self.fc_inter_se3_attn.irreps_out[0].dim, elementwise_affine=True)

        self.v_mlp = nn.Sequential(
            nn.Linear(6, 16),
            nn.SiLU(),
            nn.Linear(16, 8)
        )

    def forward(self,
                noisy_part_centers: torch.Tensor,
                latent: torch.Tensor,
                xyz: torch.Tensor,
                point_edge_index: torch.Tensor,
                part_edge_index: torch.Tensor,
                fc_part_edge_index: torch.Tensor,
                real_bones: Optional[torch.Tensor] = None            
    ):
        
        valid_P, L, _ = xyz.shape # (valid_P, L, 3)
        latent = self.latent_norm(latent)

        npc = noisy_part_centers.unsqueeze(1).repeat(1, L, 1)
        se3_states = torch.cat([latent, xyz, npc], dim=-1) # valid_P, L, 64 + 6
        se3_states = se3_states.flatten(0,1) # valid_P*L, 70

        if not real_bones is True:
            # 1. intra part attention
            pos = xyz.flatten(0,1) # valid_P*L, 3
            intra_out = self.intra_se3_attn(se3_states, pos, point_edge_index)
            se3_states = se3_states + intra_out # valid_P*L, 70
            
        pos = noisy_part_centers # valid_P, 3
        se3_states = se3_states.view(valid_P, L, -1).mean(dim=1) # valid_P, C

        scalars = se3_states[..., :self.intra_se3_attn.irreps_out[0].dim]
        scalars = self.intra_norm(scalars)
        vectors = se3_states[..., self.intra_se3_attn.irreps_out[0].dim:]
        se3_vec1, se3_vec2 = self.normalize_vectors(vectors[..., :3], vectors[..., 3:])
        se3_states = torch.cat([scalars, se3_vec1, se3_vec2], dim=-1)

        if not real_bones is True:
            # 2. inter part attention
            inter_out = self.inter_se3_attn(se3_states, pos, part_edge_index)
            se3_states = se3_states + inter_out # valid_P, 70
        
        scalars = se3_states[..., :self.inter_se3_attn.irreps_out[0].dim]
        scalars = self.inter_norm(scalars)
        vectors = se3_states[..., self.inter_se3_attn.irreps_out[0].dim:]
        se3_vec1, se3_vec2 = self.normalize_vectors(vectors[..., :3], vectors[..., 3:])
        se3_states = torch.cat([scalars, se3_vec1, se3_vec2], dim=-1)
        
        # 3. fc inter part attention
        fc_inter_out = self.fc_inter_se3_attn(se3_states, pos, fc_part_edge_index)
        se3_states = se3_states + fc_inter_out # valid_P, 70
        
        # 4. reshape
        se3_states = se3_states.unsqueeze(1).expand(-1, L, -1)
        #se3_states = se3_states.reshape(B, N*L, -1) # B, N*L, 70
        
        # 5. normalize scalars and vectors
        se3_scalars = se3_states[..., :self.inter_se3_attn.irreps_out[0].dim]
        se3_scalars = self.fc_inter_norm(se3_scalars)
        se3_vectors = se3_states[..., self.inter_se3_attn.irreps_out[0].dim:]
        se3_vec1, se3_vec2 = self.normalize_vectors(se3_vectors[..., :3], se3_vectors[..., 3:]) # valid_P, L, 4
        se3_vectors = self._prepare_se3_vector_out(se3_vec1, se3_vec2)

        return se3_scalars, se3_vectors
    
    def _prepare_se3_vector_out(self, vec1: torch.Tensor = None, vec2: torch.Tensor = None) -> torch.Tensor:
        """
        Prepare input for SE(3) equivariant attention.
        """
        cat = torch.cat([vec1, vec2], dim=-1)
        return self.v_mlp(cat)
    
    def normalize_vectors(self, vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
        """
        Normalize vectors to unit length.
        """
        return torch.nn.functional.normalize(vec1, dim=-1, eps=1e-6), torch.nn.functional.normalize(vec2, dim=-1, eps=1e-6)
        

class EncoderLayer(nn.Module):
    def __init__(self, 
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: int = None,
        attention_bias: bool = False,
        norm_elementwise_affine: bool = True,
        final_dropout: bool = False,
        se3: bool = False,
        path_drop: float = 0.0,
        irreps_out_dim: int = 8,
        irreps_in_dim: int = 64,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        self.se3 = se3
        
        #  1. self attention
        self.norm1 = MyAdaLayerNorm(dim, num_embeds_ada_norm) 

        self.self_attn_to_qkv = nn.Linear(dim, dim * 3, bias=attention_bias)
        self.self_attn_to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

        # 2. global attention
        self.norm2 = MyAdaLayerNorm(dim, num_embeds_ada_norm)

        self.global_attn_to_qkv = nn.Linear(dim, dim * 3, bias=attention_bias)
        self.global_attn_to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        
        if self.se3:
            self.norm3 = nn.LayerNorm(dim + irreps_in_dim + irreps_out_dim, elementwise_affine=norm_elementwise_affine)
            self.ff = FeedForward(
                dim + irreps_in_dim + irreps_out_dim, 
                dim,
                dropout=dropout, 
                activation_fn=activation_fn, 
                final_dropout=final_dropout,
                embedding_dim=None,
                num_embeddings=None,
            )   
        else:
            self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
            self.ff = FeedForward(
                    dim,
                    dropout=dropout, 
                    activation_fn=activation_fn, 
                    final_dropout=final_dropout,
                    embedding_dim=None,
                    num_embeddings=None,
                )
        self.drop_path1 = DropPath(drop_prob=path_drop)
        self.drop_path2 = DropPath(drop_prob=path_drop)
        self.drop_path3 = DropPath(drop_prob=path_drop)

    def forward(
        self,
        hidden_states: torch.Tensor,  # (n_points, embed_dim)
        timestep: torch.Tensor,  # (valid_P,)
        batch: torch.Tensor,  # (valid_P,)
        self_attn_seqlens: torch.Tensor,
        self_attn_cu_seqlens: torch.Tensor,
        self_attn_max_seqlen: torch.Tensor,
        global_attn_seqlens: torch.Tensor,
        global_attn_cu_seqlens: torch.Tensor,
        global_attn_max_seqlen: torch.Tensor,
        se3_scalars=None,
        se3_vectors=None,
    ):
        n_points, embed_dim = hidden_states.shape
        # 1. self attention
        norm_hidden_states = self.norm1(hidden_states, timestep, batch)

        attn_output = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv=self.self_attn_to_qkv(norm_hidden_states).half().reshape(
                n_points, 3, self.num_attention_heads, self.attention_head_dim
            ),
            cu_seqlens=self_attn_cu_seqlens,
            max_seqlen=self_attn_max_seqlen,
            dropout_p=0.0,
        ).view(n_points, embed_dim)
        attn_output = self.self_attn_to_out(attn_output.to(norm_hidden_states.dtype))
        hidden_states = hidden_states + attn_output

        # 2. global attention
        norm_hidden_states = self.norm2(hidden_states, timestep, batch)

        global_out_flash = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv=self.global_attn_to_qkv(norm_hidden_states).half().reshape(
                n_points, 3, self.num_attention_heads, self.attention_head_dim
            ),
            cu_seqlens=global_attn_cu_seqlens,
            max_seqlen=global_attn_max_seqlen,
            dropout_p=0.0,
        ).view(n_points, embed_dim)
        global_out_flash = self.global_attn_to_out(global_out_flash.to(norm_hidden_states.dtype))
        hidden_states = hidden_states + global_out_flash

        # 3. feed forward
        if self.se3:
            hidden_states_plus = torch.cat([hidden_states, se3_scalars, se3_vectors], dim=-1) 
            norm_hidden_states = self.norm3(hidden_states_plus)
        else:
            norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states

        return hidden_states  # (n_points, embed_dim)

    
        
        