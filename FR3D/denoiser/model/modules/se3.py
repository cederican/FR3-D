from e3nn.o3 import FullyConnectedTensorProduct, Irreps, spherical_harmonics, rand_matrix, Linear
from e3nn.math import soft_one_hot_linspace, soft_unit_step
from e3nn.nn import FullyConnectedNet

from torch_scatter import scatter
from torch_geometric.nn import radius_graph
import torch.nn as nn
import torch
import math

class SE3Attention(nn.Module):
    def __init__(self, irreps_in, irreps_query, irreps_key, irreps_out, num_radial=10):
        super().__init__()
        
        self.irreps_in = irreps_in
        self.irreps_query = irreps_query 
        self.irreps_key = irreps_key
        self.irreps_out = irreps_out
        self.num_radial = num_radial
        
        self.irreps_sh = Irreps.spherical_harmonics(3)
        
        self.query_proj = Linear(self.irreps_in, self.irreps_query)
        
        self.key_proj = FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_sh,
            self.irreps_key,
            shared_weights=False
        )
        self.fc_k = FullyConnectedNet([self.num_radial, 16, self.key_proj.weight_numel], act=torch.nn.functional.silu)

        self.value_proj = FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_sh,
            self.irreps_out,
            shared_weights=False
        )
        self.fc_v = FullyConnectedNet([self.num_radial, 16, self.value_proj.weight_numel], act=torch.nn.functional.silu)
        
        self.dot = FullyConnectedTensorProduct(
            self.irreps_query, 
            self.irreps_key, 
            "0e")
        
    
    def forward(self, x, pos, edge_index):
        W, C = x.shape  # (valid_P*L, num_features)
        assert pos.shape == (W, 3), "Position tensor must have shape (B, N, 3)"
        assert edge_index.shape[0] == 2, "Edge index must have shape (2, num_edges)"
        
        #x = x.flatten(0, 1)  # (B*N, C)
        #pos = pos.flatten(0, 1)  # (B*N, 3)
        
        #senders1, receivers1 = radius_graph(pos, max_radius)
        senders, receivers = edge_index  # (2, num_edges)
        pos_diff = pos[receivers] - pos[senders]  # (num_edges, 3)
        distances = pos_diff.norm(dim=-1)  # (num_edges,)
        max_radius = distances.max().item()
        
        # new try of se3 attention
        edge_attr_sph = spherical_harmonics(
            self.irreps_sh,
            x=pos_diff,
            normalize=True,
            normalization='component'
        )
        
        edge_attr_radial = soft_one_hot_linspace(
            x=distances,
            start=0.0,
            end=max_radius,
            number=self.num_radial,
            basis='smooth_finite',
            cutoff=True
        )
        edge_attr_radial = edge_attr_radial.mul(self.num_radial**0.5)
        edge_weight_cutoff = soft_unit_step(10 * (1 - distances /max_radius))

        x_s = x[senders]  # shape (num_edges, feat_dim)
        edge_sph = edge_attr_sph
        edge_rad = self.fc_v(edge_attr_radial)

        # Check inputs BEFORE tensor product
        if torch.isnan(x_s).any():
            print("NaNs in x[senders]")
            x_s = torch.nan_to_num(x_s)
        if torch.isnan(edge_sph).any():
            print("NaNs in edge_attr_sph")
            edge_sph = torch.nan_to_num(edge_sph)
        if torch.isnan(edge_rad).any():
            print("NaNs in edge_attr_radial (fc_v)")
            edge_rad = torch.nan_to_num(edge_rad)
        
        queries = self.query_proj(x) # (num_nodes, hidden_dim)
        keys = self.key_proj(x[senders], edge_attr_sph, self.fc_k(edge_attr_radial))  # (num_edges, hidden_dim)
        values = self.value_proj(x_s, edge_sph, edge_rad)  # (num_edges, out_dim)
        
        raw_logits = self.dot(queries[receivers], keys) / math.sqrt(queries.shape[-1])  # (num_edges, hidden_dim)
        # Numerically stable exp
        max_logits = scatter(raw_logits, receivers, dim=0, dim_size=x.size(0), reduce="max")
        stabilized_logits = raw_logits - max_logits[receivers]
        stabilized_logits = stabilized_logits.clamp(max=30.0)

        attention_logits = edge_weight_cutoff[:, None] * stabilized_logits.exp()  # (num_edges, hidden_dim)
        attention_weights = scatter(attention_logits, receivers, dim=0, dim_size=x.size(0))  # (num_edges, hidden_dim)
        alpha = attention_logits / attention_weights[receivers].clamp_min(1e-5) # (num_edges, hidden_dim)
        
        f_out = scatter(alpha.relu().clamp_min(1e-5).sqrt() * values, receivers, dim=0, dim_size=x.size(0))  # (num_edges, hidden_dim)
        
        return f_out

if __name__ == "__main__":
    
    irreps_in=Irreps("512x0e + 1x1e")
    irreps_query=Irreps("506x0e + 2x1e")
    irreps_key=Irreps("506x0e + 2x1e")
    irreps_out=Irreps("500x0e + 4x1e")
    
    attention = SE3Attention(
        irreps_in=irreps_in,
        irreps_query=irreps_query,
        irreps_key=irreps_key,
        irreps_out=irreps_out,
        num_radial=10
    )
    
    batch_size = 3
    f = irreps_in.randn(batch_size*20, -1)
    pos = torch.randn(batch_size*20, 3)
    edge_index = torch.randint(0, batch_size * 20, (2, 20))
    
    rot = rand_matrix()
    D_in = irreps_in.D_from_matrix(rot)
    D_out = irreps_out.D_from_matrix(rot)

    f_before = attention(f @ D_in.T, pos @ rot.T, edge_index)
    f_after = attention(f, pos, edge_index) @ D_out.T

    print(torch.allclose(f_before, f_after, atol=1e-3, rtol=1e-3))