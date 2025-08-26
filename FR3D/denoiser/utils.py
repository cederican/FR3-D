import torch


def recenter_pc(pc):
    """pc: [N, 3] (torch.Tensor)"""
    centroid = pc.mean(dim=0)
    pc = pc - centroid.unsqueeze(0)
    return pc, centroid


def random_rotation_matrix(device=None):
    """Uniform random rotation matrix using the method of Arvo (1992)."""
    u1, u2, u3 = torch.rand(3, device=device)
    q1 = torch.sqrt(1 - u1)
    q2 = torch.sqrt(u1)
    theta1 = 2 * torch.pi * u2
    theta2 = 2 * torch.pi * u3
    w = torch.cos(theta2) * q2
    x = torch.sin(theta1) * q1
    y = torch.cos(theta1) * q1
    z = torch.sin(theta2) * q2
    # Quaternion to rotation matrix
    rot_mat = torch.stack([
        torch.stack([1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w]),
        torch.stack([2*x*y + 2*z*w,     1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w]),
        torch.stack([2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2])
    ]) 
    quat = torch.stack([w, x, y, z])
    return rot_mat, quat

def rotate_pc(pc, normal=None):
    """pc: [N, 3] (torch.Tensor)"""
    rot_mat, quat_gt = random_rotation_matrix(device=pc.device)
    rotated_pc = (rot_mat @ pc.T).T
    if normal is None:
        return rotated_pc, None, quat_gt
    rotated_normal = (rot_mat @ normal.T).T
    return rotated_pc, rotated_normal, quat_gt


def shuffle_pc(pc, normal=None, locdists=None):
    """pc: [N, 3] (torch.Tensor)"""
    order = torch.randperm(pc.shape[0], device=pc.device)
    shuffled_pc = pc[order]
    if normal is None and locdists is None:
        return shuffled_pc, None, None, order
    if normal is not None and locdists is None:
        shuffled_normal = normal[order]
        return shuffled_pc, shuffled_normal, None, order
    shuffled_normal = normal[order]
    shuffled_locdists = locdists[order]
    return shuffled_pc, shuffled_normal, shuffled_locdists, order


def pad_data(data, num_points):
    """Pad data to shape [20 * num_points, ...] using torch."""
    data = torch.as_tensor(data)
    pad_shape = (20 * num_points, ) + data.shape[1:]
    pad_data = torch.zeros(pad_shape, dtype=data.dtype, device=data.device)
    pad_data[:data.shape[0]] = data
    return pad_data

def custom_schedule(epoch):
    if epoch < 400:
        return 1e-7 / 1e-4  # LR is scaled down to 1e-6
    elif 400 <= epoch < 500:
        # linear warmup: interpolate between 1e-6 and 1e-4
        scale = (epoch - 400) / 100  # 0 → 1
        lr = 1e-7 + scale * (1e-4 - 1e-7)
        return lr / 1e-4  # normalize to base LR
    elif 500 <= epoch < 800:
        return 1.0  # base LR
    else:
        return 0.5  # decay by 0.5

class InverseHuberLoss(torch.nn.Module):
    def __init__(self, delta=0.2):
        super().__init__()
        self.delta = delta

    def forward(self, pred, target):
        x = pred - target
        abs_x = torch.abs(x)
        c = self.delta * torch.max(abs_x).item()

        mask = abs_x <= c
        l1_part = abs_x[mask]
        l2_part = abs_x[~mask]

        loss = torch.cat([
            l1_part,
            (l2_part ** 2 + c ** 2) / (2 * c)
        ])
        return loss.mean()

def build_batched_fc_edge_index(part_valids):
    """
    fully connected inter part connections between parts
    """
    batch_size, _ = part_valids.shape

    edge_index_list = []
    valid_lens = [0]

    for b in range(batch_size):
        valid = part_valids[b].nonzero(as_tuple=False).squeeze(-1)
        offset = torch.sum(torch.tensor(valid_lens))
        valid_lens.append(valid.shape[0])

        sender, receiver = torch.meshgrid(valid, valid, indexing='ij')
        sender = sender.flatten()
        receiver = receiver.flatten()

        mask = sender != receiver
        sender = sender[mask]
        receiver = receiver[mask]

        sender = sender + offset
        receiver = receiver + offset

        edge_index_list.append(torch.stack([sender, receiver], dim=0))

    if edge_index_list:
        edge_index = torch.cat(edge_index_list, dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return edge_index

def build_batched_edge_index(part_conn_batch):
    """
    centers of parts are connected to each other
    """
    batch_size, _, _ = part_conn_batch.shape
    
    batch_idx, sender_idx, receiver_idx = part_conn_batch.nonzero(as_tuple=True)
    
    global_senders = torch.zeros_like(sender_idx)
    global_receivers = torch.zeros_like(receiver_idx)
    valid_lens = [0]
    
    for b in range(batch_size):
        batch_mask = batch_idx == b
        if batch_mask.any():
            nodes_in_batch = torch.cat((receiver_idx[batch_mask], sender_idx[batch_mask])).unique().numel()
            offset = torch.sum(torch.tensor(valid_lens))
            valid_lens.append(nodes_in_batch)

            global_senders[batch_mask] = sender_idx[batch_mask] + offset
            global_receivers[batch_mask] = receiver_idx[batch_mask] + offset

    edge_index = torch.stack([global_senders, global_receivers], dim=0)
    
    return edge_index

def build_fully_connected_intra_part_edge_index(part_conn_batch, num_points_per_part):
    """
    Builds edge_index connecting all points within a part
    """
    B, _, _ = part_conn_batch.shape
    b_idx, s_part, r_part = part_conn_batch.nonzero(as_tuple=True)
    
    s_part_sequential = torch.zeros_like(s_part)
    r_part_sequential = torch.zeros_like(r_part)
    valid_lens = [0]

    for b in range(B):
        batch_mask = b_idx == b
        if batch_mask.any():
            unique_parts = torch.cat((s_part[batch_mask], r_part[batch_mask])).unique()

            part_to_seq = {p.item(): i for i, p in enumerate(unique_parts)}

            s_part_sequential[batch_mask] = torch.tensor([part_to_seq[p.item()] for p in s_part[batch_mask]], device=s_part.device)
            r_part_sequential[batch_mask] = torch.tensor([part_to_seq[p.item()] for p in r_part[batch_mask]], device=r_part.device)
            valid_lens.append(len(unique_parts))

    offsets = torch.cumsum(torch.tensor(valid_lens[:-1], device=part_conn_batch.device), dim=0)
    batch_offsets = torch.zeros_like(b_idx)
    for b in range(B):
        batch_mask = b_idx == b
        if batch_mask.any():
            batch_offsets[batch_mask] = offsets[b]

    s_offset = (batch_offsets + s_part_sequential) * num_points_per_part  
    s_offset, _ = torch.unique(s_offset, return_counts=True)

    num_samples = 100
    perm = torch.randperm(s_offset.size(0), device=s_offset.device)[:num_samples]
    sorted_indices = torch.sort(perm).values
    s_offset = s_offset[sorted_indices]
    
    local_ids = torch.arange(num_points_per_part, device=part_conn_batch.device) 
    s_ids = s_offset[:, None] + local_ids[None, :]
    
    s_grid = s_ids[:, :, None].expand(-1, num_points_per_part, num_points_per_part).reshape(-1)
    r_grid = s_ids[:, None, :].expand(-1, num_points_per_part, num_points_per_part).reshape(-1)
    
    mask = s_grid != r_grid
    senders = s_grid[mask]
    receivers = r_grid[mask]
    
    edge_index = torch.stack([senders, receivers], dim=0)

    return edge_index