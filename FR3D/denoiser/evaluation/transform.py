from pytorch3d.transforms import quaternion_apply
from pytorch3d import transforms
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
        where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    # repeat to e.g. apply the same quat for all points in a point cloud
    # [4] --> [N, 4], [B, 4] --> [B, N, 4], [B, P, 4] --> [B, P, N, 4]
    if len(q.shape) == len(v.shape) - 1:
        q = q.unsqueeze(-2).repeat_interleave(v.shape[-2], dim=-2)
    assert q.shape[:-1] == v.shape[:-1]
    return quaternion_apply(q, v)


def qtransform(t, q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q,
        and then translate it by the translation described by t.
    Expects a tensor of shape (*, 3) for t, a tensor of shape (*, 4) for q and
        a tensor of shape (*, 3) for v, where * denotes any dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert t.shape[-1] == 3

    # repeat to e.g. apply the same trans for all points in a point cloud
    # [3] --> [N, 3], [B, 3] --> [B, N, 3], [B, P, 3] --> [B, P, N, 3]
    if len(t.shape) == len(v.shape) - 1:
        t = t.unsqueeze(-2).repeat_interleave(v.shape[-2], dim=-2)

    assert t.shape == v.shape

    qv = qrot(q, v)
    tqv = qv + t
    return tqv

def transform_pc(trans, rot, pc, rot_type=None):
    """Rotate and translate the 3D point cloud.

    Args:
        rot (torch.Tensor): quat
    """
    
    return qtransform(trans, rot, pc)

def transform_normal(rot, normal):
    return qrot(rot, normal)

def transform_normalize(metric):
    min = metric.min()
    max = metric.max()
    norm = (metric - min) / (max - min + 1e-6)
    scaled = norm * 2 -1
    return norm, scaled

def quaternion_to_euler(quat, to_degree=True):
    """Convert quaternion to euler angle.

    Args:
        quat: [B, 4], quat
        to_degree: bool, whether to convert to degree

    Returns:
        [B, 3], euler angle
    """

    r_mat = transforms.quaternion_to_matrix(quat)
    euler = transforms.matrix_to_euler_angles(r_mat, convention="XYZ")
    if to_degree:
        euler = torch.rad2deg(euler)

    return euler


def qeuler(q, order='xyz', epsilon=0, to_degree=True):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4

    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.view(-1, 4)

    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    if order == 'xyz':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(
            torch.clamp(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    elif order == 'yzx':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(
            torch.clamp(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
    elif order == 'zxy':
        x = torch.asin(
            torch.clamp(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'xzy':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(
            torch.clamp(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
    elif order == 'yxz':
        x = torch.asin(
            torch.clamp(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'zyx':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(
            torch.clamp(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    else:
        raise NotImplementedError

    euler = torch.stack((x, y, z), dim=1).view(original_shape)
    if to_degree:
        euler = euler * 180. / np.pi
    return euler

def translation_matrix(translation):
    batch_size = translation.shape[0]
    T = torch.eye(4).unsqueeze(0)
    T = T.repeat(batch_size, 1, 1)
    T[:, :3, 3] = translation
    T = T.to(translation.device)
    return T

def rotation_matrix_from_quaternion(quaternion):
    quaternion_np = quaternion.cpu().numpy()
    batch_size = quaternion.shape[0]
    rotation = R.from_quat(quaternion_np)
    rotation_matrix = rotation.as_matrix()
    rot_mat = np.eye(4)[np.newaxis, :, :]
    rot_mat = np.repeat(rot_mat, batch_size, axis=0) 
    rot_mat[:, :3, :3] = rotation_matrix
    rot_mat = torch.tensor(rot_mat, dtype=torch.float32).to(quaternion.device)
    return rot_mat

def pad_data(data, max_num_part):
        """Pad data to shape [`self.max_num_part`, data.shape[1], ...]."""
        data = np.array(data)
        pad_shape = (max_num_part, ) + tuple(data.shape[1:])
        pad_data = np.zeros(pad_shape, dtype=np.float32)
        pad_data[:data.shape[0]] = data
        return pad_data

def compute_final_transformation(init_pose, gt_transformation, transformation):

        trans_mat1 = translation_matrix(-init_pose[:, :3])
        rot_mat1 = rotation_matrix_from_quaternion(init_pose[:, 3:])

        trans_mat2 = translation_matrix(-gt_transformation[:, :3])
        rot_mat2 = rotation_matrix_from_quaternion(gt_transformation[:, 3:])

        trans_mat3 = translation_matrix(transformation[:, :3])
        rot_mat3 = rotation_matrix_from_quaternion(transformation[:, 3:])

        trans_mat4 = translation_matrix(init_pose[:, :3])
        rot_mat4 = rotation_matrix_from_quaternion(init_pose[:, 3:])

        # rotate -> translate -> translate -> rotate -> rotate -> translate
        final_transformation = rot_mat4 @ trans_mat4 @ trans_mat3 @ rot_mat3 @ rot_mat2 @ trans_mat2  @ trans_mat1 @ rot_mat1
        return final_transformation