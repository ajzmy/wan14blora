import torch
import numpy as np
import random


# Pure-PyTorch replacement for pytorch3d.transforms.quaternion_to_matrix.
# Converts quaternions in wxyz format to 3x3 rotation matrices.
def quaternion_to_matrix(quaternions):
    """
    Args:
        quaternions: (..., 4) tensor, [w, x, y, z]
    Returns:
        (..., 3, 3) rotation matrix tensor
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack([
        1 - two_s * (j * j + k * k),
        two_s * (i * j - k * r),
        two_s * (i * k + j * r),
        two_s * (i * j + k * r),
        1 - two_s * (i * i + k * k),
        two_s * (j * k - i * r),
        two_s * (i * k - j * r),
        two_s * (j * k + i * r),
        1 - two_s * (i * i + j * j),
    ], -1)
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def gen_batch_ray_parellel(intrinsic, c2w, W, H):
    batch_size = intrinsic.shape[0]
    fx = intrinsic[:, 0, 0].unsqueeze(1).unsqueeze(2)
    fy = intrinsic[:, 1, 1].unsqueeze(1).unsqueeze(2)
    cx = intrinsic[:, 0, 2].unsqueeze(1).unsqueeze(2)
    cy = intrinsic[:, 1, 2].unsqueeze(1).unsqueeze(2)
    i, j = torch.meshgrid(
        torch.linspace(0.5, W - 0.5, W, device=c2w.device),
        torch.linspace(0.5, H - 0.5, H, device=c2w.device),
    )
    i = i.t().unsqueeze(0).repeat(batch_size, 1, 1)
    j = j.t().unsqueeze(0).repeat(batch_size, 1, 1)
    dirs = torch.stack([(i - cx) / fx, (j - cy) / fy, torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:, np.newaxis, np.newaxis, :3, :3], -1)
    rays_o = c2w[:, :3, -1].unsqueeze(1).unsqueeze(2).repeat(1, H, W, 1)
    viewdir = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    return rays_d, rays_o, viewdir


def intrinsic_transform(intrinsic, original_res, size, transform_mode):
    fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    original_height, original_width = original_res
    if transform_mode == 'resize':
        scale_height = size[0] / original_height
        scale_width  = size[1] / original_width
        fx_new = fx * scale_width
        fy_new = fy * scale_height
        cx_new = cx * scale_width
        cy_new = cy * scale_height
    elif transform_mode == 'center_crop_resize':
        if original_height <= original_width:
            scale_ratio = min(size) / original_height
        else:
            scale_ratio = min(size) / original_width
        resize_height = scale_ratio * original_height
        resize_width  = scale_ratio * original_width
        fx_new = fx * scale_ratio
        fy_new = fy * scale_ratio
        cx_new = cx * scale_ratio
        cy_new = cy * scale_ratio
        cx_new = cx_new * (size[1] / resize_width)
        cy_new = cy_new * (size[0] / resize_height)
    else:
        raise NotImplementedError(f'No such transformation mode: {transform_mode}')
    return torch.tensor([[fx_new, 0, cx_new],
                         [0, fy_new, cy_new],
                         [0, 0, 1]])


def get_transformation_matrix_from_quat(xyz_quat):
    """
    Args:
        xyz_quat: (B, 7) tensor  [x, y, z, qx, qy, qz, qw]
    Returns:
        (B, 4, 4) transformation matrices
    """
    rot_quat = xyz_quat[:, 3:]
    # pytorch3d quaternion_to_matrix expects wxyz; convert from xyzw
    rot_quat = rot_quat[:, [3, 0, 1, 2]]
    rot   = quaternion_to_matrix(rot_quat)
    trans = xyz_quat[:, :3]
    output = torch.eye(4).unsqueeze(0).repeat(xyz_quat.shape[0], 1, 1)
    output[:, :3, :3] = rot
    output[:, :3,  3] = trans
    return output
