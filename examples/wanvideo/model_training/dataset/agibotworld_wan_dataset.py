"""
AgiBotWorld dataset for WAN 1.3B training.
- Subsamples 30fps video/actions to 5fps (stride=6)
- Returns 25 frames per episode (4*6+1, satisfies WAN 4n+1 constraint)
- Episodes with fewer than 25 frames after subsampling are filtered out
- Provides: video, first_frame, traj_maps, raymap, delta_action, prompt
"""

import os
import sys
import json
import random
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from einops import rearrange
import torchvision.transforms as transforms

# Add evac to path for baseline utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'evac'))

from lvdm.data.traj_vis_statistics import (
    ColorMapLeft, ColorMapRight, ColorListLeft, ColorListRight,
    EndEffectorPts, Gripper2EEFCvt,
)
from lvdm.data.get_actions import parse_h5
from lvdm.data.utils import gen_batch_ray_parellel, intrinsic_transform
from lvdm.data.statistics import StatisticInfo


SUBSAMPLE_STRIDE = 6        # 30fps → 5fps
TARGET_FRAMES    = 29       # 4*7+1, WAN constraint
PROMPT           = "robot arm manipulation"  #比赛未给任务描述，这个是自定义的

class AgiBotWorldWanDataset(Dataset):
    def __init__(
        self,
        data_root,
        split="train",
        sample_size=(480, 640),
        traj_radius=50,
    ):
        """
        Args:
            data_root:   path containing split directories (e.g. .../iros_challenge_2025_acwm)
            split:       'train' or 'val'
            sample_size: (H, W) to resize frames to
            traj_radius: circle radius for traj map visualisation
        """
        self.sample_size     = sample_size   # (H, W)
        self.traj_radius     = traj_radius
        self.load_from_cache = False         # required by DiffSynth launch_training_task

        split_dir = os.path.join(data_root, split)
        all_paths = sorted([
            os.path.join(split_dir, ep)
            for ep in os.listdir(split_dir)
        ])

        # Keep all valid episodes; record available frame count for padding in _get_indices
        self.episode_paths = []
        self.episode_avail = []   # available frames after subsampling (may be < TARGET_FRAMES)
        for ep in all_paths:
            h5_path = os.path.join(ep, 'proprio_stats.h5')
            try:
                with h5py.File(h5_path, 'r') as f:
                    total = f['timestamp'].shape[0]
                available = (total - 1) // SUBSAMPLE_STRIDE + 1
                if available >= 1:
                    self.episode_paths.append(ep)
                    self.episode_avail.append(available)
            except Exception:
                pass

        n_short = sum(1 for a in self.episode_avail if a < TARGET_FRAMES)
        self.norm = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )

        print(f"[AgiBotWorldWanDataset] {split}: {len(self.episode_paths)} / {len(all_paths)} episodes "
              f"({n_short} short episodes will be tail-padded to {TARGET_FRAMES} frames)")

    # ------------------------------------------------------------------
    # Frame index helpers
    # ------------------------------------------------------------------

    def _get_indices(self, total_frames):
        """
        Subsample indices: stride=6 (30fps→5fps), random start, TARGET_FRAMES long.
        If the episode is shorter than TARGET_FRAMES after subsampling,
        pad by repeating the last frame index.
        """
        indices = list(range(0, total_frames, SUBSAMPLE_STRIDE))
        available = len(indices)

        if available >= TARGET_FRAMES:
            # Normal case: random start window
            max_start = available - TARGET_FRAMES
            start = random.randint(0, max_start)
            indices = indices[start:start + TARGET_FRAMES]
        else:
            # Short episode: use all frames then pad with last frame
            pad = TARGET_FRAMES - available
            indices = indices + [indices[-1]] * pad

        return indices

    # ------------------------------------------------------------------
    # Video loading
    # ------------------------------------------------------------------

    def _load_video_frames(self, video_path, indices):
        """Return (T, H_orig, W_orig, 3) uint8 numpy array."""
        cap = cv2.VideoCapture(video_path)
        idx_set   = set(indices)
        frame_map = {}
        fi = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if fi in idx_set:
                frame_map[fi] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fi += 1
        cap.release()
        return np.stack([frame_map[i] for i in indices if i in frame_map])

    # ------------------------------------------------------------------
    # Camera parameters
    # ------------------------------------------------------------------

    def _load_camera_params(self, ep_path, indices):
        """
        Returns:
            intrinsic : (3, 3) float32 tensor (original resolution)
            c2ws      : (T, 4, 4)
            w2cs      : (T, 4, 4)
        """
        with open(os.path.join(ep_path, 'head_intrinsic_params.json')) as f:
            intr = json.load(f)['intrinsic']

        intrinsic = torch.eye(3, dtype=torch.float32)
        intrinsic[0, 0] = intr['fx']
        intrinsic[1, 1] = intr['fy']
        intrinsic[0, 2] = intr['ppx']
        intrinsic[1, 2] = intr['ppy']

        with open(os.path.join(ep_path, 'head_extrinsic_params_aligned.json')) as f:
            extr_list = json.load(f)

        c2ws, w2cs = [], []
        for i in indices:
            info  = extr_list[i]
            c2w   = torch.eye(4, dtype=torch.float32)
            c2w[:3, :3] = torch.tensor(
                info['extrinsic']['rotation_matrix'], dtype=torch.float32)
            c2w[:3,  3] = torch.tensor(
                info['extrinsic']['translation_vector'], dtype=torch.float32)
            w2c = torch.linalg.inv(c2w)
            c2ws.append(c2w)
            w2cs.append(w2c)

        return intrinsic, torch.stack(c2ws), torch.stack(w2cs)

    # ------------------------------------------------------------------
    # Action data
    # ------------------------------------------------------------------
    def get_action_bias_std(self, domain_name="agibotworld"):
        return torch.tensor(StatisticInfo[domain_name]['mean']).unsqueeze(0), torch.tensor(StatisticInfo[domain_name]['std']).unsqueeze(0)

    def _load_actions(self, ep_path, indices):
        """
        Returns:
            action       : (T, 16)   abs action  [xyz_l, quat_l(xyzw), grip_l,
                                                   xyz_r, quat_r(xyzw), grip_r]
            delta_action : (T-1, 14) delta action [xyz_l, rpy_l, grip_l,
                                                    xyz_r, rpy_r, grip_r]
                           tail-padded frames (repeated last index) have delta=0 automatically
        """
        h5_path = os.path.join(ep_path, 'proprio_stats.h5')
        abs_action, delta = parse_h5(h5_path, slices=indices, delta_act_sidx=1)
        # abs_action : (T, 16),  delta : (T-1, 14)

        action       = torch.tensor(abs_action, dtype=torch.float32)
        delta_action = torch.tensor(delta, dtype=torch.float32)  # (T-1, 14)
        # normalise delta_action using dataset statistics
        delta_act_meanv, delta_act_stdv = self.get_action_bias_std()
        delta_action[:, :6] = (delta_action[:, :6] - SUBSAMPLE_STRIDE*delta_act_meanv[:, :6]) / (SUBSAMPLE_STRIDE*delta_act_stdv[:, :6])
        delta_action[:, 7:13] = (delta_action[:, 7:13] - SUBSAMPLE_STRIDE*delta_act_meanv[:, 6:]) / (SUBSAMPLE_STRIDE*delta_act_stdv[:, 6:])
        return action, delta_action

    # ------------------------------------------------------------------
    # Trajectory map
    # ------------------------------------------------------------------

    def _get_traj(self, action, w2cs, intrinsic):
        """
        Visualise end-effector positions as coloured circles/lines.
        Adapted directly from baseline AgiBotWorldICRA26Challenge.get_traj().

        Args:
            action    : (T, 16)
            w2cs      : (T, 4, 4)
            intrinsic : (3, 3)  **already rescaled to sample_size**

        Returns:
            traj : (3, T, H, W) float32 in [0,1]
        """
        H, W = self.sample_size

        ee_key_pts = torch.tensor(
            EndEffectorPts, dtype=torch.float32).view(1, 4, 4).permute(0, 2, 1)
        cvt_matrix = torch.tensor(
            Gripper2EEFCvt, dtype=torch.float32).view(1, 4, 4)

        from lvdm.data.utils import get_transformation_matrix_from_quat
        pose_l_mat = get_transformation_matrix_from_quat(action[:, 0:7])   # (T,4,4)
        pose_r_mat = get_transformation_matrix_from_quat(action[:, 8:15])  # (T,4,4)

        ee2cam_l = torch.matmul(torch.matmul(w2cs, pose_l_mat), cvt_matrix)
        ee2cam_r = torch.matmul(torch.matmul(w2cs, pose_r_mat), cvt_matrix)

        pts_l = torch.matmul(ee2cam_l, ee_key_pts)  # (T,4,4)
        pts_r = torch.matmul(ee2cam_r, ee_key_pts)

        K = intrinsic.unsqueeze(0)  # (1,3,3)
        uvs_l = (torch.matmul(K, pts_l[:, :3, :]) / pts_l[:, 2:3, :])[:, :2, :] \
                    .permute(0, 2, 1).to(torch.int64)  # (T,4,2)
        uvs_r = (torch.matmul(K, pts_r[:, :3, :]) / pts_r[:, 2:3, :])[:, :2, :] \
                    .permute(0, 2, 1).to(torch.int64)

        img_list = []
        for i in range(action.shape[0]):
            img = np.zeros((H, W, 3), dtype=np.uint8) + 50

            nl = np.clip(action[i, 7].item() / 120.0, 0.0, 1.0)
            nr = np.clip(action[i, 15].item() / 120.0, 0.0, 1.0)
            color_l = tuple(int(c * 255) for c in ColorMapLeft(nl)[:3])
            color_r = tuple(int(c * 255) for c in ColorMapRight(nr)[:3])

            for uvs, color in [(uvs_l[i], color_l), (uvs_r[i], color_r)]:
                base = uvs[0].numpy()
                if 0 <= base[0] < W and 0 <= base[1] < H:
                    cv2.circle(img, tuple(base[:2]), self.traj_radius, color, -1)

            for uvs, colors in [(uvs_l[i], ColorListLeft), (uvs_r[i], ColorListRight)]:
                base = uvs[0].numpy()
                if 0 <= base[0] < W and 0 <= base[1] < H:
                    for j in range(1, len(uvs)):
                        pt = uvs[j].numpy()
                        cv2.line(img, tuple(base[:2]), tuple(pt[:2]), colors[j - 1], 8)

            img_list.append(img / 255.0)

        traj = rearrange(
            torch.tensor(np.stack(img_list), dtype=torch.float32),
            't h w c -> c t h w'
        )
        return traj  # (3, T, H, W) in [0,1]

    # ------------------------------------------------------------------
    # Ray map
    # ------------------------------------------------------------------

    def _get_raymap(self, intrinsic, c2ws):
        """
        Compute per-pixel ray direction and origin maps using gen_batch_ray_parellel.

        Args:
            intrinsic : (3, 3)   already rescaled to sample_size
            c2ws      : (T, 4, 4)

        Returns:
            raymap : (6, T, H, W)  [dir(3) | origin(3)]
        """
        H, W = self.sample_size
        # gen_batch_ray_parellel expects batched (B,3,3) and (B,4,4)
        K    = intrinsic.unsqueeze(0).expand(len(c2ws), -1, -1)   # (T,3,3)
        c2w  = c2ws                                                 # (T,4,4)

        rays_d, rays_o, _ = gen_batch_ray_parellel(K, c2w, W, H)
        # rays_d, rays_o : (T, H, W, 3)

        # Normalise origins to unit-length for stable network input
        rays_o_norm = rays_o / (rays_o.norm(dim=-1, keepdim=True) + 1e-8)

        raymap = torch.cat([
            rearrange(rays_d.float(),      't h w c -> c t h w'),
            rearrange(rays_o_norm.float(), 't h w c -> c t h w'),
        ], dim=0)  # (6, T, H, W)
        return raymap

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.episode_paths)

    def __getitem__(self, idx):
        while True:
            try:
                return self._load_sample(self.episode_paths[idx])
            except Exception as e:
                print(f"Error loading {self.episode_paths[idx]}: {e}")
                idx = random.randint(0, len(self.episode_paths) - 1)

    def _load_sample(self, ep_path):
        H, W = self.sample_size

        # 1. Determine frame indices
        with open(os.path.join(ep_path, 'head_extrinsic_params_aligned.json')) as f:
            total_frames = len(json.load(f))
        indices = self._get_indices(total_frames)
        T = len(indices)

        # 2. Load video frames → (3, T, H, W) float in [-1,1]
        raw_frames = self._load_video_frames(
            os.path.join(ep_path, 'head_color.mp4'), indices
        )  # (T, H_orig, W_orig, 3)
        frames = torch.from_numpy(raw_frames).permute(3, 0, 1, 2).float() / 255.0
        # Resize: treat T as batch
        frames = F.interpolate(
            frames.permute(1, 0, 2, 3), size=(H, W), mode='bilinear', align_corners=False
        ).permute(1, 0, 2, 3)  # (3, T, H, W)
        video      = self.norm(frames.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
        first_frame = video[:, 0, :, :]  # (3, H, W)

        # 3. Camera params (load with original intrinsic, then rescale)
        intrinsic_orig, c2ws, w2cs = self._load_camera_params(ep_path, indices)
        orig_h = raw_frames.shape[1]
        orig_w = raw_frames.shape[2]
        intrinsic = intrinsic_transform(
            intrinsic_orig, (orig_h, orig_w), (H, W), 'resize'
        )  # (3,3)

        # 4. Actions
        action, delta_action = self._load_actions(ep_path, indices)
        # action : (T,16),  delta_action : (T-1,14)

        # 5. Trajectory maps → (3, T, H, W) in [-1,1]
        traj_maps = self._get_traj(action, w2cs, intrinsic)
        traj_maps = self.norm(traj_maps.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)

        # 6. Ray map → (6, T, H, W)
        raymap = self._get_raymap(intrinsic, c2ws)

        return {
            'video':        video,         # (3, T, H, W)  target
            'first_frame':  first_frame,   # (3, H, W)     I2V condition
            'traj_maps':    traj_maps,     # (3, T, H, W)  action visual cond
            'raymap':       raymap,        # (6, T, H, W)  camera geometry cond
            'delta_action': delta_action,  # (T-1, 14)       action numeric cond
            'prompt':       PROMPT,
        }


# test dataset
# if __name__ == "__main__":
#     dataset = AgiBotWorldWanDataset(
#         data_root="/data/xiejunbin/AgiBotWorldChallenge-2026/mnt/public/chenshengcong/dataset/iros_challenge_2025_acwm",
#         split="train",
#         sample_size=(480, 640),
#         traj_radius=50,
#     )
#     sample = dataset[0]
#     print({k: v.shape for k, v in sample.items() if isinstance(v, torch.Tensor)})
#     delta_action = sample['delta_action']
#     print("delta_action stats: mean", delta_action.mean(dim=0), "std", delta_action.std(dim=0))