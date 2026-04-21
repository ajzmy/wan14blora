"""
AgiBotWorld GRPO Dataset.

Provides training samples for GRPO reinforcement learning on the FunControl model.
Each sample includes:
  - prompt: text description
  - reference_image: first frame (PIL Image)
  - control_video: trajectory map frames (list of PIL Images)
  - gt_frames: ground-truth video frames (list of np.ndarray, for PSNR reward)
  - gt_proprio: proprioception data (np.ndarray, for GT trajectory → NDTW reward)

Data structure expected:
    data_root/
        task_id/
            episode_id/
                clip_idx/
                    text.txt
                    annotation.json
                    proprio_stats.h5
                    head_intrinsic_params.json
                    head_extrinsic_params_aligned.json
                    videos/
                        frame_00000.jpg ...
"""

import os
import json
import random
import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# evac utilities are inside diffsynth-studio's model_training directory
EVAC_DIR = os.environ.get(
    "EVAC_DIR",
    "/root/luomingshuang/diffsynth-studio/examples/wanvideo/model_training/evac",
)
if os.path.isdir(EVAC_DIR):
    sys.path.insert(0, EVAC_DIR)

from lvdm.data.traj_vis_statistics import (
    ColorMapLeft, ColorMapRight, ColorListLeft, ColorListRight,
    EndEffectorPts, Gripper2EEFCvt,
)
from lvdm.data.get_actions import parse_h5
from lvdm.data.utils import get_transformation_matrix_from_quat, intrinsic_transform


class AgiBotGRPODataset(Dataset):
    """
    Dataset for GRPO training of FunControl on AgiBotWorld.

    Simplified from the SFT dataset:
      - Single clip sampling only (no two-clip merging)
      - Returns GT frames and proprioception for reward computation
      - No text dropout (GRPO needs consistent prompts for advantage computation)
    """

    def __init__(self, data_root, sample_size=(480, 832), target_frames=49,
                 traj_radius=50, split_file=None):
        self.data_root = data_root
        self.sample_size = sample_size  # (H, W)
        self.target_frames = target_frames
        self.traj_radius = traj_radius

        self.clips = []
        if split_file and os.path.exists(split_file):
            with open(split_file, 'r') as f:
                split_data = json.load(f)
            if "train" in split_data:
                self.clips = split_data["train"]
            else:
                self.clips = split_data
            print(f"[AgiBotGRPODataset] Loaded {len(self.clips)} clips from {split_file}")
        else:
            print(f"[AgiBotGRPODataset] Scanning {data_root} ...")
            self.clips = self._scan_all_clips(data_root)
            print(f"[AgiBotGRPODataset] Found {len(self.clips)} clips")

        # Build episode → clip map for neighbor lookups
        self._episode_clip_map = {}
        for i, clip in enumerate(self.clips):
            if isinstance(clip, dict):
                key = (clip["task_id"], clip["episode_id"])
                cidx = clip["clip_idx"]
            else:
                parts = clip.split("/")
                key = (int(parts[0]), int(parts[1]))
                cidx = int(parts[2])
            if key not in self._episode_clip_map:
                self._episode_clip_map[key] = {}
            self._episode_clip_map[key][cidx] = i

    def _scan_all_clips(self, data_root):
        clips = []
        for task_id in sorted(os.listdir(data_root)):
            task_dir = os.path.join(data_root, task_id)
            if not os.path.isdir(task_dir) or not task_id.isdigit():
                continue
            for episode_id in sorted(os.listdir(task_dir)):
                episode_dir = os.path.join(task_dir, episode_id)
                if not os.path.isdir(episode_dir):
                    continue
                for clip_idx in sorted(os.listdir(episode_dir)):
                    clip_dir = os.path.join(episode_dir, clip_idx)
                    if not os.path.isdir(clip_dir):
                        continue
                    if self._is_valid_clip(clip_dir):
                        clips.append({
                            "path": f"{task_id}/{episode_id}/{clip_idx}",
                            "task_id": int(task_id),
                            "episode_id": int(episode_id),
                            "clip_idx": int(clip_idx),
                        })
        return clips

    def _is_valid_clip(self, clip_dir):
        required = ['text.txt', 'annotation.json', 'proprio_stats.h5',
                    'head_intrinsic_params.json', 'head_extrinsic_params_aligned.json']
        for f in required:
            if not os.path.exists(os.path.join(clip_dir, f)):
                return False
        videos_dir = os.path.join(clip_dir, 'videos')
        if not os.path.isdir(videos_dir) or len(os.listdir(videos_dir)) == 0:
            return False
        return True

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        while True:
            try:
                return self._load_sample(idx)
            except Exception as e:
                print(f"[AgiBotGRPODataset] Error loading clip {idx}: {e}")
                idx = random.randint(0, len(self.clips) - 1)

    def _load_sample(self, idx):
        clip_info = self.clips[idx]
        if isinstance(clip_info, dict):
            clip_dir = os.path.join(self.data_root, clip_info["path"])
            task_id = str(clip_info["task_id"])
            episode_id = str(clip_info["episode_id"])
            cidx = clip_info["clip_idx"]
        else:
            clip_dir = os.path.join(self.data_root, clip_info)
            parts = clip_info.split("/")
            task_id, episode_id = parts[0], parts[1]
            cidx = int(parts[2])

        H, W = self.sample_size

        # Load annotation for frame indices
        with open(os.path.join(clip_dir, 'annotation.json'), 'r') as f:
            annotation = json.load(f)
        frame_indices = annotation['sampled_frame_indices']
        num_frames = len(frame_indices)

        # Select frames: random window or pad
        if num_frames >= self.target_frames:
            max_start = num_frames - self.target_frames
            start = random.randint(0, max_start)
            relative_indices = list(range(start, start + self.target_frames))
            selected_frame_numbers = frame_indices[start:start + self.target_frames]
        else:
            relative_indices = list(range(num_frames)) + [num_frames - 1] * (self.target_frames - num_frames)
            selected_frame_numbers = frame_indices + [frame_indices[-1]] * (self.target_frames - num_frames)

        # Load GT frames
        raw_frames = self._load_frames_from_jpg(clip_dir, selected_frame_numbers)
        if raw_frames is None or len(raw_frames) < self.target_frames:
            raise ValueError(f"Failed to load enough frames from {clip_dir}")

        # Resize for video generation
        gt_frames_resized = []
        video_pil = []
        for frame in raw_frames[:self.target_frames]:
            pil_img = Image.fromarray(frame).resize((W, H), Image.BILINEAR)
            video_pil.append(pil_img)
            gt_frames_resized.append(np.array(pil_img))

        # Camera params and trajectory maps
        intrinsic_orig, _, w2cs = self._load_camera_params(
            clip_dir, relative_indices[:self.target_frames])
        orig_h, orig_w = raw_frames.shape[1], raw_frames.shape[2]
        intrinsic = intrinsic_transform(intrinsic_orig, (orig_h, orig_w), (H, W), 'resize')

        h5_path = os.path.join(clip_dir, 'proprio_stats.h5')
        abs_action, _ = parse_h5(h5_path, slices=relative_indices[:self.target_frames],
                                  delta_act_sidx=1)
        action = torch.tensor(abs_action, dtype=torch.float32)
        control_video_pil, valid = self._render_traj_maps(action, w2cs, intrinsic)
        if not valid:
            raise ValueError(f"Invalid trajectory maps in {clip_dir}")

        # Reference image: try previous clip's first frame, fallback to current first frame
        ref_image = self._try_load_prev_clip_first_frame(clip_info)
        if ref_image is None:
            ref_image = video_pil[0]

        # Load text prompt
        text_path = os.path.join(clip_dir, 'text.txt')
        if os.path.exists(text_path):
            with open(text_path, 'r') as f:
                prompt = f.read().strip()
        else:
            prompt = "robot arm manipulation"

        # Proprioception for GT trajectory (used in NDTW reward)
        gt_proprio = abs_action  # shape: [T, 16]

        return {
            "prompt": prompt,
            "input_image": video_pil[0],       # GT first frame → WanVideoUnit_ImageEmbedderVAE (y tensor)
            "reference_image": ref_image,       # Reference frame → WanVideoUnit_FunReference (CLIP + ref_conv)
            "control_video": control_video_pil,
            "gt_frames": gt_frames_resized,
            "gt_proprio": gt_proprio,
            "task_id": task_id,
            "episode_id": episode_id,
        }

    # ── Helper methods (adapted from SFT dataset) ──────────────────────

    def _try_load_prev_clip_first_frame(self, clip_info):
        """Try to load the first frame from the previous clip as reference image."""
        if isinstance(clip_info, dict):
            key = (clip_info["task_id"], clip_info["episode_id"])
            cidx = clip_info["clip_idx"]
        else:
            parts = clip_info.split("/")
            key = (int(parts[0]), int(parts[1]))
            cidx = int(parts[2])

        prev_cidx = cidx - 1
        if prev_cidx < 0:
            return None
        clip_map = self._episode_clip_map.get(key, {})
        if prev_cidx not in clip_map:
            return None
        prev_clip = self.clips[clip_map[prev_cidx]]
        if isinstance(prev_clip, dict):
            prev_dir = os.path.join(self.data_root, prev_clip["path"])
        else:
            prev_dir = os.path.join(self.data_root, prev_clip)

        try:
            H, W = self.sample_size
            with open(os.path.join(prev_dir, 'annotation.json'), 'r') as f:
                anno = json.load(f)
            first_idx = anno['sampled_frame_indices'][0]
            frames = self._load_frames_from_jpg(prev_dir, [first_idx])
            if frames is not None and len(frames) > 0:
                return Image.fromarray(frames[0]).resize((W, H), Image.BILINEAR)
        except Exception:
            pass
        return None

    def _load_frames_from_jpg(self, clip_dir, frame_indices):
        videos_dir = os.path.join(clip_dir, 'videos')
        files = sorted([f for f in os.listdir(videos_dir)
                       if f.endswith('.jpg') or f.endswith('.png')])
        available_indices = [int(f.split('_')[1].split('.')[0]) for f in files]
        available_map = dict(zip(available_indices, files))

        frames = []
        for idx in frame_indices:
            if idx in available_map:
                frame_path = os.path.join(videos_dir, available_map[idx])
            else:
                closest = min(available_indices, key=lambda x: abs(x - idx))
                frame_path = os.path.join(videos_dir, available_map[closest])
            img = Image.open(frame_path).convert('RGB')
            frames.append(np.array(img))
        return np.stack(frames) if frames else None

    def _load_camera_params(self, clip_dir, indices):
        with open(os.path.join(clip_dir, 'head_intrinsic_params.json')) as f:
            intr = json.load(f)['intrinsic']
        intrinsic = torch.eye(3, dtype=torch.float32)
        intrinsic[0, 0] = intr['fx']
        intrinsic[1, 1] = intr['fy']
        intrinsic[0, 2] = intr['ppx']
        intrinsic[1, 2] = intr['ppy']

        with open(os.path.join(clip_dir, 'head_extrinsic_params_aligned.json')) as f:
            extr_list = json.load(f)

        c2ws, w2cs = [], []
        for i in indices:
            if i < len(extr_list):
                info = extr_list[i]
            else:
                info = extr_list[-1]
            c2w = torch.eye(4, dtype=torch.float32)
            c2w[:3, :3] = torch.tensor(info['extrinsic']['rotation_matrix'],
                                        dtype=torch.float32)
            c2w[:3, 3] = torch.tensor(info['extrinsic']['translation_vector'],
                                       dtype=torch.float32)
            w2c = torch.linalg.inv(c2w)
            c2ws.append(c2w)
            w2cs.append(w2c)
        return intrinsic, torch.stack(c2ws), torch.stack(w2cs)

    def _render_traj_maps(self, action, w2cs, intrinsic):
        H, W = self.sample_size
        gray_cnt = 0
        ee_key_pts = torch.tensor(EndEffectorPts, dtype=torch.float32).view(1, 4, 4).permute(0, 2, 1)
        cvt_matrix = torch.tensor(Gripper2EEFCvt, dtype=torch.float32).view(1, 4, 4)

        pose_l_mat = get_transformation_matrix_from_quat(action[:, 0:7])
        pose_r_mat = get_transformation_matrix_from_quat(action[:, 8:15])

        ee2cam_l = torch.matmul(torch.matmul(w2cs, pose_l_mat), cvt_matrix)
        ee2cam_r = torch.matmul(torch.matmul(w2cs, pose_r_mat), cvt_matrix)

        pts_l = torch.matmul(ee2cam_l, ee_key_pts)
        pts_r = torch.matmul(ee2cam_r, ee_key_pts)

        K = intrinsic.unsqueeze(0)
        uvs_l = (torch.matmul(K, pts_l[:, :3, :]) / pts_l[:, 2:3, :])[:, :2, :] \
                    .permute(0, 2, 1).to(torch.int64)
        uvs_r = (torch.matmul(K, pts_r[:, :3, :]) / pts_r[:, 2:3, :])[:, :2, :] \
                    .permute(0, 2, 1).to(torch.int64)

        pil_list = []
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

            if np.all(img == 50):
                gray_cnt += 1
            pil_list.append(Image.fromarray(img))

        valid = not (gray_cnt > int(action.shape[0] * 0.5))
        return pil_list, valid


def grpo_collate_fn(batch):
    """Custom collate: keep everything as lists since batch_size=1 typically."""
    if len(batch) == 1:
        return batch[0]
    # For batch_size > 1, return list of dicts
    return batch
