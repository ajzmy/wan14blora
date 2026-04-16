"""
AgiBotWorld Beta dataset for pure TI2V (Text+Image to Video) training.

Data format: agibot_world_beta_processed
  data_root/
    <task_id>/
      task.txt              # task-level description
      <episode_id>/
        <clip_idx>/          # 0, 1, 2, 3
          text.txt           # per-clip action description
          annotation.json    # metadata incl. sampled_frame_indices
          videos/            # frame_00048.jpg, frame_00054.jpg, ...
          head_intrinsic_params.json
          head_extrinsic_params_aligned.json
          proprio_stats.h5

Returns per sample:
  video       : (3, T, H, W) float32 in [-1, 1]
  first_frame : (3, H, W)    float32 in [-1, 1]
  prompt      : str
"""

import os
import random
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


TARGET_FRAMES = 25  # 4*6+1, satisfies WAN 4n+1 constraint


class AgiBotWorldBetaDataset(Dataset):
    def __init__(
        self,
        data_root,
        sample_size=(480, 640),
    ):
        """
        Args:
            data_root: path to agibot_world_beta_processed root
                       (contains task_id subdirectories)
            sample_size: (H, W) to resize frames to
        """
        self.sample_size = sample_size
        self.load_from_cache = False  # required by DiffSynth launch_training_task

        self.norm = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )

        # Collect all clip paths: data_root/<task_id>/<episode_id>/<clip_idx>/
        self.clip_paths = []
        self.clip_num_frames = []

        for task_id in sorted(os.listdir(data_root)):
            task_dir = os.path.join(data_root, task_id)
            if not os.path.isdir(task_dir):
                continue
            for ep_id in sorted(os.listdir(task_dir)):
                ep_dir = os.path.join(task_dir, ep_id)
                if not os.path.isdir(ep_dir):
                    continue
                for clip_idx in sorted(os.listdir(ep_dir)):
                    clip_dir = os.path.join(ep_dir, clip_idx)
                    if not os.path.isdir(clip_dir):
                        continue
                    videos_dir = os.path.join(clip_dir, 'videos')
                    text_file = os.path.join(clip_dir, 'text.txt')
                    if not os.path.isdir(videos_dir) or not os.path.isfile(text_file):
                        continue
                    # Count available frames
                    num_frames = len([
                        f for f in os.listdir(videos_dir)
                        if f.endswith('.jpg') or f.endswith('.png')
                    ])
                    if num_frames >= 2:  # need at least 2 frames
                        self.clip_paths.append(clip_dir)
                        self.clip_num_frames.append(num_frames)

        n_short = sum(1 for n in self.clip_num_frames if n < TARGET_FRAMES)
        print(f"[AgiBotWorldBetaDataset] {len(self.clip_paths)} clips loaded "
              f"({n_short} short clips will be tail-padded to {TARGET_FRAMES} frames)")

    def __len__(self):
        return len(self.clip_paths)

    def __getitem__(self, idx):
        while True:
            try:
                return self._load_sample(self.clip_paths[idx])
            except Exception as e:
                print(f"Error loading {self.clip_paths[idx]}: {e}")
                idx = random.randint(0, len(self.clip_paths) - 1)

    def _load_sample(self, clip_dir):
        H, W = self.sample_size

        # 1. Read prompt from text.txt
        with open(os.path.join(clip_dir, 'text.txt'), 'r') as f:
            prompt = f.read().strip()

        # 2. Load video frames from videos/ directory
        videos_dir = os.path.join(clip_dir, 'videos')
        frame_files = sorted([
            f for f in os.listdir(videos_dir)
            if f.endswith('.jpg') or f.endswith('.png')
        ])
        total_frames = len(frame_files)

        # 3. Select frame indices (random window or pad)
        if total_frames >= TARGET_FRAMES:
            max_start = total_frames - TARGET_FRAMES
            start = random.randint(0, max_start)
            selected_files = frame_files[start:start + TARGET_FRAMES]
        else:
            # Pad by repeating last frame
            selected_files = frame_files + [frame_files[-1]] * (TARGET_FRAMES - total_frames)

        # 4. Read and process frames
        frames = []
        for fname in selected_files:
            img = Image.open(os.path.join(videos_dir, fname)).convert('RGB')
            img_np = np.array(img, dtype=np.float32) / 255.0  # [0, 1]
            frames.append(torch.from_numpy(img_np).permute(2, 0, 1))  # (3, H_orig, W_orig)

        # Stack: (T, 3, H_orig, W_orig)
        frames_t = torch.stack(frames, dim=0)

        # Resize: (T, 3, H, W)
        frames_t = F.interpolate(
            frames_t, size=(H, W), mode='bilinear', align_corners=False
        )

        # Normalize [0,1] -> [-1,1] per frame
        # norm expects (C, H, W); apply to each frame then stack as (3, T, H, W)
        normed_frames = []
        for t in range(TARGET_FRAMES):
            normed_frames.append(self.norm(frames_t[t]))  # (3, H, W)
        video = torch.stack(normed_frames, dim=1)  # (3, T, H, W)

        first_frame = video[:, 0, :, :]  # (3, H, W)

        return {
            'video': video,              # (3, T, H, W) in [-1, 1]
            'first_frame': first_frame,  # (3, H, W) in [-1, 1]
            'prompt': prompt,
        }
