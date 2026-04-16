"""
Cached dataset that loads pre-computed VAE38 latents from .pt files.

Each .pt file (produced by preprocess_latents_5b.py) contains:
    video_latents       (48, T_lat, H_lat, W_lat)  float16
    first_frame_latents (48, 1, H_lat, W_lat)       float16
    traj_latents        (48, T_lat, H_lat, W_lat)  float16
    raymap              (6, T_lat, H_lat, W_lat)   float16
    delta_action        (T-1, 14)                    float16

A separate context.pt holds the shared T5 text embedding (L, 4096).
"""

import os
import glob
import torch
from torch.utils.data import Dataset


class AgiBotWorldWanCachedDataset(Dataset):
    def __init__(self, cache_dir):
        self.load_from_cache = False  # not using DiffSynth unified cache mode

        # Collect all clip .pt files (exclude context.pt)
        all_pt = sorted(glob.glob(os.path.join(cache_dir, '*.pt')))
        self.pt_files = [f for f in all_pt if os.path.basename(f) != 'context.pt']

        # Load shared T5 text embedding once
        context_path = os.path.join(cache_dir, 'context.pt')
        self.context = torch.load(context_path, map_location='cpu')  # (L, 4096)

        print(f"[AgiBotWorldWanCachedDataset] {len(self.pt_files)} clips, "
              f"context shape={self.context.shape}")

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        data = torch.load(self.pt_files[idx], map_location='cpu')
        data['context'] = self.context  # (L, 4096)
        return data
