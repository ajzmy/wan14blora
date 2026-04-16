import os
import json
import random
import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class AgiBotWorldBetaTextDataset(Dataset):
   
    def __init__(self, data_root, split="train", sample_size=(480, 640),
                 target_frames=25, text_dropout_prob=0.0,
                 split_file=None):
        
        self.data_root = data_root
        self.sample_size = sample_size  # (H, W)
        self.target_frames = target_frames
        self.text_dropout_prob = text_dropout_prob
        self.load_from_cache = False  # required by launch_training_task

        # Try to load split.json
        if split_file is None:
            split_file = os.path.join(data_root, "split.json")

        self.clips = []  # List of clip info dicts or paths

        if os.path.exists(split_file):
            # Load from split.json
            with open(split_file, 'r') as f:
                split_data = json.load(f)

            if split in split_data:
                self.clips = split_data[split]
                print(f"[AgiBotWorldTextControlDataset] Loaded {len(self.clips)} clips from {split_file} ({split})")
            else:
                raise ValueError(f"Split '{split}' not found in {split_file}")
        else:
            # Scan all clips in data_root
            print(f"[AgiBotWorldTextControlDataset] No split.json found, scanning {data_root}...")
            self.clips = self._scan_all_clips(data_root)
            print(f"[AgiBotWorldTextControlDataset] Found {len(self.clips)} clips")

        print(f"[AgiBotWorldTextControlDataset] {split}: {len(self.clips)} clips, "
              f"text_dropout_prob={text_dropout_prob}")


    def _scan_all_clips(self, data_root):
        """Scan data_root and return list of clip info dicts."""
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
        """Check if clip directory has all required files."""
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


    def _load_text(self, clip_dir, text_from_split=None):
        """Load text prompt, apply dropout if configured."""
        if text_from_split is not None:
            text = text_from_split
        else:
            text_path = os.path.join(clip_dir, 'text.txt')
            with open(text_path, 'r') as f:
                text = f.read().strip()

        # Text dropout for CFG training
        if self.text_dropout_prob > 0 and random.random() < self.text_dropout_prob:
            return ""
        return text
    

    def _load_annotation(self, clip_dir):
        """Load annotation.json to get sampled frame indices."""
        anno_path = os.path.join(clip_dir, 'annotation.json')
        with open(anno_path, 'r') as f:
            return json.load(f)
        

    def _load_frames(self, clip_dir, frame_indices):
        """Load pre-extracted jpg frames."""
        videos_dir = os.path.join(clip_dir, 'videos')
        frames = []

        # Get available frame files
        available_files = sorted([f for f in os.listdir(videos_dir) if f.endswith('.jpg')])
        available_indices = [int(f.split('_')[1].split('.')[0]) for f in available_files]
        available_map = dict(zip(available_indices, available_files))

        for idx in frame_indices:
            if idx in available_map:
                frame_path = os.path.join(videos_dir, available_map[idx])
            else:
                # Find closest available frame
                closest = min(available_indices, key=lambda x: abs(x - idx))
                frame_path = os.path.join(videos_dir, available_map[closest])

            img = Image.open(frame_path).convert('RGB')
            frames.append(np.array(img))

        return np.stack(frames) if frames else None
    

    def _load_sample(self, clip_dir, text_from_split=None):
        H, W = self.sample_size

        # 1. Load annotation to get frame indices (original video frame numbers)
        annotation = self._load_annotation(clip_dir)
        frame_indices = annotation['sampled_frame_indices']  # e.g., [48, 54, 60, ...]
        num_frames = len(frame_indices)

        # 2. Sample target_frames using RELATIVE indices (0, 1, 2, ...)
        if num_frames >= self.target_frames:
            max_start = num_frames - self.target_frames
            start = random.randint(0, max_start)
            relative_indices = list(range(start, start + self.target_frames))
            selected_frame_numbers = frame_indices[start:start + self.target_frames]
        else:
            relative_indices = list(range(num_frames)) + [num_frames - 1] * (self.target_frames - num_frames)
            selected_frame_numbers = frame_indices + [frame_indices[-1]] * (self.target_frames - num_frames)

        # 3. Load video frames using original frame numbers (for jpg filenames)
        raw_frames = self._load_frames(clip_dir, selected_frame_numbers)
        if raw_frames is None or len(raw_frames) < self.target_frames:
            raise ValueError(f"Failed to load enough frames from {clip_dir}")

        video_pil = [
            Image.fromarray(frame).resize((W, H), Image.BILINEAR)
            for frame in raw_frames[:self.target_frames]
        ]

        first_image_pil = video_pil[0]

        prompt = self._load_text(clip_dir, text_from_split)

        return {
            "video": video_pil,
            "input_image": first_image_pil,
            "prompt": prompt,
        }


    def __getitem__(self, idx):
        while True:
            try:
                clip_info = self.clips[idx]
                if isinstance(clip_info, dict):
                    clip_dir = os.path.join(self.data_root, clip_info["path"])
                    text_from_split = clip_info.get("text", None)
                else:
                    clip_dir = os.path.join(self.data_root, clip_info)
                    text_from_split = None
                return self._load_sample(clip_dir, text_from_split)
            except Exception as e:
                print(f"Error loading clip {idx}: {e}")
                idx = random.randint(0, len(self.clips) - 1)


# test
# if __name__ == "__main__":
#     dataset = AgiBotWorldBetaTextDataset(data_root="/root/data/agibot_world_beta_processed_main", split="train",
#                                             sample_size=(480, 640), target_frames=25,
#                                             text_dropout_prob=0.0)
#     sample = dataset[1000]
#     print(sample["prompt"])
#     print(sample["input_image"].size)
#     print(len(sample["video"]))