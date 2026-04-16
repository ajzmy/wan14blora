"""
AgiBotWorld Dataset with TEXT conditioning for Fun-Control training.

Supports:
  - Reading text.txt as prompt
  - Text dropout for CFG training
  - Pre-extracted jpg frames
  - split.json for train/val separation

Data structure expected:
    data_root/
        split.json                      # (optional) train/val split
        task_id/
            episode_id/
                clip_idx/
                    text.txt
                    annotation.json
                    proprio_stats.h5
                    head_intrinsic_params.json
                    head_extrinsic_params_aligned.json
                    videos/
                        frame_00048.jpg
                        ...
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


from lvdm.data.traj_vis_statistics import (
    ColorMapLeft, ColorMapRight, ColorListLeft, ColorListRight,
    EndEffectorPts, Gripper2EEFCvt,
)
from lvdm.data.get_actions import parse_h5
from lvdm.data.utils import get_transformation_matrix_from_quat, intrinsic_transform


class AgiBotWorldBetaDataset(Dataset):
    """
    Dataset for Fun-Control with TEXT conditioning.

    Outputs:
      - video: list of PIL Images (target frames)
      - control_video: list of PIL Images (traj maps)
      - reference_image: list with 1 PIL Image (first frame)
      - prompt: str (from text.txt, or empty string if text dropout)
    """

    def __init__(self, data_root, split="train", sample_size=(480, 640),
                 target_frames=25, traj_radius=50, text_dropout_prob=0.0,
                 split_file=None):
        """
        Args:
            data_root: Root directory containing task folders
            split: "train" or "val" (used with split.json)
            sample_size: (H, W) tuple for output size
            target_frames: Number of frames to sample
            traj_radius: Radius for trajectory visualization
            text_dropout_prob: Probability of dropping text (for CFG training)
            split_file: Path to split.json. If None, try data_root/split.json
        """
        self.data_root = data_root
        self.sample_size = sample_size  # (H, W)
        self.target_frames = target_frames
        self.traj_radius = traj_radius
        self.text_dropout_prob = text_dropout_prob
        self.load_from_cache = False  # required by launch_training_task

        # Try to load split.json
        if split_file is None:
            split_file = os.path.join(data_root, "split_no549.json")

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

    def _get_next_clip_dir(self, clip_info):
        if isinstance(clip_info, dict):
            tid, eid, cidx = clip_info["task_id"], clip_info["episode_id"], clip_info["clip_idx"]
        else:
            parts = clip_info.split("/")
            tid, eid, cidx = int(parts[0]), int(parts[1]), int(parts[2])

        next_path = os.path.join(self.data_root, str(tid), str(eid), str(cidx + 1))
        if os.path.isdir(next_path) and self._is_valid_clip(next_path):
            return next_path
        return None

    def __len__(self):
        return len(self.clips)


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
                
                # annotation = self._load_annotation(clip_dir)
                # num_frames = len(annotation['sampled_frame_indices'])

                # if num_frames < self.target_frames:
                #     next_clip_dir = self._get_next_clip_dir(clip_info)
                #     if next_clip_dir is not None:
                #         return self._load_sample_concat(clip_dir, next_clip_dir)

                return self._load_sample(clip_dir, text_from_split)
            except Exception as e:
                print(f"Error loading clip {idx}: {e}")
                idx = random.randint(0, len(self.clips) - 1)


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

    def _load_camera_params(self, clip_dir, indices):
        """Load camera intrinsic and extrinsic parameters."""
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
            c2w[:3, :3] = torch.tensor(info['extrinsic']['rotation_matrix'], dtype=torch.float32)
            c2w[:3, 3] = torch.tensor(info['extrinsic']['translation_vector'], dtype=torch.float32)
            w2c = torch.linalg.inv(c2w)
            c2ws.append(c2w)
            w2cs.append(w2c)
        return intrinsic, torch.stack(c2ws), torch.stack(w2cs)


    def get_traj_maps(self, action, w2cs, intrinsic):
        """Render trajectory maps as list of PIL Images."""
        cnt = 0
        H, W = self.sample_size
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

            img_list.append(Image.fromarray(img))

            # check
            if np.all(img == 50):
                cnt += 1
        if cnt > int(action.shape[0] * 0.5):
            print(f"Warning: {cnt}/{action.shape[0]} frames have no valid trajectory points!")
            valid = False
        else:
            valid = True
        return img_list, valid


    def _load_sample(self, clip_dir, text_from_split=None):
        H, W = self.sample_size

        # 1. Load annotation to get frame indices (original video frame numbers)
        annotation = self._load_annotation(clip_dir)
        frame_indices = annotation['sampled_frame_indices']  # e.g., [48, 54, 60, ...]
        num_frames = len(frame_indices)

        # 2. Sample target_frames using RELATIVE indices (0, 1, 2, ...)
        # The h5/extrinsic/videos are all aligned to these relative indices
        if num_frames >= self.target_frames:
            max_start = num_frames - self.target_frames
            start = random.randint(0, max_start)
            relative_indices = list(range(start, start + self.target_frames))
            selected_frame_numbers = frame_indices[start:start + self.target_frames]
        else:
            task_id = annotation["task_id"]
            episode_id = annotation["episode_id"]
            clip_idx = annotation["clip_idx"]
            next_clip_dir = os.path.join(self.data_root, f"{task_id}/{episode_id}/{clip_idx + 1}")
            if os.path.isdir(next_clip_dir):
                return self._load_sample_concat(clip_dir, next_clip_dir)
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

        # 4. Camera params using RELATIVE indices
        intrinsic_orig, c2ws, w2cs = self._load_camera_params(clip_dir, relative_indices[:self.target_frames])
        orig_h, orig_w = raw_frames.shape[1], raw_frames.shape[2]
        intrinsic = intrinsic_transform(intrinsic_orig, (orig_h, orig_w), (H, W), 'resize')

        # 5. Actions -> traj maps using RELATIVE indices
        h5_path = os.path.join(clip_dir, 'proprio_stats.h5')
        abs_action, _ = parse_h5(h5_path, slices=relative_indices[:self.target_frames], delta_act_sidx=1)
        action = torch.tensor(abs_action, dtype=torch.float32)
        control_video_pil, valid = self.get_traj_maps(action, w2cs, intrinsic)
        if not valid:
            raise ValueError(f"Invalid trajectory maps in {clip_dir}")

        # 6. first frame
        first_image_pil = video_pil[0]

        # 7. Load text prompt (with dropout)
        prompt = self._load_text(clip_dir, text_from_split)
        # prompt = "Robotic arm manipulation, high quality, realistic."

        # 8. load reference image from last clip
        task_id = annotation["task_id"]
        episode_id = annotation["episode_id"]
        clip_idx = annotation["clip_idx"]
        if clip_idx > 0:
            prev_clip_dir = os.path.join(self.data_root, f"{task_id}/{episode_id}/{clip_idx - 1}")
            prev_annotation = self._load_annotation(prev_clip_dir)
            prev_frame_indices = prev_annotation['sampled_frame_indices']
            reference_image_pil = self._load_frames(prev_clip_dir, [prev_frame_indices[0]])
            if reference_image_pil is not None and len(reference_image_pil) > 0:
                reference_image_pil = Image.fromarray(reference_image_pil[0]).resize((W, H), Image.BILINEAR)
                reference_image_pil = [reference_image_pil]
            else:
                reference_image_pil = [first_image_pil]
        else:
            reference_image_pil = [first_image_pil]

        return {
            "video": video_pil,
            "control_video": control_video_pil,
            "input_image": first_image_pil,
            "reference_image": reference_image_pil,
            "prompt": prompt,
        }
    
     # ── Two-clip concat for TRAINING (random window) ─────────────────

    def _load_sample_concat(self, clip_dir, next_clip_dir, text_from_split=None):
        """Merge two consecutive clips, then take a random contiguous window
        of `target_frames` frames.  Text from both clips is always concatenated."""

        H, W = self.sample_size

        # --- Clip A ---
        anno_a = self._load_annotation(clip_dir)
        frame_indices_a = anno_a['sampled_frame_indices']
        num_a = len(frame_indices_a)
        relative_indices_a = list(range(num_a))

        raw_frames_a = self._load_frames(clip_dir, frame_indices_a)
        if raw_frames_a is None:
            raise ValueError(f"Failed to load frames from {clip_dir}")

        intrinsic_a, c2ws_a, w2cs_a = self._load_camera_params(clip_dir, relative_indices_a)

        h5_a = os.path.join(clip_dir, 'proprio_stats.h5')
        abs_action_a, _ = parse_h5(h5_a, slices=relative_indices_a, delta_act_sidx=1)

        # --- Clip B ---
        anno_b = self._load_annotation(next_clip_dir)
        frame_indices_b = anno_b['sampled_frame_indices']
        num_b = len(frame_indices_b)
        relative_indices_b = list(range(num_b))

        raw_frames_b = self._load_frames(next_clip_dir, frame_indices_b)
        if raw_frames_b is None:
            raise ValueError(f"Failed to load frames from {next_clip_dir}")

        intrinsic_b, c2ws_b, w2cs_b = self._load_camera_params(next_clip_dir, relative_indices_b)

        h5_b = os.path.join(next_clip_dir, 'proprio_stats.h5')
        abs_action_b, _ = parse_h5(h5_b, slices=relative_indices_b, delta_act_sidx=1)

        # --- Concatenate all ---
        raw_frames_all = np.concatenate([raw_frames_a, raw_frames_b], axis=0)
        w2cs_all = torch.cat([w2cs_a, w2cs_b], dim=0)
        abs_action_all = np.concatenate([abs_action_a, abs_action_b], axis=0)
        total_frames = raw_frames_all.shape[0]

        # --- Random contiguous window ---
        if total_frames >= self.target_frames:
            max_start = total_frames - self.target_frames
            start = random.randint(0, max_start)
            end = start + self.target_frames
        else:
            # Not enough even after concat (rare) -> pad with last frame
            start = 0
            end = total_frames

        raw_frames_sel = raw_frames_all[start:end]
        w2cs_sel = w2cs_all[start:end]
        abs_action_sel = abs_action_all[start:end]

        # Pad if needed
        if raw_frames_sel.shape[0] < self.target_frames:
            pad_n = self.target_frames - raw_frames_sel.shape[0]
            raw_frames_sel = np.concatenate(
                [raw_frames_sel, np.stack([raw_frames_sel[-1]] * pad_n)], axis=0)
            w2cs_sel = torch.cat(
                [w2cs_sel, w2cs_sel[-1:].expand(pad_n, -1, -1)], dim=0)
            abs_action_sel = np.concatenate(
                [abs_action_sel, np.stack([abs_action_sel[-1]] * pad_n)], axis=0)

        orig_h, orig_w = raw_frames_sel.shape[1], raw_frames_sel.shape[2]

        video_pil = [
            Image.fromarray(frame).resize((W, H), Image.BILINEAR)
            for frame in raw_frames_sel
        ]

        # Camera intrinsic (same camera across clips in an episode)
        intrinsic = intrinsic_transform(intrinsic_a, (orig_h, orig_w), (H, W), 'resize')

        # Traj maps
        action = torch.tensor(abs_action_sel, dtype=torch.float32)
        control_video_pil, valid = self.get_traj_maps(action, w2cs_sel, intrinsic)
        if not valid:
            raise ValueError(f"Invalid trajectory maps in concat {clip_dir} + {next_clip_dir}")

        # Reference image
        task_id = anno_a["task_id"]
        episode_id = anno_a["episode_id"]
        clip_idx = anno_a["clip_idx"]
        if clip_idx > 0:
            prev_clip_dir = os.path.join(self.data_root, f"{task_id}/{episode_id}/{clip_idx - 1}")
            prev_annotation = self._load_annotation(prev_clip_dir)
            prev_frame_indices = prev_annotation['sampled_frame_indices']
            reference_image_pil = self._load_frames(prev_clip_dir, [prev_frame_indices[0]])
            if reference_image_pil is not None and len(reference_image_pil) > 0:
                reference_image_pil = Image.fromarray(reference_image_pil[0]).resize((W, H), Image.BILINEAR)
                reference_image_pil = [reference_image_pil]
            else:
                reference_image_pil = [video_pil[0]]
        else:
            reference_image_pil = [video_pil[0]]

        # Text: always concatenate both clips, then apply dropout once
        text_path_a = os.path.join(clip_dir, 'text.txt')
        with open(text_path_a, 'r') as f:
            text_a = f.read().strip()
        text_path_b = os.path.join(next_clip_dir, 'text.txt')
        with open(text_path_b, 'r') as f:
            text_b = f.read().strip()
        if text_a and text_b:
            combined_text = text_a + " " + text_b
        else:
            combined_text = text_a or text_b or ""
        if self.text_dropout_prob > 0 and random.random() < self.text_dropout_prob:
            combined_text = ""

        return {
            "video": video_pil,
            "control_video": control_video_pil,
            "reference_image": reference_image_pil,
            "prompt": combined_text,
        }


# test
# if __name__ == "__main__":
#     dataset = AgiBotWorldBetaDataset(data_root="/root/data/agibot_world_beta_processed_main", split="train",
#                                             sample_size=(480, 640), target_frames=25,
#                                             text_dropout_prob=0.0)
#     sample = dataset._load_sample_concat("/root/data/agibot_world_beta_processed_main/367/648961/1","/root/data/agibot_world_beta_processed_main/367/648961/2")
#     print(sample["prompt"])
#     # shape
#     print(len(sample["video"]), sample["video"][0].size)
#     print(len(sample["control_video"]), sample["control_video"][0].size)
#     print(len(sample["reference_image"]), sample["reference_image"][0].size)
    
#     # save as mp4 using imageio
#     import imageio
#     imageio.mimwrite("./test_video.mp4", [np.array(frame) for frame in sample["video"]], fps=5) 
#     imageio.mimwrite("./test_control_video.mp4", [np.array(frame) for frame in sample["control_video"]], fps=5)
    # clip_dir = "/root/data/agibot_world_beta_processed_main/392/674884/1"
    # sample = dataset._load_sample(clip_dir)
    # print(sample["prompt"])
    # print(len(sample["video"]), sample["video"][0].size)
    # print(len(sample["vace_video"]), sample["vace_video"][0].size)
    # print(len(sample["vace_reference_image"]), sample["vace_reference_image"][0].size)
    # print(sample["input_image"].size)
    # # 将video和trajectory map拼接在一起保存为mp4
    # import imageio
    # combined_frames = []
    # for v_frame, t_frame in zip(sample["video"], sample["vace_video"]):
    #     combined = Image.new('RGB', (sample["video"][0].size[0] * 2, sample["video"][0].size[1]))
    #     combined.paste(v_frame, (0, 0))
    #     combined.paste(t_frame, (sample["video"][0].size[0], 0))
    #     combined_frames.append(np.array(combined))
    # imageio.mimwrite("./test_combined_video.mp4", combined_frames, fps=5)