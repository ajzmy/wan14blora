"""
Reward wrapper for GRPO training.

Wraps the verified EWMReward class for use in the GRPO training loop.
Computes the competition score: (psnr/35 + ndtw + scene_consistency) / 3
"""

import os
import sys
import numpy as np
import torch
import cv2
from concurrent.futures import ThreadPoolExecutor

# Add reward module to path
REWARD_DIR = os.environ.get(
    "REWARD_DIR",
    "/root/grpo-training/diffsynth-studio/examples/wanvideo/model_training/reward",
)
if os.path.isdir(REWARD_DIR):
    sys.path.insert(0, REWARD_DIR)


class GRPORewardComputer:
    """
    Computes EWM competition rewards for GRPO training.

    Supports three modes:
      - "full": PSNR + NDTW + Scene Consistency (most accurate, slowest)
      - "psnr_only": Only PSNR (fast, no extra models needed)
      - "psnr_sc": PSNR + Scene Consistency (no YOLO needed)

    Args:
        reward_dir: Path to reward checkpoint directory (contains ckpt/, submodel/)
        device: Device for DINOv2 model (e.g., "cpu", "npu:15")
        mode: One of "full", "psnr_only", "psnr_sc"
        num_workers: Number of threads for async reward computation
    """

    def __init__(self, reward_dir=None, device="cpu", mode="psnr_only",
                 num_workers=2):
        self.device = device
        self.mode = mode
        self.num_workers = num_workers

        self.yolo_model = None
        self.dinov2_model = None
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

        if reward_dir is None:
            reward_dir = REWARD_DIR

        # Load models based on mode
        if mode in ("full",):
            self._load_yolo(os.path.join(reward_dir, "ckpt/yoloworld-EWMBench-v0.1.pt"))
        if mode in ("full", "psnr_sc"):
            self._load_dinov2(
                os.path.join(reward_dir, "ckpt/dinov2-EWMBench-v0.1.pth"),
                os.path.join(reward_dir, "ckpt/dino_config.yaml"),
                os.path.join(reward_dir, "submodel/dinov2"),
            )

    def _load_yolo(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            print(f"[RewardWrapper] YOLO checkpoint not found: {ckpt_path}")
            return
        from ultralytics import YOLO
        self.yolo_model = YOLO(ckpt_path)
        self.yolo_model.to(self.device)
        print(f"[RewardWrapper] YOLO model loaded on {self.device}")

    def _load_dinov2(self, ckpt_path, config_path, submodel_dir):
        if not os.path.exists(ckpt_path):
            print(f"[RewardWrapper] DINOv2 checkpoint not found: {ckpt_path}")
            return
        try:
            from ewm_reward import load_dinov2
            self.dinov2_model = load_dinov2(
                ckpt_path, config_path, submodel_dir, self.device)
            print(f"[RewardWrapper] DINOv2 model loaded on {self.device}")
        except Exception as e:
            print(f"[RewardWrapper] Failed to load DINOv2: {e}")

    def compute_single(self, gen_frames_np, gt_frames_np, gt_proprio=None):
        """
        Compute reward for a single generated video.

        Args:
            gen_frames_np: list of np.ndarray [H, W, 3] uint8 (generated video)
            gt_frames_np: list of np.ndarray [H, W, 3] uint8 (ground truth)
            gt_proprio: np.ndarray [T, 16] proprioception (for GT trajectory)

        Returns:
            reward: float (competition score in [0, 1])
            metadata: dict with per-metric scores
        """
        from skimage.metrics import peak_signal_noise_ratio

        # ── PSNR ──
        n = min(len(gen_frames_np), len(gt_frames_np))
        psnr_total = 0.0
        for i in range(n):
            gen = gen_frames_np[i]
            gt = gt_frames_np[i]
            if gen.shape != gt.shape:
                gen = cv2.resize(gen, (gt.shape[1], gt.shape[0]),
                                 interpolation=cv2.INTER_CUBIC)
            psnr_total += peak_signal_noise_ratio(gt, gen)
        psnr = psnr_total / n if n > 0 else 0.0
        psnr_norm = psnr / 35.0

        metadata = {"psnr": psnr, "psnr_norm": psnr_norm}

        # ── Scene Consistency ──
        sc = 0.0
        if self.dinov2_model is not None and self.mode in ("full", "psnr_sc"):
            try:
                sc = self._compute_scene_consistency(gen_frames_np)
            except Exception as e:
                print(f"[RewardWrapper] SC error: {e}")
                sc = 0.0
        metadata["scene_consistency"] = sc

        # ── NDTW ──
        ndtw = 0.0
        if self.yolo_model is not None and self.mode == "full" and len(gt_frames_np) > 0:
            try:
                ndtw = self._compute_ndtw(gen_frames_np, gt_frames_np)
            except Exception as e:
                print(f"[RewardWrapper] NDTW error: {e}")
                ndtw = 0.0
        metadata["ndtw"] = ndtw

        # ── Combined score ──
        reward = (psnr_norm + ndtw + sc) / 3.0
        metadata["reward"] = reward

        return reward, metadata

    def compute_batch(self, gen_videos, gt_data_list):
        """
        Compute rewards for a batch of generated videos.

        Args:
            gen_videos: list of (list of np.ndarray [H,W,3]) — generated videos
            gt_data_list: list of dicts with 'gt_frames' and optionally 'gt_proprio'

        Returns:
            rewards: torch.Tensor [batch_size]
            metadata_list: list of dicts
        """
        rewards = []
        metadata_list = []

        for gen_frames, gt_data in zip(gen_videos, gt_data_list):
            gt_frames = gt_data.get("gt_frames", [])
            gt_proprio = gt_data.get("gt_proprio", None)
            reward, meta = self.compute_single(gen_frames, gt_frames, gt_proprio)
            rewards.append(reward)
            metadata_list.append(meta)

        return torch.tensor(rewards, dtype=torch.float32), metadata_list

    def compute_batch_async(self, gen_videos, gt_data_list):
        """Submit batch reward computation as futures for async processing."""
        futures = []
        for gen_frames, gt_data in zip(gen_videos, gt_data_list):
            gt_frames = gt_data.get("gt_frames", [])
            gt_proprio = gt_data.get("gt_proprio", None)
            future = self.executor.submit(
                self.compute_single, gen_frames, gt_frames, gt_proprio)
            futures.append(future)
        return futures

    @staticmethod
    def collect_futures(futures):
        """Collect results from async futures."""
        rewards = []
        metadata_list = []
        for f in futures:
            reward, meta = f.result()
            rewards.append(reward)
            metadata_list.append(meta)
        return torch.tensor(rewards, dtype=torch.float32), metadata_list

    def _compute_scene_consistency(self, gen_frames_np):
        """Compute scene consistency using DINOv2."""
        import torch.nn.functional as F

        device = next(self.dinov2_model.parameters()).device
        frames_tensor = torch.from_numpy(
            np.stack(gen_frames_np)).permute(0, 3, 1, 2).to(device)

        # DINOv2 transform
        _, _, H, W = frames_tensor.shape
        max_side = max(H, W)
        pad_top = (max_side - H) // 2
        pad_bottom = max_side - H - pad_top
        pad_left = (max_side - W) // 2
        pad_right = max_side - W - pad_left
        frames_tensor = F.pad(frames_tensor,
                              (pad_left, pad_right, pad_top, pad_bottom))
        frames_tensor = F.interpolate(
            frames_tensor.float(), size=518, mode='bilinear',
            align_corners=False, antialias=False)
        frames_tensor = frames_tensor / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406],
                            device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225],
                           device=device).view(1, 3, 1, 1)
        frames_tensor = (frames_tensor - mean) / std

        sim_total = 0.0
        first_feat = None
        prev_feat = None

        with torch.no_grad():
            for i in range(len(frames_tensor)):
                frame = frames_tensor[i].unsqueeze(0)
                feat = self.dinov2_model.forward_features(frame)["x_norm_patchtokens"]
                feat = F.normalize(feat, dim=-1, p=2)

                if i == 0:
                    first_feat = feat
                else:
                    sim_prev = F.cosine_similarity(
                        prev_feat, feat, dim=1).mean(dim=-1).item()
                    sim_first = F.cosine_similarity(
                        first_feat, feat, dim=1).mean(dim=-1).item()
                    sim_total += max(0.0, (sim_prev + sim_first) / 2)

                prev_feat = feat

        return sim_total / (len(gen_frames_np) - 1)

    def _compute_ndtw(self, gen_frames_np, gt_frames_np):
        """Compute NDTW using YOLO detection + fastdtw."""
        from scipy.spatial.distance import euclidean
        from scipy.spatial import ConvexHull
        from fastdtw import fastdtw

        pred_traj = self._extract_trajectory_yolo(gen_frames_np)
        self.yolo_model.predictor = None  # Reset tracker
        gt_traj = self._extract_trajectory_yolo(gt_frames_np)
        self.yolo_model.predictor = None

        # Interpolation fill
        pred_filled, inv_pred = self._traj_interpo_fill(pred_traj)
        gt_filled, inv_gt = self._traj_interpo_fill(gt_traj)

        # Select gripper with max farthest distance in GT
        n_traj = gt_filled.shape[1]
        gt_max_distance_list = []
        valid_indices = []
        for i in range(n_traj):
            if inv_gt[i]:
                continue
            d = self._farthest_distance(gt_filled[:, i])
            gt_max_distance_list.append(d)
            valid_indices.append(i)

        if len(gt_max_distance_list) == 0:
            return 0.0

        max_idx_local = np.argmax(gt_max_distance_list)
        max_distance_index = valid_indices[max_idx_local]

        if inv_pred[max_distance_index]:
            ndtw_raw = 0.0
        else:
            di, pi = fastdtw(pred_filled[:, max_distance_index],
                             gt_filled[:, max_distance_index],
                             dist=euclidean)
            di = di / len(pi)
            ndtw_raw = 1.0 / di if di > 0 else 50.202

        ndtw_max = 50.202
        return min(ndtw_raw / ndtw_max, 1.0)

    def _extract_trajectory_yolo(self, frames_np):
        trajectory_data = []
        for frame in frames_np:
            frame_resized = cv2.resize(frame, (640, 480),
                                       interpolation=cv2.INTER_LINEAR)
            results = self.yolo_model.track(frame_resized, persist=True, conf=0.8)
            boxes = results[0].boxes
            clses = boxes.cls.cpu().tolist() if boxes.cls is not None else []
            confs = boxes.conf.cpu().tolist() if boxes.conf is not None else []

            selected_indices = {}
            for idx, (cls, conf) in enumerate(zip(clses, confs)):
                if cls not in selected_indices or conf > selected_indices[cls][1]:
                    selected_indices[cls] = (idx, conf)
            selected_idxs = [v[0] for v in selected_indices.values()]

            filtered_boxes = boxes[selected_idxs].xywh.cpu() if selected_idxs else []
            filtered_clses = [clses[i] for i in selected_idxs]

            left_data = (-1.0, -1.0)
            right_data = (-1.0, -1.0)
            for box, cls in zip(filtered_boxes, filtered_clses):
                x_center, y_center, w, h = box.tolist()
                if cls == 0.0:
                    left_data = (x_center / 640.0, y_center / 480.0)
                elif cls == 1.0:
                    right_data = (x_center / 640.0, y_center / 480.0)
            trajectory_data.append([list(left_data), list(right_data)])
        return np.array(trajectory_data, dtype=np.float32)

    @staticmethod
    def _one_traj_interpo_fill(data):
        mask = (data != [-1., -1.]).any(axis=1)
        invalid_ratio = 1 - np.mean(mask)
        if invalid_ratio > 0.80:
            return data, True
        n = data.shape[0]
        prev_arr = np.full(n, -1, dtype=int)
        next_arr = np.full(n, n, dtype=int)
        last = -1
        for j in range(n):
            if mask[j]:
                last = j
            prev_arr[j] = last
        last = n
        for j in range(n - 1, -1, -1):
            if mask[j]:
                last = j
            next_arr[j] = last
        missing = np.where(~mask)[0]
        if len(missing) == 0:
            return data, False
        p_vals = prev_arr[missing]
        q_vals = next_arr[missing]
        mask_p_invalid = (p_vals == -1)
        mask_q_invalid = (q_vals == n)
        mask_both_valid = ~mask_p_invalid & ~mask_q_invalid
        if np.any(mask_p_invalid):
            data[missing[mask_p_invalid]] = data[q_vals[mask_p_invalid]]
        if np.any(mask_q_invalid):
            data[missing[mask_q_invalid]] = data[p_vals[mask_q_invalid]]
        if np.any(mask_both_valid):
            valid_missing = missing[mask_both_valid]
            p = p_vals[mask_both_valid]
            q = q_vals[mask_both_valid]
            alpha = (valid_missing - p) / (q - p).astype(float)
            alpha = alpha[:, np.newaxis]
            data[valid_missing] = (1 - alpha) * data[p] + alpha * data[q]
        return data, False

    def _traj_interpo_fill(self, traj):
        n_traj = traj.shape[1]
        filled_trajs = []
        invalid_trajs = []
        for i in range(n_traj):
            filled, invalid = self._one_traj_interpo_fill(traj[:, i].copy())
            filled_trajs.append(filled)
            invalid_trajs.append(invalid)
        return np.stack(filled_trajs, axis=1), invalid_trajs

    @staticmethod
    def _farthest_distance(points):
        if len(points) < 2:
            return 0.0
        try:
            hull = ConvexHull(points)
        except Exception:
            return 0.0
        hull_points = points[hull.vertices]
        n = len(hull_points)
        max_dist = 0.0
        k = 1
        pts = np.vstack([hull_points, hull_points[0]])
        for i in range(n):
            j = (i + 1) % n
            while True:
                next_k = (k + 1) % n
                cross = ((pts[j, 0] - pts[i, 0]) * (pts[next_k, 1] - pts[k, 1]) -
                         (pts[j, 1] - pts[i, 1]) * (pts[next_k, 0] - pts[k, 0]))
                if cross < 0:
                    k = next_k
                else:
                    break
            max_dist = max(max_dist,
                           np.linalg.norm(pts[i] - pts[k]),
                           np.linalg.norm(pts[i] - pts[next_k]))
        return max_dist
