"""
Inference script for Fun-Control 1.3B with TEXT conditioning.

For each test episode:
  1. Load first frame (frame.png) as reference_image (fixed for ALL chunks)
  2. Load actions → render traj maps as control_video
  3. Load text.txt as prompt (fallback to generic prompt)
  4. Chunk-based autoregressive generation:
     - Chunk 0: input_image = reference_image (GT first frame)
     - Chunk N: input_image = last generated frame of chunk N-1
     - reference_image is ALWAYS the GT first frame (frame.png)

Output: submission_dataset/{task}/{ep}/{rollout}/video/frame_00000.jpg ...

Usage:
    python infer_fun_control_1_3b_text.py \
        --checkpoint ./checkpoints/agibotworld_fun_control_1_3b_text/step-4000.safetensors \
        --model_dir /root/pretrained_weights/PAI/Wan2.1-Fun-V1.1-1.3B-Control \
        --test_root /root/data/AgiBotWorldChallenge-2026/test/info_dataset \
        --output_dir ./submission_dataset \
        --cfg_scale 5.0 --seed 42
"""

import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import cv2, h5py
import numpy as np
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from PIL import Image
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, "evac"))

from diffsynth.core import load_state_dict
from diffsynth.pipelines.wan_video_cfg_zero import WanVideoPipeline, ModelConfig

from lvdm.data.traj_vis_statistics import (
    ColorMapLeft, ColorMapRight, ColorListLeft, ColorListRight,
    EndEffectorPts, Gripper2EEFCvt,
)
from lvdm.data.get_actions import parse_h5
from lvdm.data.utils import get_transformation_matrix_from_quat, intrinsic_transform

FALLBACK_PROMPT = "robot arm manipulation"
CHUNK_SIZE = 25  # must satisfy (N-1) % 4 == 0, matches training num_frames


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def load_camera_params(ep_path, num_frames):
    with open(os.path.join(ep_path, "head_intrinsic_params.json")) as f:
        intr = json.load(f)["intrinsic"]
    intrinsic = torch.eye(3, dtype=torch.float32)
    intrinsic[0, 0] = intr["fx"]
    intrinsic[1, 1] = intr["fy"]
    intrinsic[0, 2] = intr["ppx"]
    intrinsic[1, 2] = intr["ppy"]

    with open(os.path.join(ep_path, "head_extrinsic_params_aligned.json")) as f:
        extr_list = json.load(f)
    info = extr_list[0]
    c2w = torch.eye(4, dtype=torch.float32)
    c2w[:3, :3] = torch.tensor(info["extrinsic"]["rotation_matrix"], dtype=torch.float32)
    c2w[:3, 3] = torch.tensor(info["extrinsic"]["translation_vector"], dtype=torch.float32)
    w2c = torch.linalg.inv(c2w)

    return intrinsic, w2c.unsqueeze(0).expand(num_frames, -1, -1)


def render_traj_maps(action, w2cs, intrinsic, H, W, traj_radius=50):
    """Render trajectory maps as list of PIL Images."""
    ee_key_pts = torch.tensor(EndEffectorPts, dtype=torch.float32).view(1, 4, 4).permute(0, 2, 1)
    cvt_matrix = torch.tensor(Gripper2EEFCvt, dtype=torch.float32).view(1, 4, 4)

    pose_l_mat = get_transformation_matrix_from_quat(action[:, 0:7])
    pose_r_mat = get_transformation_matrix_from_quat(action[:, 8:15])

    ee2cam_l = torch.matmul(torch.matmul(w2cs, pose_l_mat), cvt_matrix)
    ee2cam_r = torch.matmul(torch.matmul(w2cs, pose_r_mat), cvt_matrix)

    pts_l = torch.matmul(ee2cam_l, ee_key_pts)
    pts_r = torch.matmul(ee2cam_r, ee_key_pts)

    K = intrinsic.unsqueeze(0)
    uvs_l = (torch.matmul(K, pts_l[:, :3, :]) / pts_l[:, 2:3, :])[:, :2, :].permute(0, 2, 1).to(torch.int64)
    uvs_r = (torch.matmul(K, pts_r[:, :3, :]) / pts_r[:, 2:3, :])[:, :2, :].permute(0, 2, 1).to(torch.int64)

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
                cv2.circle(img, tuple(base[:2]), traj_radius, color, -1)

        for uvs, colors in [(uvs_l[i], ColorListLeft), (uvs_r[i], ColorListRight)]:
            base = uvs[0].numpy()
            if 0 <= base[0] < W and 0 <= base[1] < H:
                for j in range(1, len(uvs)):
                    pt = uvs[j].numpy()
                    cv2.line(img, tuple(base[:2]), tuple(pt[:2]), colors[j - 1], 8)

        pil_list.append(Image.fromarray(img))
    return pil_list


def load_prompt(ep_path):
    """Load text.txt if available, otherwise return fallback."""
    text_path = os.path.join(ep_path, "text.txt")
    if os.path.exists(text_path):
        with open(text_path, "r") as f:
            text = f.read().strip()
        if text:
            return text
    return FALLBACK_PROMPT


def process_episode(ep_path, H, W):
    """Load all data for one test episode.

    Returns:
        reference_image: PIL Image (GT first frame, resized)
        control_video:   list of PIL Images (traj maps, one per frame)
        prompt:          str
        total_frames:    int
    """
    with h5py.File(os.path.join(ep_path, "proprio_stats.h5"), "r") as f:
        total_frames = f["timestamp"].shape[0]

    indices = list(range(total_frames))
    intrinsic_orig, w2cs = load_camera_params(ep_path, total_frames)

    ref_img = Image.open(os.path.join(ep_path, "frame.png")).convert("RGB")
    orig_w, orig_h = ref_img.size
    intrinsic = intrinsic_transform(intrinsic_orig, (orig_h, orig_w), (H, W), "resize")

    abs_action, _ = parse_h5(
        os.path.join(ep_path, "proprio_stats.h5"), slices=indices, delta_act_sidx=1
    )
    action = torch.tensor(abs_action, dtype=torch.float32)
    control_video = render_traj_maps(action, w2cs, intrinsic, H, W)

    reference_image = ref_img.resize((W, H), Image.BILINEAR)
    prompt = load_prompt(ep_path)

    return reference_image, control_video, prompt, total_frames


# ──────────────────────────────────────────────────────────────────────
# Chunk-based autoregressive generation
# ──────────────────────────────────────────────────────────────────────

def generate_video(pipe, reference_image, control_video, prompt, total_frames,
                   chunk_size, seed, height, width, cfg_scale, num_inference_steps):
    """
    Generate full video by chunked autoregressive inference.

    Conditioning strategy:
      - reference_image: ALWAYS the GT first frame (frame.png), fixed across all chunks.
                         Provides scene appearance via CLIP + VAE encoding.
      - input_image:     Chunk 0 → reference_image (GT first frame)
                         Chunk N → last generated frame of chunk N-1
                         Tells the model "continue from this frame".
      - control_video:   Traj maps for the current chunk.
      - prompt:          Text description from text.txt.
    """
    all_frames = []
    current_input_image = reference_image  # chunk 0: input_image = GT first frame

    t = 0
    chunk_idx = 0
    while t < total_frames:
        actual = min(chunk_size, total_frames - t)

        # Slice control_video for this chunk
        control_chunk = control_video[t:t + actual]

        # Pad last chunk to chunk_size if needed (replicate last traj map)
        if actual < chunk_size:
            control_chunk = control_chunk + [control_chunk[-1]] * (chunk_size - actual)

        print(f"    Chunk {chunk_idx}: frames [{t}, {t + actual}) "
              f"(actual={actual}, padded_to={chunk_size})")

        video_frames = pipe(
            prompt=prompt,
            negative_prompt="blurry, low resolution, deformation caused by movement, object twisting deformation, penetration, object suddenly disappears, grainy, noisy, pixelated, compression artifacts, distorted, unnatural, low quality",
            input_image=current_input_image,
            control_video=control_chunk,
            reference_image=reference_image,  # always GT first frame
            seed=seed,
            tiled=True,
            num_frames=chunk_size,
            height=height,
            width=width,
            cfg_scale=cfg_scale,
            num_inference_steps=num_inference_steps,
        )

        # Keep only the actual frames (discard padding)
        all_frames.extend(video_frames[:actual])

        # Next chunk: input_image = last generated frame of this chunk
        current_input_image = video_frames[actual - 1]

        t += actual
        chunk_idx += 1

    # Replace first frame with GT
    all_frames[0] = reference_image

    return all_frames


# ──────────────────────────────────────────────────────────────────────
# Save helpers
# ──────────────────────────────────────────────────────────────────────

def save_frames_as_jpg(frames, out_dir, skip_first=True):
    """Save frames as frame_00000.jpg, frame_00001.jpg, ...
    Skips the first frame if skip_first (official submission format)."""
    os.makedirs(out_dir, exist_ok=True)
    start = 1 if skip_first else 0
    for i, frame in enumerate(frames[start:]):
        path = os.path.join(out_dir, f"frame_{i:05d}.jpg")
        if isinstance(frame, Image.Image):
            frame.save(path, "JPEG", quality=95)
        else:
            Image.fromarray(frame).save(path, "JPEG", quality=95)


def save_video(frames, path, fps=5):
    """Save frames as mp4 for visualization."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    first = frames[0]
    if isinstance(first, Image.Image):
        W, H = first.size
    else:
        H, W = first.shape[:2]

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for f in frames:
        if isinstance(f, Image.Image):
            f = np.array(f)
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Inference for Fun-Control 1.3B with TEXT conditioning"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to trained DIT checkpoint (.safetensors)")
    parser.add_argument("--model_dir", required=True,
                        help="Wan2.1-Fun-V1.1-1.3B-Control model directory")
    parser.add_argument("--test_root", required=True,
                        help="Path to test/info_dataset directory")
    parser.add_argument("--output_dir", default="./submission_dataset")
    parser.add_argument("--rollout_id", type=int, default=0,
                        help="Rollout index for submission (0, 1, 2)")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE,
                        help="Frames per chunk, must satisfy (N-1) %% 4 == 0")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_mp4", action="store_true",
                        help="Also save mp4 videos for visualization")
    parser.add_argument("--only_mp4", action="store_true",
                        help="Only save mp4, skip saving frame images")
    parser.add_argument("--episodes", nargs="*", default=None,
                        help="Only generate specific episodes, e.g. --episodes 2019/650432001 2020/673471008")
    parser.add_argument("--meta_email", default="yourmailbox@mail.com")
    parser.add_argument("--device", default="npu")
    args = parser.parse_args()

    assert (args.chunk_size - 1) % 4 == 0, \
        f"chunk_size must satisfy (N-1) % 4 == 0, got {args.chunk_size}"

    # ── Load pipeline ──
    print(f"Loading pipeline from {args.model_dir} ...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=[
            ModelConfig(path=os.path.join(args.model_dir, "diffusion_pytorch_model.safetensors")),
            ModelConfig(path=os.path.join(args.model_dir, "models_t5_umt5-xxl-enc-bf16.pth")),
            ModelConfig(path=os.path.join(args.model_dir, "Wan2.1_VAE.pth")),
            ModelConfig(path=os.path.join(args.model_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")),
        ],
    )

    # ── Load trained DIT weights ──
    print(f"Loading checkpoint: {args.checkpoint}")
    sd = load_state_dict(args.checkpoint)
    pipe.dit.load_state_dict(sd)
    print(f"  Loaded {len(sd)} keys into DIT")

    # ── Collect episodes (only those with text.txt) ──
    episodes = []
    skipped = []
    for task in sorted(os.listdir(args.test_root)):
        task_dir = os.path.join(args.test_root, task)
        if not os.path.isdir(task_dir):
            continue
        for ep in sorted(os.listdir(task_dir)):
            ep_dir = os.path.join(task_dir, ep)
            if not os.path.isdir(ep_dir):
                continue
            if os.path.exists(os.path.join(ep_dir, "text.txt")):
                episodes.append((task, ep, ep_dir))
            else:
                skipped.append(f"{task}/{ep}")
    print(f"Found {len(episodes)} episodes with text.txt, skipped {len(skipped)} without")
    if skipped:
        print(f"  Skipped: {', '.join(skipped)}")

    # Filter episodes if --episodes is specified
    if args.episodes is not None:
        ep_set = set(args.episodes)
        episodes = [(t, e, p) for t, e, p in episodes if f"{t}/{e}" in ep_set]
        print(f"Filtered to {len(episodes)} episodes: {[f'{t}/{e}' for t, e, _ in episodes]}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Generate ──
    for i, (task, ep_name, ep_path) in enumerate(episodes):
        out_frame_dir = os.path.join(
            args.output_dir, task, ep_name, str(args.rollout_id), "video"
        )

        # Skip if already generated
        if os.path.exists(out_frame_dir) and len(os.listdir(out_frame_dir)) > 0:
            print(f"[{i+1}/{len(episodes)}] {task}/{ep_name} already exists, skipping")
            continue

        print(f"[{i+1}/{len(episodes)}] {task}/{ep_name}")
        try:
            reference_image, control_video, prompt, total_frames = \
                process_episode(ep_path, args.height, args.width)
            print(f"  prompt: {prompt}")
            print(f"  total_frames: {total_frames}")

            video_frames = generate_video(
                pipe=pipe,
                reference_image=reference_image,
                control_video=control_video,
                prompt=prompt,
                total_frames=total_frames,
                chunk_size=args.chunk_size,
                seed=args.seed,
                height=args.height,
                width=args.width,
                cfg_scale=args.cfg_scale,
                num_inference_steps=args.num_inference_steps,
            )

            # Save submission frames (skip first frame per official format)
            if not args.only_mp4:
                save_frames_as_jpg(video_frames, out_frame_dir, skip_first=True)
                print(f"  -> saved {total_frames - 1} frames to {out_frame_dir}")

            # Save mp4
            if args.save_mp4 or args.only_mp4:
                mp4_path = os.path.join(args.output_dir, "mp4", f"{task}_{ep_name}.mp4")
                save_video(video_frames, mp4_path, fps=5)
                print(f"  -> saved mp4 to {mp4_path}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # ── meta_info.txt ──
    meta_path = os.path.join(args.output_dir, "meta_info.txt")
    with open(meta_path, "w") as f:
        f.write(args.meta_email + "\n")

    print(f"\nDone. Output at: {args.output_dir}")


if __name__ == "__main__":
    main()
