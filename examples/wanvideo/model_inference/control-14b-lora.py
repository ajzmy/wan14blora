import os, sys, json, argparse
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import h5py
import numpy as np
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from PIL import Image
from tqdm import tqdm
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.core.device.npu_compatible_device import get_device_name
from diffsynth.core import load_state_dict

from lvdm.data.traj_vis_statistics import (
    ColorMapLeft, ColorMapRight, ColorListLeft, ColorListRight,
    EndEffectorPts, Gripper2EEFCvt,
)
from lvdm.data.get_actions import parse_h5
from lvdm.data.utils import get_transformation_matrix_from_quat, intrinsic_transform


def get_traj_maps(action, w2cs, intrinsic, H, W, traj_radius=50):
    
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


def load_test_episode(ep_path, sample_size):
    # first frame
    H, W = sample_size
    first_frame_pil = Image.open(os.path.join(ep_path, 'frame.png')).convert('RGB')
    orig_h, orig_w = np.array(first_frame_pil).shape[:2]
    first_frame_pil = first_frame_pil.resize((W, H), Image.BILINEAR)

    # reference frame 
    ref_frame_pil = Image.open(os.path.join(ep_path, 'frame.png')).convert('RGB')
    ref_frame_pil = ref_frame_pil.resize((W, H), Image.BILINEAR)

    # action
    h5_path = os.path.join(ep_path, 'proprio_stats.h5')
    abs_action, _ = parse_h5(h5_path, slices=None, delta_act_sidx=1)
    T = abs_action.shape[0]
    action = torch.tensor(abs_action, dtype=torch.float32)

    # camera parameters
    with open(os.path.join(ep_path, 'head_intrinsic_params.json')) as f:
        intr = json.load(f)['intrinsic']
    intrinsic_orig = torch.eye(3, dtype=torch.float32)
    intrinsic_orig[0, 0] = intr['fx']
    intrinsic_orig[1, 1] = intr['fy']
    intrinsic_orig[0, 2] = intr['ppx']
    intrinsic_orig[1, 2] = intr['ppy']
    intrinsic = intrinsic_transform(intrinsic_orig, (orig_h, orig_w), (H, W), 'resize')

    with open(os.path.join(ep_path, 'head_extrinsic_params_aligned.json')) as f:
        extr_list = json.load(f)
    info = extr_list[0]
    c2w = torch.eye(4, dtype=torch.float32)
    c2w[:3, :3] = torch.tensor(info['extrinsic']['rotation_matrix'])
    c2w[:3, 3]  = torch.tensor(info['extrinsic']['translation_vector'])
    w2c  = torch.linalg.inv(c2w)
    w2cs = w2c.unsqueeze(0).expand(T, -1, -1)

    # traj maps
    traj_maps = get_traj_maps(action, w2cs, intrinsic, H, W)

    # text
    text_path = os.path.join(ep_path, "text.txt")
    if os.path.exists(text_path):
        with open(text_path, "r") as f:
            text = f.read().strip()
    else:
        print(f"Warning: text.txt in {ep_path} is empty or not exists. Using default prompt.")
        text = "Robot arms manipulation"

    return first_frame_pil, ref_frame_pil, traj_maps, text, T


def prepare_model(args):
    # here we don't use the vram_config since it will change the checkpoint loading logic(adding module prefix)
    vram_config = {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "npu",
        "computation_dtype": torch.bfloat16,
        "computation_device": "npu",
    }

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="npu",
        model_configs=[
            ModelConfig(path="/root/pretrained_weights/Wan2.1-v1.1-Control-14B/diffusion_pytorch_model.safetensors", **vram_config),
            ModelConfig(path="/root/pretrained_weights/PAI/Wan2.1-Fun-V1.1-1.3B-Control/models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
            ModelConfig(path="/root/pretrained_weights/PAI/Wan2.1-Fun-V1.1-1.3B-Control/Wan2.1_VAE.pth", **vram_config),
            ModelConfig(path="/root/pretrained_weights/PAI/Wan2.1-Fun-V1.1-1.3B-Control/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", **vram_config)
        ],
        tokenizer_config=ModelConfig(path="/root/pretrained_weights/PAI/Wan2.1-Fun-V1.1-1.3B-Control/google/umt5-xxl"),
        # vram_limit=torch.npu.mem_get_info(get_device_name())[1] / (1024 ** 3) - 2,
    )


    # load post-trained weights
    print(f"Loading post-trained checkpoint from: {args.checkpoint}")
    pipe.load_lora(pipe.dit, args.checkpoint, alpha=1)
    return pipe

# generate video chunk by chunk
def generate_video(pipe, first_frame_pil, ref_frame_pil, traj_maps, text, T, args):
    all_frames = []
    current_first_frame = first_frame_pil

    t = 0
    chunk_idx = 0
    while t < T:
        actual = min(args.chunk, T - t)

        # Slice vace_video for this chunk
        control_video_chunk = traj_maps[t:t + actual]
        print(f"    Chunk {chunk_idx}: frames [{t}, {t + actual}) ")

        # Pad last chunk to chunk_size if needed (replicate last traj map)
        if actual < args.chunk:
            control_video_chunk = control_video_chunk + [control_video_chunk[-1]] * (args.chunk - actual)
            print(f"(actual={actual}, padded_to={args.chunk})")
        
        frames = pipe(
            prompt=text,
            negative_prompt="blurry, Motion blur, object twisting deformation, penetration, object suddenly appears or disappears, violate physics, grainy, noisy, pixelated, compression artifacts, distorted, unnatural, low quality",
            input_image=current_first_frame,
            reference_image=ref_frame_pil,
            control_video=control_video_chunk,
            seed=args.seed, tiled=True,
            height=args.height,
            width=args.width,
            num_frames=args.chunk,
            cfg_scale=args.cfg_scale,
        )

        all_frames.extend(frames[:actual])
        current_first_frame = frames[actual - 1]
        t += actual
        chunk_idx += 1

    # Replace first frame with GT
    all_frames[0] = first_frame_pil
    return all_frames

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


def main():
    parser = argparse.ArgumentParser(
        description="Inference for Fun-Control 1.3B with TEXT conditioning"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to trained post-trained checkpoint (.safetensors)")
    parser.add_argument("--test_root", required=True,
                        help="Path to test/info_dataset directory")
    parser.add_argument("--output_dir", default="./submission_dataset")
    parser.add_argument("--rollout_id", type=int, default=0,
                        help="Rollout index for submission (0, 1, 2)")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--chunk", type=int, default=25,
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

    args = parser.parse_args()

    assert (args.chunk - 1) % 4 == 0, \
        f"chunk_size must satisfy (N-1) % 4 == 0, got {args.chunk}"
    
    print("Preparing model...")
    pipe = prepare_model(args)
    
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
    print("Starting inference...")
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
            first_frame_pil, ref_frame_pil, traj_maps, text, T = load_test_episode(ep_path, (args.height, args.width))
            print(f"  prompt: {text}")
            print(f"  total_frames: {T}, chunk_size: {args.chunk}")
            frames = generate_video(pipe, first_frame_pil, ref_frame_pil, traj_maps, text, T, args)

            if not args.only_mp4:
                save_frames_as_jpg(frames, out_frame_dir, skip_first=True)
                print(f"  -> saved {T - 1} frames to {out_frame_dir}")

            if args.save_mp4 or args.only_mp4:
                video_path = os.path.join(args.output_dir, f"{task}_{ep_name}.mp4")
                # os.makedirs(os.path.dirname(video_path), exist_ok=True)
                save_video(frames, video_path, fps=5, quality=5)
                print(f"  -> saved mp4 to {video_path}")
        except Exception as e:
            print(f"  Error processing {task}/{ep_name}: {e}")
            import traceback
            traceback.print_exc()

    print("Inference completed.")


if __name__ == "__main__":
    main()