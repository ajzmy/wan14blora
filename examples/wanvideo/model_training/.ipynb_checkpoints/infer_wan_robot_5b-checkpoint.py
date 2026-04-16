"""
Inference script for WanRobotModel5B on AgiBotWorld test set.

Key differences from infer_wan_robot.py (1.3B):
  - No CLIP image encoder (5B uses TI2V conditioning)
  - VAE38 (Wan2.2_VAE.pth): 48-channel latents, 16x spatial downsampling
  - First frame conditioned via latent fusion (frame 0 = clean VAE latent, not y-mask)
  - Noise shape: (B, 48, T_lat, H/16, W/16)
  - Default resolution: 480x640 (matches training)

Usage (run from DiffSynth-Studio root):
  python examples/wanvideo/model_training/infer_wan_robot_5b.py \\
    --dit_weight_path ./checkpoints/wan_robot_5b_backup_20epoch_attempt/step-4000.safetensors \\
    --model_dir       /data/xiejunbin/Wan2.2-TI2V-5B \\
    --test_root       /data/xiejunbin/AgiBotWorldChallenge-2026/test/test/info_dataset \\
    --output_dir      ./test_outputs_5b \\
    --height 480 --width 640
"""

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from einops import rearrange
from tqdm import tqdm

# ── paths ───────────────────────────────────────────────────────────────
THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
DIFFSYNTH = os.path.dirname(os.path.dirname(os.path.dirname(THIS_DIR)))
sys.path.insert(0, DIFFSYNTH)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(THIS_DIR, 'evac'))

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion.flow_match import FlowMatchScheduler

from dataset.agibotworld_wan_dataset import AgiBotWorldWanDataset, SUBSAMPLE_STRIDE
from models.wan_robot_model_5b import WanRobotModel5B

import torchvision.transforms as transforms

# evac utilities
from lvdm.data.get_actions import parse_h5
from lvdm.data.utils import gen_batch_ray_parellel, intrinsic_transform
from lvdm.data.traj_vis_statistics import (
    ColorMapLeft, ColorMapRight, ColorListLeft, ColorListRight,
    EndEffectorPts, Gripper2EEFCvt,
)
from lvdm.data.utils import get_transformation_matrix_from_quat
from lvdm.data.statistics import StatisticInfo


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def encode_video(vae, video, device, dtype):
    """video: (B, 3, T, H, W) in [-1,1]. Returns (B, 48, T_lat, H/16, W/16)."""
    B = video.shape[0]
    with torch.no_grad():
        latents = vae.encode(
            [video[b].to(device=device, dtype=dtype) for b in range(B)],
            device=device,
        )
    if isinstance(latents, (list, tuple)):
        latents = torch.stack(latents)
    return latents.to(device=device, dtype=dtype)


def subsample_to_lat(tensor, T_lat):
    """Subsample T → T_lat by taking indices 0, 4, 8, ..."""
    if tensor.dim() == 5:
        indices = [min(4 * i, tensor.shape[2] - 1) for i in range(T_lat)]
        return tensor[:, :, indices, :, :]
    elif tensor.dim() == 3:
        indices = [min(4 * i, tensor.shape[1] - 1) for i in range(T_lat)]
        return tensor[:, indices, :]
    else:
        raise ValueError(f"Unexpected tensor dim: {tensor.dim()}")


# ──────────────────────────────────────────────────────────────────────
# Test episode loader  (same logic as 1.3B version)
# ──────────────────────────────────────────────────────────────────────

def load_test_episode(ep_path, sample_size, traj_radius=50):
    """
    Load a test episode and return all conditioning tensors.

    Returns:
        first_frame_t  : (1, 3, H, W) normalised to [-1,1]
        traj_maps      : (1, 3, T, H, W) normalised to [-1,1]
        raymap         : (1, 6, T, H, W)
        delta_action   : (1, T-1, 14)  normalised
        T              : number of frames
    """
    H, W = sample_size
    norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    # ── first frame ──
    first_frame_pil = Image.open(os.path.join(ep_path, 'frame.png')).convert('RGB')
    first_frame_np  = np.array(first_frame_pil, dtype=np.float32) / 255.0
    orig_h, orig_w  = first_frame_np.shape[:2]

    first_frame_t = torch.from_numpy(first_frame_np).permute(2, 0, 1)
    first_frame_t = F.interpolate(
        first_frame_t.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
    )
    first_frame_t = norm(first_frame_t[0]).unsqueeze(0)  # (1,3,H,W)

    # ── load actions (5fps) ──
    h5_path = os.path.join(ep_path, 'proprio_stats.h5')
    abs_action, delta = parse_h5(h5_path, slices=None, delta_act_sidx=1)
    T = abs_action.shape[0]

    action       = torch.tensor(abs_action, dtype=torch.float32)
    delta_action = torch.tensor(delta, dtype=torch.float32)  # (T-1, 14)

    # ── camera params ──
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
    c2w  = torch.eye(4, dtype=torch.float32)
    c2w[:3, :3] = torch.tensor(info['extrinsic']['rotation_matrix'])
    c2w[:3,  3] = torch.tensor(info['extrinsic']['translation_vector'])
    w2c  = torch.linalg.inv(c2w)
    c2ws = c2w.unsqueeze(0).expand(T, -1, -1)
    w2cs = w2c.unsqueeze(0).expand(T, -1, -1)

    # ── traj maps + raymap ──
    dummy_ds = AgiBotWorldWanDataset.__new__(AgiBotWorldWanDataset)
    dummy_ds.sample_size = sample_size
    dummy_ds.traj_radius = traj_radius
    dummy_ds.norm        = norm

    # normalise delta_action (matches training)
    stats = StatisticInfo["agibotworld"]
    act_mean = torch.tensor(stats["mean"], dtype=torch.float32)
    act_std  = torch.tensor(stats["std"],  dtype=torch.float32)
    delta_action[:, :6]   = (delta_action[:, :6]   - SUBSAMPLE_STRIDE * act_mean[:6])  / (SUBSAMPLE_STRIDE * act_std[:6])
    delta_action[:, 7:13] = (delta_action[:, 7:13] - SUBSAMPLE_STRIDE * act_mean[6:]) / (SUBSAMPLE_STRIDE * act_std[6:])

    traj_maps = dummy_ds._get_traj(action, w2cs, intrinsic)
    traj_maps = dummy_ds.norm(
        traj_maps.permute(1, 0, 2, 3)
    ).permute(1, 0, 2, 3)  # (3,T,H,W)

    raymap = dummy_ds._get_raymap(intrinsic, c2ws)  # (6,T,H,W)

    return (
        first_frame_t,              # (1,3,H,W)
        traj_maps.unsqueeze(0),     # (1,3,T,H,W)
        raymap.unsqueeze(0),        # (1,6,T,H,W)
        delta_action.unsqueeze(0),  # (1,T-1,14)
        T,
    )


# ──────────────────────────────────────────────────────────────────────
# Sampling / denoising  (5B: no CLIP, fuse first frame into latents)
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample(
    dit, vae, text_encoder, tokenizer,
    first_frame_t,
    traj_maps, raymap, delta_action,
    T, H, W,
    prompt="robot arm manipulation",
    num_steps=50,
    cfg_scale=7.5,
    device=None,
    dtype=torch.bfloat16,
):
    """
    Flow-matching denoising for WanRobotModel5B.

    First-frame conditioning: encode first frame → clean latent, fuse into
    noisy latents at frame 0 and re-fuse after every denoising step.
    No CLIP encoder needed.
    """
    B = 1

    # ── encode first frame (pad video to T frames for VAE) ──
    first_frame_5d = torch.cat([
        first_frame_t.unsqueeze(2).to(device=device, dtype=dtype),
        torch.zeros(B, 3, T - 1, H, W, device=device, dtype=dtype),
    ], dim=2)
    all_latents = encode_video(vae, first_frame_5d, device, dtype)
    T_lat = all_latents.shape[2]
    H_lat = all_latents.shape[3]
    W_lat = all_latents.shape[4]
    first_frame_latents = all_latents[:, :, 0:1]   # (B, 48, 1, H_lat, W_lat)

    # ── VAE-encode traj maps ──
    traj_latents = encode_video(
        vae, traj_maps.to(device=device, dtype=dtype), device, dtype
    )

    # ── raymap at latent resolution ──
    raymap_lat = subsample_to_lat(raymap.to(device=device, dtype=dtype), T_lat)
    raymap_lat = raymap_lat.permute(0, 2, 1, 3, 4).flatten(0, 1)
    raymap_lat = F.interpolate(
        raymap_lat.float(), size=(H_lat, W_lat), mode='bilinear', align_corners=False
    ).to(dtype)
    raymap_lat = raymap_lat.view(B, T_lat, 6, H_lat, W_lat).permute(0, 2, 1, 3, 4)

    # ── delta_action at latent temporal resolution ──
    delta_action_lat = subsample_to_lat(
        delta_action.to(device=device, dtype=dtype), T_lat
    )

    # ── text encoding ──
    text_inputs = tokenizer([prompt])
    context     = text_encoder(text_inputs.to(device))

    # ── noise scheduler ──
    scheduler = FlowMatchScheduler(template="Wan")
    scheduler.set_timesteps(num_steps)
    sigmas, timesteps = scheduler.sigmas, scheduler.timesteps

    # ── initial latents: noise + fuse clean first frame ──
    latents = torch.randn(B, 48, T_lat, H_lat, W_lat, device=device, dtype=dtype)
    latents[:, :, 0:1] = first_frame_latents

    # ── denoising loop ──
    for sigma, t in zip(sigmas, timesteps):
        t_batch = torch.tensor([t.item()], device=device, dtype=dtype)

        if cfg_scale != 1.0:
            empty_tokens = tokenizer([""])
            ctx_unc  = text_encoder(empty_tokens.to(device))
            ctx_all  = torch.cat([context, ctx_unc], dim=0)
            tl_all   = traj_latents.expand(2, -1, -1, -1, -1)
            rm_all   = raymap_lat.expand(2, -1, -1, -1, -1)
            da_all   = delta_action_lat.expand(2, -1, -1)
            lat_in   = latents.expand(2, -1, -1, -1, -1)
            pred_all = dit(
                x=lat_in, timestep=t_batch, context=ctx_all,
                traj_latents=tl_all, raymap=rm_all, delta_action=da_all,
            )
            pred_cond, pred_unc = pred_all.chunk(2, dim=0)
            pred = pred_unc + cfg_scale * (pred_cond - pred_unc)
        else:
            pred = dit(
                x=latents, timestep=t_batch, context=context,
                traj_latents=traj_latents, raymap=raymap_lat,
                delta_action=delta_action_lat,
            )

        latents = scheduler.step(pred, t, latents)
        # re-fuse first frame after each step (TI2V conditioning)
        latents[:, :, 0:1] = first_frame_latents

    # ── decode ──
    frames = vae.decode(latents, device=device)[0]  # (3, T', H, W)
    if frames.shape[1] > T:
        frames = frames[:, :T]

    frames = ((frames.float() * 0.5 + 0.5).clamp(0, 1) * 255).byte()
    frames = frames.permute(1, 2, 3, 0).cpu().numpy()  # (T, H, W, 3)
    return frames


CHUNK_SIZE = 49  # 4*12+1, WAN 4n+1 constraint (same as 1.3B)


@torch.no_grad()
def sample_ar(
    dit, vae, text_encoder, tokenizer,
    first_frame_t,
    traj_maps, raymap, delta_action,
    T_total, H, W,
    chunk_size=CHUNK_SIZE,
    **kwargs,
):
    """
    Chunk-based autoregressive generation.
    Last generated frame becomes first frame for the next chunk.
    """
    all_frames = []
    norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    # delta_action is (B, T_total-1, 14): delta_action[j] = frame j -> frame j+1
    # For a chunk of `actual` frames (t..t+actual-1), the within-chunk transitions
    # are delta_action[t..t+actual-2], i.e. actual-1 deltas (matching training:
    # T frames -> T-1 deltas).

    t = 0
    while t < T_total:
        actual = min(chunk_size, T_total - t)

        traj_chunk = traj_maps[:, :, t:t + actual]
        ray_chunk  = raymap[:, :, t:t + actual]
        # actual-1 deltas for actual frames (frame t->t+1, ..., t+actual-2->t+actual-1)
        da_end   = min(t + actual - 1, delta_action.shape[1])
        da_chunk = delta_action[:, t:da_end, :]

        # pad last chunk to chunk_size
        if actual < chunk_size:
            pad = chunk_size - actual
            traj_chunk = F.pad(traj_chunk, (0, 0, 0, 0, 0, pad), mode='replicate')
            ray_chunk  = F.pad(ray_chunk,  (0, 0, 0, 0, 0, pad), mode='replicate')
        # pad da_chunk to chunk_size-1 (match training: T-1 deltas for T frames)
        da_target = chunk_size - 1
        if da_chunk.shape[1] < da_target:
            da_pad = da_target - da_chunk.shape[1]
            if da_chunk.shape[1] == 0:
                da_chunk = torch.zeros(
                    delta_action.shape[0], da_target, delta_action.shape[2],
                    dtype=delta_action.dtype, device=delta_action.device,
                )
            else:
                da_chunk = F.pad(da_chunk, (0, 0, 0, da_pad), mode='replicate')

        frames = sample(
            dit=dit, vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            first_frame_t=first_frame_t,
            traj_maps=traj_chunk,
            raymap=ray_chunk,
            delta_action=da_chunk,
            T=chunk_size, H=H, W=W,
            **kwargs,
        )  # (chunk_size, H, W, 3) uint8

        all_frames.append(frames[:actual])

        # update first frame for next chunk
        last_frame_np = frames[actual - 1]
        first_frame_t = norm(
            torch.from_numpy(last_frame_np.astype(np.float32) / 255.0).permute(2, 0, 1)
        ).unsqueeze(0)  # (1, 3, H, W)

        t += actual

    return np.concatenate(all_frames, axis=0)  # (T_total, H, W, 3)


def save_video(frames, path, fps=5):
    T, H, W, _ = frames.shape
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H)
    )
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype  = torch.bfloat16

    print("Loading pipeline (T5 + VAE38) …")
    model_configs = [
        ModelConfig(path=os.path.join(args.model_dir, 'models_t5_umt5-xxl-enc-bf16.pth')),
        ModelConfig(path=os.path.join(args.model_dir, 'Wan2.2_VAE.pth')),
    ]
    tokenizer_config = ModelConfig(path=os.path.join(args.model_dir, 'google', 'umt5-xxl'))
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=dtype, device=device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
        redirect_common_files=False,
    )
    vae          = pipe.vae.eval()
    text_encoder = pipe.text_encoder.eval()
    tokenizer    = pipe.tokenizer

    print("Loading WanRobotModel5B …")
    dit = WanRobotModel5B.from_pretrained(args.dit_weight_path).to(
        device=device, dtype=dtype
    ).eval()

    sample_size = (args.height, args.width)
    os.makedirs(args.output_dir, exist_ok=True)

    task_dirs = sorted(os.listdir(args.test_root))
    for task_id in tqdm(task_dirs, desc='tasks'):
        task_path = os.path.join(args.test_root, task_id)
        ep_dirs   = sorted(os.listdir(task_path))
        for ep_id in tqdm(ep_dirs, desc=f'task {task_id}', leave=False):
            ep_path  = os.path.join(task_path, ep_id)
            out_path = os.path.join(args.output_dir, f"{task_id}_{ep_id}.mp4")
            if os.path.exists(out_path):
                continue
            try:
                first_frame_t, traj_maps, raymap, delta_action, T = \
                    load_test_episode(ep_path, sample_size)

                frames = sample_ar(
                    dit=dit, vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    first_frame_t=first_frame_t,
                    traj_maps=traj_maps,
                    raymap=raymap,
                    delta_action=delta_action,
                    T_total=T, H=args.height, W=args.width,
                    num_steps=args.num_steps,
                    cfg_scale=args.cfg_scale,
                    device=device, dtype=dtype,
                )
                save_video(frames, out_path, fps=5)

            except Exception as e:
                print(f"[WARN] {task_id}/{ep_id} failed: {e}")
                import traceback; traceback.print_exc()
                continue

    print(f"Done. Videos saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dit_weight_path", required=True,
                        help="Path to trained 5B checkpoint (.safetensors)")
    parser.add_argument("--model_dir",       required=True,
                        help="Wan2.2-TI2V-5B model directory (T5 + VAE38 + tokenizer)")
    parser.add_argument("--test_root",       required=True,
                        help="Path to test/info_dataset directory")
    parser.add_argument("--output_dir",      default="./test_outputs_5b")
    parser.add_argument("--height",          type=int,   default=480)
    parser.add_argument("--width",           type=int,   default=640)
    parser.add_argument("--num_steps",       type=int,   default=50)
    parser.add_argument("--cfg_scale",       type=float, default=7.5)
    args = parser.parse_args()
    main(args)
