"""
Train WanRobotModel5B on AgiBotWorld dataset using DiffSynth-Studio framework.

Architecture
  - WanRobotModel5B: Wan2.2-TI2V-5B with expanded patch_embedding (48→102)
      + 48ch traj_map_vae38 + 6ch raymap
      + delta_action injected via Resampler(14→8 tokens→3072) into cross-attention context
  - Training module: DiffusionTrainingModule subclass
  - Loss: DiffSynth FlowMatchSFTLoss (flow matching SFT)
  - Training loop: DiffSynth launch_training_task

Key differences from 1.3B version:
  - Uses WanVideoVAE38 (48ch latents, 16x spatial downsampling)
  - No CLIP image encoder
  - First frame conditioning via first_frame_latents (not y=mask+vae)
  - Separated timestep handled inside WanRobotModel5B.forward()

Launch (single node, 8 GPUs):
  accelerate launch --num_processes 8 \\
      examples/wanvideo/model_training/train_wan_robot_5b.py \\
      --dit_weight_path /path/to/Wan2.2-TI2V-5B/diffusion_pytorch_model*.safetensors \\
      --model_dir       /path/to/Wan2.2-TI2V-5B \\
      --data_root       /data/.../iros_challenge_2025_acwm \\
      --output_path     ./checkpoints/wan_robot_5b \\
      --height 480 --width 640 \\
      --num_epochs 10 --learning_rate 1e-5 --save_steps 5000
"""

import os, sys, argparse
import torch
import torch.nn.functional as F
import accelerate
from accelerate import Accelerator

# ── paths ──────────────────────────────────────────────────────────────
THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
DIFFSYNTH = os.path.dirname(os.path.dirname(os.path.dirname(THIS_DIR)))
sys.path.insert(0, DIFFSYNTH)   # makes `diffsynth` importable
sys.path.insert(0, THIS_DIR)    # makes `dataset`, `models` importable
sys.path.insert(0, os.path.join(THIS_DIR, 'evac'))  # makes `lvdm` importable

# ── DiffSynth imports ──────────────────────────────────────────────────
from diffsynth.pipelines.wan_video       import WanVideoPipeline, ModelConfig
from diffsynth.diffusion                 import FlowMatchSFTLoss
from diffsynth.diffusion.training_module import DiffusionTrainingModule
from diffsynth.diffusion.runner          import launch_training_task
from diffsynth.diffusion.logger          import ModelLogger

# ── AgiBotWorld imports ────────────────────────────────────────────────
from dataset.agibotworld_wan_dataset import AgiBotWorldWanDataset
from models.wan_robot_model_5b       import WanRobotModel5B


# ──────────────────────────────────────────────────────────────────────
# Tensor helpers
# ──────────────────────────────────────────────────────────────────────

def encode_video(vae, video, device, dtype):
    """
    VAE-encode a video tensor.

    Args:
        video: (B, 3, T, H, W) float in [-1, 1]
    Returns:
        (B, C_lat, T_lat, H_lat, W_lat)
        For VAE38: C_lat=48, H_lat=H/16, W_lat=W/16
    """
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
    """
    Subsample temporal dimension T → T_lat by taking indices {0, 4, 8, ...}.

    Supports (B, C, T, H, W) and (B, T, D) tensors.
    """
    if tensor.dim() == 5:
        indices = [min(4 * i, tensor.shape[2] - 1) for i in range(T_lat)]
        return tensor[:, :, indices, :, :]
    elif tensor.dim() == 3:
        indices = [min(4 * i, tensor.shape[1] - 1) for i in range(T_lat)]
        return tensor[:, indices, :]
    else:
        raise ValueError(f"subsample_to_lat: unexpected tensor dim {tensor.dim()}")


# ──────────────────────────────────────────────────────────────────────
# Custom model function
# ──────────────────────────────────────────────────────────────────────

def model_fn_wan_robot_5b(
    dit,
    # --- unused pipeline models (kept for DiffSynth compatibility) ---
    motion_controller=None,
    vace=None,
    animate_adapter=None,
    vap=None,
    # --- standard WAN inputs ---
    latents=None,
    timestep=None,
    context=None,
    # --- robot-specific inputs ---
    traj_latents=None,
    raymap=None,
    delta_action=None,
    # --- flags ---
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs,
):
    """
    Replaces model_fn_wan_video for 5B robot model.

    Delegates directly to WanRobotModel5B.forward(), which handles
    time-embedding, text-embedding, patchify, separated timestep, and
    transformer blocks internally.
    """
    return dit(
        x=latents,
        timestep=timestep,
        context=context,
        traj_latents=traj_latents,
        raymap=raymap,
        delta_action=delta_action,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )


# ──────────────────────────────────────────────────────────────────────
# Training module
# ──────────────────────────────────────────────────────────────────────

class WanRobotTrainingModule5B(DiffusionTrainingModule):
    """
    DiffusionTrainingModule for WanRobotModel5B.

    Frozen  : VAE38, text_encoder (T5)
    Trainable: pipe.dit  (WanRobotModel5B, full fine-tune)

    Key differences from 1.3B version:
      - No CLIP image encoder (5B doesn't use it)
      - Uses WanVideoVAE38 (48ch, 16x spatial)
      - First frame = clean latent (fuse_vae_embedding_in_latents)
      - No explicit y (mask+vae) construction
    """

    def __init__(
        self,
        dit_weight_path: str,
        model_dir: str,
        use_gradient_checkpointing: bool = True,
        use_gradient_checkpointing_offload: bool = False,
        device: str = 'cpu',
    ):
        super().__init__()

        # ── 1. Load shared encoders via DiffSynth ──────────────────────
        # 5B only needs T5 text encoder + VAE38. No CLIP.
        model_configs = [
            ModelConfig(path=os.path.join(model_dir, 'models_t5_umt5-xxl-enc-bf16.pth')),
            ModelConfig(path=os.path.join(model_dir, 'Wan2.2_VAE.pth')),
        ]
        tokenizer_config = ModelConfig(path=os.path.join(model_dir, 'google', 'umt5-xxl'))

        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            redirect_common_files=False,
        )

        # ── 2. Replace dit with WanRobotModel5B ──────────────────────
        self.pipe.dit = WanRobotModel5B.from_pretrained(dit_weight_path)
        self.pipe.dit = self.pipe.dit.to(device=device, dtype=torch.bfloat16)

        # ── 3. Use robot-specific model function ───────────────────────
        self.pipe.model_fn = model_fn_wan_robot_5b

        # ── 4. Freeze encoders; keep only dit trainable ────────────────
        self.pipe.freeze_except(["dit"])

        # ── 5. Noise scheduler for training ───────────────────────────
        self.pipe.scheduler.set_timesteps(1000, training=True)

        self.use_gradient_checkpointing         = True
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload

    # ------------------------------------------------------------------
    # forward  (called once per step by launch_training_task)
    # ------------------------------------------------------------------

    def forward(self, data):
        """
        data: single sample dict from AgiBotWorldWanDataset
            video        (3, T, H, W)  target frames in [-1,1]
            first_frame  (3, H, W)     first frame in [-1,1]
            traj_maps    (3, T, H, W)  traj visualisation in [-1,1]
            raymap       (6, T, H, W)  ray dir + origin
            delta_action (T, 14)       per-frame delta actions
            prompt       str
        """
        device = self.pipe.device
        dtype  = self.pipe.torch_dtype

        # add batch dim (launch_training_task uses collate_fn = x[0])
        video        = data['video'].unsqueeze(0).to(device=device, dtype=dtype)
        traj_maps    = data['traj_maps'].unsqueeze(0).to(device=device, dtype=dtype)
        raymap       = data['raymap'].unsqueeze(0).to(device=device, dtype=dtype)
        delta_action = data['delta_action'].unsqueeze(0).to(device=device, dtype=dtype)

        B, C, T, H, W = video.shape

        # ── encode with frozen networks ──────────────────────────────
        with torch.no_grad():

            # target video → latents (VAE38: 48ch, 16x spatial)
            video_latents = encode_video(self.pipe.vae, video, device, dtype)
            T_lat = video_latents.shape[2]
            H_lat = video_latents.shape[3]
            W_lat = video_latents.shape[4]

            # first_frame_latents: frame 0 clean latent for fuse_vae_embedding_in_latents
            first_frame_latents = video_latents[:, :, 0:1]  # (B, 48, 1, H_lat, W_lat)

            # traj maps → latents (same VAE38)
            traj_latents = encode_video(self.pipe.vae, traj_maps, device, dtype)

            # text → T5 embeddings
            text_inputs = self.pipe.tokenizer([data['prompt']])
            context     = self.pipe.text_encoder(text_inputs.to(device))

        # ── prepare raymap at latent resolution ──────────────────────
        # raymap: (1, 6, T, H, W) → subsample T → T_lat, then resize H,W → H_lat, W_lat
        raymap_lat = subsample_to_lat(raymap, T_lat)              # (1, 6, T_lat, H, W)
        raymap_lat = raymap_lat.permute(0, 2, 1, 3, 4)           # (1, T_lat, 6, H, W)
        raymap_lat = raymap_lat.flatten(0, 1)                     # (T_lat, 6, H, W)
        raymap_lat = F.interpolate(
            raymap_lat.float(), size=(H_lat, W_lat),
            mode='bilinear', align_corners=False,
        ).to(dtype)
        raymap_lat = (
            raymap_lat
            .view(B, T_lat, 6, H_lat, W_lat)
            .permute(0, 2, 1, 3, 4)                               # (1, 6, T_lat, H_lat, W_lat)
        )

        # delta_action: (1, T, 14) → feed directly to Resampler
        delta_action_lat = delta_action  # (1, T, 14)

        # ── FlowMatchSFTLoss (DiffSynth standard) ─────────────────────
        # first_frame_latents triggers: frame 0 in noisy latents replaced
        # with clean encoding, and frame 0 excluded from loss computation
        inputs = {
            "input_latents":       video_latents,        # (1, 48, T_lat, H_lat, W_lat)
            "first_frame_latents": first_frame_latents,  # (1, 48, 1, H_lat, W_lat)
            "context":             context,               # (1, L_text, 4096)
            "traj_latents":        traj_latents,          # (1, 48, T_lat, H_lat, W_lat)
            "raymap":              raymap_lat,            # (1, 6, T_lat, H_lat, W_lat)
            "delta_action":        delta_action_lat,      # (1, T, 14)
            "use_gradient_checkpointing":         self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "max_timestep_boundary": 1.0,
            "min_timestep_boundary": 0.0,
        }
        return FlowMatchSFTLoss(self.pipe, **inputs)


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train WanRobotModel5B on AgiBotWorld dataset"
    )
    parser.add_argument("--dit_weight_path", required=True,
                        help="Path to 5B diffusion_pytorch_model*.safetensors (file, glob, or dir)")
    parser.add_argument("--model_dir",       required=True,
                        help="Directory with T5 / VAE38 / tokenizer weights")
    parser.add_argument("--data_root",       required=True,
                        help="AgiBotWorld dataset root (contains train/ subdir)")
    parser.add_argument("--output_path",     default="./checkpoints/wan_robot_5b",
                        help="Where to save checkpoints")
    parser.add_argument("--height",          type=int,   default=480)
    parser.add_argument("--width",           type=int,   default=640)

    # launch_training_task reads these from args
    parser.add_argument("--num_epochs",              type=int,   default=10)
    parser.add_argument("--learning_rate",           type=float, default=1e-5)
    parser.add_argument("--weight_decay",            type=float, default=1e-2)
    parser.add_argument("--save_steps",              type=int,   default=5000)
    parser.add_argument("--dataset_num_workers",     type=int,   default=4)

    # accelerate settings
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--find_unused_parameters",
                        action="store_true", default=True)
    args = parser.parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=args.find_unused_parameters
            )
        ],
    )

    dataset = AgiBotWorldWanDataset(
        data_root=args.data_root,
        split='train',
        sample_size=(args.height, args.width),
    )

    # Models are loaded on CPU; launch_training_task moves to accelerator.device
    model = WanRobotTrainingModule5B(
        dit_weight_path=args.dit_weight_path,
        model_dir=args.model_dir,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=True,
        device='cpu',
    )

    # ModelLogger saves only trainable (dit) params.
    # remove_prefix_in_ckpt strips "pipe.dit." so inference loads directly.
    model_logger = ModelLogger(
        output_path=args.output_path,
        remove_prefix_in_ckpt="pipe.dit.",
    )

    launch_training_task(accelerator, dataset, model, model_logger, args=args)
