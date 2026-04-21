"""
GRPO Training Script for AgiBotWorld FunControl.

Self-contained script for Group Relative Policy Optimization on the
Wan2.1-Fun-V1.1-1.3B-Control model. Uses the diffsynth framework.

Usage:
    torchrun --nproc_per_node=16 grpo_main.py \
        --model_dir /root/pretrained_weights/PAI/Wan2.1-Fun-V1.1-1.3B-Control \
        --sft_checkpoint /path/to/step-24000.safetensors \
        --data_root /root/data/agibot_world_beta_processed_main \
        --output_dir ./grpo_output
"""

import os
import math
import time
import argparse
import warnings
from collections import deque

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from diffusers.optimization import get_scheduler

# NPU support
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    HAS_NPU = True
except ImportError:
    HAS_NPU = False

# diffsynth imports
from diffsynth.core import ModelConfig
from diffsynth.pipelines.wan_video import WanVideoPipeline, model_fn_wan_video

# Local imports
from dataset_agibot_grpo import AgiBotGRPODataset, grpo_collate_fn
from torch.utils.data.distributed import DistributedSampler
from reward_wrapper import GRPORewardComputer


# ── Utility ──────────────────────────────────────────────────────────

def get_device_type():
    if HAS_NPU:
        return "npu"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def empty_cache():
    if HAS_NPU:
        torch.npu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def autocast_device():
    return "npu" if HAS_NPU else "cuda"


def main_print(*args, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if HAS_NPU:
        torch.npu.manual_seed_all(seed)
    elif torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── SDE Step with Log Probability ─────────────────────────────────────

def sd3_time_shift(shift, t):
    """Apply time shift to sigma schedule (same as Wan's default shift=5)."""
    return (shift * t) / (1 + (shift - 1) * t)


def sde_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    sigmas: torch.Tensor,
    index: int,
    prev_sample: torch.Tensor = None,
):
    """
    SDE step with log probability computation for GRPO.

    Same math as DanceGRPO's flux_step() and teammate's wan_step():
      - Euler ODE step: prev_sample_mean = latents + dsigma * model_output
      - SDE correction: add score-based log term
      - Log probability: Gaussian log-prob of prev_sample given prev_sample_mean

    Args:
        model_output: DIT output (velocity prediction)
        latents: current noisy latents
        eta: noise coefficient for SDE solver
        sigmas: full sigma schedule [num_steps + 1]
        index: current step index
        prev_sample: if provided, use this as the sample (for training);
                     if None, sample fresh noise (for reference model sampling)

    Returns:
        prev_sample: denoised sample at next timestep
        pred_original: predicted clean sample
        log_prob: log probability of the sample (per-batch)
    """
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma
    prev_sample_mean = latents + dsigma * model_output

    pred_original_sample = latents - sigma * model_output

    delta_t = sigma - sigmas[index + 1]
    delta_t_val = max(delta_t.item() if isinstance(delta_t, torch.Tensor) else float(delta_t), 1e-10)
    std_dev_t = eta * math.sqrt(delta_t_val)
    std_dev_t = max(std_dev_t, 1e-8)  # prevent log(0) and division by zero

    # SDE correction term
    score_estimate = -(latents - pred_original_sample * (1 - sigma)) / sigma**2
    log_term = -0.5 * eta**2 * score_estimate
    prev_sample_mean = prev_sample_mean + log_term * dsigma

    # Sample if no prev_sample provided (reference model sampling)
    if prev_sample is None:
        prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t

    # Log probability
    log_prob = (
        -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2)
        / (2 * (std_dev_t ** 2))
        - math.log(std_dev_t)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample, pred_original_sample, log_prob


# ── FunControl Model Forward ──────────────────────────────────────────

def dit_forward(dit, latents, timestep, context, clip_feature, y,
                reference_latents, cfg_scale=1.0, negative_context=None,
                use_gradient_checkpointing=False):
    """
    Run the FunControl DIT forward pass with optional CFG.

    Uses model_fn_wan_video from diffsynth.
    FunControl requires fuse_vae_embedding_in_latents=False.
    """
    common_kwargs = dict(
        dit=dit, latents=latents, timestep=timestep,
        clip_feature=clip_feature, y=y,
        reference_latents=reference_latents,
        fuse_vae_embedding_in_latents=False,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )

    if cfg_scale > 1.0 and negative_context is not None:
        # CFG: run both conditional and unconditional
        noise_pred_posi = model_fn_wan_video(context=context, **common_kwargs)
        noise_pred_nega = model_fn_wan_video(context=negative_context, **common_kwargs)
        noise_pred = noise_pred_nega.to(torch.float32) + cfg_scale * (
            noise_pred_posi.to(torch.float32) - noise_pred_nega.to(torch.float32))
        return noise_pred
    else:
        return model_fn_wan_video(context=context, **common_kwargs)


# ── Conditioning Preparation ──────────────────────────────────────────

@torch.no_grad()
def prepare_conditioning(pipe, sample, device):
    """
    Prepare all conditioning tensors from a dataset sample using the
    diffsynth pipeline's unit processing.

    Args:
        pipe: WanVideoPipeline instance
        sample: dict from AgiBotGRPODataset.__getitem__()
        device: target device

    Returns:
        dict with: context, negative_context, clip_feature, y,
                   reference_latents, control_latents
    """
    prompt = sample["prompt"]
    input_image = sample["input_image"]          # GT first frame → y tensor (WanVideoUnit_ImageEmbedderVAE)
    reference_image = sample["reference_image"]  # Reference frame → CLIP + ref_conv (WanVideoUnit_FunReference)
    control_video = sample["control_video"]
    negative_prompt = "blurry, low resolution, grainy, noisy, pixelated, compression artifacts, distorted, unnatural, low quality"

    # ── Text encoding (tokenizer + text_encoder forward) ──
    # Text encoder and image encoder may have been offloaded to CPU after the
    # previous step. Move them back to device for encoding.
    if pipe.text_encoder is not None:
        pipe.text_encoder.to(device)
    if pipe.image_encoder is not None:
        pipe.image_encoder.to(device)

    # Encode positive prompt
    ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
    ids = ids.to(pipe.device)
    mask = mask.to(pipe.device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    context = pipe.text_encoder(ids, mask)
    for i, v in enumerate(seq_lens):
        context[:, v:] = 0
    context = context.to(device=device, dtype=pipe.torch_dtype)

    # Encode negative prompt
    neg_ids, neg_mask = pipe.tokenizer(negative_prompt, return_mask=True, add_special_tokens=True)
    neg_ids = neg_ids.to(pipe.device)
    neg_mask = neg_mask.to(pipe.device)
    neg_seq_lens = neg_mask.gt(0).sum(dim=1).long()
    negative_context = pipe.text_encoder(neg_ids, neg_mask)
    for i, v in enumerate(neg_seq_lens):
        negative_context[:, v:] = 0
    negative_context = negative_context.to(device=device, dtype=pipe.torch_dtype)

    # ── Reference image encoding ──
    # VAE encode reference image → reference_latents
    ref_video = pipe.preprocess_video([reference_image])
    reference_latents = pipe.vae.encode(ref_video, device=device)
    reference_latents = reference_latents.to(device=device, dtype=pipe.torch_dtype)

    # CLIP encode reference image → clip_feature
    clip_feature = None
    if pipe.image_encoder is not None:
        clip_input = pipe.preprocess_image(reference_image)
        clip_feature = pipe.image_encoder.encode_image([clip_input])
        clip_feature = clip_feature.to(device=device, dtype=pipe.torch_dtype)

    # ── Control video encoding ──
    control_video_tensor = pipe.preprocess_video(control_video)
    control_latents = pipe.vae.encode(
        control_video_tensor, device=device,
        tiled=True, tile_size=(30, 52), tile_stride=(15, 26)
    )
    control_latents = control_latents.to(device=device, dtype=pipe.torch_dtype)

    # ── Build y tensor ──
    # Replicate the logic from WanVideoUnit_ImageEmbedderVAE + WanVideoUnit_FunControl.
    #
    # WanVideoUnit_ImageEmbedderVAE encodes input_image (GT first frame) as:
    #   vae_input = [input_image_3ch, zeros_for_remaining_frames]  → VAE encode → 16ch
    #   mask = [1, 0, 0, ..., 0] reshaped → 4ch
    #   y_image = concat(mask_4ch, vae_16ch) → 20ch [1, 20, T, H, W]
    #
    # WanVideoUnit_FunControl then:
    #   y_dim = in_dim(48) - control_ch(16) - noise_ch(16) = 16
    #   y_remaining = y_image[:, -16:]  (last 16ch from the 20ch = the VAE part)
    #   y = concat(control_latents_16ch, y_remaining_16ch) → 32ch
    dit = pipe.dit
    latent_channels = 16
    y_dim = dit.in_dim - control_latents.shape[1] - latent_channels
    if clip_feature is None:
        clip_feature = torch.zeros((1, 257, 1280), dtype=pipe.torch_dtype, device=device)
    num_frames = control_latents.shape[2]
    height_latent = control_latents.shape[3]
    width_latent = control_latents.shape[4]

    if y_dim > 0 and dit.require_vae_embedding:
        # Replicate WanVideoUnit_ImageEmbedderVAE:
        # Build a video tensor with input_image as the first frame, rest zeros.
        # input_image is the GT first frame (same as SFT training's input_image).
        H_pixel = height_latent * 8  # pixel height from latent dims
        W_pixel = width_latent * 8   # pixel width from latent dims
        img_preprocessed = pipe.preprocess_image(
            input_image.resize((W_pixel, H_pixel))
        ).to(pipe.device)  # [1, 3, H, W]
        # Build video: input_image as first frame, zeros for rest
        num_frames_pixel = len(control_video)
        vae_input = torch.cat([
            img_preprocessed.transpose(0, 1),  # [3, 1, H, W]
            torch.zeros(3, num_frames_pixel - 1, H_pixel, W_pixel,
                        device=pipe.device, dtype=pipe.torch_dtype)
        ], dim=1)  # [3, T, H, W]

        # Build mask: 1 for first frame, 0 for rest, then reshape to latent temporal dim
        msk = torch.ones(1, num_frames_pixel, height_latent, width_latent,
                         device=device, dtype=pipe.torch_dtype)
        msk[:, 1:] = 0
        # Temporal compression: first frame repeated 4x, then rest
        msk = torch.cat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
            msk[:, 1:]
        ], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height_latent, width_latent)
        msk = msk.transpose(1, 2)[0]  # [4, T_latent, H_latent, W_latent]

        # VAE encode the image-padded video
        y_vae = pipe.vae.encode(
            [vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)],
            device=device, tiled=True, tile_size=(30, 52), tile_stride=(15, 26)
        )[0]  # [16, T_latent, H_latent, W_latent]
        y_vae = y_vae.to(dtype=pipe.torch_dtype, device=device)

        # y_image = concat(mask_4ch, vae_16ch) → 20ch
        y_image = torch.cat([msk, y_vae], dim=0).unsqueeze(0)  # [1, 20, T, H, W]

        # FunControl takes last y_dim channels from y_image
        y_remaining = y_image[:, -y_dim:]  # [1, 16, T, H, W]
        y = torch.cat([control_latents, y_remaining], dim=1)  # [1, 32, T, H, W]
    elif y_dim > 0:
        y_padding = torch.zeros(
            (1, y_dim, num_frames, height_latent, width_latent),
            dtype=pipe.torch_dtype, device=device)
        y = torch.cat([control_latents, y_padding], dim=1)
    else:
        y = control_latents

    empty_cache()

    return {
        "context": context,
        "negative_context": negative_context,
        "clip_feature": clip_feature,
        "y": y,
        "reference_latents": reference_latents,
        "control_latents": control_latents,
        "num_frames_latent": num_frames,
        "height_latent": height_latent,
        "width_latent": width_latent,
    }


# ── Sampling (Reference Model) ───────────────────────────────────────

@torch.no_grad()
def sample_reference_model(args, dit, cond, device, sigma_schedule):
    """
    Generate a video using the current model (no gradients).
    Collects latents and log_probs at each timestep for GRPO training.

    Returns:
        final_latents: predicted clean latents (for VAE decoding)
        all_latents: [1, num_steps+1, C, T, H, W] — latent trajectory
        all_log_probs: [1, num_steps] — log probabilities
    """
    num_steps = args.sampling_steps
    C = 16
    T = cond["num_frames_latent"]
    H = cond["height_latent"]
    W = cond["width_latent"]

    # Initialize noise(init_same_noise)
    z = torch.randn((1, C, T, H, W), device=device, dtype=torch.bfloat16)

    all_latents = [z]
    all_log_probs = []

    dit.eval()
    for i in range(num_steps):
        sigma = sigma_schedule[i]
        timestep_value = sigma * 1000.0
        timestep = torch.tensor([timestep_value], device=device, dtype=args._torch_dtype)

        with torch.autocast(autocast_device(), torch.bfloat16):
            pred = dit_forward(
                dit=dit,
                latents=z,
                timestep=timestep,
                context=cond["context"],
                clip_feature=cond["clip_feature"],
                y=cond["y"],
                reference_latents=cond["reference_latents"],
                cfg_scale=args.cfg_scale,
                negative_context=cond["negative_context"],
            )

        z_new, pred_original, log_prob = sde_step(
            pred, z.to(torch.float32), args.eta,
            sigmas=sigma_schedule, index=i, prev_sample=None,
        )
        z = z_new.to(torch.bfloat16)
        all_latents.append(z)
        all_log_probs.append(log_prob)

    all_latents = torch.stack(all_latents, dim=1)  # [1, steps+1, C, T, H, W]
    all_log_probs = torch.stack(all_log_probs, dim=1)  # [1, steps]

    return pred_original, all_latents, all_log_probs


# ── VAE Decoding ──────────────────────────────────────────────────────

@torch.no_grad()
def decode_latents_to_frames(pipe, latents, device):
    """
    Decode latents to video frames as numpy arrays.

    Args:
        pipe: WanVideoPipeline
        latents: [1, C, T, H, W] predicted clean latents

    Returns:
        frames: list of np.ndarray [H, W, 3] uint8
    """
    latents = latents.to(device=device, dtype=torch.bfloat16)

    # VAE decode
    video_tensor = pipe.vae.decode(
        latents, device=device,
        tiled=True, tile_size=(30, 52), tile_stride=(15, 26)
    )
    # video_tensor shape: [1, 3, T, H, W] in [-1, 1]
    video_tensor = video_tensor.float().cpu()
    video_tensor = (video_tensor.clamp(-1, 1) + 1) / 2 * 255
    video_tensor = video_tensor[0].permute(1, 2, 3, 0).numpy().astype(np.uint8)  # [T, H, W, 3]

    frames = [video_tensor[i] for i in range(video_tensor.shape[0])]
    return frames


# ── GRPO Training Step ─────────────────────────────────────────────────

def grpo_one_step(args, dit, latents, pre_latents, cond, timestep, index,
                  sigma_schedule):
    """
    Re-compute log probability at a specific timestep during training.

    Args:
        latents: noisy latents at timestep index
        pre_latents: target latents at timestep index+1 (from reference trajectory)
    """
    dit.train()
    sigma = sigma_schedule[index]
    timestep_val = sigma * 1000.0
    timestep_tensor = torch.tensor([timestep_val], device=latents.device,
                                    dtype=args._torch_dtype)

    with torch.autocast(autocast_device(), torch.bfloat16):
        pred = dit_forward(
            dit=dit,
            latents=latents,
            timestep=timestep_tensor,
            context=cond["context"],
            clip_feature=cond["clip_feature"],
            y=cond["y"],
            reference_latents=cond["reference_latents"],
            cfg_scale=args.cfg_scale,
            negative_context=cond["negative_context"],
            use_gradient_checkpointing=args.gradient_checkpointing,
        )

    _, _, log_prob = sde_step(
        pred, latents.to(torch.float32), args.eta,
        sigma_schedule, index,
        prev_sample=pre_latents.to(torch.float32),
    )
    return log_prob


def train_one_step(args, device, dit, pipe, cond, gt_data, reward_computer,
                   optimizer, lr_scheduler, sigma_schedule):
    """
    Complete GRPO training step (DanceGRPO style):
      1. Sample K videos from reference model (no grad)
      2. Compute EWM rewards
      3. Compute advantages within local K videos (no cross-rank gather)
      4. PPO clipped loss over shuffled timesteps
      5. Update weights

    Each GPU processes a different prompt. Advantage is computed within
    the K videos generated on this GPU for this prompt.

    Args:
        cond: conditioning dict from prepare_conditioning()
        gt_data: dict with gt_frames and gt_proprio
        reward_computer: GRPORewardComputer instance

    Returns:
        total_loss: float
        mean_reward: float
        grad_norm: float
    """
    total_loss = 0.0
    optimizer.zero_grad()
    rank0 = (not dist.is_initialized()) or dist.get_rank() == 0

    num_generations = args.num_generations

    # ── 1. Sample K videos ──
    all_latents_list = []
    all_log_probs_list = []
    all_gen_frames = []
    sampling_t0 = time.time()
    if rank0:
        print(f"  [timing] sampling start: num_generations={num_generations}, "
              f"sampling_steps={args.sampling_steps}")

    for k in range(num_generations):
        generation_t0 = time.time()
        pred_original, latent_traj, log_probs = sample_reference_model(
            args, dit, cond, device, sigma_schedule)

        all_latents_list.append(latent_traj)
        all_log_probs_list.append(log_probs)

        # Decode to frames for reward
        gen_frames = decode_latents_to_frames(pipe, pred_original, device)
        all_gen_frames.append(gen_frames)
        if rank0:
            print(f"  [timing] generation {k + 1}/{num_generations}: "
                  f"{time.time() - generation_t0:.1f}s")
    if rank0:
        print(f"  [timing] sampling total: {time.time() - sampling_t0:.1f}s")

    # ── 2. Compute rewards ──
    reward_t0 = time.time()
    gt_data_list = [gt_data] * num_generations
    rewards_tensor, reward_metadata = reward_computer.compute_batch(
        all_gen_frames, gt_data_list)
    rewards_tensor = rewards_tensor.to(device)
    if rank0:
        print(f"  [timing] reward total: {time.time() - reward_t0:.1f}s")

    # ── 3. Compute advantages (local, within this GPU's K videos) ──
    mean_reward = rewards_tensor.mean().item()
    group_mean = rewards_tensor.mean()
    group_std = rewards_tensor.std() + 1e-8
    advantages = (rewards_tensor - group_mean) / group_std

    # Log rewards (all_reduce for monitoring only, not used in training)
    if dist.is_initialized():
        reward_for_log = rewards_tensor.mean().clone()
        dist.all_reduce(reward_for_log, op=dist.ReduceOp.SUM)
        global_mean_reward = (reward_for_log / dist.get_world_size()).item()
    else:
        global_mean_reward = mean_reward

    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"  local rewards: {rewards_tensor.tolist()}, "
              f"global mean: {global_mean_reward:.4f}")
        meta = reward_metadata[0] if reward_metadata else {}
        print(f"  psnr: {meta.get('psnr', 'N/A')}, ndtw: {meta.get('ndtw', 'N/A')}, "
              f"sc: {meta.get('scene_consistency', 'N/A')}")

    # ── 4. PPO Loss ──
    backward_t0 = time.time()
    if rank0:
        print(f"  [timing] backward start: timestep_fraction={args.timestep_fraction}")
    for k in range(num_generations):
        latent_traj = all_latents_list[k]  # [1, steps+1, C, T, H, W]
        old_log_probs = all_log_probs_list[k]  # [1, steps]
        adv = advantages[k]

        num_steps = old_log_probs.shape[1]
        train_timesteps = int(num_steps * args.timestep_fraction)

        # Shuffle timestep order
        perm = torch.randperm(num_steps, device=device)

        for t_idx in range(train_timesteps):
            step_i = perm[t_idx].item()

            # Get latents before and after this timestep
            z_t = latent_traj[:, step_i].detach()      # latent at step i
            z_next = latent_traj[:, step_i + 1].detach()  # latent at step i+1

            # Recompute log prob with current model
            new_log_prob = grpo_one_step(
                args, dit, z_t, z_next, cond,
                None, step_i, sigma_schedule,
            )

            # PPO clipped loss
            old_lp = old_log_probs[:, step_i].detach()
            ratio = torch.exp(new_log_prob - old_lp)

            adv_clamped = torch.clamp(adv, -args.adv_clip_max, args.adv_clip_max)
            unclipped = -adv_clamped * ratio
            clipped = -adv_clamped * torch.clamp(
                ratio, 1.0 - args.clip_range, 1.0 + args.clip_range)

            loss = torch.mean(torch.maximum(unclipped, clipped)) / (
                args.gradient_accumulation_steps * num_generations * train_timesteps)
            loss.backward()

            if dist.is_initialized():
                avg_loss = loss.detach().clone()
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                total_loss += (avg_loss / dist.get_world_size()).item()
            else:
                total_loss += loss.item()
    if rank0:
        print(f"  [timing] backward total: {time.time() - backward_t0:.1f}s")

    # ── 5. Gradient sync + Optimizer step ──
    # All-reduce gradients across GPUs (critical for distributed training).
    # Each GPU processed a different prompt, so gradients must be averaged
    # before updating weights to keep all GPUs in sync.
    opt_t0 = time.time()
    if dist.is_initialized():
        world_size = dist.get_world_size()
        for param in dit.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(world_size)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        dit.parameters(), max_norm=args.max_grad_norm)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    if rank0:
        print(f"  [timing] optimizer+sync total: {time.time() - opt_t0:.1f}s")

    return total_loss, global_mean_reward, grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm


# ── Model Loading ─────────────────────────────────────────────────────

def load_pipeline(args, device):
    """Load the diffsynth WanVideoPipeline and apply SFT checkpoint."""
    model_dir = args.model_dir

    # Build model configs from individual weight files (same as SFT training)
    model_paths = [
        os.path.join(model_dir, "diffusion_pytorch_model.safetensors"),
        os.path.join(model_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
        os.path.join(model_dir, "Wan2.1_VAE.pth"),
        os.path.join(model_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
    ]
    model_configs = [ModelConfig(path=p) for p in model_paths if os.path.exists(p)]

    # Tokenizer config (umt5-xxl, inside model_dir/google/umt5-xxl/)
    tokenizer_path = os.path.join(model_dir, "google/umt5-xxl")
    if os.path.isdir(tokenizer_path):
        tokenizer_config = ModelConfig(path=tokenizer_path)
    else:
        tokenizer_config = ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="google/umt5-xxl/",
        )

    main_print(f"Loading pipeline from {model_dir} ...")
    main_print(f"  Model files: {[os.path.basename(p) for p in model_paths]}")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )

    # Apply SFT checkpoint (overwrites DIT weights)
    if args.sft_checkpoint and os.path.exists(args.sft_checkpoint):
        main_print(f"Loading SFT checkpoint from {args.sft_checkpoint} ...")
        from safetensors.torch import load_file
        sft_state_dict = load_file(args.sft_checkpoint)
        pipe.dit.load_state_dict(sft_state_dict, strict=False)
        main_print(f"  Loaded {len(sft_state_dict)} tensors from SFT checkpoint")

    # Freeze non-DIT models
    pipe.vae.requires_grad_(False)
    if pipe.text_encoder is not None:
        pipe.text_encoder.requires_grad_(False)
    if pipe.image_encoder is not None:
        pipe.image_encoder.requires_grad_(False)

    # With ModelConfig(path=...) defaults (no offload settings), VRAM management
    # is not enabled by the loader. Explicitly disable as a safety measure in case
    # the loader behavior changes in the future — we need DIT to stay on device.
    pipe.vram_management_enabled = False
    for name, model in pipe.named_children():
        if hasattr(model, "vram_management_enabled"):
            model.vram_management_enabled = False

    return pipe


# ── Main ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="GRPO Training for AgiBotWorld FunControl")

    # Model
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to pretrained Wan2.1-Fun-V1.1-1.3B-Control")
    parser.add_argument("--sft_checkpoint", type=str, default=None,
                        help="Path to SFT checkpoint (.safetensors)")

    # Data
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to agibot_world_beta_processed_main")
    parser.add_argument("--split_file", type=str, default=None,
                        help="Optional split JSON file")
    parser.add_argument("--num_workers", type=int, default=4)

    # Video dimensions
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=49)

    # Sampling
    parser.add_argument("--sampling_steps", type=int, default=None)
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--eta", type=float, default=1.0,
                        help="SDE noise coefficient")
    parser.add_argument("--shift", type=float, default=5.0,
                        help="Sigma schedule shift")

    # GRPO
    parser.add_argument("--num_generations", type=int, default=4,
                        help="Number of videos per prompt for advantage computation")
    parser.add_argument("--clip_range", type=float, default=1e-4,
                        help="PPO clip range")
    parser.add_argument("--adv_clip_max", type=float, default=5.0,
                        help="Advantage clipping maximum")
    parser.add_argument("--timestep_fraction", type=float, default=1.0,
                        help="Fraction of timesteps to train on")

    # Training
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=2.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--max_train_steps", type=int, default=10000)
    # lr_scheduler
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=10,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of cycles in the learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )

    # Checkpointing & Logging
    parser.add_argument("--output_dir", type=str, default="./grpo_output")
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true")

    # Reward
    parser.add_argument("--reward_dir", type=str, default=None,
                        help="Path to reward model checkpoints")
    parser.add_argument("--reward_mode", type=str, default="psnr_only",
                        choices=["full", "psnr_only", "psnr_sc"],
                        help="Reward computation mode")
    parser.add_argument("--reward_device", type=str, default="cpu",
                        help="Device for reward models (cpu, npu:15, or auto to follow local_rank)")

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Distributed setup ──
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        backend = "hccl" if HAS_NPU else "nccl"
        dist.init_process_group(backend)
        if HAS_NPU:
            torch.npu.set_device(local_rank)
            device = torch.device(f"npu:{local_rank}")
        else:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
    else:
        device_type = get_device_type()
        device = torch.device(f"{device_type}:0" if device_type != "cpu" else "cpu")

    if args.seed is not None:
        set_seed(args.seed + rank)

    if rank == 0 and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    main_print("=" * 60)
    main_print("GRPO Training for AgiBotWorld FunControl")
    main_print("=" * 60)

    # ── Load pipeline ──
    pipe = load_pipeline(args, device)
    dit = pipe.dit
    dit.to(device)

    # Store torch_dtype for timestep creation in sampling/training functions
    args._torch_dtype = pipe.torch_dtype

    # ── Optimizer ──
    params_to_optimize = list(filter(lambda p: p.requires_grad, dit.parameters()))
    num_trainable = sum(p.numel() for p in params_to_optimize)
    main_print(f"Trainable parameters: {num_trainable / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    # from torch.optim.lr_scheduler import LambdaLR
    # def warmup_lambda(step):
    #     if step < args.lr_warmup_steps:
    #         return float(step) / float(max(1, args.lr_warmup_steps))
    #     return 1.0
    # lr_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    init_steps = 0
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=1000000,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=init_steps - 1,
    )


    # ── Dataset ──
    dataset = AgiBotGRPODataset(
        data_root=args.data_root,
        sample_size=(args.height, args.width),
        target_frames=args.num_frames,
        split_file=args.split_file,
    )
    main_print(f"Dataset: {len(dataset)} clips")

    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank,
            shuffle=True, seed=args.seed,
        )
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=grpo_collate_fn,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
    )

    # ── Reward model ──
    reward_device = args.reward_device
    if reward_device == "auto":
        reward_device = str(device)
    reward_computer = GRPORewardComputer(
        reward_dir=args.reward_dir,
        device=reward_device,
        mode=args.reward_mode,
    )
    main_print(f"Reward mode: {args.reward_mode}, reward device: {reward_device}")

    # ── Sigma schedule ──
    sigma_schedule = torch.linspace(1, 0, args.sampling_steps + 1)
    sigma_schedule = sd3_time_shift(args.shift, sigma_schedule)
    sigma_schedule = sigma_schedule.to(device)

    # ── WandB ──
    if args.use_wandb and rank == 0:
        import wandb
        wandb.init(project="agibot_grpo", config=vars(args))

    # ── Training loop ──
    main_print(f"\nStarting training: {args.max_train_steps} steps")
    main_print(f"  Sampling steps: {args.sampling_steps}")
    main_print(f"  Num generations per prompt: {args.num_generations}")
    main_print(f"  CFG scale: {args.cfg_scale}")
    main_print(f"  Learning rate: {args.learning_rate}")
    main_print(f"  Clip range: {args.clip_range}")

    step_times = deque(maxlen=50)
    global_step = 0

    for epoch in range(100000):  # Effectively infinite, controlled by max_train_steps
        if sampler is not None:
            sampler.set_epoch(epoch)

        for sample in dataloader:
            if global_step >= args.max_train_steps:
                break

            start_time = time.time()

            # ── Checkpoint ──
            if global_step > 0 and global_step % args.checkpointing_steps == 0:
                if rank == 0:
                    save_dir = os.path.join(args.output_dir,
                                            f"checkpoint-{global_step}")
                    os.makedirs(save_dir, exist_ok=True)
                    from safetensors.torch import save_file
                    state_dict = dit.state_dict()
                    save_file(state_dict, os.path.join(save_dir, "dit.safetensors"))
                    main_print(f"  Checkpoint saved: {save_dir}")
                if dist.is_initialized():
                    dist.barrier()

            # ── Prepare conditioning ──
            cond_t0 = time.time()
            if rank == 0:
                print("  [timing] conditioning start")
            cond = prepare_conditioning(pipe, sample, device)
            if rank == 0:
                print(f"  [timing] conditioning total: {time.time() - cond_t0:.1f}s")

            # Offload text_encoder and image_encoder to CPU to free device memory
            # for the memory-intensive sampling/training phase.
            # They are only needed during prepare_conditioning().
            if pipe.text_encoder is not None:
                pipe.text_encoder.to("cpu")
            if pipe.image_encoder is not None:
                pipe.image_encoder.to("cpu")
            empty_cache()

            # ── GT data for reward ──
            gt_data = {
                "gt_frames": sample["gt_frames"],
                "gt_proprio": sample.get("gt_proprio", None),
            }

            # ── Train one step ──
            loss, mean_reward, grad_norm = train_one_step(
                args, device, dit, pipe, cond, gt_data, reward_computer,
                optimizer, lr_scheduler, sigma_schedule,
            )

            step_time = time.time() - start_time
            step_times.append(step_time)

            if rank == 0:
                print(f"Step {global_step}: loss={loss:.4f}, "
                      f"reward={mean_reward:.4f}, grad_norm={grad_norm:.4f}, "
                      f"time={step_time:.1f}s")

                if args.use_wandb:
                    import wandb
                    wandb.log({
                        "loss": loss,
                        "reward": mean_reward,
                        "grad_norm": grad_norm,
                        "step_time": step_time,
                        "lr": lr_scheduler.get_last_lr()[0],
                    }, step=global_step)

            global_step += 1
            empty_cache()

        if global_step >= args.max_train_steps:
            break

    # ── Final save ──
    if rank == 0:
        save_dir = os.path.join(args.output_dir, f"checkpoint-final-{global_step}")
        os.makedirs(save_dir, exist_ok=True)
        from safetensors.torch import save_file
        save_file(dit.state_dict(), os.path.join(save_dir, "dit.safetensors"))
        main_print(f"Final checkpoint saved: {save_dir}")

    if dist.is_initialized():
        dist.destroy_process_group()

    main_print("Training complete!")


if __name__ == "__main__":
    main()
