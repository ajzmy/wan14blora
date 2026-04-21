import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os, argparse, accelerate, warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import math
import numpy as np
import time
from collections import deque
from tqdm.auto import tqdm
from PIL import Image
from einops import repeat, reduce
from diffusers.optimization import get_scheduler

from diffsynth.models.wan_video_dit import WanModel
from diffsynth.models.wan_video_vae import WanVideoVAE
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.core import load_state_dict
from diffsynth.utils.data import save_video
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = get_logger(__name__)


def vae_output_to_video(vae_output, pattern="B C T H W", min_value=-1, max_value=1):
    # Transform a torch.Tensor to list of PIL.Image
    if pattern != "T H W C":
        vae_output = reduce(vae_output, f"{pattern} -> T H W C", reduction="mean")

    def vae_output_to_image(vae_output, pattern="B C H W", min_value=-1, max_value=1):
        # Transform a torch.Tensor to PIL.Image
        if pattern != "H W C":
            vae_output = reduce(vae_output, f"{pattern} -> H W C", reduction="mean")
        image = ((vae_output - min_value) * (255 / (max_value - min_value))).clip(0, 255)
        image = image.to(device="cpu", dtype=torch.uint8)
        image = Image.fromarray(image.numpy())
        return image
    
    video = [vae_output_to_image(image, pattern="H W C", min_value=min_value, max_value=max_value) for image in vae_output]
    return video


# wan scheduler step using sde
def wan_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    sigmas: torch.Tensor,
    index: int,
    prev_sample: torch.Tensor,
    grpo: bool=True,
    sde_solver: bool=True,
):
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma
    prev_sample_mean = latents + dsigma * model_output

    pred_original_sample = latents - sigma * model_output

    delta_t = sigma - sigmas[index + 1]
    std_dev_t = eta * math.sqrt(delta_t)

    if sde_solver:
        score_estimate = -(latents-pred_original_sample*(1 - sigma))/sigma**2
        log_term = -0.5 * eta**2 * score_estimate
        prev_sample_mean = prev_sample_mean + log_term * dsigma

    if grpo and prev_sample is None:
        prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t 
        

    if grpo:
        # log prob of prev_sample given prev_sample_mean and std_dev_t
        log_prob = ((
            -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2) / (2 * (std_dev_t**2))
        )
        - math.log(std_dev_t)- torch.log(torch.sqrt(2 * torch.as_tensor(math.pi))))

        # mean along all but batch dimension
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return prev_sample, pred_original_sample, log_prob
    else:
        return prev_sample_mean,pred_original_sample
    
# sample 
def run_sample_step(
        args,
        z,
        first_image_latents,
        control_latents,
        reference_latents,
        clip_feature,
        progress_bar,
        sigma_schedule,
        transformer,
        encoder_hidden_states, 
        negative_prompt_embeds, 
        grpo_sample,
    ):
    if grpo_sample:
        all_latents = [z]
        all_log_probs = []
        for i in progress_bar:  # Add progress bar
            B = encoder_hidden_states.shape[0]
            sigma = sigma_schedule[i]
            timestep_value = int(sigma * 1000)
            timesteps = torch.full([encoder_hidden_states.shape[0]], timestep_value, device=z.device, dtype=torch.long)
            transformer.eval()
            with torch.autocast("npu", torch.bfloat16):
                z = torch.cat([z, control_latents, first_image_latents], dim=1)
                if clip_feature is not None and transformer.require_clip_embedding:
                    clip_embdding = transformer.img_emb(clip_feature)
                    context = torch.cat([clip_embdding, encoder_hidden_states], dim=1)
                    nega_context = torch.cat([clip_embdding, negative_prompt_embeds], dim=1)
                model_pred= transformer(
                    x=z,
                    timestep=timesteps,
                    context=context,
                    reference_latents=reference_latents
                )[0]
                if args.cfg_infer > 1.0:
                    uncond_pred = transformer(
                        x=z,
                        timestep=timesteps,
                        context=nega_context,
                        reference_latents=reference_latents
                    )[0]
                    pred  =  uncond_pred.to(torch.float32) + args.cfg_infer * (model_pred.to(torch.float32) - uncond_pred.to(torch.float32))
                else:
                    pred = model_pred.to(torch.float32)
            
            z, pred_original, log_prob = wan_step(pred, z.to(torch.float32), args.eta, sigmas=sigma_schedule, index=i, prev_sample=None, grpo=True, sde_solver=True)
            z.to(torch.bfloat16)
            all_latents.append(z)
            all_log_probs.append(log_prob)
        latents = pred_original
        all_latents = torch.stack(all_latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
        all_log_probs = torch.stack(all_log_probs, dim=1)  # (batch_size, num_steps, 1)
        return z, latents, all_latents, all_log_probs
    

def sample_reference_model(
    args,
    device, 
    transformer,
    vae,
    first_image_latents,
    control_latents,
    reference_latents,
    clip_feature,
    encoder_hidden_states, 
    negative_prompt_embeds, 
    reward_model,
    tokenizer,
    caption,
):
    w, h, t = args.w, args.h, args.t
    sample_steps = args.sampling_steps
    sigma_schedule = torch.linspace(1, 0, args.sampling_steps + 1)
    # shift
    sigma_schedule = args.shift * sigma_schedule / (1 + (args.shift - 1) * sigma_schedule)

    assert len(sigma_schedule) == sample_steps + 1, "sigma_schedule must have length sample_steps + 1"

    B = encoder_hidden_states.shape[0]
    SPATIAL_DOWNSAMPLE = 8
    TEMPORAL_DOWNSAMPLE = 4
    IN_CHANNELS = 16
    latent_t = ((t - 1) // TEMPORAL_DOWNSAMPLE) + 1
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE

    batch_size = 1  
    batch_indices = torch.chunk(torch.arange(B), B // batch_size)

    all_latents = []
    all_log_probs = []
    all_rewards = []  
    if args.init_same_noise:
        input_latents = torch.randn(
                (1, IN_CHANNELS, latent_t, latent_h, latent_w),  #（c,t,h,w)
                device=device,
                dtype=torch.bfloat16,
            )

    for index, batch_idx in enumerate(batch_indices):
        batch_encoder_hidden_states = encoder_hidden_states[batch_idx]
        batch_negative_prompt_embeds = negative_prompt_embeds[batch_idx]
        batch_caption = [caption[i] for i in batch_idx]
        if not args.init_same_noise:
            input_latents = torch.randn(
                    (1, IN_CHANNELS, latent_t, latent_h, latent_w),  #（c,t,h,w)
                    device=device,
                    dtype=torch.bfloat16,
                )
        grpo_sample=True
        progress_bar = tqdm(range(0, sample_steps), desc="Sampling Progress")
        with torch.no_grad():
            z, latents, batch_latents, batch_log_probs = run_sample_step(
                args,
                input_latents.clone(),
                first_image_latents,
                control_latents,
                reference_latents,
                clip_feature,
                progress_bar,
                sigma_schedule,
                transformer,
                batch_encoder_hidden_states,
                batch_negative_prompt_embeds, 
                grpo_sample,
            )
        all_latents.append(batch_latents)
        all_log_probs.append(batch_log_probs)
        
        # save video for reward
        rank = int(os.environ["RANK"])
        with torch.inference_mode():
            with torch.autocast("npu", dtype=torch.bfloat16):
                video = vae.decode(latents, device=device, tiled=True, tile_size=(30, 52), tile_stride=(15, 26))
                video = vae_output_to_video(video)
        save_video(video, f"./sample_videos_output/wan_control_{rank}_{index}.mp4", fps=5)

    # TODO: compute reward using reward_model

    all_latents = torch.cat(all_latents, dim=0)
    all_log_probs = torch.cat(all_log_probs, dim=0)
    all_rewards = torch.cat(all_rewards, dim=0)
    
    return all_rewards, all_latents, all_log_probs, sigma_schedule


def grpo_one_step(
    args,
    latents,
    pre_latents,
    first_image_latents,
    control_latents,
    reference_latents,
    clip_feature,
    encoder_hidden_states, 
    negative_prompt_embeds, 
    transformer,
    timesteps,
    i,
    sigma_schedule,
):
    B = encoder_hidden_states.shape[0]
    transformer.train()
    with torch.autocast("npu", torch.bfloat16):
        z = torch.cat([z, control_latents, first_image_latents], dim=1)
        if clip_feature is not None and transformer.require_clip_embedding:
            clip_embdding = transformer.img_emb(clip_feature)
            context = torch.cat([clip_embdding, encoder_hidden_states], dim=1)
            nega_context = torch.cat([clip_embdding, negative_prompt_embeds], dim=1)
        model_pred= transformer(
            x=z,
            timestep=timesteps,
            context=context,
            reference_latents=reference_latents
        )[0]
        if args.cfg_infer > 1.0:
            uncond_pred = transformer(
                x=z,
                timestep=timesteps,
                context=nega_context,
                reference_latents=reference_latents
            )[0]
            pred  =  uncond_pred.to(torch.float32) + args.cfg_infer * (model_pred.to(torch.float32) - uncond_pred.to(torch.float32))
        else:
            pred = model_pred.to(torch.float32)
    z, pred_original, log_prob = wan_step(pred, latents.to(torch.float32), args.eta, sigma_schedule, i, prev_sample=pre_latents.to(torch.float32), grpo=True, sde_solver=True)
    return log_prob


def train_one_step(
    args,
    accelerator,
    transformer,
    vae,
    reward_model,
    tokenizer,
    optimizer,
    lr_scheduler,
    encoder_hidden_states, 
    negative_prompt_embeds, 
    caption,
    noise_scheduler,
    max_grad_norm,
    first_image_latents,
    control_latents,
    reference_latents,
    clip_feature,
):
    total_loss = 0.0
    optimizer.zero_grad()
    device = accelerator.device
    if args.use_group:
        def repeat_tensor(tensor):
            if tensor is None:
                return None
            return torch.repeat_interleave(tensor, args.num_generations, dim=0)

        encoder_hidden_states = repeat_tensor(encoder_hidden_states)  # (B, C) -> (B*num_generations, C)
        negative_prompt_embeds = repeat_tensor(negative_prompt_embeds)


        if isinstance(caption, str):
            caption = [caption] * args.num_generations
        elif isinstance(caption, list):
            caption = [item for item in caption for _ in range(args.num_generations)]
        else:
            raise ValueError(f"Unsupported caption type: {type(caption)}")

    reward, all_latents, all_log_probs, sigma_schedule = sample_reference_model(
            args,
            device, 
            transformer,
            vae,
            first_image_latents,
            control_latents,
            reference_latents,
            clip_feature,
            encoder_hidden_states, 
            negative_prompt_embeds, 
            reward_model,
            tokenizer,
            caption,
        )
    batch_size = all_latents.shape[0]
    timestep_value = [int(sigma * 1000) for sigma in sigma_schedule][:args.sampling_steps]
    timestep_values = [timestep_value[:] for _ in range(batch_size)]
    device = all_latents.device
    timesteps =  torch.tensor(timestep_values, device=all_latents.device, dtype=torch.long)

    samples = {
        "timesteps": timesteps.detach().clone()[:, :-1],
        "latents": all_latents[
            :, :-1
        ][:, :-1],  # each entry is the latent before timestep t
        "next_latents": all_latents[
            :, 1:
        ][:, :-1],  # each entry is the latent after timestep t
        "log_probs": all_log_probs[:, :-1],
        "rewards": reward.to(torch.float32),
        "encoder_hidden_states": encoder_hidden_states,
        "negative_prompt_embeds": negative_prompt_embeds,
        "first_image_latents": first_image_latents,
        "control_latents": control_latents,
        "reference_latents": reference_latents,
        "clip_feature": clip_feature,
    }
    gathered_reward = accelerator.gather(samples["rewards"])
    if accelerator.is_main_process:
        print("gathered_reward", gathered_reward)
        with open('./reward.txt', 'a') as f: 
            f.write(f"{gathered_reward.mean().item()}\n")

    # caculate advantage
    if args.use_group:
        n = len(samples["rewards"]) // (args.num_generations)
        advantages = torch.zeros_like(samples["rewards"])
        
        for i in range(n):
            start_idx = i * args.num_generations
            end_idx = (i + 1) * args.num_generations
            group_rewards = samples["rewards"][start_idx:end_idx]
            group_mean = group_rewards.mean()
            group_std = group_rewards.std() + 1e-8
            advantages[start_idx:end_idx] = (group_rewards - group_mean) / group_std
        
        samples["advantages"] = advantages
    else:
        advantages = (samples["rewards"] - gathered_reward.mean())/(gathered_reward.std()+1e-8)
        samples["advantages"] = advantages

    
    perms = torch.stack(
        [
            torch.randperm(len(samples["timesteps"][0]))
            for _ in range(batch_size)
        ]
    ).to(device) 
    for key in ["timesteps", "latents", "next_latents", "log_probs"]:
        samples[key] = samples[key][
            torch.arange(batch_size).to(device) [:, None],
            perms,
        ]
    samples_batched = {
        k: v.unsqueeze(1)
        for k, v in samples.items()
    }
    # dict of lists -> list of dicts for easier iteration
    samples_batched_list = [
        dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
    ]
    train_timesteps = int(len(samples["timesteps"][0])*args.timestep_fraction)
    for i,sample in list(enumerate(samples_batched_list)):
        for _ in range(train_timesteps):
            clip_range = args.clip_range
            adv_clip_max = args.adv_clip_max
            new_log_probs = grpo_one_step(
                args,
                sample["latents"][:,_],
                sample["next_latents"][:,_],
                sample["first_image_latents"],
                sample["control_latents"],
                sample["reference_latents"],
                sample["clip_feature"],
                sample["encoder_hidden_states"],
                sample["negative_prompt_embeds"],
                transformer,
                sample["timesteps"][:,_],
                perms[i][_],
                sigma_schedule,
            )

            advantages = torch.clamp(
                sample["advantages"],
                -adv_clip_max,
                adv_clip_max,
            )

            ratio = torch.exp(new_log_probs - sample["log_probs"][:,_])

            unclipped_loss = -advantages * ratio
            clipped_loss = -advantages * torch.clamp(
                ratio,
                1.0 - clip_range,
                1.0 + clip_range,
            )
            loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss)) / (args.gradient_accumulation_steps * train_timesteps)

            accelerator.backward(loss)
            avg_loss = loss.detach().clone()
            avg_loss = accelerator.reduce(loss, reduction="mean")
            total_loss += avg_loss.item()

        if (i+1) % args.gradient_accumulation_steps==0:
            grad_norm = accelerator.clip_grad_norm_(transformer.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if accelerator.process_index % 8==0:
            print("reward", sample["rewards"].item())
            print("ratio", ratio)
            print("advantage", sample["advantages"].item())
            print("final loss", loss.item())

        accelerator.wait_for_everyone()
    return total_loss, grad_norm.item()


def main(args):
    accelerator = accelerate.Accelerator(
        log_with="tensorboard",
        project_dir=args.output_path,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    
    # 初始化 tracking project
    if accelerator.is_main_process:
        accelerator.init_trackers("wan_grpo_training_logs")

    if args.seed is not None:
        # TODO: t within the same seq parallel group should be the same. Noise should be different.
        set_seed(args.seed + accelerator.process_index)

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="npu",
        model_configs=[
            ModelConfig(path=os.path.join(args.model_dir, "diffusion_pytorch_model.safetensors")),
            ModelConfig(path=os.path.join(args.model_dir, "models_t5_umt5-xxl-enc-bf16.pth")),
            ModelConfig(path=os.path.join(args.model_dir, "Wan2.1_VAE.pth")),
            ModelConfig(path=os.path.join(args.model_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")),
        ],
    )
    sd = load_state_dict(args.checkpoint)
    m, u = pipe.dit.load_state_dict(sd, strict=False)
    print(f"### missing keys: {m}")
    print(f"### unexpected keys: {u}")
    transformer = pipe.dit.copy().to(accelerator.device)
    vae = pipe.vae.copy().to(accelerator.device)
    del pipe

     # Set model as trainable.
    transformer.train()

    noise_scheduler = None

    params_to_optimize = transformer.parameters()
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

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

    # TODO: get dataset and distributed sampler, then use accelerator.prepare to prepare

    world_size = accelerator.num_processes
    total_batch_size = (
        world_size
        * args.gradient_accumulation_steps
        / args.sp_size
        * args.train_sp_batch_size
    )
    accelerator.print("***** Running training *****")
    accelerator.print(f"  Num examples = {len(train_dataset)}")
    accelerator.print(f"  Dataloader size = {len(train_dataloader)}")
    accelerator.print(f"  Resume training from step {init_steps}")
    accelerator.print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    accelerator.print(
        f"  Total train batch size (w. data & sequence parallel, accumulation) = {total_batch_size}"
    )
    accelerator.print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    accelerator.print(f"  Total optimization steps per epoch = {args.max_train_steps}")
    accelerator.print(
        f"  Total training parameters per FSDP shard = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9} B"
    )
    # print dtype
    accelerator.print(f"  Master weight dtype: {transformer.parameters().__next__().dtype}")

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        assert NotImplementedError("resume_from_checkpoint is not supported now.")
        # TODO

    progress_bar = tqdm(
        range(0, 100000),
        initial=init_steps,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )


    step_times = deque(maxlen=100)
    
