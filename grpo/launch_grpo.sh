#!/bin/bash
# Launch GRPO training for AgiBotWorld FunControl on 16 NPUs.
#
# Usage:
#   bash grpo/launch_grpo.sh
#
# Before running, ensure:
#   1. All grpo/*.py files are in /root/xiejunbin/Diffsynth-Studio/grpo/
#   2. EVAC_DIR and REWARD_DIR environment variables are set (or use defaults below)
#   3. pip install safetensors wandb fastdtw scikit-image ultralytics (for full reward)

set -e

# ── Environment ──
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PYTHONPATH="/root/xiejunbin/Diffsynth-Studio:${PYTHONPATH}"

# EVAC dataset utilities (for trajectory rendering)
export EVAC_DIR="/root/luomingshuang/diffsynth-studio/examples/wanvideo/model_training/evac"

# Reward model checkpoints
export REWARD_DIR="/root/grpo-training/diffsynth-studio/examples/wanvideo/model_training/reward"

# Disable tokenizer parallelism warnings
export TOKENIZERS_PARALLELISM=false

# HCCL settings for Ascend NPU communication
export HCCL_CONNECT_TIMEOUT=7200

cd /root/xiejunbin/Diffsynth-Studio

# ── Launch ──
torchrun \
    --nproc_per_node=16 \
    --master_port=29500 \
    grpo/grpo_main.py \
    --model_dir /root/pretrained_weights/PAI/Wan2.1-Fun-V1.1-1.3B-Control \
    --sft_checkpoint /root/luomingshuang/diffsynth-studio/checkpoints/agibotworld_fun_control_1_3b_text_chunk49_batchsize32_v2/step-24000.safetensors \
    --data_root /root/data/agibot_world_beta_processed_main \
    --reward_dir "${REWARD_DIR}" \
    --reward_mode psnr_only \
    --reward_device cpu \
    --output_dir /root/xiejunbin/Diffsynth-Studio/grpo/output \
    --height 480 \
    --width 640 \
    --num_frames 49 \
    --sampling_steps 20 \
    --cfg_scale 3.0 \
    --shift 5.0 \
    --eta 0.3 \
    --num_generations 4 \
    --learning_rate 1e-5 \
    --weight_decay 0.0001 \
    --max_grad_norm 1.0 \
    --clip_range 1e-4 \
    --adv_clip_max 5.0 \
    --timestep_fraction 0.6 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --max_train_steps 10000 \
    --checkpointing_steps 500 \
    --lr_warmup_steps 0 \
    --num_workers 4 \
    --seed 42 \
    --use_wandb \
    --split_file /root/data/agibot_world_beta_processed_main/split_no549.json
