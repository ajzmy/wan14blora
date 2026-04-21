#!/bin/bash
# Smoke test for GRPO on a single Ascend NPU (device 15).
# Goal: verify one minimal end-to-end train loop works:
# prepare_conditioning -> sample -> decode -> reward -> backward -> checkpoint.

set -e

export ASCEND_RT_VISIBLE_DEVICES=15
export PYTHONPATH="/root/xiejunbin/Diffsynth-Studio:${PYTHONPATH}"
export EVAC_DIR="/root/luomingshuang/diffsynth-studio/examples/wanvideo/model_training/evac"
export REWARD_DIR="/root/grpo-training/diffsynth-studio/examples/wanvideo/model_training/reward"
export TOKENIZERS_PARALLELISM=false
export HCCL_CONNECT_TIMEOUT=7200

cd /root/xiejunbin/Diffsynth-Studio

mkdir -p /root/xiejunbin/Diffsynth-Studio/grpo/output_smoke_npu15

torchrun \
    --nproc_per_node=1 \
    --master_port=29515 \
    grpo/grpo_main.py \
    --model_dir /root/pretrained_weights/PAI/Wan2.1-Fun-V1.1-1.3B-Control \
    --sft_checkpoint /root/luomingshuang/diffsynth-studio/checkpoints/agibotworld_fun_control_1_3b_text_chunk49_batchsize32_v2/step-24000.safetensors \
    --data_root /root/data/agibot_world_beta_processed_main \
    --split_file /root/data/agibot_world_beta_processed_main/split_no549.json \
    --reward_dir "${REWARD_DIR}" \
    --reward_mode psnr_only \
    --reward_device cpu \
    --output_dir /root/xiejunbin/Diffsynth-Studio/grpo/output_smoke_npu15 \
    --height 480 \
    --width 640 \
    --num_frames 49 \
    --sampling_steps 4 \
    --cfg_scale 3.0 \
    --shift 5.0 \
    --eta 0.3 \
    --num_generations 2 \
    --learning_rate 1e-5 \
    --weight_decay 0.0001 \
    --max_grad_norm 1.0 \
    --clip_range 1e-4 \
    --adv_clip_max 5.0 \
    --timestep_fraction 0.25 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --max_train_steps 2 \
    --checkpointing_steps 1 \
    --lr_warmup_steps 0 \
    --num_workers 0 \
    --seed 42
