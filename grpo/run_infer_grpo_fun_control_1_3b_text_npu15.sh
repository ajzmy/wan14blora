#!/bin/bash
# Test inference for a GRPO checkpoint on a single Ascend NPU (device 15).
# Default checkpoint points to the latest full-reward smoke result; replace it with
# a larger GRPO checkpoint when available.

set -e

export ASCEND_RT_VISIBLE_DEVICES=15
export PYTHONPATH="/root/luomingshuang/diffsynth-studio:${PYTHONPATH}"

CHECKPOINT="/root/xiejunbin/Diffsynth-Studio/grpo/output_full_8gpusv2/checkpoint-75/dit.safetensors"
MODEL_DIR="/root/pretrained_weights/PAI/Wan2.1-Fun-V1.1-1.3B-Control"
TEST_ROOT="/root/data/AgiBotWorldChallenge-2026/test/info_dataset"
OUTPUT_DIR="/root/xiejunbin/Diffsynth-Studio/grpo/test_output_grpo_fun_control_1_3b_text/125/cfg3.0_seed42"

cd /root/luomingshuang/diffsynth-studio

python examples/wanvideo/model_training/infer_fun_control_1_3b_text.py \
    --checkpoint "${CHECKPOINT}" \
    --model_dir "${MODEL_DIR}" \
    --test_root "${TEST_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --cfg_scale 3.0 \
    --only_mp4 \
    --seed 42 \
    --num_inference_steps 50 \
    --chunk_size 49 \
    --save_mp4 \
    --device npu
    # If you only want a fast sanity check, append for example:
    # --episodes 2022/12389210000000348
