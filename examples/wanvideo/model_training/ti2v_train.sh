# ── NPU environment ──────────────────────────────────────────────────
export TORCHINDUCTOR_DISABLE=1
export NPU_DISABLE_TORCHINDUCTOR=1
export HCCL_CONNECT_TIMEOUT=600
export HCCL_IF_BASE_PORT=10000
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export MASTER_PORT=29500

# ── Paths ─────────────────────────────────────────────
DATA_ROOT="/root/data/agibot_world_beta_processed_main"
OUTPUT_DIR="./checkpoints/agibotworld_ti2v_5b"

# ── Training ─────────────────────────────────────────────────────────
accelerate launch --num_processes 8 \
  --config_file examples/wanvideo/model_training/full/accelerate_config_zero2_8gpu.yaml \
  examples/wanvideo/model_training/ti2v_train.py \
  --dataset_base_path "${DATA_ROOT}" \
  --height 480 \
  --width 640 \
  --num_frames 25 \
  --model_paths '[
  [
    "/root/pretrained_weights/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors",
    "/root/pretrained_weights/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors",
    "/root/pretrained_weights/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors"
  ],
    "/root/pretrained_weights/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth",
    "/root/pretrained_weights/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
]' \
  --tokenizer_path "/root/pretrained_weights/Wan-AI/Wan2.2-TI2V-5B/google/umt5-xxl" \
  --learning_rate 1e-5 \
  --num_epochs 100 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_DIR}" \
  --save_steps 4000 \
  --trainable_models "dit" \
  --extra_inputs "input_image" \
  --text_dropout_prob 0.1 \
  --gradient_accumulation_steps 4 \
  --use_gradient_checkpointing_offload
