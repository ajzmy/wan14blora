# ── NPU environment ──────────────────────────────────────────────────
export TORCHINDUCTOR_DISABLE=1
export NPU_DISABLE_TORCHINDUCTOR=1
export HCCL_CONNECT_TIMEOUT=600
export HCCL_IF_BASE_PORT=10000
# export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export MASTER_PORT=29500
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export CPU_AFFINITY_CONF=1

# ── Paths ─────────────────────────────────────────────
DATA_ROOT="/root/data/agibot_world_beta_processed_main"
OUTPUT_DIR="./checkpoints/agibotworld_control_14b_lora"

# ── Training ─────────────────────────────────────────────────────────
accelerate launch --num_processes 14 \
  --config_file examples/wanvideo/model_training/full/accelerate_config_zero2_8gpu.yaml \
  examples/wanvideo/model_training/lora_train_14b.py \
  --dataset_base_path "${DATA_ROOT}" \
  --height 480 \
  --width 640 \
  --num_frames 49 \
  --model_paths '[
    "/root/pretrained_weights/Wan2.1-v1.1-Control-14B/diffusion_pytorch_model.safetensors",
    "/root/pretrained_weights/PAI/Wan2.1-Fun-V1.1-1.3B-Control/models_t5_umt5-xxl-enc-bf16.pth",
    "/root/pretrained_weights/PAI/Wan2.1-Fun-V1.1-1.3B-Control/Wan2.1_VAE.pth",
    "/root/pretrained_weights/PAI/Wan2.1-Fun-V1.1-1.3B-Control/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
]' \
  --tokenizer_path "/root/pretrained_weights/PAI/Wan2.1-Fun-V1.1-1.3B-Control/google/umt5-xxl" \
  --learning_rate 1e-4 \
  --num_epochs 100 \
  --save_steps 2000 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_DIR}" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 64 \
  --extra_inputs "control_video,reference_image,input_image" \
  --text_dropout_prob 0.1 \
  --gradient_accumulation_steps 2 \
  --initialize_model_on_cpu