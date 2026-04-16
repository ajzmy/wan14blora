# ── NPU environment ──────────────────────────────────────────────────
export TORCHINDUCTOR_DISABLE=1
export NPU_DISABLE_TORCHINDUCTOR=1
export HCCL_CONNECT_TIMEOUT=600
export HCCL_IF_BASE_PORT=10000
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export MASTER_PORT=29500

# ── Paths ─────────────────────────────────────────────
DATA_ROOT="/root/data/agibot_world_beta_processed_main"
OUTPUT_DIR="./checkpoints/agibotworld_vace_5b_no_text"

# ── Training ─────────────────────────────────────────────────────────
accelerate launch --num_processes 8 \
  examples/wanvideo/model_training/vace_train.py \
  --dataset_base_path "${DATA_ROOT}" \
  --height 480 \
  --width 640 \
  --num_frames 33 \
  --model_paths '[
    "/root/xiejunbin/Diffsynth-Studio/checkpoints/agibotworld_ti2v_5b/step-40000.safetensors",
    "/root/pretrained_weights/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth",
    "/root/pretrained_weights/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
]' \
  --tokenizer_path "/root/pretrained_weights/Wan-AI/Wan2.2-TI2V-5B/google/umt5-xxl" \
  --learning_rate 5e-5 \
  --num_epochs 100 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "${OUTPUT_DIR}" \
  --save_steps 4000 \
  --trainable_models "vace" \
  --extra_inputs "vace_video,input_image" \
  --text_dropout_prob 0.0 \
  --gradient_accumulation_steps 2 \
  --use_gradient_checkpointing_offload
# The learning rate is kept consistent with the settings in the original paper
