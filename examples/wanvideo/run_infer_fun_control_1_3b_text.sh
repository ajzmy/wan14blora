#\!/bin/bash
set -e

export ASCEND_RT_VISIBLE_DEVICES=15


python examples/wanvideo/model_inference/infer_fun_control_1_3b_text.py \
    --checkpoint /root/luomingshuang/diffsynth-studio/checkpoints/agibotworld_fun_control_1_3b_text_chunk49_batchsize32_v2/step-20000.safetensors \
    --model_dir /root/pretrained_weights/PAI/Wan2.1-Fun-V1.1-1.3B-Control \
    --test_root /root/data/AgiBotWorldChallenge-2026/test/info_dataset \
    --output_dir ./test_gen_cfg_zero_2022 \
    --cfg_scale 5.0 \
    --only_mp4 \
    --seed 42 \
    --num_inference_steps 50 \
    --chunk_size 57 \
    --save_mp4 \
    --episodes 2022/12389210000000348 2022/12398870006000540 2022/12399220080501095
