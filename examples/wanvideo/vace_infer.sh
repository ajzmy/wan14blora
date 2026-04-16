#!/bin/bash

export ASCEND_RT_VISIBLE_DEVICES=3

python examples/wanvideo/model_inference/vace_infer.py \
    --checkpoint /root/xiejunbin/Diffsynth-Studio/checkpoints/agibotworld_vace_5b/step-24000.safetensors \
    --test_root /root/data/AgiBotWorldChallenge-2026/test/info_dataset \
    --output_dir ./vace5b_test_gen_24000 \
    --chunk 25 \
    --cfg_scale 3.0 \
    --seed 42 \
    --only_mp4 \
    --episodes 2019/12394160052800854
