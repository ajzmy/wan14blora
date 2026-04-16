#!/bin/bash

export ASCEND_RT_VISIBLE_DEVICES=15

python examples/wanvideo/model_inference/control-14b-lora.py \
    --checkpoint /root/xiejunbin/Diffsynth-Studio/checkpoints/agibotworld_control_14b_lora/step-10000.safetensors \
    --test_root /root/data/AgiBotWorldChallenge-2026/test/info_dataset \
    --output_dir ./14blora_test_gen_10000 \
    --chunk 49 \
    --cfg_scale 3.0 \
    --seed 42 \
    --only_mp4 \
    --episodes 2015/12397530039300756 2017/12398120030000900 2017/12398380163002048 2020/12393530080201171 2019/12395520178102272