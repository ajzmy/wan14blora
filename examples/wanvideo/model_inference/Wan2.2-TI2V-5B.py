import os
import numpy as np
import torch
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.core import load_state_dict

from lvdm.data.get_actions import parse_h5

pretrain_path = "/root/xiejunbin/Diffsynth-Studio/checkpoints/agibotworld_ti2v_5b/step-40000.safetensors"



def load_test_episode(ep_path, sample_size):
    # first frame
    H, W = sample_size
    first_frame_pil = Image.open(os.path.join(ep_path, 'frame.png')).convert('RGB')
    first_frame_pil = first_frame_pil.resize((W, H), Image.BILINEAR)
    # text
    text_path = os.path.join(ep_path, "text.txt")
    if os.path.exists(text_path):
        with open(text_path, "r") as f:
            text = f.read().strip()
    else:
        print(f"Warning: text.txt in {ep_path} is empty or not exists. Using default prompt.")
        text = "Robot arms manipulation"

    h5_path = os.path.join(ep_path, 'proprio_stats.h5')
    abs_action, _ = parse_h5(h5_path, slices=None, delta_act_sidx=1)
    T = abs_action.shape[0]

    return first_frame_pil, text, T


def generate_video(pipe, input_image, text, T):
    all_frames = []
    current_first_frame = input_image

    t = 0
    chunk_idx = 0
    while t < T:
        actual = min(25, T - t)
        
        frames = pipe(
            prompt="Robotic arm manipulation, high quality, realistic.",
            negative_prompt="blurry, low resolution, grainy, noisy, pixelated, compression artifacts, distorted, unnatural, low quality",
            seed=0, tiled=True,
            height=480, width=640,
            input_image=current_first_frame,
            num_frames=25,
            cfg_scale=3.0
        )

        all_frames.extend(frames[:actual])
        current_first_frame = frames[actual - 1]
        t += actual
        chunk_idx += 1

    # Replace first frame with GT
    all_frames[0] = input_image
    return all_frames



if __name__ == "__main__":
    test_root = "/root/data/AgiBotWorldChallenge-2026/test/info_dataset"
    output_dir = "./ti2v_test_notext"

    print("Loading model...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="npu",
        model_configs=[
            ModelConfig(path=pretrain_path),
            ModelConfig(path="/root/pretrained_weights/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(path="/root/pretrained_weights/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
        ],
        tokenizer_config=ModelConfig(path="/root/pretrained_weights/Wan-AI/Wan2.2-TI2V-5B/google/umt5-xxl"),
    )
    # ── Collect episodes (only those with text.txt) ──
    episodes = []
    skipped = []
    for task in sorted(os.listdir(test_root)):
        task_dir = os.path.join(test_root, task)
        if not os.path.isdir(task_dir):
            continue
        for ep in sorted(os.listdir(task_dir)):
            ep_dir = os.path.join(task_dir, ep)
            if not os.path.isdir(ep_dir):
                continue
            if os.path.exists(os.path.join(ep_dir, "text.txt")):
                episodes.append((task, ep, ep_dir))
            else:
                skipped.append(f"{task}/{ep}")
    print(f"Found {len(episodes)} episodes with text.txt, skipped {len(skipped)} without")
    if skipped:
        print(f"  Skipped: {', '.join(skipped)}")


    os.makedirs(output_dir, exist_ok=True)

    # ── Generate ──
    print("Starting inference...")
    for i, (task, ep_name, ep_path) in enumerate(episodes):
        
        print(f"[{i+1}/{len(episodes)}] {task}/{ep_name}")
        try:
            first_frame_pil, text, T = load_test_episode(ep_path, (480,640))
            print(f"  prompt: {text}")
            print(f"  total_frames: {T}")
            frames = generate_video(pipe, first_frame_pil, text, T)

            video_path = os.path.join(output_dir, f"{task}_{ep_name}.mp4")
            save_video(frames, video_path, fps=5, quality=5)
            print(f"  -> saved mp4 to {video_path}")
        except Exception as e:
            print(f"  Error processing {task}/{ep_name}: {e}")
            import traceback
            traceback.print_exc()

    print("Inference completed.")