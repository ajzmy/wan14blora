#!/usr/bin/env python3
"""
Convert /root/luomingshuang/4_6_submission to submission format.

Input format: 2012_12385780013200460.mp4
Output format: submission_dataset/2012/12385780013200460/{0,1,2}/video/frame_XXXXX.jpg
"""
import cv2
import os
import shutil
import glob

INPUT_DIR = '/root/xiejunbin/Diffsynth-Studio/14blora_test_gen_10000'
OUTPUT_DIR = './submission_dataset_4_15'
META_EMAIL = '739314837@qq.com'

# Clean old output
if os.path.exists(OUTPUT_DIR):
    print(f'Removing old output directory: {OUTPUT_DIR}')
    shutil.rmtree(OUTPUT_DIR)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find all mp4 files
mp4_files = sorted(glob.glob(os.path.join(INPUT_DIR, '*.mp4')))
print(f'Found {len(mp4_files)} mp4 files')

for mp4_path in mp4_files:
    filename = os.path.basename(mp4_path)
    name = filename.replace('.mp4', '')

    # Parse filename: 2012_12385780013200460.mp4 -> year=2012, video_id=12385780013200460
    parts = name.split('_', 1)
    if len(parts) != 2:
        print(f'  WARNING: Unexpected filename format: {filename}, skipping')
        continue

    year = parts[0]
    video_id = parts[1]

    print(f'Processing {year}/{video_id} ...')

    # Read video
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        print(f'  ERROR: Cannot open video {mp4_path}, skipping')
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Skip first frame (as in original conversion script)
    ret, _ = cap.read()
    if not ret:
        print(f'  ERROR: Cannot read first frame, skipping')
        cap.release()
        continue

    # Read ALL remaining frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    print(f'  Got {len(frames)} frames (skipped first frame, total in video={total_frames})')

    if len(frames) == 0:
        print(f'  WARNING: No frames after skipping first frame, skipping this video')
        continue

    # Write to clip 0
    clip0_dir = os.path.join(OUTPUT_DIR, year, video_id, '0', 'video')
    os.makedirs(clip0_dir, exist_ok=True)

    for idx, frame in enumerate(frames):
        # Use frame_XXXXX.jpg format (5 digits, zero-padded)
        out_path = os.path.join(clip0_dir, f'frame_{idx:05d}.jpg')
        cv2.imwrite(out_path, frame)

    # Copy to clip 1 and clip 2
    for clip_idx in [1, 2]:
        clip_dir = os.path.join(OUTPUT_DIR, year, video_id, str(clip_idx), 'video')
        os.makedirs(clip_dir, exist_ok=True)

        for idx in range(len(frames)):
            src = os.path.join(clip0_dir, f'frame_{idx:05d}.jpg')
            dst = os.path.join(clip_dir, f'frame_{idx:05d}.jpg')
            shutil.copy2(src, dst)

# Write meta_info.txt
meta_path = os.path.join(OUTPUT_DIR, 'meta_info.txt')
with open(meta_path, 'w') as f:
    f.write(META_EMAIL + '\n')

print(f'\nDone! Output at {OUTPUT_DIR}')
print(f'Total videos processed: {len(mp4_files)}')
