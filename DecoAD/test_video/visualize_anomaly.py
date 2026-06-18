import sys

# Extract our custom arguments from sys.argv to avoid ModuleNotFoundError/UnrecognizedArguments during import of dataset.py
custom_args = {}
filtered_argv = []
i = 1
while i < len(sys.argv):
    arg = sys.argv[i]
    if arg in ['--checkpoint', '--video_name', '--threshold', '--window_size', '--output', '--pose_json', '--frames_dir']:
        if i + 1 < len(sys.argv):
            custom_args[arg] = sys.argv[i+1]
            i += 2
        else:
            custom_args[arg] = 'True'
            i += 1
    else:
        filtered_argv.append(arg)
        i += 1

# Reset sys.argv for standard args parser inside dataset.py
sys.argv = [sys.argv[0]] + filtered_argv

import os
import json
import cv2
import torch
import numpy as np
import argparse
from tqdm import tqdm

from WSVAD.stage1.fusion import Model
from WSVAD.stage1.dataset import gen_fusion_dataset_dataloader
from WSVAD.stage1.test import smooth_scores
from WSVAD.stage1.args import init_parser, init_sub_args

def main():
    parser = init_parser()
    args = parser.parse_args()
    
    # Restore our custom arguments manually
    args.checkpoint = custom_args.get('--checkpoint', None)
    args.video_name = custom_args.get('--video_name', 'abnormal_scene_1_scenario2')
    args.threshold = float(custom_args.get('--threshold', 0.5))
    args.window_size = int(custom_args.get('--window_size', 78))
    args.output = custom_args.get('--output', 'output_visualized.mp4')
    args.pose_json = custom_args.get('--pose_json', None)
    args.frames_dir = custom_args.get('--frames_dir', None)
    
    args, model_args = init_sub_args(args)
    
    device = torch.device('cuda:0' if 'cuda' in args.device and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Model
    print("Loading model and weights...")
    model = Model().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # 2. Get Dataloader to collect segment-level scores
    print("Loading test dataset and running inference to collect scores...")
    _, _, _, _, _, test_loader = gen_fusion_dataset_dataloader()
    
    # Structure to hold: segment_scores[clip_id][person_id][frame_idx] = list of scores
    person_segment_scores = {}
    
    with torch.no_grad():
        for pose_np, mate, scene_np, _, path_data in tqdm(test_loader, desc="Running inference"):
            pose_input = torch.stack(
                [d.clone().detach().to(dtype=torch.float) for d in pose_np]
            ).to(device)

            path_input = torch.stack(
                [p.reshape(24, 2).clone().detach().to(dtype=torch.float) for p in path_data]
            ).to(device)

            scene_input = torch.stack(
                [s.squeeze(0).clone().detach().to(dtype=torch.float) for s in scene_np]
            ).to(device)
            
            logits_dir = model(pose_input, path_input, scene_input)
            
            # Map batch outputs to individual segments
            for i in range(len(mate[0])):
                scene_id = int(mate[0][i])
                clip_id = mate[1][i]  # e.g., 'abnormal_2' or 'normal_1'
                person_id = int(mate[2][i])
                frame_start = int(mate[3][i])
                score = float(logits_dir[i].item())
                
                person_segment_scores.setdefault(clip_id, {}).setdefault(person_id, []).append((frame_start, score))

    # 3. Determine target clip_id from video_name
    # Mapping abnormal_scene_1_scenario2 -> abnormal_2, normal_scene_1_scenario1 -> normal_1
    parts = args.video_name.split('_')
    prefix = parts[0] # abnormal or normal
    scenario = parts[-1].replace('scenario', '') # e.g. 2
    target_clip_id = f"{prefix}_{scenario}"
    print(f"Target clip_id mapped from '{args.video_name}': '{target_clip_id}'")
    
    if target_clip_id not in person_segment_scores:
        print(f"Warning: clip_id '{target_clip_id}' not found in inference results. Available: {list(person_segment_scores.keys())}")
        return
        
    clip_scores = person_segment_scores[target_clip_id]
    
    # 4. Load tracking JSON to get coordinates and find the total frame count
    json_path = args.pose_json if args.pose_json else os.path.join(args.data_dir, args.dataset, "pose", "test", f"{args.video_name}_alphapose_tracked_person.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Tracking JSON not found at: {json_path}")
        
    print(f"Reading tracking JSON from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        tracking_data = json.load(f)
        
    # Get total frames in the video
    all_frame_indices = []
    for track_id, frames_dict in tracking_data.items():
        all_frame_indices.extend([int(f) for f in frames_dict.keys()])
    num_frames = max(all_frame_indices) + 1 if all_frame_indices else 0
    print(f"Video has {num_frames} frames according to JSON.")
    
    # 5. Compute smoothed person-level frame scores
    person_frame_scores = {} # person_id -> smoothed scores array of shape [num_frames]
    
    for person_id, segments in clip_scores.items():
        # Initialize raw frame scores and counts
        raw_scores = np.zeros(num_frames)
        counts = np.zeros(num_frames)
        
        for frame_start, score in segments:
            # Each segment covers 24 frames
            end_frame = min(frame_start + 24, num_frames)
            for f in range(frame_start, end_frame):
                raw_scores[f] += score
                counts[f] += 1
                
        # Calculate average score per frame where counts > 0
        for f in range(num_frames):
            if counts[f] > 0:
                raw_scores[f] = raw_scores[f] / counts[f]
                
        # Smooth scores
        smoothed = smooth_scores(raw_scores, args.window_size)
        person_frame_scores[person_id] = smoothed

    # 6. Read frames, draw boxes and compile to video
    frames_dir = args.frames_dir if args.frames_dir else os.path.join(args.data_dir, args.dataset, "test", "frames")
    print(f"Reading frames from {frames_dir}...")
    
    # Verify first frame to get size
    first_frame_path = os.path.join(frames_dir, "frame_000000.jpg")
    if not os.path.exists(first_frame_path):
        raise FileNotFoundError(f"First frame not found at: {first_frame_path}. Ensure preprocessing was run.")
        
    img = cv2.imread(first_frame_path)
    height, width, _ = img.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, 25.0, (width, height))
    
    print(f"Writing annotated video to {args.output}...")
    for f in tqdm(range(num_frames), desc="Annotating frames"):
        frame_name = f"frame_{f:06d}.jpg"
        frame_path = os.path.join(frames_dir, frame_name)
        
        if not os.path.exists(frame_path):
            # If downsampled, some frames might not exist physically
            continue
            
        frame_img = cv2.imread(frame_path)
        
        # Draw bounding box for each person active in this frame
        for track_id, frames_dict in tracking_data.items():
            f_str = str(f)
            if f_str in frames_dict:
                person_data = frames_dict[f_str]
                keypoints = np.array(person_data["keypoints"]).reshape(-1, 3)
                
                # Compute bounding box from active keypoints
                valid_kps = keypoints[keypoints[:, 2] > 0.1]
                if len(valid_kps) == 0:
                    valid_kps = keypoints # fallback if all have low conf
                    
                x_coords = valid_kps[:, 0]
                y_coords = valid_kps[:, 1]
                
                x1, y1 = int(np.min(x_coords)), int(np.min(y_coords))
                x2, y2 = int(np.max(x_coords)), int(np.max(y_coords))
                
                # Get smoothed score
                person_idx = int(track_id)
                score = 0.0
                if person_idx in person_frame_scores:
                    score = person_frame_scores[person_idx][f]
                    
                # Choose color and text based on threshold
                if score >= args.threshold:
                    color = (0, 0, 255) # Red (BGR)
                    label = f"Abnormal! ID: {track_id} ({score:.2f})"
                else:
                    color = (0, 255, 0) # Green (BGR)
                    label = f"Normal ID: {track_id} ({score:.2f})"
                    
                # Draw box
                cv2.rectangle(frame_img, (x1, y1), (x2, y2), color, 2)
                # Draw text background
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame_img, (x1, y1 - 20), (x1 + w, y1), color, -1)
                # Draw text
                cv2.putText(frame_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
        out.write(frame_img)
        
    out.release()
    print("Done! Video visualization saved successfully.")

if __name__ == "__main__":
    main()
