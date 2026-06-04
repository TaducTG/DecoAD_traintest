import argparse
import os
import sys
import subprocess
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end Raw Video Anomaly Detection using AlphaPose & STG-NF")
    parser.add_argument("--video", type=str, required=True, help="Path to input raw video file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/UBnormal_supervised_79_2.tar", help="Path to STG-NF model checkpoint")
    parser.add_argument("--output-dir", type=str, default="data/custom_test", help="Directory for generated artifacts")
    parser.add_argument("--threshold", type=float, default=0.35, help="Anomaly detection threshold (normality score < threshold)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference (cpu or cuda:0)")
    parser.add_argument("--alphapose-root", type=str, default="d:/CV/DecoAD_prj/DecoAD/AlphaPose", help="Path to AlphaPose repository")
    parser.add_argument("--alphapose-cfg", type=str, default="d:/CV/DecoAD_prj/DecoAD/AlphaPose/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml", help="Path to AlphaPose config")
    parser.add_argument("--alphapose-ckpt", type=str, default="d:/CV/DecoAD_prj/DecoAD/AlphaPose/pretrained_models/fast_res50_256x192.pth", help="Path to AlphaPose checkpoint")
    return parser.parse_args()

def extract_frames(video_path: Path, frames_dir: Path) -> list:
    print(f"[*] Extracting frames from {video_path}...")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    if frames_dir.exists():
        import shutil
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    saved_index = 0
    
    while True:
        success, frame = capture.read()
        if not success:
            break
        
        frame_path = frames_dir / f"{saved_index:06d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        frame_paths.append(frame_path)
        saved_index += 1
        
    capture.release()
    print(f"[+] Extracted {saved_index} frames to {frames_dir}")
    return frame_paths

def run_alphapose(frames_dir: Path, alphapose_root: Path, output_dir: Path, cfg_path: Path, checkpoint_path: Path, device: str) -> Path:
    print(f"[*] Running AlphaPose pose extraction...")
    demo_script = alphapose_root / "scripts" / "demo_inference.py"
    if not demo_script.exists():
        raise FileNotFoundError(f"AlphaPose demo script not found: {demo_script}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Map device to gpus parameter
    gpu_param = "-1" if device.startswith("cpu") else device.split(":")[-1]
    
    cmd = [
        sys.executable,
        "scripts/demo_inference.py",
        "--cfg", str(cfg_path),
        "--checkpoint", str(checkpoint_path),
        "--indir", str(frames_dir),
        "--outdir", str(output_dir),
        "--detector", "yolo",
        "--sp",
        "--gpus", gpu_param
    ]
    
    # Check if Reid weight file exists
    reid_weight_path = alphapose_root / "trackers" / "weights" / "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth"
    if reid_weight_path.exists():
        cmd.append("--pose_track")
        print("[*] Found Reid tracker weights. Using Reid-based tracking (--pose_track).")
    else:
        cmd.append("--pose_flow")
        print("[!] Reid tracker weights not found. Falling back to PoseFlow tracking (--pose_flow).")
    
    run_env = os.environ.copy()
    existing_pythonpath = run_env.get("PYTHONPATH", "")
    if existing_pythonpath:
        run_env["PYTHONPATH"] = f"{alphapose_root}{os.pathsep}{existing_pythonpath}"
    else:
        run_env["PYTHONPATH"] = str(alphapose_root)
        
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(alphapose_root), env=run_env, check=False)
    
    if result.returncode != 0:
        raise RuntimeError(f"AlphaPose execution failed with exit code: {result.returncode}")
        
    # Locate alphapose output JSON
    results_json = output_dir / "alphapose-results.json"
    if not results_json.exists():
        json_files = sorted(output_dir.glob("*.json"))
        if json_files:
            results_json = json_files[0]
        else:
            raise FileNotFoundError(f"No AlphaPose JSON output found in {output_dir}")
            
    print(f"[+] AlphaPose finished. Output: {results_json}")
    return results_json

def convert_alphapose_results_to_tracks(result_json: Path, frame_paths: list) -> dict:
    print(f"[*] Converting AlphaPose output to tracking format...")
    frame_index_map = {path.name: index for index, path in enumerate(frame_paths)}
    
    with result_json.open("r", encoding="utf-8") as file:
        raw_results = json.load(file)
        
    tracks = {}
    
    # Handle both list and dict formats of AlphaPose output
    items_list = raw_results if isinstance(raw_results, list) else []
    if isinstance(raw_results, dict):
        for frame_name, people in raw_results.items():
            frame_key = str(frame_index_map.get(Path(frame_name).name, len(tracks)))
            for idx, person in enumerate(people, start=1):
                track_id = str(person.get("idx", idx))
                keypoints = np.asarray(person.get("keypoints", []), dtype=float).reshape(-1, 3).tolist()
                score = float(person.get("scores", person.get("score", 1.0)))
                tracks.setdefault(track_id, {})[frame_key] = {"keypoints": keypoints, "scores": score}
        return tracks

    for idx, item in enumerate(items_list, start=1):
        image_id = item.get("image_id", "")
        frame_name = Path(image_id).name
        frame_key = str(frame_index_map.get(frame_name, idx - 1))
        track_id = str(item.get("idx", idx))
        keypoints = np.asarray(item.get("keypoints", []), dtype=float).reshape(-1, 3).tolist()
        score = float(item.get("score", 1.0))
        tracks.setdefault(track_id, {})[frame_key] = {"keypoints": keypoints, "scores": score}
        
    print(f"[+] Converted pose tracks for {len(tracks)} people.")
    return tracks

def main():
    args = parse_args()
    
    # 1. Setup Directories
    output_root = Path(args.output_dir).resolve()
    frames_dir = output_root / "test" / "frames"
    pose_dir = output_root / "pose" / "test"
    alpha_output_dir = output_root / "alphapose_output"
    
    pose_dir.mkdir(parents=True, exist_ok=True)
    
    # Output pose JSON name matching UBnormal regex format:
    # (abnormal|normal)_scene_(\d+)_scenario(.*)_alphapose_tracked_person.json
    pose_json_path = pose_dir / "normal_scene_1_scenario1_alphapose_tracked_person.json"
    scores_npy_path = output_root / "normality_scores.npy"
    
    # 2. Extract Frames
    frame_paths = extract_frames(Path(args.video).resolve(), frames_dir)
    if not frame_paths:
        raise RuntimeError("No frames extracted.")
        
    # 3. Run AlphaPose
    alphapose_json = run_alphapose(
        frames_dir=frames_dir,
        alphapose_root=Path(args.alphapose_root).resolve(),
        output_dir=alpha_output_dir,
        cfg_path=Path(args.alphapose_cfg).resolve(),
        checkpoint_path=Path(args.alphapose_ckpt).resolve(),
        device=args.device
    )
    
    # 4. Format Pose Tracks
    tracks = convert_alphapose_results_to_tracks(alphapose_json, frame_paths)
    with pose_json_path.open("w", encoding="utf-8") as file:
        json.dump(tracks, file, indent=2)
    print(f"[+] Converted pose tracks saved to {pose_json_path}")
    # 5. Run STG-NF Inference
    # Check if there is at least one track with length > 16 (since UBnormal uses seg_len=16)
    has_valid_tracks = False
    seg_len = 16
    for track_id, track_data in tracks.items():
        if len(track_data) > seg_len:
            has_valid_tracks = True
            break

    if has_valid_tracks:
        print(f"[*] Running STG-NF inference...")
        eval_cmd = [
            sys.executable,
            "train_eval.py",
            "--dataset", "UBnormal",
            "--seg_len", str(seg_len),
            "--pose_path_test", str(pose_dir),
            "--vid_path_test", str(frames_dir),
            "--device", args.device,
            "--num_workers", "0",
            "--checkpoint", args.checkpoint,
            "--save_scores_path", str(scores_npy_path)
        ]
        # For supervised model checkpoint, we need the matching R argument
        if "supervised" in args.checkpoint:
            eval_cmd += ["--R", "10"]
            
        print(f"Executing evaluation: {' '.join(eval_cmd)}")
        # Run subprocess with cwd set to the root of STG-NF repository (parent directory of this script's directory)
        stg_nf_root = Path(__file__).resolve().parent.parent
        eval_result = subprocess.run(eval_cmd, cwd=str(stg_nf_root), check=False)
        if eval_result.returncode != 0:
            raise RuntimeError(f"STG-NF evaluation failed with exit code: {eval_result.returncode}")
    else:
        print("[!] WARNING: No valid human pose tracks of length > 16 were detected by AlphaPose.")
        print("[*] Creating a default 'normal' scores array...")
        # Save a dummy scores array (all ones = completely normal) to scores_npy_path
        np.save(scores_npy_path, np.full(len(frame_paths), 1.0))
        
    # 6. Analyze Scores & Plot
    if not scores_npy_path.exists():
        print("[-] Error: Normality scores file not found. Inference might have failed.")
        return
        
    scores = np.load(scores_npy_path)
    total_video_frames = len(frame_paths)
    if len(scores) < total_video_frames:
        pad_val = scores.max() if len(scores) > 0 else 1.0
        padding = np.full(total_video_frames - len(scores), pad_val)
        scores = np.concatenate([scores, padding])
        print(f"[*] Normality scores padded from length {len(scores) - len(padding)} to {len(scores)} to match video frame count.")
        
    print(f"[+] Normality scores loaded. Length: {len(scores)}")
    
    # Find abnormal segments
    is_anomaly = scores < args.threshold
    suspicious_frames = []
    in_anomaly_segment = False
    start_frame = 0
    
    for idx, anomaly_flag in enumerate(is_anomaly):
        if anomaly_flag:
            if not in_anomaly_segment:
                start_frame = idx
                in_anomaly_segment = True
        else:
            if in_anomaly_segment:
                suspicious_frames.append((start_frame, idx - 1))
                in_anomaly_segment = False
    if in_anomaly_segment:
        suspicious_frames.append((start_frame, len(is_anomaly) - 1))
        
    # Plot abnormality scores
    chart_path = output_root / "anomaly_chart.png"
    plt.figure(figsize=(12, 5))
    plt.plot(scores, label="Normality Score", color='blue')
    plt.axhline(y=args.threshold, color='red', linestyle='--', label=f"Anomaly Threshold ({args.threshold})")
    
    # Highlight anomaly zones
    for start, end in suspicious_frames:
        plt.axvspan(start, end, color='red', alpha=0.2)
        
    plt.xlabel("Frame Index")
    plt.ylabel("Score")
    plt.title(f"Anomaly Detection Results (Normality Curve)")
    plt.legend()
    plt.grid(True)
    plt.savefig(str(chart_path))
    print(f"[+] Saved anomaly visualization chart to: {chart_path}")
    
    print("\n" + "="*30 + " ANOMALY DETECTION REPORT " + "="*30)
    if suspicious_frames:
        print("\033[91m[!] WARNING: Anomalous/Suspicious events detected in the video:\033[0m")
        for start, end in suspicious_frames:
            print(f"  - Frame {start} to Frame {end} (Normality Score drops below threshold)")
    else:
        print("\033[92m[+] SUCCESS: No anomalies detected in the video. All frames are normal.\033[0m")
    print("="*86 + "\n")

if __name__ == "__main__":
    main()
