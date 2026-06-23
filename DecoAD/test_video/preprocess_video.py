from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18
from typing import List, Dict, Tuple
from torchvision.models.detection import (
    KeypointRCNN_ResNet50_FPN_Weights,
    keypointrcnn_resnet50_fpn,
)


COCO_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a raw video into frames, pose JSON, and scene features.")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--output-root", type=str, default="output", help="Directory for generated artifacts.")
    parser.add_argument("--dataset-name", type=str, default="Demo", help="Dataset folder name used in WSVAD-style output.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Output split name.")
    parser.add_argument("--scene-id", type=int, default=1, help="Scene id used in output names.")
    parser.add_argument("--clip-id", type=str, default="1", help="Clip id used in output names.")
    parser.add_argument("--prefix", type=str, default="normal", choices=["normal", "abnormal"], help="Prefix for the pose JSON file name.")
    parser.add_argument("--frame-step", type=int, default=1, help="Save every Nth frame from the video.")
    parser.add_argument("--max-frames", type=int, default=180, help="Maximum number of frames to keep after sampling. Use 0 to keep all sampled frames.")
    parser.add_argument("--pose-score-threshold", type=float, default=0.70, help="Detection score threshold for people.")
    parser.add_argument("--track-iou-threshold", type=float, default=0.25, help="IoU threshold for track assignment.")
    parser.add_argument("--scene-sample-stride", type=int, default=8, help="Sample every Nth frame for scene features.")
    parser.add_argument("--pose-track", action="store_true", help="Enable AlphaPose tracking pipeline (requires tracker weights).")
    parser.add_argument("--save-vis-img", action="store_true", help="Save AlphaPose visualized output images (can be slow).")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device, e.g. cpu or cuda:0.")
    return parser.parse_args()


import shutil

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def clear_dir(path: Path) -> None:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def extract_frames(video_path: Path, frames_dir: Path, frame_step: int = 1) -> list[Path]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    sampled_indices: list[int]
    if total_frames > 0:
        sampled_indices = list(range(0, total_frames, max(1, frame_step)))
    else:
        sampled_indices = []

    frame_paths: list[Path] = []
    frame_index = 0
    saved_index = 0
    sampled_set = set(sampled_indices)

    while True:
        success, frame = capture.read()
        if not success:
            break

        if total_frames == 0:
            should_save = frame_index % max(1, frame_step) == 0
        else:
            should_save = frame_index in sampled_set

        if should_save:
            frame_path = frames_dir / f"frame_{saved_index:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(frame_path)
            saved_index += 1

        frame_index += 1

    capture.release()
    return frame_paths


def build_pose_model(device: torch.device) -> torch.nn.Module:
    raise NotImplementedError("AlphaPose is used for pose extraction in this script.")


def find_alphapose_output_json(output_dir: Path) -> Path:
    candidates = [
        output_dir / "alphapose-results.json",
        output_dir / "alphapose-results-forvis-tracked.json",
        output_dir / "alphapose-results-forvis.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    json_files = sorted(output_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No AlphaPose json output found in {output_dir}")
    return json_files[0]


def resolve_cfg_path(cfg_path: Path) -> Path:
    if cfg_path.exists() and cfg_path.is_file() and cfg_path.suffix.lower() in {".yaml", ".yml"}:
        return cfg_path

    if cfg_path.exists() and cfg_path.is_dir():
        preferred_patterns = [
            "**/*res50*256x192*.yaml",
            "**/*res50*256x192*.yml",
            "**/*.yaml",
            "**/*.yml",
        ]
        for pattern in preferred_patterns:
            matches = sorted(cfg_path.glob(pattern))
            if matches:
                return matches[0]

    raise FileNotFoundError(
        f"ALPHAPOSE_CFG must be a .yaml/.yml file or directory containing one, got: {cfg_path}"
    )


def resolve_checkpoint_path(checkpoint_path: Path) -> Path:
    if checkpoint_path.exists() and checkpoint_path.is_file() and checkpoint_path.suffix.lower() in {".pth", ".pt"}:
        return checkpoint_path

    if checkpoint_path.exists() and checkpoint_path.is_dir():
        preferred_patterns = [
            "**/*res50*256x192*.pth",
            "**/*.pth",
            "**/*.pt",
        ]
        for pattern in preferred_patterns:
            matches = sorted(checkpoint_path.glob(pattern))
            if matches:
                return matches[0]

    raise FileNotFoundError(
        f"ALPHAPOSE_CKPT must be a .pth/.pt file or directory containing one, got: {checkpoint_path}"
    )


def run_alphapose(
    frames_dir: Path,
    alphapose_root: Path,
    output_dir: Path,
    cfg_path: Path,
    checkpoint_path: Path,
    device: str,
    pose_track: bool,
    save_vis_img: bool = False,
) -> Path:
    demo_script = alphapose_root / "scripts" / "demo_inference.py"
    if not demo_script.exists():
        raise FileNotFoundError(f"AlphaPose demo script not found: {demo_script}")

    cfg_path = resolve_cfg_path(cfg_path)
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    # alpha_python = os.environ.get("ALPHAPOSE_PYTHON", "").strip()
    # python_executable = alpha_python if alpha_python else sys.executable
    python_executable = sys.executable
    cmd = [
        python_executable,
        "scripts/demo_inference.py",
        "--cfg",
        str(cfg_path),
        "--checkpoint",
        str(checkpoint_path),
        "--indir",
        str(frames_dir),
        "--outdir",
        str(output_dir),
        "--detector",
        "yolo",
        "--qsize",
        "64",
        "--sp",
        "--gpus",
        "-1" if device.startswith("cpu") else device.split(":")[-1],
    ]

    if save_vis_img:
        cmd.append("--save_img")

    if pose_track:
        cmd.append("--pose_track")

    run_env = os.environ.copy()
    existing_pythonpath = run_env.get("PYTHONPATH", "")
    if existing_pythonpath:
        run_env["PYTHONPATH"] = f"{alphapose_root}{os.pathsep}{existing_pythonpath}"
    else:
        run_env["PYTHONPATH"] = str(alphapose_root)

    result = subprocess.run(
        cmd,
        cwd=str(alphapose_root),
        env=run_env,
        check=False,
        #capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        stderr_text = result.stderr or ""
        stdout_text = result.stdout or ""

        if "No module named 'natsort'" in stderr_text:
            raise RuntimeError(
                "AlphaPose dependency missing: natsort. "
                "Install into the AlphaPose environment, e.g. `pip install natsort`, "
                "or set ALPHAPOSE_PYTHON to the Python executable of your AlphaPose env.\n"
                f"Current python used: {python_executable}"
            )

        raise RuntimeError(
            "AlphaPose failed to run.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Exit code: {result.returncode}\n"
            f"STDOUT:\n{stdout_text}\n"
            f"STDERR:\n{stderr_text}"
        )

    return find_alphapose_output_json(output_dir)


def clean_track_id(idx_val, fallback):
    if idx_val is None:
        return str(fallback)
    if isinstance(idx_val, (list, tuple, np.ndarray)):
        if len(idx_val) > 0:
            val = idx_val
            while isinstance(val, (list, tuple, np.ndarray)):
                if len(val) > 0:
                    val = val[0]
                else:
                    return str(fallback)
            idx_val = val
        else:
            return str(fallback)
    try:
        return str(int(float(idx_val)))
    except (ValueError, TypeError):
        return str(fallback)


def convert_alphapose_results_to_tracks(result_json: Path, frame_paths: list[Path]) -> dict:
    frame_index_map = {path.name: index for index, path in enumerate(frame_paths)}

    with result_json.open("r", encoding="utf-8") as file:
        raw_results = json.load(file)

    tracks: dict[str, dict[str, dict[str, object]]] = {}

    if isinstance(raw_results, dict):
        for frame_name, people in raw_results.items():
            frame_key = str(frame_index_map.get(Path(frame_name).name, len(tracks)))
            for person_index, person in enumerate(people, start=1):
                idx_val = person.get("idx")
                track_id = clean_track_id(idx_val, person_index)
                keypoints = np.asarray(person.get("keypoints", []), dtype=float).reshape(-1, 3).tolist()
                score = float(person.get("scores", person.get("score", 1.0)))
                tracks.setdefault(track_id, {})[frame_key] = {"keypoints": keypoints, "scores": score}
        return tracks

    for fallback_index, item in enumerate(raw_results, start=1):
        frame_name = Path(item.get("image_id", f"frame_{fallback_index:06d}.jpg")).name
        frame_key = str(frame_index_map.get(frame_name, fallback_index - 1))
        idx_val = item.get("idx")
        track_id = clean_track_id(idx_val, fallback_index)
        keypoints = np.asarray(item.get("keypoints", []), dtype=float).reshape(-1, 3).tolist()
        score = float(item.get("score", 1.0))
        tracks.setdefault(track_id, {})[frame_key] = {"keypoints": keypoints, "scores": score}

    return tracks


def build_scene_backbone(device: torch.device) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT
    backbone = resnet18(weights=weights)
    feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
    feature_extractor.to(device)
    feature_extractor.eval()
    return feature_extractor


def image_to_tensor(image_bgr: np.ndarray) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).to(torch.float32) / 255.0
    return tensor


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def pick_tracks(detections: dict, active_tracks: dict[int, np.ndarray], next_track_id: int, iou_threshold: float):
    assignments: list[tuple[int, int]] = []
    used_tracks: set[int] = set()

    for det_index, det_box in enumerate(detections["boxes"]):
        best_track_id = None
        best_iou = 0.0
        for track_id, track_box in active_tracks.items():
            if track_id in used_tracks:
                continue
            score = compute_iou(track_box, det_box)
            if score > best_iou:
                best_iou = score
                best_track_id = track_id

        if best_track_id is not None and best_iou >= iou_threshold:
            assignments.append((det_index, best_track_id))
            used_tracks.add(best_track_id)
        else:
            assignments.append((det_index, next_track_id))
            used_tracks.add(next_track_id)
            next_track_id += 1

    return assignments, next_track_id


def run_pose_inference(
    frame_paths: list[Path],
    pose_model: torch.nn.Module,
    device: torch.device,
    pose_score_threshold: float,
    iou_threshold: float,
) -> dict:
    tracks: dict[str, dict[str, dict[str, object]]] = {}
    active_tracks: dict[int, np.ndarray] = {}
    next_track_id = 1

    for frame_index, frame_path in enumerate(frame_paths):
        image_bgr = cv2.imread(str(frame_path))
        if image_bgr is None:
            raise RuntimeError(f"Cannot read frame: {frame_path}")

        image_tensor = image_to_tensor(image_bgr).to(device)

        with torch.no_grad():
            output = pose_model([image_tensor])[0]

        boxes = output.get("boxes", torch.empty((0, 4), device=device)).detach().cpu().numpy()
        scores = output.get("scores", torch.empty((0,), device=device)).detach().cpu().numpy()
        keypoints = output.get("keypoints", torch.empty((0, 17, 3), device=device)).detach().cpu().numpy()

        keep_indices = [index for index, score in enumerate(scores) if float(score) >= pose_score_threshold]
        if not keep_indices:
            continue

        detections = {
            "boxes": boxes[keep_indices],
            "scores": scores[keep_indices],
            "keypoints": keypoints[keep_indices],
        }

        assignments, next_track_id = pick_tracks(detections, active_tracks, next_track_id, iou_threshold)
        current_tracks: dict[int, np.ndarray] = {}

        for det_index, track_id in assignments:
            track_key = str(track_id)
            frame_key = str(frame_index)
            tracks.setdefault(track_key, {})[frame_key] = {
                "keypoints": detections["keypoints"][det_index].tolist(),
                "scores": float(detections["scores"][det_index]),
            }

            current_tracks[track_id] = detections["boxes"][det_index]

        active_tracks = current_tracks

    return tracks


def extract_scene_feature(frame_paths: list[Path], backbone: torch.nn.Module, device: torch.device, stride: int) -> torch.Tensor:
    features = []
    selected_paths = frame_paths[:: max(1, stride)]
    if not selected_paths:
        return torch.zeros(1, 512)

    with torch.no_grad():
        for frame_path in selected_paths:
            image_bgr = cv2.imread(str(frame_path))
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).to(torch.float32) / 255.0
            image_tensor = image_tensor.unsqueeze(0).to(device)
            feature_map = backbone(image_tensor)
            feature_vector = feature_map.flatten(1).squeeze(0).detach().cpu()
            features.append(feature_vector)

    if not features:
        return torch.zeros(1, 1, 512)

    stacked = torch.stack(features, dim=0)
    return stacked.mean(dim=0, keepdim=True).unsqueeze(1)


def save_pose_json(tracks: dict, pose_json_path: Path) -> None:
    with pose_json_path.open("w", encoding="utf-8") as file:
        json.dump(tracks, file, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    video_path = Path(args.video).resolve()
    output_root = Path(args.output_root).resolve()

    dataset_root = ensure_dir(output_root / args.dataset_name)
    
    # Clear frames_dir to prevent old frames from contaminating the new run
    frames_dir = dataset_root / args.split / "frames"
    clear_dir(frames_dir)
    
    pose_dir = ensure_dir(dataset_root / "pose" / args.split)
    scene_dir = ensure_dir(output_root / f"{args.dataset_name}_scene_feature")

    pose_json_path = pose_dir / f"{args.prefix}_scene_{args.scene_id}_scenario{args.clip_id}_alphapose_tracked_person.json"
    scene_feature_path = scene_dir / f"scene{args.scene_id}_features.pth"

    frame_paths = extract_frames(video_path, frames_dir, frame_step=args.frame_step)
    if not frame_paths:
        raise RuntimeError("No frames were extracted from the input video.")

    if args.max_frames and args.max_frames > 0 and len(frame_paths) > args.max_frames:
        sample_indices = np.linspace(0, len(frame_paths) - 1, args.max_frames, dtype=int)
        selected_paths = {frame_paths[index] for index in sample_indices}
        for path in frame_paths:
            if path not in selected_paths and path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass
        frame_paths = [path for path in frame_paths if path in selected_paths]

    alphapose_root_env = os.environ.get("ALPHAPOSE_ROOT")
    alphapose_root = Path(alphapose_root_env).resolve() if alphapose_root_env else None
    cfg_env = os.environ.get("ALPHAPOSE_CFG")
    ckpt_env = os.environ.get("ALPHAPOSE_CKPT")
    if alphapose_root is None:
        raise RuntimeError("Set ALPHAPOSE_ROOT to your AlphaPose repo path before running this script.")
    if not cfg_env or not ckpt_env:
        raise RuntimeError("Set ALPHAPOSE_CFG and ALPHAPOSE_CKPT to an AlphaPose config and checkpoint.")

    cfg_path = Path(cfg_env).resolve()
    ckpt_path = Path(ckpt_env).resolve()

    device = torch.device(args.device)
    alpha_output_dir = output_root / args.dataset_name / "alphapose_output"
    clear_dir(alpha_output_dir)
    pose_result_json = run_alphapose(
        frames_dir=frames_dir,
        alphapose_root=alphapose_root,
        output_dir=alpha_output_dir,
        cfg_path=cfg_path,
        checkpoint_path=ckpt_path,
        device=args.device,
        pose_track=args.pose_track,
        save_vis_img=args.save_vis_img,
    )
    tracks = convert_alphapose_results_to_tracks(pose_result_json, frame_paths)
    save_pose_json(tracks, pose_json_path)

    scene_backbone = build_scene_backbone(device)

    scene_feature = extract_scene_feature(frame_paths, scene_backbone, device, stride=args.scene_sample_stride)
    torch.save(scene_feature, scene_feature_path)

    print(f"Frames saved to: {frames_dir}")
    print(f"Pose JSON saved to: {pose_json_path}")
    print(f"Scene feature saved to: {scene_feature_path}")
    print(f"AlphaPose raw output saved in: {alpha_output_dir}")
    print(f"WSVAD dataset root: {dataset_root}")


if __name__ == "__main__":
    main()