import numpy as np
import os
from pathlib import Path

def main():
    # Use relative path so it works on both Windows and Google Colab
    gt_dir = Path(__file__).resolve().parent.parent / "data" / "Demo" / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    
    # normal_scene_1_scenario1 has 1364 frames
    num_frames_normal = 1364
    gt_normal = np.zeros(num_frames_normal, dtype=np.float64)
    gt_normal_path = gt_dir / "normal_scene_1_scenario1_tracks.npy"
    np.save(gt_normal_path, gt_normal)
    print(f"Created GT file at: {gt_normal_path} with shape {gt_normal.shape}")

    # abnormal_scene_1_scenario2 has 1439 frames
    num_frames_abnormal = 1439
    gt_abnormal = np.zeros(num_frames_abnormal, dtype=np.float64)
    # Mark frames 300 to 700 as abnormal (1.0) since the video only has 1439 frames
    gt_abnormal[300:700] = 1.0
    gt_abnormal_path = gt_dir / "abnormal_scene_1_scenario2_tracks.npy"
    np.save(gt_abnormal_path, gt_abnormal)
    print(f"Created GT file at: {gt_abnormal_path} with shape {gt_abnormal.shape}")

if __name__ == "__main__":
    main()
