import numpy as np
import os
from pathlib import Path

def main():
    gt_dir = Path("d:/CV/DecoAD_prj/DecoAD/data/Demo/gt")
    gt_dir.mkdir(parents=True, exist_ok=True)
    
    # normal_scene_1_scenario1 has frames up to 8090 (so 8091 frames)
    num_frames_normal = 8091
    gt_normal = np.zeros(num_frames_normal, dtype=np.float64)
    gt_normal_path = gt_dir / "normal_scene_1_scenario1_tracks.npy"
    np.save(gt_normal_path, gt_normal)
    print(f"Created GT file at: {gt_normal_path} with shape {gt_normal.shape}")

    # abnormal_scene_1_scenario2 has frames up to 9808 (so 9809 frames)
    # We must include at least some abnormal frames (label=1) so that ROC AUC calculation doesn't crash
    num_frames_abnormal = 9809
    gt_abnormal = np.zeros(num_frames_abnormal, dtype=np.float64)
    # Mark frames 3000 to 5000 as abnormal (1.0)
    gt_abnormal[3000:5000] = 1.0
    gt_abnormal_path = gt_dir / "abnormal_scene_1_scenario2_tracks.npy"
    np.save(gt_abnormal_path, gt_abnormal)
    print(f"Created GT file at: {gt_abnormal_path} with shape {gt_abnormal.shape}")

if __name__ == "__main__":
    main()
