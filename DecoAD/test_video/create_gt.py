import numpy as np
import os
from pathlib import Path
import scipy.io

def main():
    # Use relative path so it works on both Windows and Google Colab
    project_root = Path(__file__).resolve().parents[2]
    gt_dir = Path(__file__).resolve().parent.parent / "data" / "Demo" / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Create ground truth for normal_scene_1_scenario1 (1364 frames, all normal)
    num_frames_normal = 1364
    gt_normal = np.zeros(num_frames_normal, dtype=np.float64)
    gt_normal_path = gt_dir / "normal_scene_1_scenario1_tracks.npy"
    np.save(gt_normal_path, gt_normal)
    print(f"Created normal GT file at: {gt_normal_path} with shape {gt_normal.shape}")

    # 2. Convert all 21 .mat files from ground_truth_demo/testing_label_mask
    mask_dir = project_root / "ground_truth_demo" / "testing_label_mask"
    
    for i in range(1, 22):
        mat_path = mask_dir / f"{i}_label.mat"
        if not mat_path.exists():
            print(f"Warning: Label file {mat_path} not found. Skipping.")
            continue
            
        # Load .mat file
        data = scipy.io.loadmat(str(mat_path))
        label_cell = data['volLabel'][0] # shape (num_frames,) containing 2D arrays
        num_frames = len(label_cell)
        
        # Collapse 2D masks to 1D frame-level labels
        gt_array = np.zeros(num_frames, dtype=np.float64)
        for t in range(num_frames):
            if np.any(label_cell[t]):
                gt_array[t] = 1.0
                
        gt_path = gt_dir / f"abnormal_scene_1_scenario{i}_tracks.npy"
        np.save(gt_path, gt_array)
        print(f"Converted {mat_path.name} -> {gt_path.name} (shape: {gt_array.shape}, abnormal frames: {int(np.sum(gt_array))})")

if __name__ == "__main__":
    main()

