import numpy as np
import os
from pathlib import Path

def main():
    gt_dir = Path("d:/CV/DecoAD_prj/DecoAD/data/Demo/gt")
    gt_dir.mkdir(parents=True, exist_ok=True)
    
    # Số lượng frames thực tế là 157
    num_frames = 157
    gt_dummy = np.zeros(num_frames, dtype=np.float64)
    
    gt_path = gt_dir / "normal_scene_1_scenario1_tracks.npy"
    np.save(gt_path, gt_dummy)
    print(f"Created dummy GT file at: {gt_path} with shape {gt_dummy.shape}")

if __name__ == "__main__":
    main()
