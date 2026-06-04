import json
import numpy as np
from pathlib import Path

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

def main():
    raw_json_path = Path("data/Demo/alphapose_output/alphapose-results.json")
    pose_json_path = Path("data/Demo/pose/test/normal_scene_1_scenario1_alphapose_tracked_person.json")
    frames_dir = Path("data/Demo/test/frames")
    
    frame_paths = sorted(list(frames_dir.glob("*.jpg")))
    frame_index_map = {path.name: index for index, path in enumerate(frame_paths)}
    
    with raw_json_path.open("r", encoding="utf-8") as file:
        raw_results = json.load(file)
        
    tracks = {}
    
    # Ở đây raw_results là một list
    for fallback_index, item in enumerate(raw_results, start=1):
        frame_name = Path(item.get("image_id", f"frame_{fallback_index:06d}.jpg")).name
        frame_key = str(frame_index_map.get(frame_name, fallback_index - 1))
        
        idx_val = item.get("idx")
        track_id = clean_track_id(idx_val, fallback_index)
        
        keypoints = np.asarray(item.get("keypoints", []), dtype=float).reshape(-1, 3).tolist()
        score = float(item.get("score", 1.0))
        
        tracks.setdefault(track_id, {})[frame_key] = {"keypoints": keypoints, "scores": score}
        
    with pose_json_path.open("w", encoding="utf-8") as file:
        json.dump(tracks, file, ensure_ascii=False, indent=2)
        
    print(f"Fixed tracked person JSON file at: {pose_json_path}")
    print("New track keys:", list(tracks.keys()))

if __name__ == "__main__":
    main()
