import argparse
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Thêm thư mục hiện tại vào python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from WSVAD.stage1.dataset import get_dataset_and_loader, trans_list
from WSVAD.stage1.fusion import Model
from WSVAD.stage1.args import init_parser, init_sub_args

def parse_args():
    parser = argparse.ArgumentParser(description="Run WSVAD inference on preprocessed video data.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint (.pkl).")
    parser.add_argument("--dataset", type=str, default="Demo", help="Dataset name (e.g. Demo)")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory holding datasets.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference (cpu or cuda:0).")
    parser.add_argument("--window-size", type=int, default=12, help="Smoothing window size.")
    parser.add_argument("--output-chart", type=str, default="anomaly_score.png", help="Path to save the anomaly score chart.")
    return parser.parse_args()

def smooth_scores(scores, window_size=12):
    if window_size > len(scores):
        return scores
    if window_size < 1:
        return scores
    return np.convolve(scores, np.ones(window_size) / window_size, mode='same')

def main():
    args_opt = parse_args()
    
    # Thay thế tham số hệ thống để init_parser hoạt động tốt
    sys.argv = [
        sys.argv[0],
        "--dataset", args_opt.dataset,
        "--data_dir", args_opt.data_dir,
        "--device", args_opt.device,
        "--only_test"
    ]
    
    parser = init_parser()
    for action in parser._actions:
        if action.dest == 'dataset':
            action.choices = ['NWPUC', 'UFSR', 'UBnormal', 'ShanghaiTech', 'Demo']
            
    args = parser.parse_args()
    args, model_args = init_sub_args(args)
    
    # Chỉ load dataset test
    dataset_dict, loader_dict = get_dataset_and_loader(args, trans_list=trans_list, only_test=True)
    test_dataset = dataset_dict['test']
    
    # Chuẩn bị test data format tương tự gen_fusion_dataset_dataloader nhưng không cần train set
    dataset_t = []
    for test_item in test_dataset:
        data, tran, mate, label, path_data = test_item
        data = data[:2, :, :]
        
        scene = mate[0]
        scene_feat_path = os.path.join(args.data_dir, f"{args.dataset}_scene_feature", f"scene{scene}_features.pth")
        if not os.path.exists(scene_feat_path):
            raise FileNotFoundError(f"Scene feature not found: {scene_feat_path}")
        scene_feat = torch.load(scene_feat_path, map_location='cpu')
        scene_feat = scene_feat.expand(1, 1, 512)
        
        t = [data, mate, scene_feat, label, path_data]
        dataset_t.append(t)
        
    from torch.utils.data import DataLoader
    from WSVAD.stage1.dataset import UbnormalDataset
    
    loader_t = DataLoader(UbnormalDataset(dataset_t), batch_size=args.batch_size, shuffle=False)
    
    device = torch.device(args_opt.device)
    model = Model().to(device)
    
    # Load model checkpoint
    if not os.path.exists(args_opt.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args_opt.checkpoint}")
    
    checkpoint = torch.load(args_opt.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    
    # Một số checkpoint lưu keys của GCNConv không có '.lin.' hoặc ngược lại tùy phiên bản PyG, 
    # nhưng chúng ta hãy thử load trực tiếp state_dict trước.
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"Successfully loaded checkpoint from: {args_opt.checkpoint}")
    
    # Run Inference
    total_frames = 0
    for test_item in test_dataset:
        data, tran, mate, label, path_data = test_item
        total_frames = max(total_frames, int(mate[3]) + args.seg_len)
        
    frames_dir = os.path.join(args.data_dir, args.dataset, "test/frames")
    if os.path.exists(frames_dir):
        total_frames = len([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    frame_scores = np.zeros(total_frames)
    
    with torch.no_grad():
        for pose_np, mate, scene_np, _, path_data in tqdm(loader_t, desc="Running inference"):
            pose_input = torch.stack([d.clone().detach().to(dtype=torch.float) for d in pose_np]).to(device)
            path_input = torch.stack([p.reshape(24, 2).clone().detach().to(dtype=torch.float) for p in path_data]).to(device)
            scene_input = torch.stack([s.squeeze(0).clone().detach().to(dtype=torch.float) for s in scene_np]).to(device)
            
            logits_dir = model(pose_input, path_input, scene_input)
            
            for i in range(len(mate[0])):
                frame_start = int(mate[3][i])
                logits = logits_dir[i].cpu().numpy()
                
                for idx in range(24):
                    f_idx = frame_start + idx
                    if f_idx < total_frames:
                        frame_scores[f_idx] = max(frame_scores[f_idx], float(logits[idx]))
                        
    # Smooth scores
    smoothed = smooth_scores(frame_scores, args_opt.window_size)
    
    # Vẽ và lưu kết quả
    plt.figure(figsize=(12, 5))
    plt.plot(frame_scores, label="Raw Anomaly Score", alpha=0.4, color='blue')
    plt.plot(smoothed, label=f"Smoothed Score (Window: {args_opt.window_size})", color='red')
    plt.xlabel("Frame Index")
    plt.ylabel("Anomaly Score")
    plt.title(f"Anomaly Detection Results for {args_opt.dataset}")
    plt.legend()
    plt.grid(True)
    plt.savefig(args_opt.output_chart)
    print(f"Saved anomaly chart to: {args_opt.output_chart}")
    
    # Lọc ra các frame đáng ngờ (score > 0.5)
    suspicious_frames = []
    in_anomaly_segment = False
    start_frame = 0
    for idx, score in enumerate(smoothed):
        if score > 0.5:
            if not in_anomaly_segment:
                start_frame = idx
                in_anomaly_segment = True
        else:
            if in_anomaly_segment:
                suspicious_frames.append((start_frame, idx - 1))
                in_anomaly_segment = False
    if in_anomaly_segment:
        suspicious_frames.append((start_frame, len(smoothed) - 1))
        
    print("\n=== KẾT QUẢ PHÂN TÍCH VIDEO ===")
    if suspicious_frames:
        print("Phát hiện các khoảng thời gian nghi ngờ có hành vi bất thường:")
        for start, end in suspicious_frames:
            print(f"- Từ Frame {start} đến Frame {end}")
    else:
        print("Không phát hiện hành vi bất thường nào (tất cả các frame đều dưới ngưỡng 0.5).")

if __name__ == "__main__":
    main()
