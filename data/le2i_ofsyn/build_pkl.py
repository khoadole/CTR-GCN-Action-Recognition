#!/usr/bin/env python3
"""Convert prepared CSV splits into PKL format for CTR-GCN training.
Output PKL format: (features, labels)
- features: (N, T, 17, 3)
- labels: (N, C) one-hot/soft labels

FIXED: 
- normalize_pose_sequence is now applied PER WINDOW to match inference.
- Added temporal_drop_ratio to simulate YOLO dropping frames in production.
"""

from __future__ import annotations
import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

MAIN_PARTS = [
    "left_shoulder_x", "left_shoulder_y", "left_shoulder_s",
    "right_shoulder_x", "right_shoulder_y", "right_shoulder_s",
    "left_hip_x", "left_hip_y", "left_hip_s",
    "right_hip_x", "right_hip_y", "right_hip_s",
    "left_knee_x", "left_knee_y", "left_knee_s",
    "right_knee_x", "right_knee_y", "right_knee_s",
]

MAIN_IDX_PARTS = [5, 6, 11, 12, 13, 14]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PKL files from prepared CSV splits")
    parser.add_argument("--config", default="/home/cxviewlab2/data/khoa.do/CTR-GCN/data/le2i_ofsyn/config/build_pkl.yaml", help="Path to PKL build config")
    return parser.parse_args()

def resolve_path(base_dir: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()

def normalize_pose_sequence(kps_seq: np.ndarray) -> np.ndarray:
    """
    Shifts the coordinate system so the hip midpoint at frame 0 is the origin (0,0).
    kps_seq: (T, V, 3). Tọa độ x,y hiện tại đã là [0, 1] từ bước extract.
    """
    out = kps_seq.copy().astype(np.float32)
    
    lhip = out[0, 11, :2]
    rhip = out[0, 12, :2]

    # Guard against invalid root in frame 0 (NaN or all-zero hips).
    if np.isnan(lhip).any() or np.isnan(rhip).any() or (lhip == 0).all() or (rhip == 0).all():
        hip_xy = out[:, [11, 12], :2]
        root = np.nanmean(hip_xy, axis=(0, 1))
        if np.isnan(root).any():
            return out
    else:
        root = (lhip + rhip) / 2.0
    
    # Dời toàn bộ hệ quy chiếu của cả chuỗi (sequence) về tâm
    out[:, :, 0] = out[:, :, 0] - root[0]
    out[:, :, 1] = out[:, :, 1] - root[1]
    
    return out

def seq_label_smoothing(labels: np.ndarray, max_step: int = 10) -> np.ndarray:
    steps = 0
    remain_step = 0
    target_label = 0
    active_label = 0
    start_change = 0
    max_val = float(np.max(labels))
    min_val = float(np.min(labels))
    for i in range(labels.shape[0]):
        if remain_step > 0:
            if i >= start_change:
                labels[i][active_label] = max_val * remain_step / steps
                next_val = max_val * (steps - remain_step) / steps
                labels[i][target_label] = next_val if next_val else min_val
            remain_step -= 1
            continue
        end = min(i + max_step, labels.shape[0])
        diff_index = np.where(np.argmax(labels[i:end], axis=1) - np.argmax(labels[i]) != 0)[0]
        if len(diff_index) > 0:
            steps = int(diff_index[0])
            if steps <= 0:
                continue
            remain_step = steps
            start_change = i + remain_step // 2
            target_label = int(np.argmax(labels[i + remain_step]))
            active_label = int(np.argmax(labels[i]))
    return labels

def simulate_production_drop(fs: list[int], drop_ratio: float = 0.35, min_keep: int = 8) -> list[int]:
    """Mô phỏng Roulette: drop ngẫu nhiên nhưng không drop liên tục quá mạnh."""
    if drop_ratio <= 0 or len(fs) <= min_keep:
        return fs
    rng = np.random.default_rng()
    n = len(fs)
    keep_mask = rng.random(n) > drop_ratio
    
    # Đảm bảo ít nhất min_keep frame
    if keep_mask.sum() < min_keep:
        keep_mask[:min_keep] = True
        
    # Tránh drop 2 frame liên tiếp quá nhiều
    for i in range(1, n):
        if not keep_mask[i] and not keep_mask[i-1]:
            keep_mask[i] = True
            
    kept_indices = np.where(keep_mask)[0]
    return [fs[i] for i in kept_indices]

# ====================== PROCESS CSV ======================
def process_csv(
    csv_path: Path,
    class_count: int,
    n_frames: int,
    skip_frame: int,
    smooth_labels_step: int,
    label_smoothing_eps: float,
    max_frame_gap: int,
    temporal_drop_ratio: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    
    print(f"[INFO] Reading CSV: {csv_path}")
    annot = pd.read_csv(csv_path)
    in_rows = int(len(annot))

    required = {"video", "frame", "label"}
    missing = required - set(annot.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")

    if any(c not in annot.columns for c in MAIN_PARTS):
        missing_main = [c for c in MAIN_PARTS if c not in annot.columns]
        raise ValueError(f"Missing main keypoint columns in {csv_path}: {missing_main}")

    # Drop NaN torso/leg keypoints (We stitch good frames together)
    idx_nan = annot[MAIN_PARTS].isna().any(axis=1)
    annot = annot[~idx_nan].reset_index(drop=True)
    after_nan_rows = int(len(annot))

    labels_int = annot["label"].astype(int)
    all_classes = list(range(class_count))
    unknown_labels = sorted(set(labels_int.unique()) - set(all_classes))
    if unknown_labels:
        raise ValueError(f"Unknown labels in {csv_path}: {unknown_labels}")

    label_onehot = pd.get_dummies(labels_int, prefix="cls")
    cols = [f"cls_{c}" for c in all_classes]
    for col in cols:
        if col not in label_onehot.columns:
            label_onehot[col] = 0
    label_onehot = label_onehot[cols]

    data = pd.concat([annot.drop(columns=["label"]), label_onehot], axis=1)

    feature_set: list[np.ndarray] = []
    labels_set: list[np.ndarray] = []
    
    coord_cols = [c for c in data.columns if c not in {"video", "frame", "width", "height"} and not c.startswith("cls_")]
    if len(coord_cols) != len(KEYPOINT_NAMES) * 3:
        raise ValueError(f"Expected {len(KEYPOINT_NAMES)*3} coord columns but got {len(coord_cols)}")

    for _, group in data.groupby("video", sort=False):
        group = group.sort_values("frame").reset_index(drop=True)
        y_data = group[cols].values.astype(np.float32)

        if label_smoothing_eps > 0:
            y_data = y_data * (1.0 - label_smoothing_eps) + (1.0 - y_data) * label_smoothing_eps / max(1, class_count - 1)
        if smooth_labels_step > 0:
            y_data = seq_label_smoothing(y_data, smooth_labels_step)

        frames = group["frame"].values.astype(np.int64)
        frame_sets: list[list[int]] = []
        fs = [0]
        for i in range(1, len(frames)):
            if frames[i] < frames[i - 1] + max_frame_gap:
                fs.append(i)
            else:
                frame_sets.append(fs)
                fs = [i]
        frame_sets.append(fs)

        all_coords = group[coord_cols].values.astype(np.float32)

        for fs in frame_sets:
            if len(fs) < n_frames:
                continue

            # SIMULATE PRODUCTION DROP BEFORE WINDOWING
            if temporal_drop_ratio > 0:
                fs = simulate_production_drop(fs, drop_ratio=temporal_drop_ratio)
                if len(fs) < n_frames:
                    continue

            # Extract raw coordinates for the chunk
            xys = all_coords[fs].reshape(-1, len(KEYPOINT_NAMES), 3)
            
            # Confidence Scores for weighting
            scr = np.clip(xys[:, :, 2].copy(), 0.0, 1.0)
            scr[:, MAIN_IDX_PARTS] = np.minimum(scr[:, MAIN_IDX_PARTS] * 1.5, 1.0)
            scr_mean = scr.mean(axis=1)

            # Build windows FIRST, Normalize SECOND
            for i in range(0, xys.shape[0] - n_frames + 1, skip_frame):
                # 1. Slice the 36-frame window
                window_xys = xys[i:i + n_frames]
                
                # 2. Apply origin shift PER WINDOW (Fixes train/test leakage)
                window_scaled = normalize_pose_sequence(window_xys)
                feature_set.append(window_scaled)
                
                # 3. Handle Labels
                window_scr = scr_mean[i:i + n_frames]
                window_lb = y_data[fs][i:i + n_frames] * window_scr[:, None]
                lb_mean = window_lb.mean(axis=0)
                lb_sum = float(lb_mean.sum())
                lb_mean = lb_mean / lb_sum if lb_sum > 1e-6 else lb_mean
                labels_set.append(lb_mean)

    X = np.asarray(feature_set, dtype=np.float32)
    y = np.asarray(labels_set, dtype=np.float32)
    print(f"[INFO] Built sequences: {X.shape[0]} from {csv_path.name} (temporal_drop_ratio={temporal_drop_ratio})")

    stats = {
        "input_rows": in_rows,
        "rows_after_nan_drop": after_nan_rows,
        "num_sequences": int(X.shape[0]),
        "feature_shape": list(X.shape),
        "label_shape": list(y.shape),
        "temporal_drop_ratio": temporal_drop_ratio,
    }
    return X, y, stats

def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    base = cfg_path.parent
    class_names = cfg.get("class_names", [])
    class_count = int(len(class_names))
    if class_count <= 0:
        raise ValueError("class_names must be non-empty")

    proc = cfg.get("processing", {})
    temporal_drop_ratio = float(proc.get("temporal_drop_ratio", 0.0))
    n_frames = int(proc.get("n_frames", 30))  # Set to 36 in your YAML
    skip_frame = int(proc.get("skip_frame", 1))
    smooth_labels_step = int(proc.get("smooth_labels_step", 8))
    label_smoothing_eps = float(proc.get("label_smoothing_eps", 0.1))
    max_frame_gap = int(proc.get("max_frame_gap", 10))
    overwrite = bool(proc.get("overwrite", False))

    datasets = cfg.get("datasets", [])
    if not datasets:
        raise ValueError("datasets list is empty in config")

    summary: dict = {
        "config_path": str(cfg_path),
        "class_names": class_names,
        "processing": {
            "n_frames": n_frames,
            "skip_frame": skip_frame,
            "smooth_labels_step": smooth_labels_step,
            "label_smoothing_eps": label_smoothing_eps,
            "max_frame_gap": max_frame_gap,
            "temporal_drop_ratio": temporal_drop_ratio,
            "overwrite": overwrite,
        },
        "datasets": {},
    }

    for item in datasets:
        name = str(item["name"])
        csv_path = resolve_path(base, str(item["csv_path"]))
        pkl_path = resolve_path(base, str(item["pkl_path"]))

        if (not overwrite) and pkl_path.exists():
            print(f"[SKIP] {name}: {pkl_path} already exists")
            summary["datasets"][name] = {"csv_path": str(csv_path), "pkl_path": str(pkl_path), "status": "skipped_exists"}
            continue

        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV for {name}: {csv_path}")

        X, y, stats = process_csv(
            csv_path=csv_path,
            class_count=class_count,
            n_frames=n_frames,
            skip_frame=skip_frame,
            smooth_labels_step=smooth_labels_step,
            label_smoothing_eps=label_smoothing_eps,
            max_frame_gap=max_frame_gap,
            temporal_drop_ratio=temporal_drop_ratio,
        )

        pkl_path.parent.mkdir(parents=True, exist_ok=True)
        with pkl_path.open("wb") as f:
            pickle.dump((X, y), f)

        summary["datasets"][name] = {
            "csv_path": str(csv_path),
            "pkl_path": str(pkl_path),
            "status": "ok",
            **stats,
        }
        print(f"[OK] Saved: {pkl_path}")

    summary_path = resolve_path(base, str(cfg.get("io", {}).get("summary_yaml", "build_pkl_summary.yaml")))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False, allow_unicode=False)

    print(f"[DONE] Summary: {summary_path}")

if __name__ == "__main__":
    main()