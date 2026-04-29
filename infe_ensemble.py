#!/usr/bin/env python3
"""
CTR-GCN Multi-Stream Ensemble Inference
3 streams: joint + bone + velocity

Usage:
    python infe_ensemble.py --config infe_ensemble.yaml --input video.mp4
    python infe_ensemble.py --config infe_ensemble.yaml --input ./videos/ --output-dir ./results
    python infe_ensemble.py --config infe_ensemble.yaml --input video.mp4 --joint-weights runs-44.pt
    python infe_ensemble.py --config infe_ensemble.yaml --input video.mp4 --no-vel
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import OrderedDict, deque
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpeg", ".mpg"}

SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# 0-indexed (child, parent) — direct conversion of coco17_pairs in feeders/bone_pairs.py
# Feeder iterates pairs and sets bone[v1-1] = joint[v1-1] - joint[v2-1]; same here.
# Note: joint 0 appears twice → second assignment (0,2) wins, matching feeder behaviour.
# Joints 9, 10, 15, 16 never appear as child → their bone stays 0, matching feeder zeros_like.
COCO17_BONE_PAIRS = [
    (0, 1),  (0, 2),   # nose (assigned twice, second wins)
    (1, 3),  (2, 4),   # eyes → ears
    (3, 5),  (4, 6),   # ears → shoulders
    (5, 6),            # left shoulder → right shoulder
    (5, 7),  (7, 9),   # left shoulder → elbow → wrist (wrist stays 0)
    (6, 8),  (8, 10),  # right shoulder → elbow → wrist (wrist stays 0)
    (5, 11), (6, 12),  # shoulders → hips
    (11, 12),          # left hip → right hip
    (11, 13), (13, 15), # left hip → knee → ankle (ankle stays 0)
    (12, 14), (14, 16), # right hip → knee → ankle (ankle stays 0)
]

TRACK_COLORS = [
    (255, 80,  80 ), (80,  255, 80 ), (80,  180, 255),
    (255, 200, 50 ), (200, 80,  255), (80,  255, 220),
    (255, 120, 200), (140, 255, 80 ),
]
CORE_KPT = [5, 6, 11, 12, 13, 14]


# -----------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------
def import_class(import_str: str):
    mod_str, _, class_str = import_str.rpartition(".")
    __import__(mod_str)
    return getattr(sys.modules[mod_str], class_str)


def normalize_pose_sequence(kps_seq: np.ndarray) -> np.ndarray:
    """Robust root-centred normalisation — mirrors build_pkl.py + infe_v4.py."""
    out = kps_seq.copy().astype(np.float32)
    valid_mask = (out[:, 11, 2] > 0.10) & (out[:, 12, 2] > 0.10)
    if int(valid_mask.sum()) >= 3:
        valid_frames = out[valid_mask]
        root_x = float(np.median(valid_frames[:, [11, 12], 0]))
        root_y = float(np.median(valid_frames[:, [11, 12], 1]))
    else:
        root_x = (out[0, 11, 0] + out[0, 12, 0]) / 2.0
        root_y = (out[0, 11, 1] + out[0, 12, 1]) / 2.0
    out[:, :, 0] -= root_x
    out[:, :, 1] -= root_y
    return out


def to_bone(seq: np.ndarray) -> np.ndarray:
    """seq: (T, V, 3) -> bone vectors (T, V, 3)"""
    bone = np.zeros_like(seq)
    for child, parent in COCO17_BONE_PAIRS:
        bone[:, child] = seq[:, child] - seq[:, parent]
    return bone


def to_vel(seq: np.ndarray) -> np.ndarray:
    """seq: (T, V, 3) -> frame-difference velocity (T, V, 3)"""
    vel = np.zeros_like(seq)
    vel[:-1] = seq[1:] - seq[:-1]
    return vel


def to_tensor(seq: np.ndarray, device: torch.device) -> torch.Tensor:
    """(T, V, 3) -> (1, 3, T, V, 1)"""
    x = seq.transpose(2, 0, 1)[None, ..., None]
    return torch.from_numpy(x).float().to(device)


def is_valid_kpt(kps: np.ndarray, thresh: float = 0.05) -> bool:
    return all(kps[k, 2] >= thresh for k in CORE_KPT)


# -----------------------------------------------------------------
# Drawing
# -----------------------------------------------------------------
def get_track_color(tid: int) -> tuple:
    return TRACK_COLORS[tid % len(TRACK_COLORS)]


def draw_skeleton(frame, kpts_norm, orig_w, orig_h, color, thresh=0.2):
    kpts = kpts_norm.copy()
    kpts[:, 0] *= orig_w
    kpts[:, 1] *= orig_h
    for i, j in SKELETON_PAIRS:
        if kpts[i, 2] > thresh and kpts[j, 2] > thresh:
            cv2.line(frame,
                     (int(kpts[i, 0]), int(kpts[i, 1])),
                     (int(kpts[j, 0]), int(kpts[j, 1])),
                     color, 2, cv2.LINE_AA)
    for p in kpts:
        if p[2] > thresh:
            cv2.circle(frame, (int(p[0]), int(p[1])), 4, color, -1)


def draw_label_box(frame, bbox_xyxy, label, color, conf):
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf:.0%}"
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    bg_y1 = max(y1 - th - bl - 6, 0)
    cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, text, (x1 + 4, y1 - bl - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)


# -----------------------------------------------------------------
# Model loading + prediction
# -----------------------------------------------------------------
def load_model(cfg: dict, weights_path: str, device: torch.device):
    Model = import_class(cfg.get("model", "model.ctrgcn.Model"))
    model = Model(**cfg.get("model_args", {}))
    state = torch.load(weights_path, map_location=device)
    state = OrderedDict([[k.replace("module.", ""), v] for k, v in state.items()])
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def ensemble_predict(
    window: np.ndarray,
    models: dict,
    alphas: dict,
    device: torch.device,
) -> np.ndarray:
    """
    window : (T, V, 3)  raw [0,1] keypoints (not yet root-centred)
    returns: probs (num_class,)
    """
    seq = normalize_pose_sequence(window)

    inputs = {
        "joint": to_tensor(seq,         device),
        "bone":  to_tensor(to_bone(seq), device),
        "vel":   to_tensor(to_vel(seq),  device),
    }

    total = None
    weight_sum = 0.0
    with torch.no_grad():
        for name, model in models.items():
            logits = model(inputs[name])
            probs  = torch.softmax(logits, dim=1)
            w      = float(alphas.get(name, 1.0))
            weight_sum += w
            total  = w * probs if total is None else total + w * probs

    return (total / weight_sum).cpu().numpy()[0]


def apply_bbox_rule(probs: np.ndarray, bbox_xyxy, class_names: list, rules: dict) -> int:
    """Reject 'lying' prediction when bbox aspect ratio is tall (standing person)."""
    pred_id = int(np.argmax(probs))
    if not rules.get("enable", False):
        return pred_id

    pred_name = class_names[pred_id]
    if pred_name not in rules.get("lying_classes", []):
        return pred_id

    x1, y1, x2, y2 = bbox_xyxy
    h, w = float(y2 - y1), float(x2 - x1)
    if w > 0 and h / w > float(rules.get("min_h_to_w_ratio", 1.3)):
        sorted_ids = np.argsort(probs)[::-1]
        return int(sorted_ids[1])

    return pred_id


# -----------------------------------------------------------------
# Per-track state
# -----------------------------------------------------------------
class TrackState:
    def __init__(self, window_size: int):
        self.kpt_buffer: deque[np.ndarray] = deque(maxlen=window_size)
        self.label: str = "..."
        self.conf: float = 0.0


# -----------------------------------------------------------------
# Single-video processing
# -----------------------------------------------------------------
def process_video(
    input_path: str,
    output_path: str,
    models: dict,
    yolo,
    cfg: dict,
    device: torch.device,
    show: bool,
) -> None:
    window_size  = int(cfg.get("window_size", 36))
    class_names  = cfg["class_names"]
    yolo_imgsz   = int(cfg.get("yolo_imgsz", 640))
    min_kpt_conf = float(cfg.get("min_kpt_conf", 0.05))
    tracker_cfg  = cfg.get("tracker", "bytetrack.yaml")
    alphas       = cfg.get("alpha", {"joint": 1.0, "bone": 1.0, "vel": 0.5})
    bbox_rules   = cfg.get("bbox_rules", {"enable": False})

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open: {input_path}")
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))

    tracks: dict[int, TrackState] = {}
    frame_idx = 0
    t0 = time.time()

    print(f"  Input  : {input_path}")
    print(f"  Output : {output_path}")
    print(f"  Size   : {orig_w}x{orig_h}  FPS:{fps:.1f}  Frames:{total}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = yolo.track(frame, imgsz=yolo_imgsz, persist=True,
                             tracker=tracker_cfg, verbose=False)
        result = results[0]
        active_ids: set[int] = set()

        if result.boxes is not None and result.keypoints is not None:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            track_ids  = result.boxes.id
            kpts_data  = result.keypoints.data.cpu().numpy()  # (P, 17, 3)

            if track_ids is not None:
                track_ids = track_ids.int().cpu().numpy()

                for pidx in range(len(boxes_xyxy)):
                    tid = int(track_ids[pidx])
                    active_ids.add(tid)

                    kpts_norm = kpts_data[pidx].copy().astype(np.float32)
                    kpts_norm[:, 0] /= orig_w
                    kpts_norm[:, 1] /= orig_h
                    if not is_valid_kpt(kpts_norm, min_kpt_conf):
                        kpts_norm[:] = 0.0

                    if tid not in tracks:
                        tracks[tid] = TrackState(window_size)
                    ts = tracks[tid]
                    ts.kpt_buffer.append(kpts_norm)

                    if len(ts.kpt_buffer) == window_size:
                        window = np.stack(list(ts.kpt_buffer), axis=0)
                        probs  = ensemble_predict(window, models, alphas, device)
                        pred_id = apply_bbox_rule(probs, boxes_xyxy[pidx],
                                                  class_names, bbox_rules)
                        ts.label = class_names[pred_id]
                        ts.conf  = float(probs[pred_id])

                    color = get_track_color(tid)
                    draw_skeleton(frame, kpts_norm, orig_w, orig_h, color)
                    draw_label_box(frame, boxes_xyxy[pidx],
                                   f"#{tid} {ts.label}", color, ts.conf)

        for k in [k for k in tracks if k not in active_ids]:
            del tracks[k]

        writer.write(frame)
        if show:
            cv2.imshow("CTR-GCN Ensemble", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_idx % 100 == 0:
            elapsed = time.time() - t0
            pct = 100 * frame_idx / total if total > 0 else 0
            print(f"    frame {frame_idx}/{total} ({pct:.1f}%)  "
                  f"{frame_idx / elapsed:.1f} fps")

    cap.release()
    writer.release()
    elapsed = time.time() - t0
    print(f"  Done  : {frame_idx} frames in {elapsed:.1f}s "
          f"({frame_idx / elapsed:.1f} fps avg)\n")


# -----------------------------------------------------------------
# Job resolution
# -----------------------------------------------------------------
def resolve_jobs(input_val: str, output_val: str, output_dir_val: str) -> list[tuple[str, str]]:
    inp = Path(input_val)

    def make_out(src: Path, out_dir: Path) -> str:
        return str(out_dir / (src.stem + "_ensemble.mp4"))

    if inp.is_file():
        if output_val:
            return [(str(inp), output_val)]
        out_dir = Path(output_dir_val) if output_dir_val else inp.parent
        return [(str(inp), make_out(inp, out_dir))]

    if inp.is_dir():
        out_dir = Path(output_dir_val) if output_dir_val else inp / "annotated"
        videos = sorted(p for p in inp.iterdir()
                        if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
        if not videos:
            raise FileNotFoundError(f"No video files found in: {inp}")
        return [(str(v), make_out(v, out_dir)) for v in videos]

    raise FileNotFoundError(f"Input not found: {inp}")


# -----------------------------------------------------------------
# CLI
# -----------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CTR-GCN multi-stream ensemble inference")
    p.add_argument("--config",        required=True,       help="Path to infe_ensemble.yaml")
    p.add_argument("--input",         default="",          help="Override input video or folder")
    p.add_argument("--output",        default="",          help="Override output path (single file)")
    p.add_argument("--output-dir",    default="",          dest="output_dir")
    p.add_argument("--device",        default="",          help="Override device: 0 / cpu")
    p.add_argument("--show",          action="store_true", help="Show live preview")
    p.add_argument("--joint-weights", default="",          help="Override joint stream weights")
    p.add_argument("--bone-weights",  default="",          help="Override bone stream weights")
    p.add_argument("--vel-weights",   default="",          help="Override vel stream weights")
    p.add_argument("--no-bone",       action="store_true", help="Disable bone stream")
    p.add_argument("--no-vel",        action="store_true", help="Disable vel stream")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    input_val      = args.input      or str(cfg.get("input", ""))
    output_dir_val = args.output_dir or str(cfg.get("output_dir", ""))
    if not input_val:
        raise ValueError("No input. Set 'input' in config or use --input.")
    jobs = resolve_jobs(input_val, args.output, output_dir_val)

    device_str = args.device or str(cfg.get("device", "0"))
    device = torch.device(f"cuda:{device_str}" if device_str.isdigit() else device_str)

    # Merge CLI weight overrides into config
    weights = dict(cfg.get("weights", {}))
    if args.joint_weights: weights["joint"] = args.joint_weights
    if args.bone_weights:  weights["bone"]  = args.bone_weights
    if args.vel_weights:   weights["vel"]   = args.vel_weights

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Loading models...")

    models: dict = {}
    if weights.get("joint"):
        print(f"  joint : {weights['joint']}")
        models["joint"] = load_model(cfg, weights["joint"], device)
    if weights.get("bone") and not args.no_bone:
        print(f"  bone  : {weights['bone']}")
        models["bone"] = load_model(cfg, weights["bone"], device)
    if weights.get("vel") and not args.no_vel:
        print(f"  vel   : {weights['vel']}")
        models["vel"] = load_model(cfg, weights["vel"], device)

    if not models:
        raise ValueError("No models loaded. Check 'weights' in config.")
    print(f"[INFO] Active streams: {list(models.keys())}\n")

    from ultralytics import YOLO
    yolo = YOLO(cfg.get("yolo_model", "yolo11m-pose.pt"))
    print(f"[INFO] YOLO ready  |  {len(jobs)} job(s)\n")

    for i, (inp, out) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}]")
        process_video(inp, out, models, yolo, cfg, device, show=args.show)

    if args.show:
        cv2.destroyAllWindows()
    print("[ALL DONE]")


if __name__ == "__main__":
    main()
