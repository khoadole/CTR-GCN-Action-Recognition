#!/usr/bin/env python3
"""
CTR-GCN Inference: Video → Annotated Video (multi-person, YOLO ByteTrack)

Usage:
    # All from config (input + output_dir defined in yaml)
    python infer.py --config infer_config.yaml

    # Override input/output at runtime
    python infer.py --config infer_config.yaml --input video.mp4 --output-dir ./results
    python infer.py --config infer_config.yaml --input ./videos/  --output-dir ./results
    python infer.py --config infer_config.yaml --input video.mp4  --output out.mp4

    # Override weights / device
    python infer.py --config infer_config.yaml --weights runs-65-12345.pt --device 0

Pipeline per frame:
    1. YOLO11m-pose + ByteTrack  -> per-person bbox + 17 keypoints
    2. Per-track: normalise keypoints (root-centred hip, same as training)
    3. Rolling window buffer (T=window_size) -> CTR-GCN forward pass
    4. Draw skeleton, bbox, class label + confidence on frame
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml


# -----------------------------------------------------------------
# Constants
# -----------------------------------------------------------------
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpeg", ".mpg"}

SKELETON_PAIRS = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]

TRACK_COLORS = [
    (255, 80,  80 ), (80,  255, 80 ), (80,  180, 255),
    (255, 200, 50 ), (200, 80,  255), (80,  255, 220),
    (255, 120, 200), (140, 255, 80 ),
]

CORE_KPT = [5, 6, 11, 12, 13, 14]   # shoulders + hips + knees


# -----------------------------------------------------------------
# CLI
# -----------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CTR-GCN video inference")
    parser.add_argument("--config",     required=True, help="Path to infer_config.yaml")
    parser.add_argument("--input",      default="",    help="Override input: video file or folder")
    parser.add_argument("--output",     default="",    help="Override output: exact path (single-file only)")
    parser.add_argument("--output-dir", default="",    dest="output_dir",
                        help="Override output_dir from config")
    parser.add_argument("--weights",    default="",    help="Override weights path from config")
    parser.add_argument("--device",     default="",    help="Override device: 0 / cpu")
    parser.add_argument("--show",       action="store_true", help="Show live preview (cv2.imshow)")
    return parser.parse_args()


# -----------------------------------------------------------------
# Dynamic model import
# -----------------------------------------------------------------
def import_class(import_str: str):
    mod_str, _, class_str = import_str.rpartition(".")
    __import__(mod_str)
    import sys
    return getattr(sys.modules[mod_str], class_str)


# -----------------------------------------------------------------
# Keypoint normalisation  (mirrors build_pkl.py exactly)
# -----------------------------------------------------------------
# def normalize_pose_sequence(kps_seq: np.ndarray) -> np.ndarray:
#     """
#     Root-centred normalisation identical to training.
#     kps_seq : (T, V, 3)  x,y in [0,1], conf in [0,1]
#     Returns  : (T, V, 3)  x,y shifted so hip-midpoint@t=0 is origin
#     """
#     out = kps_seq.copy().astype(np.float32)
#     root_x = (out[0, 11, 0] + out[0, 12, 0]) / 2.0
#     root_y = (out[0, 11, 1] + out[0, 12, 1]) / 2.0
#     out[:, :, 0] -= root_x
#     out[:, :, 1] -= root_y
#     return out

def normalize_pose_sequence(kps_seq: np.ndarray) -> np.ndarray:
    """
    Robust root-centered normalization.
    Uses median hip position from all VALID frames in the window (much more stable than using only frame 0).
    """
    out = kps_seq.copy().astype(np.float32)
    
    # Frames where BOTH hips have decent confidence
    valid_mask = (out[:, 11, 2] > 0.10) & (out[:, 12, 2] > 0.10)
    valid_count = int(valid_mask.sum())
    
    if valid_count >= 3:
        # Correct indexing
        valid_frames = out[valid_mask]                    # (N_valid, 17, 3)
        hip_x = valid_frames[:, [11, 12], 0]              # (N_valid, 2)
        hip_y = valid_frames[:, [11, 12], 1]
        root_x = float(np.median(hip_x))
        root_y = float(np.median(hip_y))
    else:
        # Fallback: use first frame (original behavior)
        root_x = (out[0, 11, 0] + out[0, 12, 0]) / 2.0
        root_y = (out[0, 11, 1] + out[0, 12, 1]) / 2.0
    
    # Apply shift
    out[:, :, 0] -= root_x
    out[:, :, 1] -= root_y
    return out


def is_valid_kpt(kps: np.ndarray, thresh: float = 0.05) -> bool:
    return all(kps[k, 2] >= thresh for k in CORE_KPT)


# -----------------------------------------------------------------
# CTR-GCN input tensor  (1, C, T, V, M)
# -----------------------------------------------------------------
def build_ctrgcn_input(window: np.ndarray, device: torch.device) -> torch.Tensor:
    """window: (T, V, 3) -> tensor (1, 3, T, V, 1)"""
    seq = normalize_pose_sequence(window)      # (T, V, 3)
    x = seq.transpose(2, 0, 1)               # (3, T, V)
    x = x[np.newaxis, :, :, :, np.newaxis]   # (1, 3, T, V, 1)
    return torch.from_numpy(x).float().to(device)


# -----------------------------------------------------------------
# Drawing helpers
# -----------------------------------------------------------------
def get_track_color(track_id: int) -> tuple:
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]


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
# Per-track rolling state
# -----------------------------------------------------------------
class TrackState:
    def __init__(self, window_size: int):
        self.kpt_buffer: deque[np.ndarray] = deque(maxlen=window_size)
        self.label: str = "..."
        self.conf: float = 0.0


# -----------------------------------------------------------------
# Single-video inference
# -----------------------------------------------------------------
def process_video(
    input_path: str,
    output_path: str,
    model,
    yolo,
    cfg: dict,
    device: torch.device,
    show: bool,
) -> None:
    window_size    = int(cfg.get("window_size", 36))
    class_names    = cfg["class_names"]
    yolo_imgsz     = int(cfg.get("yolo_imgsz", 640))
    min_kpt_conf   = float(cfg.get("min_kpt_conf", 0.05))
    min_valid_kpts = int(cfg.get("min_valid_kpts", 0))
    tracker_cfg    = cfg.get("tracker", "bytetrack.yaml")

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
    print(f"  Size   : {orig_w}x{orig_h}  FPS: {fps:.1f}  Frames: {total}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = yolo.track(
            frame,
            imgsz=yolo_imgsz,
            persist=True,
            tracker=tracker_cfg,
            verbose=False,
        )
        result = results[0]
        active_ids: set[int] = set()

        if result.boxes is not None and result.keypoints is not None:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            track_ids  = result.boxes.id
            kpts_data  = result.keypoints.data.cpu().numpy()  # (P,17,3) pixel+conf

            if track_ids is not None:
                track_ids = track_ids.int().cpu().numpy()

                for pidx in range(len(boxes_xyxy)):
                    tid = int(track_ids[pidx])
                    active_ids.add(tid)

                    # normalise keypoints to [0, 1]
                    kpts_norm = kpts_data[pidx].copy().astype(np.float32)
                    kpts_norm[:, 0] /= orig_w
                    kpts_norm[:, 1] /= orig_h
                    if min_valid_kpts > 0 and int((kpts_norm[:, 2] >= min_kpt_conf).sum()) < min_valid_kpts:
                        continue
                    if not is_valid_kpt(kpts_norm, min_kpt_conf):
                        continue

                    if tid not in tracks:
                        tracks[tid] = TrackState(window_size)
                    ts = tracks[tid]
                    ts.kpt_buffer.append(kpts_norm)

                    # predict when rolling window is full
                    if len(ts.kpt_buffer) == window_size:
                        window = np.stack(list(ts.kpt_buffer), axis=0)  # (T,17,3)
                        with torch.no_grad():
                            inp    = build_ctrgcn_input(window, device)
                            logits = model(inp)
                            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
                        pred_id  = int(np.argmax(probs))
                        ts.label = class_names[pred_id]
                        ts.conf  = float(probs[pred_id])

                    color = get_track_color(tid)
                    draw_skeleton(frame, kpts_norm, orig_w, orig_h, color)
                    draw_label_box(frame, boxes_xyxy[pidx],
                                   f"#{tid} {ts.label}", color, ts.conf)

        # remove tracks that disappeared this frame
        for k in [k for k in tracks if k not in active_ids]:
            del tracks[k]

        writer.write(frame)

        if show:
            cv2.imshow("CTR-GCN Inference", frame)
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
# Resolve input -> list of (input_path, output_path) jobs
# -----------------------------------------------------------------
def resolve_jobs(
    input_val: str,
    output_val: str,
    output_dir_val: str,
) -> list[tuple[str, str]]:
    """
    Returns list of (input_video_path, output_video_path).

    Rules:
      - input is a FILE + --output given   -> exact output path
      - input is a FILE + output_dir given -> output_dir/<stem>_annotated.mp4
      - input is a FOLDER                  -> all videos -> output_dir/<stem>_annotated.mp4
      - fallback (no output_dir)           -> alongside input file
    """
    inp = Path(input_val)

    def make_out(src: Path, out_dir: Path) -> str:
        return str(out_dir / (src.stem + "_annotated.mp4"))

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
            raise FileNotFoundError(f"No video files found in folder: {inp}")
        return [(str(v), make_out(v, out_dir)) for v in videos]

    raise FileNotFoundError(f"Input not found: {inp}")


# -----------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------
def main() -> None:
    args = parse_args()

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # CLI overrides config values
    input_val      = args.input      or str(cfg.get("input", ""))
    output_val     = args.output     or ""
    output_dir_val = args.output_dir or str(cfg.get("output_dir", ""))

    if not input_val:
        raise ValueError("No input specified. Set 'input' in config or use --input.")

    jobs = resolve_jobs(input_val, output_val, output_dir_val)

    # Weights + device
    weights_path = args.weights or str(cfg.get("weights", ""))
    if not weights_path:
        raise ValueError("No weights specified. Set 'weights' in config or use --weights.")

    device_str = args.device or str(cfg.get("device", "0"))
    device = torch.device(f"cuda:{device_str}" if device_str.isdigit() else device_str)

    print(f"[INFO] Device  : {device}")
    print(f"[INFO] Weights : {weights_path}")
    print(f"[INFO] Jobs    : {len(jobs)} video(s)\n")

    # Load CTR-GCN once
    from collections import OrderedDict
    Model = import_class(cfg.get("model", "model.ctrgcn.Model"))
    model = Model(**cfg.get("model_args", {}))
    state = torch.load(weights_path, map_location=device)
    state = OrderedDict([[k.replace("module.", ""), v] for k, v in state.items()])
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"[INFO] CTR-GCN ready  "
          f"({cfg.get('model_args', {}).get('num_class', '?')} classes)")

    # Load YOLO once
    from ultralytics import YOLO
    yolo = YOLO(cfg.get("yolo_model", "yolo11m-pose.pt"))
    print(f"[INFO] YOLO ready\n")

    # Process all jobs
    for i, (inp, out) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}]")
        process_video(inp, out, model, yolo, cfg, device, show=args.show)

    if args.show:
        cv2.destroyAllWindows()
    print("[ALL DONE]")


if __name__ == "__main__":
    main()