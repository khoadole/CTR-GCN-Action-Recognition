#!/usr/bin/env python3
"""
CTR-GCN Multi-Stream Ensemble Inference — FAST GPU pipeline.

Optimisations vs infe_ensemble.py:
  * Per-track keypoint buffer kept on GPU as a circular torch.Tensor
  * Multiple tracks batched into a single forward per stream
  * normalize / bone / vel computed with torch on GPU (no numpy round-trip)
  * Configurable predict_stride: skip inference on intermediate frames
  * Optional FP16 autocast on CUDA

Extra config keys (yaml or CLI override):
  predict_stride : int  (default 3)   - run ensemble every N frames
  use_amp        : bool (default true)- enable torch.cuda.amp.autocast

Usage:
    python infe_ensemble_fast.py --config infe_ensemble.yaml
    python infe_ensemble_fast.py --config infe_ensemble.yaml --predict-stride 4 --no-amp
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import OrderedDict
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

COCO17_BONE_PAIRS = [
    (0, 1),  (0, 2),
    (1, 3),  (2, 4),
    (3, 5),  (4, 6),
    (5, 6),
    (5, 7),  (7, 9),
    (6, 8),  (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

TRACK_COLORS = [
    (255, 80,  80 ), (80,  255, 80 ), (80,  180, 255),
    (255, 200, 50 ), (200, 80,  255), (80,  255, 220),
    (255, 120, 200), (140, 255, 80 ),
]
CORE_KPT = [5, 6, 11, 12, 13, 14]
NUM_KPT = 17


# -----------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------
def import_class(import_str: str):
    mod_str, _, class_str = import_str.rpartition(".")
    __import__(mod_str)
    return getattr(sys.modules[mod_str], class_str)


# -----------------------------------------------------------------
# GPU pose ops — all operate on (B, T, V, 3) tensors
# -----------------------------------------------------------------
def normalize_batch_gpu(seq: torch.Tensor, conf_thr: float = 0.10) -> torch.Tensor:
    """Per-sample root-centred normalisation using median hip across valid frames."""
    out = seq.clone()
    valid = (out[:, :, 11, 2] > conf_thr) & (out[:, :, 12, 2] > conf_thr)  # (B, T)
    valid_count = valid.sum(dim=1).tolist()  # one sync, B values

    for b in range(out.shape[0]):
        if valid_count[b] >= 3:
            frames = out[b][valid[b]]                          # (N, V, 3)
            hip_xy = frames[:, [11, 12], :2].reshape(-1, 2)    # (2N, 2)
            root = hip_xy.median(dim=0).values                 # (2,)
        else:
            root = (out[b, 0, 11, :2] + out[b, 0, 12, :2]) / 2.0
        out[b, :, :, 0] -= root[0]
        out[b, :, :, 1] -= root[1]
    return out


def to_bone_batch_gpu(seq: torch.Tensor) -> torch.Tensor:
    """Bone vectors with COCO-17 last-write-wins semantics (matches feeder)."""
    bone = torch.zeros_like(seq)
    for child, parent in COCO17_BONE_PAIRS:
        bone[:, :, child, :] = seq[:, :, child, :] - seq[:, :, parent, :]
    return bone


def to_vel_batch_gpu(seq: torch.Tensor) -> torch.Tensor:
    """Frame-to-frame velocity; last frame stays zero."""
    vel = torch.zeros_like(seq)
    vel[:, :-1] = seq[:, 1:] - seq[:, :-1]
    return vel


def seq_to_model_input(seq: torch.Tensor) -> torch.Tensor:
    """(B, T, V, 3) → (B, 3, T, V, 1) contiguous."""
    return seq.permute(0, 3, 1, 2).unsqueeze(-1).contiguous()


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
# Model loading + batched ensemble prediction
# -----------------------------------------------------------------
def load_model(cfg: dict, weights_path: str, device: torch.device):
    Model = import_class(cfg.get("model", "model.ctrgcn.Model"))
    model = Model(**cfg.get("model_args", {}))
    state = torch.load(weights_path, map_location=device)
    state = OrderedDict([[k.replace("module.", ""), v] for k, v in state.items()])
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def ensemble_predict_batch(
    seq_batch: torch.Tensor,
    models: dict,
    alphas: dict,
    use_amp: bool,
) -> torch.Tensor:
    """
    seq_batch : (B, T, V, 3) raw [0,1] keypoints, NOT yet root-centred.
    Returns probs (B, num_class) properly normalised.
    """
    seq = normalize_batch_gpu(seq_batch)

    streams = {}
    if "joint" in models:
        streams["joint"] = seq_to_model_input(seq)
    if "bone" in models:
        streams["bone"] = seq_to_model_input(to_bone_batch_gpu(seq))
    if "vel" in models:
        streams["vel"] = seq_to_model_input(to_vel_batch_gpu(seq))

    total = None
    weight_sum = 0.0
    amp_ctx = torch.cuda.amp.autocast(enabled=use_amp)
    with torch.no_grad(), amp_ctx:
        for name, model in models.items():
            logits = model(streams[name])
            probs  = torch.softmax(logits.float(), dim=1)
            w      = float(alphas.get(name, 1.0))
            weight_sum += w
            total  = w * probs if total is None else total + w * probs

    return total / weight_sum


def apply_bbox_rule(probs: np.ndarray, bbox_xyxy, class_names: list, rules: dict) -> int:
    pred_id = int(np.argmax(probs))
    if not rules.get("enable", False):
        return pred_id
    if class_names[pred_id] not in rules.get("lying_classes", []):
        return pred_id
    x1, y1, x2, y2 = bbox_xyxy
    h, w = float(y2 - y1), float(x2 - x1)
    if w > 0 and h / w > float(rules.get("min_h_to_w_ratio", 1.3)):
        return int(np.argsort(probs)[::-1][1])
    return pred_id


# -----------------------------------------------------------------
# GPU-resident circular buffer per track
# -----------------------------------------------------------------
class TrackState:
    def __init__(self, window_size: int, device: torch.device):
        self.buffer = torch.zeros(window_size, NUM_KPT, 3, device=device)
        self.size   = window_size
        self.write  = 0
        self.filled = False
        self.label  = "..."
        self.conf   = 0.0

    def append(self, kpts: torch.Tensor) -> None:
        """kpts : (V, 3) on same device as buffer."""
        self.buffer[self.write].copy_(kpts)
        self.write += 1
        if self.write >= self.size:
            self.write = 0
            self.filled = True

    def get_window(self) -> torch.Tensor:
        """Returns (T, V, 3) in chronological order."""
        if self.write == 0:
            return self.buffer
        return torch.roll(self.buffer, shifts=-self.write, dims=0)


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
    window_size    = int(cfg.get("window_size", 36))
    class_names    = cfg["class_names"]
    yolo_imgsz     = int(cfg.get("yolo_imgsz", 640))
    min_kpt_conf   = float(cfg.get("min_kpt_conf", 0.05))
    min_valid_kpts = int(cfg.get("min_valid_kpts", 0))
    tracker_cfg    = cfg.get("tracker", "bytetrack.yaml")
    alphas         = cfg.get("alpha", {"joint": 1.0, "bone": 1.0, "vel": 0.5})
    bbox_rules     = cfg.get("bbox_rules", {"enable": False})
    predict_stride = int(cfg.get("predict_stride", 3))
    use_amp        = bool(cfg.get("use_amp", True)) and device.type == "cuda"

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
    print(f"  Stride : {predict_stride}  AMP:{use_amp}  Streams:{list(models.keys())}")
    print(f"  Gate   : min_kpt_conf={min_kpt_conf}  min_valid_kpts={min_valid_kpts}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = yolo.track(frame, imgsz=yolo_imgsz, persist=True,
                             tracker=tracker_cfg, verbose=False)
        result = results[0]
        active_ids: set[int] = set()

        boxes_xyxy_cpu = None
        track_ids_cpu  = None
        kpts_norm_cpu  = None

        if (result.boxes is not None
                and result.keypoints is not None
                and result.boxes.id is not None):
            boxes_xyxy_cpu = result.boxes.xyxy.cpu().numpy()
            track_ids_cpu  = result.boxes.id.int().cpu().numpy()

            kpts_gpu = result.keypoints.data.to(device).float().clone()
            kpts_gpu[..., 0] /= orig_w
            kpts_gpu[..., 1] /= orig_h

            # Per-person gates
            n_valid_kpts = (kpts_gpu[..., 2] >= min_kpt_conf).sum(dim=1)        # (P,)
            core_invalid = (kpts_gpu[:, CORE_KPT, 2] < min_kpt_conf).any(dim=1) # (P,)
            if min_valid_kpts > 0:
                skip_mask = n_valid_kpts < min_valid_kpts
            else:
                skip_mask = torch.zeros_like(core_invalid)

            # Buffer-only copy: zero out persons that pass skip gate but have bad core kpts
            zero_mask = core_invalid & ~skip_mask
            kpts_for_buf = kpts_gpu.clone()
            if zero_mask.any():
                kpts_for_buf[zero_mask] = 0.0

            skip_cpu = skip_mask.cpu().numpy()

            for pidx in range(boxes_xyxy_cpu.shape[0]):
                tid = int(track_ids_cpu[pidx])
                active_ids.add(tid)
                if tid not in tracks:
                    tracks[tid] = TrackState(window_size, device)
                if not skip_cpu[pidx]:
                    tracks[tid].append(kpts_for_buf[pidx])

            # Single CPU transfer for drawing only
            kpts_norm_cpu = kpts_gpu.cpu().numpy()

        # ---- Batched ensemble inference at stride boundaries ----
        if frame_idx % predict_stride == 0 and tracks:
            ready_tids = [tid for tid in active_ids
                          if tid in tracks and tracks[tid].filled]
            if ready_tids:
                windows = torch.stack(
                    [tracks[tid].get_window() for tid in ready_tids], dim=0
                )  # (B, T, V, 3)
                probs = ensemble_predict_batch(windows, models, alphas, use_amp)
                probs_cpu = probs.cpu().numpy()

                tid_to_pidx = {int(track_ids_cpu[i]): i
                               for i in range(len(track_ids_cpu))}
                for i, tid in enumerate(ready_tids):
                    pidx = tid_to_pidx.get(tid)
                    if pidx is None:
                        pred_id = int(np.argmax(probs_cpu[i]))
                    else:
                        pred_id = apply_bbox_rule(
                            probs_cpu[i], boxes_xyxy_cpu[pidx],
                            class_names, bbox_rules,
                        )
                    tracks[tid].label = class_names[pred_id]
                    tracks[tid].conf  = float(probs_cpu[i, pred_id])

        # ---- Draw ----
        if boxes_xyxy_cpu is not None and kpts_norm_cpu is not None:
            for pidx in range(boxes_xyxy_cpu.shape[0]):
                tid = int(track_ids_cpu[pidx])
                ts  = tracks.get(tid)
                if ts is None:
                    continue
                color = get_track_color(tid)
                draw_skeleton(frame, kpts_norm_cpu[pidx], orig_w, orig_h, color)
                draw_label_box(frame, boxes_xyxy_cpu[pidx],
                               f"#{tid} {ts.label}", color, ts.conf)

        for k in [k for k in tracks if k not in active_ids]:
            del tracks[k]

        writer.write(frame)
        if show:
            cv2.imshow("CTR-GCN Ensemble Fast", frame)
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
        return str(out_dir / (src.stem + "_ensemble_fast.mp4"))

    if inp.is_file():
        if output_val:
            return [(str(inp), output_val)]
        out_dir = Path(output_dir_val) if output_dir_val else inp.parent
        return [(str(inp), make_out(inp, out_dir))]

    if inp.is_dir():
        out_dir = Path(output_dir_val) if output_dir_val else inp / "annotated"
        videos = sorted(p for p in inp.rglob("*")
                        if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
        if not videos:
            raise FileNotFoundError(f"No video files found in: {inp}")
        jobs = []
        for v in videos:
            rel = v.relative_to(inp)
            jobs.append((str(v), str(out_dir / rel.parent / (v.stem + "_ensemble_fast.mp4"))))
        return jobs

    raise FileNotFoundError(f"Input not found: {inp}")


# -----------------------------------------------------------------
# CLI
# -----------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CTR-GCN ensemble inference (fast GPU)")
    p.add_argument("--config",          required=True)
    p.add_argument("--input",           default="")
    p.add_argument("--output",          default="")
    p.add_argument("--output-dir",      default="", dest="output_dir")
    p.add_argument("--device",          default="")
    p.add_argument("--show",            action="store_true")
    p.add_argument("--joint-weights",   default="")
    p.add_argument("--bone-weights",    default="")
    p.add_argument("--vel-weights",     default="")
    p.add_argument("--no-bone",         action="store_true")
    p.add_argument("--no-vel",          action="store_true")
    p.add_argument("--predict-stride",  default=None, type=int, dest="predict_stride")
    p.add_argument("--no-amp",          action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if args.predict_stride is not None:
        cfg["predict_stride"] = args.predict_stride
    if args.no_amp:
        cfg["use_amp"] = False

    input_val      = args.input      or str(cfg.get("input", ""))
    output_dir_val = args.output_dir or str(cfg.get("output_dir", ""))
    if not input_val:
        raise ValueError("No input. Set 'input' in config or use --input.")
    jobs = resolve_jobs(input_val, args.output, output_dir_val)

    device_str = args.device or str(cfg.get("device", "0"))
    device = torch.device(f"cuda:{device_str}" if device_str.isdigit() else device_str)

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
