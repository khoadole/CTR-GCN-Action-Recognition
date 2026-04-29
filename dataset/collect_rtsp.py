#!/usr/bin/env python3
"""
Thu thập skeleton từ RTSP (hoặc video file) để tạo training data mới.

Xử lý nhiều người: mỗi track_id (ByteTrack) → buffer riêng → sample riêng.
Hai chế độ label:
  - pseudo    : CTR-GCN tự gán nhãn, chỉ lưu sample có confidence >= PSEUDO_CONF_THRESH
  - unlabeled : lưu hết, gán label=-1, review thủ công sau

Xem output qua browser (SSH-friendly):
  --stream-port 8003  →  http://<server-ip>:8003/

Usage:
    # Thu từ RTSP, pseudo-label, xem qua browser
    python collect_rtsp.py \
        --source rtsp://user:pass@192.168.1.10/stream1 \
        --weights ../work_dir/custom/coco17_8class_v4/ctrgcn_joint/runs-3-8757.pt \
        --cam-id cam_topdown_1 \
        --mode pseudo \
        --out-csv ./rtsp_data/cam1_pseudo.csv \
        --stream-port 8003

    # Thu từ video file, pseudo-label
    python collect_rtsp.py \
        --source /home/cxviewlab2/data/khoa.do/CTR-GCN/sources/input/hard_case/truoc_xuong_3_03.avi \
        --mode pseudo \
        --out-csv ./rtsp_data/cam1_pseudo.csv \
        --stream-port 8003
"""

from __future__ import annotations

import argparse
import csv
import os
import time
import threading
from collections import deque, defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import cv2
import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS    = "../work_dir/custom/coco17_8class_v4/ctrgcn_joint/runs-3-8757.pt"
DEFAULT_YOLO       = "../dataset/yolo11m-pose.pt"
WINDOW_SIZE        = 36
WINDOW_STRIDE      = 18          # thu window mỗi 18 frame (50% overlap)
PSEUDO_CONF_THRESH = 0.70
MIN_KPT_CONF       = 0.05
CORE_KPT           = [5, 6, 11, 12, 13, 14]

CLASS_NAMES = [
    "Sit down", "Lying Down", "Walking", "Stand up",
    "Standing", "Fall Down", "Sitting", "Other",
]

TRACK_COLORS = [
    (255, 80, 80), (80, 255, 80), (80, 180, 255), (255, 200, 50),
    (200, 80, 255), (80, 255, 220), (255, 120, 200), (140, 255, 80),
]

SKELETON_PAIRS = [
    (0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),
]

KPT_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle",
]
ALL_COLS = (
    ["video", "frame"]
    + [f"{n}_{c}" for n in KPT_NAMES for c in ("x","y","s")]
    + ["label", "pseudo_conf"]
)


# ─────────────────────────────────────────────────────────────────
# MJPEG streaming server (SSH-friendly browser preview)
# ─────────────────────────────────────────────────────────────────
class MjpegServer:
    """
    Server MJPEG đơn giản, chạy trong thread riêng.
    Browser mở http://<ip>:<port>/ là xem được stream ngay.
    """

    _HTML = b"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>collect_rtsp preview</title>
  <style>
    body { margin:0; background:#111; display:flex; flex-direction:column;
           align-items:center; font-family:monospace; color:#eee; }
    img  { max-width:100%; max-height:90vh; margin-top:8px; }
    #hud { padding:6px 12px; background:#222; width:100%; box-sizing:border-box; font-size:13px; }
  </style>
</head>
<body>
  <div id="hud">CTR-GCN collect_rtsp &mdash; MJPEG live preview</div>
  <img src="/stream" alt="stream">
</body>
</html>"""

    def __init__(self, port: int, jpeg_quality: int = 75):
        self._port    = port
        self._quality = jpeg_quality
        self._lock    = threading.Lock()
        self._jpeg    = b""          # frame hiện tại dạng JPEG bytes
        self._server  = None

    def push(self, frame_bgr: np.ndarray) -> None:
        ok, buf = cv2.imencode(
            ".jpg", frame_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, self._quality],
        )
        if ok:
            with self._lock:
                self._jpeg = buf.tobytes()

    def _get_jpeg(self) -> bytes:
        with self._lock:
            return self._jpeg

    def start(self) -> None:
        server_ref = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):   # tắt access log
                pass

            def do_GET(self):
                if self.path == "/":
                    body = server_ref._HTML
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header(
                        "Content-Type",
                        "multipart/x-mixed-replace; boundary=mjpegframe",
                    )
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    try:
                        while True:
                            jpeg = server_ref._get_jpeg()
                            if jpeg:
                                chunk = (
                                    b"--mjpegframe\r\n"
                                    b"Content-Type: image/jpeg\r\n"
                                    b"\r\n"
                                    + jpeg
                                    + b"\r\n"
                                )
                                self.wfile.write(chunk)
                                self.wfile.flush()
                            time.sleep(0.033)   # ~30 fps cap
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass

                else:
                    self.send_response(404)
                    self.end_headers()

        httpd = HTTPServer(("0.0.0.0", self._port), _Handler)
        httpd.allow_reuse_address = True
        self._server = httpd
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        print(f"[INFO] MJPEG stream  → http://0.0.0.0:{self._port}/")
        print(f"[INFO]   trên mạng   → http://100.77.54.40:{self._port}/  (thay IP nếu khác)")

    def stop(self):
        if self._server:
            self._server.shutdown()


# ─────────────────────────────────────────────────────────────────
# Keypoint / pose helpers
# ─────────────────────────────────────────────────────────────────
def is_valid_kpt(kpts: np.ndarray, thresh: float = MIN_KPT_CONF) -> bool:
    return all(kpts[k, 2] >= thresh for k in CORE_KPT)


def normalize_pose_sequence(kps_seq: np.ndarray) -> np.ndarray:
    """Root-centred norm — giống hệt infe_v4.py và build_pkl.py."""
    out   = kps_seq.copy().astype(np.float32)
    valid = (out[:, 11, 2] > 0.10) & (out[:, 12, 2] > 0.10)
    if int(valid.sum()) >= 3:
        vf     = out[valid]
        root_x = float(np.median(vf[:, [11, 12], 0]))
        root_y = float(np.median(vf[:, [11, 12], 1]))
    else:
        root_x = (out[0, 11, 0] + out[0, 12, 0]) / 2.0
        root_y = (out[0, 11, 1] + out[0, 12, 1]) / 2.0
    out[:, :, 0] -= root_x
    out[:, :, 1] -= root_y
    return out


def build_model_input(window: np.ndarray, device: torch.device) -> torch.Tensor:
    seq = normalize_pose_sequence(window)
    x   = seq.transpose(2, 0, 1)[np.newaxis, :, :, :, np.newaxis]
    return torch.from_numpy(x).float().to(device)


# ─────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────
def draw_skeleton(frame, kpts_norm, W, H, color):
    k = kpts_norm.copy()
    k[:, 0] *= W
    k[:, 1] *= H
    for i, j in SKELETON_PAIRS:
        if k[i, 2] > 0.2 and k[j, 2] > 0.2:
            cv2.line(frame,
                     (int(k[i, 0]), int(k[i, 1])),
                     (int(k[j, 0]), int(k[j, 1])),
                     color, 2, cv2.LINE_AA)
    for p in k:
        if p[2] > 0.2:
            cv2.circle(frame, (int(p[0]), int(p[1])), 4, color, -1)


def draw_box(frame, bbox, tid, label, conf, color):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"#{tid} {label} {conf:.0%}"
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    bg_y1 = max(y1 - th - bl - 6, 0)
    cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, text, (x1 + 4, y1 - bl - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)


def draw_hud(frame, frame_idx, n_tracks, saved, skipped, rows):
    lines = [
        f"Frame {frame_idx}   Tracks: {n_tracks}",
        f"Saved windows: {saved}   LowConf skipped: {skipped}   CSV rows: {rows}",
    ]
    y = 24
    for line in lines:
        cv2.putText(frame, line, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22


# ─────────────────────────────────────────────────────────────────
# Per-track state
# ─────────────────────────────────────────────────────────────────
class TrackState:
    def __init__(self, track_id: int, cam_id: str):
        self.tid          = track_id
        self.cam_id       = cam_id
        self.kpt_buffer   : deque[np.ndarray] = deque(maxlen=WINDOW_SIZE)
        self.raw_buffer   : deque[tuple]      = deque()   # (frame_idx, kpts_norm)
        self.frames_seen  = 0
        self.segments_out = 0
        self.label        = "..."
        self.conf         = 0.0
        self.color        = TRACK_COLORS[track_id % len(TRACK_COLORS)]

    def segment_key(self) -> str:
        return f"{self.cam_id}_track{self.tid:04d}_seg{self.segments_out:04d}"


# ─────────────────────────────────────────────────────────────────
# CSV writer
# ─────────────────────────────────────────────────────────────────
class CsvWriter:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(ALL_COLS)
        self.rows_written = 0

    def write_window(self, seg_key: str, raw_window: list[tuple],
                     label_id: int, pseudo_conf: float):
        for local_f, (_, kpts) in enumerate(raw_window):
            row = [seg_key, local_f + 1]
            row.extend(kpts.flatten().tolist())
            row.append(label_id)
            row.append(round(pseudo_conf, 4))
            self._w.writerow(row)
        self.rows_written += len(raw_window)
        self._f.flush()

    def close(self):
        self._f.close()


# ─────────────────────────────────────────────────────────────────
# CTR-GCN loader
# ─────────────────────────────────────────────────────────────────
def load_ctrgcn(weights_path: str, device: torch.device):
    import sys
    from collections import OrderedDict
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from model.ctrgcn import Model

    model = Model(
        num_class=len(CLASS_NAMES),
        num_point=17,
        num_person=1,
        graph="graph.coco17.Graph",
        graph_args={"labeling_mode": "spatial"},
    )
    state = torch.load(weights_path, map_location=device)
    state = OrderedDict([[k.replace("module.", ""), v] for k, v in state.items()])
    model.load_state_dict(state)
    model.to(device).eval()
    return model


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def run(args):
    device    = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    cam_id    = args.cam_id or Path(args.source).stem
    use_model = (args.mode == "pseudo")

    print(f"[INFO] Source      : {args.source}")
    print(f"[INFO] Cam ID      : {cam_id}")
    print(f"[INFO] Mode        : {args.mode}")
    print(f"[INFO] Window      : {WINDOW_SIZE} frames, stride {WINDOW_STRIDE}")
    print(f"[INFO] Output CSV  : {args.out_csv}")

    # MJPEG server
    mjpeg = None
    if args.stream_port:
        mjpeg = MjpegServer(port=args.stream_port, jpeg_quality=args.jpeg_quality)
        mjpeg.start()

    # YOLO
    from ultralytics import YOLO
    yolo = YOLO(args.yolo)
    print("[INFO] YOLO ready")

    # CTR-GCN
    ctrgcn = None
    if use_model:
        ctrgcn = load_ctrgcn(args.weights, device)
        print(f"[INFO] CTR-GCN ready  (conf_thresh={PSEUDO_CONF_THRESH})")

    csv_writer = CsvWriter(args.out_csv)

    # Mở stream
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "hwaccel;none")
    cap = cv2.VideoCapture(args.source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được source: {args.source}")

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"[INFO] Stream      : {W}x{H} @ {fps:.1f} fps\n")

    tracks        : dict[int, TrackState] = {}
    frame_idx     = 0
    saved_windows = 0
    skipped_low   = 0
    t0            = time.time()
    is_file       = not str(args.source).startswith("rtsp")

    while True:
        ret, frame = cap.read()
        if not ret:
            if is_file:
                break
            time.sleep(0.05)
            continue

        frame_idx += 1

        # ── YOLO + ByteTrack ──────────────────────────────────────
        results    = yolo.track(frame, persist=True, tracker=args.tracker,
                                imgsz=640, verbose=False)
        res        = results[0]
        active_ids : set[int] = set()

        if res.boxes is not None and res.keypoints is not None:
            track_ids_raw = res.boxes.id
            if track_ids_raw is not None:
                track_ids  = track_ids_raw.cpu().numpy().astype(int)
                boxes_xyxy = res.boxes.xyxy.cpu().numpy()
                kpts_data  = res.keypoints.data.cpu().numpy()   # (N,17,3) pixel

                for pidx, tid in enumerate(track_ids):
                    if pidx >= len(kpts_data):
                        continue
                    active_ids.add(tid)

                    kpts_norm = kpts_data[pidx].copy()
                    kpts_norm[:, 0] /= W
                    kpts_norm[:, 1] /= H
                    if not is_valid_kpt(kpts_norm):
                        kpts_norm[:] = 0.0

                    if tid not in tracks:
                        tracks[tid] = TrackState(tid, cam_id)
                    ts = tracks[tid]

                    ts.kpt_buffer.append(kpts_norm.copy())
                    ts.raw_buffer.append((frame_idx, kpts_norm.copy()))
                    ts.frames_seen += 1

                    # ── Xuất window ──────────────────────────────
                    ready = (ts.frames_seen >= WINDOW_SIZE and
                             (ts.frames_seen - WINDOW_SIZE) % WINDOW_STRIDE == 0)
                    if ready:
                        window_np  = np.stack(list(ts.kpt_buffer), axis=0)   # (T,17,3)
                        raw_window = list(ts.raw_buffer)[-WINDOW_SIZE:]

                        label_id, pseudo_conf = -1, -1.0

                        if use_model and ctrgcn is not None:
                            with torch.no_grad():
                                logits = ctrgcn(build_model_input(window_np, device))
                                probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
                            pred_id     = int(np.argmax(probs))
                            pseudo_conf = float(probs[pred_id])
                            ts.label    = CLASS_NAMES[pred_id]
                            ts.conf     = pseudo_conf

                            if pseudo_conf >= PSEUDO_CONF_THRESH:
                                label_id = pred_id
                            else:
                                skipped_low += 1
                                ts.segments_out += 1
                                # vẫn tiếp tục vẽ, chỉ không lưu
                        else:
                            ts.label = "?"
                            ts.conf  = 0.0
                            label_id = -1

                        if label_id >= 0 or not use_model:
                            seg_key = ts.segment_key()
                            csv_writer.write_window(seg_key, raw_window,
                                                    label_id, pseudo_conf)
                            saved_windows += 1
                            ts.segments_out += 1

                    # ── Vẽ skeleton + box ────────────────────────
                    draw_skeleton(frame, kpts_norm, W, H, ts.color)
                    draw_box(frame, boxes_xyxy[pidx], tid, ts.label, ts.conf, ts.color)

        # ── Xóa track mất khỏi frame ──────────────────────────────
        for k in [k for k in tracks if k not in active_ids]:
            del tracks[k]

        # ── HUD ──────────────────────────────────────────────────
        draw_hud(frame, frame_idx, len(active_ids),
                 saved_windows, skipped_low, csv_writer.rows_written)

        # ── Push lên MJPEG server ─────────────────────────────────
        if mjpeg is not None:
            mjpeg.push(frame)

        if args.show:
            cv2.imshow("collect_rtsp", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_idx % 300 == 0:
            elapsed = time.time() - t0
            print(f"  [{frame_idx}f | {elapsed:.0f}s]  "
                  f"tracks={len(active_ids)}  "
                  f"saved={saved_windows}  "
                  f"skipped={skipped_low}  "
                  f"rows={csv_writer.rows_written}")

    # ── Kết thúc ──────────────────────────────────────────────────
    cap.release()
    csv_writer.close()
    if mjpeg:
        mjpeg.stop()
    if args.show:
        cv2.destroyAllWindows()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE  —  {frame_idx} frames  {elapsed:.1f}s  ({frame_idx/max(elapsed,1):.1f} fps)")
    print(f"  Windows saved    : {saved_windows}")
    print(f"  Skipped low conf : {skipped_low}")
    print(f"  CSV rows         : {csv_writer.rows_written}")
    print(f"  Output           : {args.out_csv}")
    if use_model:
        try:
            import pandas as pd
            df  = pd.read_csv(args.out_csv)
            dfw = df[df["frame"] == 1]
            print("  Label distribution:")
            for lid, name in enumerate(CLASS_NAMES):
                n = (dfw["label"] == lid).sum()
                if n > 0:
                    print(f"    [{lid}] {name:15s}: {n} windows")
        except Exception:
            pass
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Thu thập skeleton từ RTSP/video, xem qua browser"
    )
    p.add_argument("--source",       required=True,
                   help="RTSP URL hoặc đường dẫn video file")
    p.add_argument("--out-csv",      required=True,
                   help="File CSV output (giống format le2i_train_kpts.csv)")
    p.add_argument("--mode",         choices=["pseudo", "unlabeled"], default="pseudo",
                   help="pseudo=auto-label bằng CTR-GCN | unlabeled=lưu hết, label=-1")
    p.add_argument("--weights",      default=DEFAULT_WEIGHTS,
                   help="Checkpoint CTR-GCN (.pt) — dùng trong chế độ pseudo")
    p.add_argument("--yolo",         default=DEFAULT_YOLO)
    p.add_argument("--cam-id",       default="",
                   help="Prefix camera (vd: cam_topdown_1)")
    p.add_argument("--device",       default="0",
                   help="'0' = cuda:0 | 'cpu'")
    p.add_argument("--tracker",      default="bytetrack.yaml")
    p.add_argument("--stream-port",  type=int, default=0, dest="stream_port",
                   help="Port HTTP để xem MJPEG qua browser (vd: 8003). 0 = tắt")
    p.add_argument("--jpeg-quality", type=int, default=75, dest="jpeg_quality",
                   help="JPEG quality cho MJPEG stream (1-100, default 75)")
    p.add_argument("--show",         action="store_true",
                   help="Hiển thị cv2.imshow (chỉ dùng khi có màn hình)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
