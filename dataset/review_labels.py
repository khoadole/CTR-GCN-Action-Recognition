#!/usr/bin/env python3
"""
Tool review + sửa label cho CSV pseudo-labeled.

Chạy: python review_labels.py --csv rtsp_data/cam1_pseudo.csv --port 8004
Mở browser: http://<ip>:8004/

Tính năng:
  - Hiện tất cả segment, nhóm theo label
  - Click segment → xem skeleton animation (loop)
  - Chọn label đúng → Save
  - Export CSV đã sửa
"""

from __future__ import annotations

import argparse
import io
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import cv2
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Sit down", "Lying Down", "Walking", "Stand up",
    "Standing", "Fall Down", "Sitting", "Other",
]
CLASS_COLORS_HEX = [
    "#e74c3c", "#9b59b6", "#2ecc71", "#f39c12",
    "#3498db", "#e67e22", "#1abc9c", "#95a5a6",
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
KPT_COLS = [f"{n}_{c}" for n in KPT_NAMES for c in ("x","y","s")]

CANVAS_W, CANVAS_H = 480, 360
FPS_LOOP = 12   # skeleton animation fps


# ─────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────
class LabelStore:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df       = pd.read_csv(csv_path)
        self._lock    = threading.Lock()

        # Build segment index: seg_key → {label, pseudo_conf, frames: list of (17,3)}
        self._segments: dict[str, dict] = {}
        for seg_key, grp in self.df.groupby("video", sort=False):
            grp_sorted = grp.sort_values("frame")
            frames_kpts = []
            for _, row in grp_sorted.iterrows():
                kpts = np.array([[row[f"{n}_x"], row[f"{n}_y"], row[f"{n}_s"]]
                                 for n in KPT_NAMES], dtype=np.float32)
                frames_kpts.append(kpts)
            self._segments[seg_key] = {
                "label":       int(grp_sorted["label"].iloc[0]),
                "pseudo_conf": float(grp_sorted["pseudo_conf"].iloc[0]),
                "frames":      frames_kpts,
                "corrected":   False,
            }

    def keys(self):
        return list(self._segments.keys())

    def get(self, seg_key: str) -> dict | None:
        return self._segments.get(seg_key)

    def update_label(self, seg_key: str, new_label: int):
        with self._lock:
            if seg_key in self._segments:
                self._segments[seg_key]["label"]     = new_label
                self._segments[seg_key]["corrected"] = True
                # Cập nhật DataFrame
                mask = self.df["video"] == seg_key
                self.df.loc[mask, "label"] = new_label

    def save(self, out_path: str | None = None):
        path = out_path or self.csv_path
        with self._lock:
            self.df.to_csv(path, index=False)
        return path

    def summary(self):
        result = {}
        for seg_key, info in self._segments.items():
            lid = info["label"]
            result.setdefault(lid, []).append({
                "key":       seg_key,
                "conf":      round(info["pseudo_conf"], 3),
                "corrected": info["corrected"],
            })
        return result


# ─────────────────────────────────────────────────────────────────
# Skeleton renderer
# ─────────────────────────────────────────────────────────────────
def render_skeleton_frame(kpts: np.ndarray, label: int,
                          W: int = CANVAS_W, H: int = CANVAS_H) -> np.ndarray:
    """kpts: (17,3) normalized 0-1 coords → BGR image"""
    img   = np.zeros((H, W, 3), dtype=np.uint8)
    color = tuple(int(CLASS_COLORS_HEX[label][i:i+2], 16)
                  for i in (5, 3, 1))  # hex → BGR

    # Scale keypoints to canvas — center với padding
    pad   = 40
    xs    = kpts[:, 0]
    ys    = kpts[:, 1]
    valid = kpts[:, 2] > 0.05
    if valid.sum() >= 4:
        x_min, x_max = xs[valid].min(), xs[valid].max()
        y_min, y_max = ys[valid].min(), ys[valid].max()
    else:
        x_min, x_max, y_min, y_max = 0.0, 1.0, 0.0, 1.0

    x_range = max(x_max - x_min, 0.001)
    y_range = max(y_max - y_min, 0.001)

    def to_px(x, y):
        px = int(pad + (x - x_min) / x_range * (W - 2 * pad))
        py = int(pad + (y - y_min) / y_range * (H - 2 * pad))
        return px, py

    # Vẽ bones
    for i, j in SKELETON_PAIRS:
        if kpts[i, 2] > 0.1 and kpts[j, 2] > 0.1:
            cv2.line(img, to_px(xs[i], ys[i]), to_px(xs[j], ys[j]),
                     color, 2, cv2.LINE_AA)
    # Vẽ joints
    for k in range(17):
        if kpts[k, 2] > 0.1:
            cv2.circle(img, to_px(xs[k], ys[k]), 5, color, -1)

    return img


def build_animation_jpeg_list(frames: list[np.ndarray], label: int) -> list[bytes]:
    """Trả về list JPEG bytes cho từng frame của skeleton animation."""
    jpegs = []
    for kpts in frames:
        img    = render_skeleton_frame(kpts, label)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            jpegs.append(buf.tobytes())
    return jpegs


# ─────────────────────────────────────────────────────────────────
# HTML page
# ─────────────────────────────────────────────────────────────────
def build_html(store: LabelStore) -> str:
    summary  = store.summary()
    cls_opts = "".join(
        f'<option value="{i}">{i}: {name}</option>'
        for i, name in enumerate(CLASS_NAMES)
    )

    # Sidebar: segments nhóm theo label, sorted by label id
    sidebar_html = ""
    for lid in sorted(summary.keys()):
        name  = CLASS_NAMES[lid] if 0 <= lid < len(CLASS_NAMES) else f"label={lid}"
        color = CLASS_COLORS_HEX[lid] if 0 <= lid < len(CLASS_COLORS_HEX) else "#888"
        segs  = summary[lid]
        sidebar_html += f'<div class="group"><div class="group-hdr" style="background:{color}22;border-left:3px solid {color}">'
        sidebar_html += f'<b style="color:{color}">{name}</b> <span class="cnt">({len(segs)})</span></div>'
        for s in segs:
            corrected = "✓ " if s["corrected"] else ""
            sidebar_html += (
                f'<div class="seg-item" data-key="{s["key"]}" data-label="{lid}" '
                f'onclick="selectSeg(this)">'
                f'{corrected}<span class="seg-key">{s["key"].split("_seg")[1] if "_seg" in s["key"] else s["key"]}</span>'
                f'<span class="conf" style="color:{color}">{s["conf"]:.2f}</span>'
                f'</div>'
            )
        sidebar_html += '</div>'

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<title>Label Review</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{display:flex;height:100vh;font-family:monospace;background:#1a1a2e;color:#eee;font-size:13px}}
  #sidebar{{width:280px;min-width:280px;overflow-y:auto;border-right:1px solid #333;display:flex;flex-direction:column}}
  #sidebar h2{{padding:10px;background:#16213e;font-size:14px;border-bottom:1px solid #333;flex-shrink:0}}
  .group{{margin:4px 0}}
  .group-hdr{{padding:5px 10px;font-size:12px}}
  .cnt{{opacity:.6}}
  .seg-item{{padding:4px 12px;cursor:pointer;display:flex;justify-content:space-between;border-radius:3px;margin:1px 4px}}
  .seg-item:hover{{background:#ffffff15}}
  .seg-item.active{{background:#ffffff25;outline:1px solid #aaa}}
  .seg-key{{flex:1}}
  .conf{{font-size:11px;opacity:.8}}
  #main{{flex:1;display:flex;flex-direction:column;padding:16px;gap:12px}}
  #top-bar{{display:flex;align-items:center;gap:12px;background:#16213e;padding:10px;border-radius:6px}}
  #seg-title{{flex:1;font-size:15px;font-weight:bold}}
  #label-sel{{padding:5px;background:#2a2a4a;color:#eee;border:1px solid #555;border-radius:4px;font-size:13px}}
  #btn-save{{padding:6px 18px;background:#2ecc71;color:#000;border:none;border-radius:4px;cursor:pointer;font-weight:bold}}
  #btn-save:hover{{background:#27ae60}}
  #btn-export{{padding:6px 14px;background:#3498db;color:#fff;border:none;border-radius:4px;cursor:pointer}}
  #anim-wrap{{flex:1;display:flex;gap:12px}}
  #skeleton-box{{background:#0d0d1a;border-radius:6px;display:flex;align-items:center;justify-content:center;flex:1;min-height:300px;position:relative}}
  #skeleton-img{{max-width:100%;max-height:100%;border-radius:4px}}
  #info-box{{width:220px;background:#16213e;border-radius:6px;padding:14px;font-size:12px;line-height:1.8}}
  .info-row{{display:flex;justify-content:space-between;border-bottom:1px solid #333;padding:3px 0}}
  #msg{{padding:6px 10px;background:#27ae60;color:#fff;border-radius:4px;display:none;font-size:12px}}
  #speed-ctrl{{display:flex;align-items:center;gap:6px;font-size:11px}}
  #speed-slider{{width:80px}}
</style>
</head>
<body>
<div id="sidebar">
  <h2>Segments ({len(store.keys())} total)</h2>
  {sidebar_html}
</div>
<div id="main">
  <div id="top-bar">
    <div id="seg-title">← Chọn segment từ danh sách</div>
    <select id="label-sel">{cls_opts}</select>
    <button id="btn-save" onclick="saveLabel()">💾 Save Label</button>
    <button id="btn-export" onclick="exportCsv()">⬇ Export CSV</button>
    <span id="msg"></span>
  </div>
  <div id="anim-wrap">
    <div id="skeleton-box">
      <img id="skeleton-img" src="" style="display:none" alt="">
      <div id="placeholder" style="opacity:.3;font-size:14px">click segment để xem skeleton animation</div>
    </div>
    <div id="info-box">
      <div style="font-weight:bold;margin-bottom:8px;font-size:13px">Thông tin segment</div>
      <div class="info-row"><span>Key</span><span id="i-key">—</span></div>
      <div class="info-row"><span>Label cũ</span><span id="i-label-orig">—</span></div>
      <div class="info-row"><span>Confidence</span><span id="i-conf">—</span></div>
      <div class="info-row"><span>Frames</span><span id="i-frames">—</span></div>
      <br>
      <div id="speed-ctrl">
        <span>Speed</span>
        <input id="speed-slider" type="range" min="1" max="30" value="{FPS_LOOP}" oninput="setSpeed(this.value)">
        <span id="speed-val">{FPS_LOOP} fps</span>
      </div>
      <br>
      <div style="opacity:.6;font-size:11px;line-height:1.6">
        Skeleton animation tái tạo từ keypoints trong CSV.<br>
        Màu = label hiện tại.
      </div>
    </div>
  </div>
</div>
<script>
let currentKey = null;
let currentLabel = null;
let frameTimer  = null;
let frameList   = [];
let frameIdx    = 0;
let fps         = {FPS_LOOP};

function selectSeg(el) {{
  document.querySelectorAll('.seg-item').forEach(e => e.classList.remove('active'));
  el.classList.add('active');
  currentKey   = el.dataset.key;
  currentLabel = parseInt(el.dataset.label);
  document.getElementById('seg-title').textContent = currentKey;
  document.getElementById('label-sel').value = currentLabel;
  document.getElementById('i-key').textContent = currentKey.split('_track')[1] || currentKey;

  fetch('/info/' + encodeURIComponent(currentKey))
    .then(r => r.json())
    .then(d => {{
      document.getElementById('i-label-orig').textContent = d.label_name + ' (' + d.label + ')';
      document.getElementById('i-conf').textContent = d.conf;
      document.getElementById('i-frames').textContent = d.n_frames + ' frames';
    }});

  // Load frame list then start animation
  fetch('/framelist/' + encodeURIComponent(currentKey))
    .then(r => r.json())
    .then(list => {{
      frameList = list;
      frameIdx  = 0;
      startAnimation();
    }});

  document.getElementById('placeholder').style.display = 'none';
  document.getElementById('skeleton-img').style.display = 'block';
}}

function startAnimation() {{
  if (frameTimer) clearInterval(frameTimer);
  if (frameList.length === 0) return;
  frameTimer = setInterval(() => {{
    const img = document.getElementById('skeleton-img');
    img.src = frameList[frameIdx % frameList.length] + '?t=' + Date.now();
    frameIdx++;
  }}, Math.round(1000 / fps));
}}

function setSpeed(v) {{
  fps = parseInt(v);
  document.getElementById('speed-val').textContent = fps + ' fps';
  if (frameList.length > 0) startAnimation();
}}

function saveLabel() {{
  if (!currentKey) {{ showMsg('Chưa chọn segment!', '#e74c3c'); return; }}
  const newLabel = parseInt(document.getElementById('label-sel').value);
  fetch('/update', {{
    method: 'POST',
    headers: {{'Content-Type':'application/json'}},
    body: JSON.stringify({{key: currentKey, label: newLabel}})
  }})
  .then(r => r.json())
  .then(d => {{
    showMsg('Đã lưu: ' + d.label_name, '#2ecc71');
    // Cập nhật sidebar
    const el = document.querySelector(`.seg-item[data-key="${{currentKey}}"]`);
    if (el) el.dataset.label = newLabel;
    currentLabel = newLabel;
    // Reload để cập nhật grouping
    setTimeout(() => location.reload(), 800);
  }});
}}

function exportCsv() {{
  fetch('/export', {{method:'POST'}})
    .then(r => r.json())
    .then(d => showMsg('Đã export: ' + d.path, '#3498db'));
}}

function showMsg(text, color) {{
  const m = document.getElementById('msg');
  m.textContent = text;
  m.style.background = color;
  m.style.display = 'block';
  setTimeout(() => m.style.display = 'none', 2500);
}}
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────
# HTTP server
# ─────────────────────────────────────────────────────────────────
def make_handler(store: LabelStore, csv_path: str):

    # Pre-render tất cả frames cho mỗi segment (cache)
    frame_cache: dict[str, list[bytes]] = {}
    cache_lock  = threading.Lock()

    def get_frames(seg_key: str) -> list[bytes]:
        with cache_lock:
            if seg_key not in frame_cache:
                info   = store.get(seg_key)
                label  = info["label"] if info else 0
                frames = info["frames"] if info else []
                frame_cache[seg_key] = build_animation_jpeg_list(frames, label)
            return frame_cache[seg_key]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass

        def do_GET(self):
            parsed = urlparse(self.path)
            path   = parsed.path

            if path == "/":
                body = build_html(store).encode()
                self._respond(200, "text/html; charset=utf-8", body)

            elif path.startswith("/framelist/"):
                seg_key = path[len("/framelist/"):]
                jpegs   = get_frames(seg_key)
                # Trả về list URL của từng frame
                urls = [f"/frame/{seg_key}/{i}" for i in range(len(jpegs))]
                self._respond(200, "application/json", json.dumps(urls).encode())

            elif path.startswith("/frame/"):
                # /frame/<seg_key>/<idx>
                parts   = path[len("/frame/"):].rsplit("/", 1)
                seg_key = parts[0]
                idx     = int(parts[1]) if len(parts) > 1 else 0
                jpegs   = get_frames(seg_key)
                if jpegs:
                    data = jpegs[idx % len(jpegs)]
                    self._respond(200, "image/jpeg", data)
                else:
                    self._respond(404, "text/plain", b"not found")

            elif path.startswith("/info/"):
                seg_key = path[len("/info/"):]
                info    = store.get(seg_key)
                if info:
                    resp = {
                        "label":      info["label"],
                        "label_name": CLASS_NAMES[info["label"]] if 0 <= info["label"] < len(CLASS_NAMES) else "?",
                        "conf":       round(info["pseudo_conf"], 4),
                        "n_frames":   len(info["frames"]),
                    }
                    self._respond(200, "application/json", json.dumps(resp).encode())
                else:
                    self._respond(404, "application/json", b"{}")

            else:
                self._respond(404, "text/plain", b"not found")

        def do_POST(self):
            parsed = urlparse(self.path)
            path   = parsed.path
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)

            if path == "/update":
                data      = json.loads(body)
                seg_key   = data["key"]
                new_label = int(data["label"])
                store.update_label(seg_key, new_label)
                label_name = CLASS_NAMES[new_label] if 0 <= new_label < len(CLASS_NAMES) else "?"
                # Invalidate cache cho segment này
                with cache_lock:
                    frame_cache.pop(seg_key, None)
                resp = {"ok": True, "label": new_label, "label_name": label_name}
                self._respond(200, "application/json", json.dumps(resp).encode())

            elif path == "/export":
                stem     = Path(csv_path).stem
                out_path = str(Path(csv_path).parent / "finetune" / f"{stem}_reviewed.csv")
                saved    = store.save(out_path)
                resp     = {"ok": True, "path": saved}
                self._respond(200, "application/json", json.dumps(resp).encode())

            else:
                self._respond(404, "text/plain", b"not found")

        def _respond(self, code: int, ctype: str, body: bytes):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

    return Handler


# ─────────────────────────────────────────────────────────────────
def run(args):
    print(f"[INFO] Loading CSV: {args.csv}")
    store = LabelStore(args.csv)
    segs  = store.keys()
    print(f"[INFO] {len(segs)} segments loaded")

    summary = store.summary()
    for lid in sorted(summary.keys()):
        name = CLASS_NAMES[lid] if 0 <= lid < len(CLASS_NAMES) else f"label={lid}"
        print(f"  [{lid}] {name:15s}: {len(summary[lid])} segments")

    Handler = make_handler(store, args.csv)
    httpd   = HTTPServer(("0.0.0.0", args.port), Handler)
    httpd.allow_reuse_address = True

    print(f"\n[INFO] Review UI  → http://0.0.0.0:{args.port}/")
    print(f"[INFO] Trên mạng  → http://100.77.54.40:{args.port}/")
    print(f"[INFO] Ctrl+C để dừng, label đã sửa sẽ export vào *_reviewed.csv\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Đang lưu...")
        saved = store.save()
        print(f"[INFO] Đã lưu: {saved}")


def parse_args():
    p = argparse.ArgumentParser(description="Review + sửa label CSV pseudo-labeled")
    p.add_argument("--csv",  required=True, help="File CSV cần review")
    p.add_argument("--port", type=int, default=8004, help="HTTP port (default: 8004)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
