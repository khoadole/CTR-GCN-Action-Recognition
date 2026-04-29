"""
Extract skeleton joints (x, y, score) from OF-SYN videos using YOLO11m-pose.

Behavior:
- Traverse videos under ./of-syn by class folders (no train/test split)
- Use letterbox resize to preserve aspect ratio
- Normalize keypoints to original video resolution
- Output one CSV at ./of-syn/csv/ofsyn_yolo11m.csv
- Video key format: ofsyn_<folder_name>_<file_name_without_ext>
"""

import os
import cv2
import numpy as np
import pandas as pd
import hashlib
import shutil
import subprocess
import tempfile
from collections import defaultdict
from ultralytics import YOLO


# ==================== CONFIG ====================
MODEL_PATH = "yolo11m-pose.pt"
FRAME_SIZE = 640
MIN_KPT_CONF = 0.05
DRAW_KPT_CONF = 0.2
AUTO_FALLBACK_TRANSCODE = True
FALLBACK_CACHE_DIR = "./of-syn/csv/_decode_cache"
KEEP_FALLBACK_CACHE = False

# OF-SYN mode (single root, no train/test split)
INPUT_ROOT = "./of-syn"
OUTPUT_CSV = "./of-syn/csv/ofsyn_yolo11m.csv"
VIS_ROOT = "./of-syn/csv/visulize"
LABEL_CSV = "./omnifall_data/labels/of-syn.csv"
RESUME_IF_CSV_EXISTS = True

# On remote/SSH Linux, AV1 hardware decoding may be unavailable.
# Ask OpenCV-FFmpeg to prefer software decode.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "hwaccel;none")

VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv", ".wmv", ".mpeg", ".mpg")

CLASS_NAMES = [
    "walk",
    "fall",
    "fallen",
    "sit_down",
    "sitting",
    "lie_down",
    "lying",
    "stand_up",
    "standing",
    "other",
]
CLASS_TO_LABEL = {name: idx for idx, name in enumerate(CLASS_NAMES)}
OTHER_LABEL = CLASS_TO_LABEL["other"]

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

COLUMNS = [
    "video", "frame",
    "nose_x", "nose_y", "nose_s",
    "left_eye_x", "left_eye_y", "left_eye_s",
    "right_eye_x", "right_eye_y", "right_eye_s",
    "left_ear_x", "left_ear_y", "left_ear_s",
    "right_ear_x", "right_ear_y", "right_ear_s",
    "left_shoulder_x", "left_shoulder_y", "left_shoulder_s",
    "right_shoulder_x", "right_shoulder_y", "right_shoulder_s",
    "left_elbow_x", "left_elbow_y", "left_elbow_s",
    "right_elbow_x", "right_elbow_y", "right_elbow_s",
    "left_wrist_x", "left_wrist_y", "left_wrist_s",
    "right_wrist_x", "right_wrist_y", "right_wrist_s",
    "left_hip_x", "left_hip_y", "left_hip_s",
    "right_hip_x", "right_hip_y", "right_hip_s",
    "left_knee_x", "left_knee_y", "left_knee_s",
    "right_knee_x", "right_knee_y", "right_knee_s",
    "left_ankle_x", "left_ankle_y", "left_ankle_s",
    "right_ankle_x", "right_ankle_y", "right_ankle_s",
    "label",
]


# ==================== HELPER FUNCTIONS ====================
def normalize_keypoints(kpts_xy, orig_w, orig_h):
    """Normalize keypoints to [0, 1] relative to original video size."""
    kpts_xy = kpts_xy.copy()
    kpts_xy[:, 0] /= orig_w
    kpts_xy[:, 1] /= orig_h
    return kpts_xy


def is_video_file(name):
    return name.lower().endswith(VIDEO_EXTS)


def map_raw_label_to_target(raw_label):
    """Map OmniFall raw labels to 10-class target taxonomy."""
    raw = int(raw_label)
    if 0 <= raw <= 9:
        return raw
    if 10 <= raw <= 15:
        return OTHER_LABEL
    return OTHER_LABEL


def load_ofsyn_intervals(label_csv_path):
    """Load per-video temporal label intervals from OF-SYN CSV.

    Returns dict: {"folder/stem": [(start_sec, end_sec, target_label), ...]}
    """
    if not os.path.exists(label_csv_path):
        raise FileNotFoundError(f"Label CSV not found: {label_csv_path}")

    df = pd.read_csv(label_csv_path)
    need_cols = {"path", "label", "start", "end"}
    if not need_cols.issubset(df.columns):
        raise ValueError(f"Label CSV must contain columns: {sorted(need_cols)}")

    intervals = defaultdict(list)
    for _, r in df.iterrows():
        key = str(r["path"]).strip().replace("\\", "/")
        if not key:
            continue

        start_sec = float(r["start"])
        end_sec = float(r["end"])
        if end_sec < start_sec:
            start_sec, end_sec = end_sec, start_sec

        target_label = map_raw_label_to_target(r["label"])
        intervals[key].append((start_sec, end_sec, target_label))

    for key in intervals:
        intervals[key].sort(key=lambda x: x[0])

    return intervals


def resolve_frame_label(frame_idx, fps, intervals):
    """Resolve frame label from temporal intervals using FPS.

    Frame index is 1-based. Time uses start-inclusive, end-inclusive epsilon.
    """
    if not intervals:
        return None

    if fps <= 0:
        fps = 30.0

    t = (frame_idx - 1) / fps
    eps = 1e-6
    for start_sec, end_sec, lb in intervals:
        if start_sec <= t <= (end_sec + eps):
            return lb
    return None


def ffmpeg_available():
    return shutil.which("ffmpeg") is not None


def transcode_to_h264(video_path, cache_dir):
    if not ffmpeg_available():
        print("[Fallback] ffmpeg not found. Please install ffmpeg to decode AV1 videos.")
        return ""

    stem = os.path.splitext(os.path.basename(video_path))[0]
    path_hash = hashlib.md5(video_path.encode("utf-8")).hexdigest()[:8]
    if KEEP_FALLBACK_CACHE:
        os.makedirs(cache_dir, exist_ok=True)
        out_path = os.path.join(cache_dir, f"{stem}_{path_hash}_h264.mp4")
        if os.path.exists(out_path):
            return out_path
    else:
        os.makedirs(cache_dir, exist_ok=True)
        fd, out_path = tempfile.mkstemp(
            prefix=f"{stem}_{path_hash}_",
            suffix="_h264.mp4",
            dir=cache_dir,
        )
        os.close(fd)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        return out_path
    except subprocess.CalledProcessError as e:
        err = (e.stderr or "").strip().splitlines()
        tail = err[-2:] if err else ["unknown ffmpeg error"]
        print(f"[Fallback] Transcode failed for {video_path}: {' | '.join(tail)}")
        return ""
    except Exception as e:
        print(f"[Fallback] Transcode exception for {video_path}: {e}")
        return ""


def try_open_capture(video_path):
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return None

    ok, _ = cap.read()
    if ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return cap

    cap.release()
    return None


def open_capture_with_fallback(video_path):
    cap = try_open_capture(video_path)
    if cap is not None:
        return cap, ""

    if not AUTO_FALLBACK_TRANSCODE:
        return None, ""

    trans_path = transcode_to_h264(video_path, FALLBACK_CACHE_DIR)
    if not trans_path:
        return None, ""

    cap2 = try_open_capture(trans_path)
    if cap2 is not None:
        print(f"[Fallback] AV1 -> H264: {video_path} -> {trans_path}")
        return cap2, trans_path

    print(f"[Fallback] Still cannot open after transcode: {video_path}")
    return None, ""


def cleanup_transcoded_file(path):
    if KEEP_FALLBACK_CACHE:
        return
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def kpts_to_xy_score(kpts, thresh=0.05):
    """Extract 17 keypoints and check if core body keypoints are confident."""
    result = np.zeros((17, 3), dtype=np.float32)
    result[:kpts.shape[0], :] = kpts[:17, :3]

    core_kpts = [5, 6, 11, 12, 13, 14]
    cf = all(result[kid, 2] >= thresh for kid in core_kpts)
    return result, cf


def select_main_person(result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    xyxy = boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    return int(np.argmax(areas))


def draw_pose(img, kpts, color=(125, 255, 255), thresh=DRAW_KPT_CONF):
    for i, j in SKELETON:
        if kpts[i, 2] > thresh and kpts[j, 2] > thresh:
            cv2.line(
                img,
                (int(kpts[i, 0]), int(kpts[i, 1])),
                (int(kpts[j, 0]), int(kpts[j, 1])),
                color,
                2,
            )
    for p in kpts:
        if p[2] > thresh:
            cv2.circle(img, (int(p[0]), int(p[1])), 3, color, -1)


def extract_main_person_kpts(result):
    if result.keypoints is None:
        return None, False

    kpts_all = result.keypoints.data.cpu().numpy()
    person_idx = select_main_person(result)
    if person_idx is None or person_idx >= len(kpts_all):
        return None, False

    kpts = kpts_all[person_idx]
    return kpts_to_xy_score(kpts, thresh=MIN_KPT_CONF)


def extract_row_from_frame(model, frame, video_key, frame_idx, cls_idx, orig_w, orig_h):
    """Extract one frame by running YOLO directly; Ultralytics handles letterbox/scale."""
    row = [video_key, frame_idx]

    result = model(frame, imgsz=FRAME_SIZE, verbose=False)[0]

    kpt_result, cf = extract_main_person_kpts(result)
    if kpt_result is not None and cf:
        pt_norm = normalize_keypoints(kpt_result[:, :2], orig_w, orig_h)
        combined = np.column_stack([pt_norm, kpt_result[:, 2]])
        row.extend(combined.flatten().tolist())
    else:
        row.extend([np.nan] * (17 * 3))

    row.append(cls_idx)
    return row


def process_video_single_label(model, video_path, video_key, cls_idx):
    rows = []
    cap, trans_path = open_capture_with_fallback(video_path)
    if cap is None:
        print(f"[Error] Cannot open video: {video_path}")
        return rows

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_idx = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        row = extract_row_from_frame(model, frame, video_key, frame_idx, cls_idx, orig_w, orig_h)
        rows.append(row)
        frame_idx += 1

    cap.release()
    cleanup_transcoded_file(trans_path)
    return rows


def process_video_with_intervals(model, video_path, video_key, intervals):
    rows = []
    cap, trans_path = open_capture_with_fallback(video_path)
    if cap is None:
        print(f"[Error] Cannot open video: {video_path}")
        return rows

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0

    frame_idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cls_idx = resolve_frame_label(frame_idx, fps, intervals)
        if cls_idx is not None:
            row = extract_row_from_frame(model, frame, video_key, frame_idx, cls_idx, orig_w, orig_h)
            rows.append(row)
        frame_idx += 1

    cap.release()
    cleanup_transcoded_file(trans_path)
    return rows


def write_rows(rows, csv_path, write_header):
    if not rows:
        return
    df = pd.DataFrame(rows, columns=COLUMNS)
    mode = "w" if write_header else "a"
    df.to_csv(csv_path, mode=mode, header=write_header, index=False)


def load_processed_video_keys(csv_path):
    """Return set of video keys already present in output CSV."""
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return set(), 0

    if "video" not in pd.read_csv(csv_path, nrows=0).columns:
        return set(), 0

    processed = set()
    row_count = 0
    for chunk in pd.read_csv(csv_path, usecols=["video"], chunksize=200_000):
        videos = chunk["video"].dropna().astype(str)
        row_count += len(videos)
        processed.update(videos.tolist())
    return processed, row_count


def visualize_one_video(model, video_path, class_name, output_root, intervals=None):
    cap, trans_path = open_capture_with_fallback(video_path)
    if cap is None:
        print(f"[Warn] Cannot open video for visualize: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    file_stem = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(output_root, class_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{file_stem}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model(frame, imgsz=FRAME_SIZE, verbose=False)[0]
        kpt_result, _ = extract_main_person_kpts(result)
        if kpt_result is not None:
            draw_pose(frame, kpt_result)

        cls_idx = resolve_frame_label(frame_idx, fps, intervals)
        if cls_idx is not None:
            cls_name = CLASS_NAMES[cls_idx] if 0 <= cls_idx < len(CLASS_NAMES) else "other"
            cv2.putText(
                frame,
                f"label: {cls_name} ({cls_idx})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    cleanup_transcoded_file(trans_path)
    print(f"[Visualize] Saved: {out_path}")


def collect_ofsyn_videos(root_dir):
    tasks = []
    if not os.path.isdir(root_dir):
        return tasks

    skip_dirs = {"csv", "_decode_cache"}
    class_dirs = sorted(
        [
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d)) and d not in skip_dirs
        ]
    )

    for class_name in class_dirs:
        class_path = os.path.join(root_dir, class_name)

        for filename in sorted(os.listdir(class_path)):
            if not is_video_file(filename):
                continue

            video_path = os.path.join(class_path, filename)
            file_stem = os.path.splitext(filename)[0]
            video_key = f"ofsyn_{class_name}_{file_stem}"
            ann_key = f"{class_name}/{file_stem}"
            tasks.append((video_path, video_key, class_name, ann_key))

    return tasks


# ==================== MAIN ====================
def main():
    if not os.path.isdir(INPUT_ROOT):
        raise FileNotFoundError(f"Input folder not found: {INPUT_ROOT}")

    model = YOLO(MODEL_PATH)
    interval_map = load_ofsyn_intervals(LABEL_CSV)

    tasks = collect_ofsyn_videos(INPUT_ROOT)
    if not tasks:
        print(f"No videos found under: {INPUT_ROOT}")
        return

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    os.makedirs(VIS_ROOT, exist_ok=True)

    processed_video_keys = set()
    existing_rows = 0
    write_header = True
    if os.path.exists(OUTPUT_CSV):
        if RESUME_IF_CSV_EXISTS:
            processed_video_keys, existing_rows = load_processed_video_keys(OUTPUT_CSV)
            write_header = os.path.getsize(OUTPUT_CSV) == 0
        else:
            os.remove(OUTPUT_CSV)

    print(f"OF-SYN mode: {len(tasks)} videos")
    print(
        "Resume status: "
        f"csv_rows={existing_rows}, "
        f"processed_videos={len(processed_video_keys)}, "
        f"pending_videos={max(0, len(tasks) - len(processed_video_keys))}"
    )

    if processed_video_keys:
        pending_tasks = [t for t in tasks if t[1] not in processed_video_keys]
        if pending_tasks:
            print(f"Next pending video: {pending_tasks[0][1]}")
    else:
        pending_tasks = tasks

    if not pending_tasks:
        print("All videos already processed. Nothing to do.")
        return

    visualized_classes = set()
    missing_ann = 0
    total_pending = len(pending_tasks)
    for idx, (video_path, video_key, class_name, ann_key) in enumerate(pending_tasks, start=1):
        print(f"[{idx}/{total_pending}] {video_key}")
        intervals = interval_map.get(ann_key, [])
        if not intervals:
            missing_ann += 1
            print(f"[Warn] Missing temporal labels for: {ann_key}. Skip this video.")
            continue

        rows = process_video_with_intervals(model, video_path, video_key, intervals)
        write_rows(rows, OUTPUT_CSV, write_header)
        if rows:
            write_header = False

        if class_name not in visualized_classes:
            visualize_one_video(model, video_path, class_name, VIS_ROOT, intervals)
            visualized_classes.add(class_name)

    print(f"Missing temporal annotations: {missing_ann}")
    print(f"Done. Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
