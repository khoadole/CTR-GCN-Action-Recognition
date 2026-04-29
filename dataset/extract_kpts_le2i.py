"""
Extract skeleton joints (x, y, score) from videos using YOLO11m-pose.

- YOLO handles letterbox resize internally; original frame is passed directly.
- Keypoints are normalized to [0, 1] relative to original video resolution.
- Frames where core body keypoints (shoulders, hips, knees) are below MIN_KPT_CONF
  are stored as NaN rows and later dropped in build_pkl.py.
"""

import os
import cv2
import numpy as np
import pandas as pd
import hashlib
import shutil
import subprocess
import tempfile
from ultralytics import YOLO


# ==================== CONFIG ====================
MODEL_PATH = "yolo11m-pose.pt"
FRAME_SIZE = 640
MIN_KPT_CONF = 0.05
AUTO_FALLBACK_TRANSCODE = True
FALLBACK_CACHE_DIR = "./le2i/csv/_decode_cache"
KEEP_FALLBACK_CACHE = False

# Ask OpenCV-FFmpeg to prefer software decode on remote/SSH Linux.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "hwaccel;none")

# Annotation mode (old pipeline)
ANNOT_FILE = "../dataset/dataset_create_2/annotation.csv"
VIDEO_FOLDER = "../dataset/dataset_create_2/videos"
ANNOT_OUTPUT_CSV = "../outputs_data/output_create_yolo11/pose_and_score.csv"

# Auto split mode (new project structure)
DATASET_SPLIT_ROOT = "./le2i"
TRAIN_OUTPUT_CSV = "./le2i/csv/le2i_train_yolo11m.csv"
TEST_OUTPUT_CSV = "./le2i/csv/le2i_test_yolo11m.csv"

CLASS_NAMES = [
    "Sit down",   # 0
    "Lying Down", # 1
    "Walking",    # 2
    "Stand up",   # 3
    "Standing",   # 4
    "Fall Down",  # 5
    "Sitting",    # 6
]
CLASS_TO_LABEL = {name: idx for idx, name in enumerate(CLASS_NAMES)}

VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv", ".wmv", ".mpeg", ".mpg")

# Correct column names (fixed typo)
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

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


# ==================== HELPER FUNCTIONS ====================
def normalize_keypoints(kpts_xy, orig_w, orig_h):
    """Normalize keypoints to [0, 1] relative to original video size"""
    kpts_xy = kpts_xy.copy()
    kpts_xy[:, 0] /= orig_w
    kpts_xy[:, 1] /= orig_h
    return kpts_xy


def is_video_file(name):
    return name.lower().endswith(VIDEO_EXTS)


def kpts_to_xy_score(kpts, thresh=0.05):
    """Extract 17 keypoints and check if core body keypoints are confident"""
    result = np.zeros((17, 3), dtype=np.float32)
    result[:kpts.shape[0], :] = kpts[:17, :3]

    # Check core keypoints (shoulders, hips, knees, ankles)
    core_kpts = [5, 6, 11, 12, 13, 14]
    cf = all(result[kid, 2] >= thresh for kid in core_kpts)
    return result, cf


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


def draw_pose(img, kpts, box_xyxy, color=(125, 255, 255), thresh=0.2):
    for i, j in SKELETON:
        if kpts[i, 2] > thresh and kpts[j, 2] > thresh:
            cv2.line(img, (int(kpts[i, 0]), int(kpts[i, 1])),
                     (int(kpts[j, 0]), int(kpts[j, 1])), color, 2)
    for p in kpts:
        if p[2] > thresh:
            cv2.circle(img, (int(p[0]), int(p[1])), 3, color, -1)
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


def select_main_person(result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    xyxy = boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    return int(np.argmax(areas))


def extract_row_from_frame(model, frame, video_key, frame_idx, cls_idx, orig_w, orig_h):
    """Pass original frame directly — YOLO handles letterbox and rescales keypoints internally."""
    row = [video_key, frame_idx]

    result = model(frame, imgsz=FRAME_SIZE, verbose=False)[0]

    person_idx = select_main_person(result)
    if person_idx is not None and result.keypoints is not None:
        kpts_all = result.keypoints.data.cpu().numpy()
        if person_idx < len(kpts_all):
            kpts = kpts_all[person_idx]
            kpt_result, cf = kpts_to_xy_score(kpts, thresh=MIN_KPT_CONF)
            if cf:
                # Keypoints đã ở tọa độ frame gốc — normalize thẳng
                pt_norm = normalize_keypoints(kpt_result[:, :2], orig_w, orig_h)
                combined = np.column_stack([pt_norm, kpt_result[:, 2]])
                row.extend(combined.flatten().tolist())
            else:
                row.extend([np.nan] * (17 * 3))
        else:
            row.extend([np.nan] * (17 * 3))
    else:
        row.extend([np.nan] * (17 * 3))

    row.append(cls_idx)
    return row


# ==================== PROCESSING FUNCTIONS ====================
def process_video_with_label_map(model, video_path, video_key, label_map):
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

        if frame_idx in label_map:
            cls_idx = int(label_map[frame_idx])
            row = extract_row_from_frame(model, frame, video_key, frame_idx, cls_idx, orig_w, orig_h)
            rows.append(row)

        frame_idx += 1

    cap.release()
    cleanup_transcoded_file(trans_path)
    return rows


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


def write_rows(rows, csv_path, write_header):
    if not rows:
        return
    df = pd.DataFrame(rows, columns=COLUMNS)
    mode = "w" if write_header else "a"
    df.to_csv(csv_path, mode=mode, header=write_header, index=False)


# ==================== MAIN MODES ====================
def run_annotation_mode(model):
    print(f"Annotation mode: {ANNOT_FILE}")
    annot = pd.read_csv(ANNOT_FILE)
    if not os.path.isdir(VIDEO_FOLDER):
        raise FileNotFoundError(f"Video folder not found: {VIDEO_FOLDER}")

    os.makedirs(os.path.dirname(ANNOT_OUTPUT_CSV), exist_ok=True)
    if os.path.exists(ANNOT_OUTPUT_CSV):
        os.remove(ANNOT_OUTPUT_CSV)

    vid_list = annot["video"].unique()
    write_header = True

    for idx, vid in enumerate(vid_list, start=1):
        frames_label = annot[annot["video"] == vid].reset_index(drop=True)
        label_map = dict(zip(frames_label["frame"].astype(int), frames_label["label"].astype(int)))

        video_path = os.path.join(VIDEO_FOLDER, vid)
        if not os.path.exists(video_path):
            print(f"[Skip] Missing video: {video_path}")
            continue

        print(f"[{idx}/{len(vid_list)}] {vid}")
        rows = process_video_with_label_map(model, video_path, vid, label_map)
        write_rows(rows, ANNOT_OUTPUT_CSV, write_header)
        write_header = False

    print(f"Done annotation mode. Saved: {ANNOT_OUTPUT_CSV}")


def collect_split_videos(split_dir):
    tasks = []
    if not os.path.isdir(split_dir):
        return tasks

    class_dirs = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
    for class_name in class_dirs:
        if class_name not in CLASS_TO_LABEL:
            print(f"[Warn] Unknown class folder, skip: {class_name}")
            continue

        cls_idx = CLASS_TO_LABEL[class_name]
        class_path = os.path.join(split_dir, class_name)
        for filename in sorted(os.listdir(class_path)):
            if not is_video_file(filename):
                continue
            video_path = os.path.join(class_path, filename)
            video_key = f"{class_name}/{filename}"
            tasks.append((video_path, video_key, cls_idx))
    return tasks


def run_auto_split_mode(model):
    train_dir = os.path.join(DATASET_SPLIT_ROOT, "train")
    test_dir = os.path.join(DATASET_SPLIT_ROOT, "test")

    if not os.path.isdir(train_dir) and not os.path.isdir(test_dir):
        raise FileNotFoundError(
            f"Cannot find train/test folders. Expected: {train_dir} and/or {test_dir}"
        )

    os.makedirs(os.path.dirname(TRAIN_OUTPUT_CSV), exist_ok=True)
    if os.path.exists(TRAIN_OUTPUT_CSV):
        os.remove(TRAIN_OUTPUT_CSV)
    if os.path.exists(TEST_OUTPUT_CSV):
        os.remove(TEST_OUTPUT_CSV)

    for split_name, split_dir, out_csv in [
        ("train", train_dir, TRAIN_OUTPUT_CSV),
        ("test", test_dir, TEST_OUTPUT_CSV),
    ]:
        tasks = collect_split_videos(split_dir)
        if not tasks:
            print(f"No videos found in split: {split_name}")
            continue

        print(f"Auto split mode: {split_name} ({len(tasks)} videos)")
        write_header = True
        for idx, (video_path, video_key, cls_idx) in enumerate(tasks, start=1):
            print(f"[{split_name} {idx}/{len(tasks)}] {video_key}")
            rows = process_video_single_label(model, video_path, video_key, cls_idx)
            write_rows(rows, out_csv, write_header)
            write_header = False

        print(f"Saved {split_name}: {out_csv}")


def main():
    model = YOLO(MODEL_PATH)

    if ANNOT_FILE and os.path.exists(ANNOT_FILE):
        run_annotation_mode(model)
    else:
        print(f"Annotation file not found: {ANNOT_FILE}")
        print("Switching to auto split mode from class folders (train/test).")
        run_auto_split_mode(model)


if __name__ == "__main__":
    main()