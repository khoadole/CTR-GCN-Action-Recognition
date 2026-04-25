"""
verify_csv_overlay.py
Đọc CSV đã extract → denormalize → vẽ skeleton lên frame gốc → xuất video
Dùng để confirm: tọa độ đúng chỗ, không bị flip, không bị offset
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

SKELETON_EDGES = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]

KEYPOINT_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle",
]

COORD_COLS = []
for name in KEYPOINT_NAMES:
    COORD_COLS.extend([f"{name}_x", f"{name}_y", f"{name}_s"])


def draw_skeleton_on_frame(frame, kpts_pixel, conf_thresh=0.05):
    """
    kpts_pixel: (17, 3) — x_pixel, y_pixel, score (chưa normalize)
    """
    h, w = frame.shape[:2]

    for i, j in SKELETON_EDGES:
        if kpts_pixel[i, 2] > conf_thresh and kpts_pixel[j, 2] > conf_thresh:
            pt1 = (int(kpts_pixel[i, 0]), int(kpts_pixel[i, 1]))
            pt2 = (int(kpts_pixel[j, 0]), int(kpts_pixel[j, 1]))
            # Kiểm tra trong bounds
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

    for kid, (x, y, s) in enumerate(kpts_pixel):
        if s > conf_thresh:
            px, py = int(x), int(y)
            if 0 <= px < w and 0 <= py < h:
                color = (0, 255, 0) if s > 0.3 else (0, 165, 255)
                cv2.circle(frame, (px, py), 4, color, -1)
                # Vẽ tên khớp (optional, bỏ nếu rối)
                # cv2.putText(frame, str(kid), (px+3, py-3),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    return frame


def verify_one_video(video_path: str, csv_path: str, video_key: str,
                     output_path: str, max_frames: int = 300):
    """
    So sánh trực tiếp: frame gốc vs skeleton từ CSV
    """
    # Load CSV rows cho video này
    df = pd.read_csv(csv_path)
    rows = df[df["video"] == video_key].sort_values("frame").reset_index(drop=True)

    if len(rows) == 0:
        print(f"[Error] Không tìm thấy video_key='{video_key}' trong CSV")
        print(f"Sample keys: {df['video'].unique()[:5].tolist()}")
        return

    print(f"Found {len(rows)} frames for '{video_key}'")

    # Kiểm tra nhanh giá trị CSV
    xy_cols = [c for c in COORD_COLS if c.endswith("_x") or c.endswith("_y")]
    s_cols  = [c for c in COORD_COLS if c.endswith("_s")]
    xy_vals = rows[xy_cols].values.astype(float)
    s_vals  = rows[s_cols].values.astype(float)

    print(f"\n--- CSV Quick Stats ---")
    print(f"XY range   : [{np.nanmin(xy_vals):.4f}, {np.nanmax(xy_vals):.4f}]  ← phải trong [0, 1]")
    print(f"Score range: [{np.nanmin(s_vals):.4f},  {np.nanmax(s_vals):.4f}]  ← phải trong [0, 1]")
    print(f"NaN frames : {rows[COORD_COLS].isna().all(axis=1).sum()} / {len(rows)}")

    # Mở video gốc
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] Không mở được video: {video_path}")
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"Video size : {orig_w}x{orig_h}, fps={fps:.1f}")

    # Setup writer — 2 panel: trái gốc, phải overlay
    out_w = orig_w * 2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, orig_h))

    # Build frame→row lookup
    frame_lookup = {int(r["frame"]): r for _, r in rows.iterrows()}

    frame_idx = 1
    written = 0

    while written < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        overlay = frame.copy()

        if frame_idx in frame_lookup:
            row = frame_lookup[frame_idx]
            kpt_vals = row[COORD_COLS].values.astype(float)  # (51,)

            if not np.isnan(kpt_vals).all():
                kpts = kpt_vals.reshape(17, 3)

                # Denormalize x,y về pixel
                kpts_pixel = kpts.copy()
                kpts_pixel[:, 0] *= orig_w   # x → pixel
                kpts_pixel[:, 1] *= orig_h   # y → pixel
                # score giữ nguyên

                draw_skeleton_on_frame(overlay, kpts_pixel)

                # Hiển thị label
                label = int(row["label"])
                cv2.putText(overlay, f"label={label} frame={frame_idx}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            else:
                cv2.putText(overlay, f"NaN frame={frame_idx}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # 2-panel: gốc bên trái, overlay bên phải
        combined = np.hstack([frame, overlay])
        writer.write(combined)
        written += 1
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"\nSaved: {output_path}  ({written} frames)")
    print("Kiểm tra: skeleton có bám đúng người không? Có bị lệch/flip không?")


def verify_multiple(csv_path: str, video_root: str, output_dir: str,
                    n_per_class: int = 1):
    """Tự động verify N video mỗi class"""
    df = pd.read_csv(csv_path, usecols=["video", "label"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group theo class
    from collections import defaultdict
    class_to_keys = defaultdict(list)
    for video_key in df["video"].unique():
        label = df[df["video"] == video_key]["label"].mode()[0]
        class_to_keys[int(label)].append(video_key)

    print(f"Classes found: {sorted(class_to_keys.keys())}")

    for cls, keys in sorted(class_to_keys.items()):
        for video_key in keys[:n_per_class]:
            # Tìm video file
            # video_key format: "ofsyn_<class>_<stem>" hoặc "ClassName/filename.avi"
            stem = video_key.split("/")[-1] if "/" in video_key else video_key
            # Tìm video trong thư mục
            matches = list(Path(video_root).rglob(f"{stem}.*"))
            matches = [m for m in matches if m.suffix.lower() in
                       ('.avi','.mp4','.mov','.mkv')]

            if not matches:
                print(f"[Skip] Video file not found for key: {video_key}")
                continue

            video_path = str(matches[0])
            out_path = str(output_dir / f"verify_cls{cls}_{stem}.mp4")
            print(f"\n[Class {cls}] {video_key}")
            verify_one_video(video_path, csv_path, video_key, out_path)


if __name__ == "__main__":
    # --- Option 1: verify 1 video cụ thể ---
    verify_one_video(
        video_path="/home/cxviewlab2/data/khoa.do/CTR-GCN/dataset/le2i/train/Fall Down/video_0.avi",
        csv_path="/home/cxviewlab2/data/khoa.do/CTR-GCN/dataset/le2i/csv/le2i_train_yolo11m.csv",
        video_key="Fall Down/video_0.avi",
        output_path="/home/cxviewlab2/data/khoa.do/CTR-GCN/dataset/le2i/video/verify_video_0.mp4",
    )

    # --- Option 2: auto verify 1 video mỗi class ---
    # verify_multiple(
    #     csv_path="./of-syn/csv/ofsyn_yolo11m.csv",
    #     video_root="./of-syn",
    #     output_dir="./verify_output",
    #     n_per_class=1,
    # )