"""verify_pkl.py — chạy ngay sau build_pkl"""
import pickle
import numpy as np

def verify_pkl(pkl_path: str, class_count: int):
    with open(pkl_path, "rb") as f:
        X, y = pickle.load(f)

    print(f"\n{'='*50}")
    print(f"File: {pkl_path}")
    print(f"X shape : {X.shape}")   # expect (N, T, 17, 3)
    print(f"y shape : {y.shape}")   # expect (N, C)

    errors = []

    # --- Shape ---
    if X.ndim != 4:
        errors.append(f"X phải là 4D, got {X.ndim}D")
    if X.shape[2] != 17:
        errors.append(f"X dim2 phải là 17 keypoints, got {X.shape[2]}")
    if X.shape[3] != 3:
        errors.append(f"X dim3 phải là 3 (x,y,conf), got {X.shape[3]}")
    if y.shape[1] != class_count:
        errors.append(f"y dim1 phải là {class_count}, got {y.shape[1]}")

    # --- NaN / Inf ---
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    if nan_count > 0:
        errors.append(f"X có {nan_count} NaN values!")
    if inf_count > 0:
        errors.append(f"X có {inf_count} Inf values!")

    # --- Confidence range [0, 1] ---
    conf = X[:, :, :, 2]
    if conf.min() < -0.01 or conf.max() > 1.01:
        errors.append(f"Confidence ngoài [0,1]: min={conf.min():.4f}, max={conf.max():.4f}")

    # --- Label sum (sau normalize nên gần 1.0) ---
    label_sums = y.sum(axis=1)
    bad_labels = (label_sums < 0.01).sum()
    if bad_labels > 0:
        errors.append(f"{bad_labels} samples có label sum ≈ 0 (weighting bug?)")

    # --- Tọa độ sau normalize_pose_window: phần lớn nên nằm trong [-3, 3] ---
    xy = X[:, :, :, :2]
    pct_outlier = (np.abs(xy) > 5).mean() * 100
    if pct_outlier > 5:
        errors.append(f"{pct_outlier:.1f}% tọa độ ngoài [-5,5] — normalize có vấn đề")

    # Report
    if errors:
        print("\n❌ LỖI PHÁT HIỆN:")
        for e in errors:
            print(f"  • {e}")
    else:
        print("\n✅ Shape & range OK")

    # Stats
    print(f"\nX stats : min={X.min():.4f}, max={X.max():.4f}, mean={X.mean():.4f}")
    print(f"y stats : min={y.min():.4f}, max={y.max():.4f}")
    print(f"Label distribution:")
    label_ids = np.argmax(y, axis=1)
    unique, counts = np.unique(label_ids, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  class {cls:>2}: {cnt:>6} samples ({100*cnt/len(y):.1f}%)")

    return len(errors) == 0


if __name__ == "__main__":
    verify_pkl("/home/cxviewlab2/data/khoa.do/CTR-GCN/data/le2i_ofsyn/combine/csv/10_to_8_100_v2/pkl/v4_36f/train_v4_36f.pkl", class_count=8)
    verify_pkl("/home/cxviewlab2/data/khoa.do/CTR-GCN/data/le2i_ofsyn/combine/csv/10_to_8_100_v2/pkl/v4_36f/test_v4_36f.pkl",  class_count=8)