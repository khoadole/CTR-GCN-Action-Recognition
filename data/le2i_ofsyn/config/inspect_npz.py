# python inspect_npz.py /path/to/coco17_8class_v4.npz

import numpy as np
import os

def inspect_npz(path):
    size_mb = os.path.getsize(path) / 1e6
    print(f"File: {path}")
    print(f"Size: {size_mb:.1f} MB")

    d = np.load(path)
    print(f"Keys: {list(d.keys())}")

    CLASS_NAMES = ["walk", "standing", "sitting", "sit_down", "stand_up", "lie_down", "lying", "other"]

    for split in ("train", "test"):
        x = d.get(f"x_{split}")
        y = d.get(f"y_{split}")
        if x is None:
            continue
        print(f"\n--- {split} ---")
        print(f"  x shape: {x.shape}  dtype: {x.dtype}")
        print(f"  y shape: {y.shape}  dtype: {y.dtype}")

        labels = y.argmax(axis=1) if y.ndim == 2 else y
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  label distribution:")
        for cls_idx, cnt in zip(unique, counts):
            name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"class_{cls_idx}"
            print(f"    [{cls_idx}] {name}: {cnt:,}")

        print(f"  x range: [{x.min():.4f}, {x.max():.4f}]  mean={x.mean():.4f}")

        # Check if x-coords are centred (root-normalised) or in [0,1]
        x_channel = x[:, 0] if x.ndim == 5 else x[..., 0]
        print(f"  x-channel range: [{x_channel.min():.4f}, {x_channel.max():.4f}]")
        if x_channel.min() < -0.1:
            print("  => root-centred (build_pkl normalisation applied)")
        else:
            print("  => raw [0,1] coords (no root-centering?)")

        # Heuristic: flip data doubles direction-sensitive classes
        train_y = d.get("y_train")
        if train_y is not None and split == "train":
            train_labels = train_y.argmax(axis=1) if train_y.ndim == 2 else train_y
            total = len(train_labels)
            print(f"  total train samples: {total:,}")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/le2i_ofsyn/combine/csv/10_to_8_100_v2/npz/coco17_8class_v4.npz"
    inspect_npz(path)
