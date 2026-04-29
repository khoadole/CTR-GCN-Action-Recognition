#!/usr/bin/env python3
"""Convert ST-GCN style PKL/CSV into CTR-GCN NPZ format.

Output NPZ keys:
- x_train, y_train
- x_test, y_test

Default x shape is (N, C, T, V, M), which is directly supported by
feeders.feeder_custom.Feeder.

python data/le2i_ofsyn/build_pkl.py

python data/le2i_ofsyn/pkl_to_npz.py \
  --train-pkl /home/cxviewlab2/data/khoa.do/CTR-GCN/data/le2i_ofsyn/combine/csv/10_to_8_100_v2/pkl/v4_36f/train_v4_36f.pkl \
  --test-pkl  /home/cxviewlab2/data/khoa.do/CTR-GCN/data/le2i_ofsyn/combine/csv/10_to_8_100_v2/pkl/v4_36f/test_v4_36f.pkl \
  --output    /home/cxviewlab2/data/khoa.do/CTR-GCN/data/le2i_ofsyn/combine/csv/10_to_8_100_v2/npz/coco17_8class_v4.npz \
  --num-class 8 \
  --num-person 1

PYTHONPATH=./torchlight python main.py \
  --config config/custom/coco17_8class_v4.yaml \
  --work-dir work_dir/custom/coco17_8class_v4/ctrgcn_joint \
  --device 0

"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:  # CSV mode only
    pd = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Convert ST-GCN dataset to CTR-GCN NPZ')

    src = parser.add_argument_group('source')
    src.add_argument('--train-pkl', type=Path, default=None, help='Path to train PKL (tuple: X,y)')
    src.add_argument('--test-pkl', type=Path, default=None, help='Path to test PKL (tuple: X,y)')
    src.add_argument('--train-csv', type=Path, default=None, help='Path to train CSV (video,frame,label,17*3 coords)')
    src.add_argument('--test-csv', type=Path, default=None, help='Path to test CSV (video,frame,label,17*3 coords)')

    csv = parser.add_argument_group('csv options')
    csv.add_argument('--window-size', type=int, default=30, help='Temporal window size for CSV mode')
    csv.add_argument('--stride', type=int, default=1, help='Temporal stride for CSV mode')
    csv.add_argument('--max-frame-gap', type=int, default=10, help='Split sequence when frame gap is too large')
    csv.add_argument('--normalize-xy', action='store_true', help='Normalize x,y to [-1,1] per frame')

    out = parser.add_argument_group('output')
    out.add_argument('--num-class', type=int, default=0, help='Optional fixed num classes (auto infer if 0)')
    out.add_argument('--num-person', type=int, default=1, help='Number of persons (M) in output tensor')
    out.add_argument('--output', type=Path, required=True, help='Output NPZ path')

    return parser.parse_args()


def scale_pose_xy(xy: np.ndarray) -> np.ndarray:
    # xy: (T, V, 2)
    xy_min = np.nanmin(xy, axis=1, keepdims=True)
    xy_max = np.nanmax(xy, axis=1, keepdims=True)
    diff = xy_max - xy_min
    diff[diff == 0] = 1e-6
    return ((xy - xy_min) / diff) * 2 - 1


def labels_to_index(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 1:
        if y.dtype.kind not in 'iu':
            # Handle string/object labels by mapping to class indices.
            _, inv = np.unique(y.astype(str), return_inverse=True)
            return inv.astype(np.int64)
        return y.astype(np.int64)
    if y.ndim == 2:
        return np.argmax(y, axis=1).astype(np.int64)
    if y.ndim == 3:
        # Sequence labels (N,T,C) -> clip label by averaging over time.
        return np.argmax(y.mean(axis=1), axis=1).astype(np.int64)
    raise ValueError(f'Unsupported label shape: {y.shape}')


def to_ctrgcn_tensor(x: np.ndarray, num_person: int) -> np.ndarray:
    """Convert to (N, C, T, V, M)."""
    x = np.asarray(x, dtype=np.float32)

    # Already in CTR-GCN style (N,C,T,V,M)
    if x.ndim == 5:
        if x.shape[1] not in (2, 3):
            raise ValueError(f'Expected channel dim C=2/3 for 5D input, got shape={x.shape}')
        if x.shape[1] == 2:
            n, _, t, v, m = x.shape
            z = np.zeros((n, 1, t, v, m), dtype=np.float32)
            x = np.concatenate([x, z], axis=1)
        return x

    if x.ndim != 4:
        raise ValueError(f'Expected 4D/5D features, got {x.shape}')

    # Common layouts:
    # 1) (N,T,V,C)
    # 2) (N,C,T,V)
    if x.shape[-1] in (2, 3):
        n, t, v, c = x.shape
        x = np.transpose(x, (0, 3, 1, 2))  # (N,C,T,V)
    elif x.shape[1] in (2, 3):
        n, c, t, v = x.shape
    else:
        raise ValueError(f'Cannot infer channel axis for 4D input shape={x.shape}')

    if c == 2:
        z = np.zeros((n, 1, t, v), dtype=np.float32)
        x = np.concatenate([x, z], axis=1)

    if num_person < 1:
        raise ValueError('num_person must be >= 1')

    x = x[..., None]  # (N,C,T,V,1)
    if num_person > 1:
        x = np.repeat(x, num_person, axis=4)
    return x


def one_hot(y_idx: np.ndarray, num_class: int) -> np.ndarray:
    y = np.zeros((len(y_idx), num_class), dtype=np.float32)
    y[np.arange(len(y_idx)), y_idx] = 1.0
    return y


def _is_numeric_like(value) -> bool:
    try:
        arr = np.asarray(value)
    except Exception:
        return False
    return arr.dtype.kind in 'iufb'


def _extract_xy_from_obj(obj) -> tuple[np.ndarray, np.ndarray]:
    """Extract (X, y) from common PKL payload variants."""
    # Dict payload
    if isinstance(obj, dict):
        x_keys = ['x', 'X', 'features', 'feature', 'data', 'inputs']
        y_keys = ['y', 'Y', 'labels', 'label', 'targets', 'target']

        x_val = next((obj[k] for k in x_keys if k in obj), None)
        y_val = next((obj[k] for k in y_keys if k in obj), None)
        if x_val is not None and y_val is not None:
            return np.asarray(x_val), np.asarray(y_val)

        raise ValueError(f'Unsupported dict PKL keys: {sorted(obj.keys())}')

    # Tuple/list payload
    if isinstance(obj, (tuple, list)):
        if len(obj) == 0:
            raise ValueError('Empty tuple/list PKL payload')

        numeric_items = []
        for i, item in enumerate(obj):
            if _is_numeric_like(item):
                arr = np.asarray(item)
                if arr.ndim >= 1:
                    numeric_items.append((i, arr))

        if not numeric_items:
            raise ValueError('Tuple/list PKL contains no numeric array-like items')

        # Choose X as a high-dimensional tensor (prefer ndim>=4).
        x_candidates = [(i, arr) for i, arr in numeric_items if arr.ndim >= 3]
        if not x_candidates:
            raise ValueError('Cannot find feature tensor X (expected ndim>=3) in tuple/list PKL payload')

        def x_score(item):
            _, arr = item
            per_sample = int(np.prod(arr.shape[1:])) if arr.ndim > 1 else 1
            return (arr.ndim >= 4, arr.ndim, per_sample)

        x_idx, x_arr = max(x_candidates, key=x_score)
        n_samples = int(x_arr.shape[0])

        # Prefer y as low-dimensional labels with matching N.
        y_candidates = []
        for i, arr in numeric_items:
            if i == x_idx:
                continue
            if int(arr.shape[0]) != n_samples:
                continue
            if arr.shape == x_arr.shape:
                continue

            # Best-case labels: (N,) or (N,C)
            if arr.ndim <= 2:
                score = (2, -arr.ndim, -int(np.prod(arr.shape[1:])) if arr.ndim > 1 else 0)
                y_candidates.append((score, i, arr))
                continue

            # Fallback: sequence labels (N,T,C), but avoid skeleton-like tails (e.g. 17x3)
            if arr.ndim == 3 and arr.shape[-1] <= 128:
                if not (arr.shape[-2] == 17 and arr.shape[-1] in (2, 3, 4)):
                    score = (1, -arr.shape[-1], -arr.shape[-2])
                    y_candidates.append((score, i, arr))

        if y_candidates:
            _, y_idx, y_arr = max(y_candidates, key=lambda t: t[0])
            return x_arr, y_arr

        raise ValueError(
            'Unsupported tuple/list PKL payload. Expected (X,y) or a payload containing one array-like X (ndim>=3) '
            'and one array-like y with matching first dimension.'
        )

    raise ValueError(f'Unsupported PKL payload type: {type(obj)}')


def load_from_pkl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open('rb') as f:
        obj = pickle.load(f)

    x, y = _extract_xy_from_obj(obj)
    if x.ndim < 4:
        raise ValueError(f'Expected X ndim >= 4, got shape {x.shape}')

    y_idx = labels_to_index(y)
    return x, y_idx


def load_from_csv(
    path: Path,
    window_size: int,
    stride: int,
    max_frame_gap: int,
    normalize_xy: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if pd is None:
        raise ImportError('pandas is required for CSV mode. Please install pandas.')

    df = pd.read_csv(path)
    required = {'video', 'frame', 'label'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns in {path}: {sorted(missing)}')

    coord_cols = [c for c in df.columns if c not in {'video', 'frame', 'label'}]
    if len(coord_cols) % 3 != 0:
        raise ValueError(f'Expected coordinate columns grouped by 3 (x,y,s), got {len(coord_cols)} columns')

    num_joint = len(coord_cols) // 3
    if num_joint != 17:
        raise ValueError(f'Expected 17 joints from ST-GCN COCO, got {num_joint}')

    features = []
    labels = []

    for _, group in df.groupby('video', sort=False):
        group = group.sort_values('frame').reset_index(drop=True)

        frames = group['frame'].to_numpy(np.int64)
        all_y = group['label'].to_numpy(np.int64)
        all_x = group[coord_cols].to_numpy(np.float32).reshape(-1, num_joint, 3)

        # Split into contiguous chunks by frame continuity
        chunks = []
        current = [0]
        for i in range(1, len(frames)):
            if frames[i] - frames[i - 1] < max_frame_gap:
                current.append(i)
            else:
                chunks.append(current)
                current = [i]
        chunks.append(current)

        for idxs in chunks:
            if len(idxs) < window_size:
                continue

            seq = all_x[idxs].copy()
            if normalize_xy:
                seq[:, :, :2] = scale_pose_xy(seq[:, :, :2])

            seq_labels = all_y[idxs]
            for start in range(0, len(idxs) - window_size + 1, stride):
                end = start + window_size
                clip = seq[start:end]
                clip_labels = seq_labels[start:end]

                features.append(clip)
                labels.append(int(np.bincount(clip_labels).argmax()))

    if not features:
        raise ValueError(f'No windows generated from CSV: {path}')

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return x, y


def load_split(
    pkl_path: Path | None,
    csv_path: Path | None,
    window_size: int,
    stride: int,
    max_frame_gap: int,
    normalize_xy: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if pkl_path is not None and csv_path is not None:
        raise ValueError('Provide either PKL or CSV for a split, not both')
    if pkl_path is None and csv_path is None:
        raise ValueError('Each split must provide one source: PKL or CSV')

    if pkl_path is not None:
        return load_from_pkl(pkl_path)

    assert csv_path is not None
    return load_from_csv(csv_path, window_size, stride, max_frame_gap, normalize_xy)


def main() -> None:
    args = parse_args()

    x_train, y_train_idx = load_split(
        pkl_path=args.train_pkl,
        csv_path=args.train_csv,
        window_size=args.window_size,
        stride=args.stride,
        max_frame_gap=args.max_frame_gap,
        normalize_xy=args.normalize_xy,
    )
    x_test, y_test_idx = load_split(
        pkl_path=args.test_pkl,
        csv_path=args.test_csv,
        window_size=args.window_size,
        stride=args.stride,
        max_frame_gap=args.max_frame_gap,
        normalize_xy=args.normalize_xy,
    )

    inferred_num_class = int(max(y_train_idx.max(initial=0), y_test_idx.max(initial=0)) + 1)
    num_class = args.num_class if args.num_class > 0 else inferred_num_class

    if y_train_idx.max(initial=0) >= num_class or y_test_idx.max(initial=0) >= num_class:
        raise ValueError('num_class is smaller than max label id in data')

    x_train_out = to_ctrgcn_tensor(x_train, num_person=args.num_person)
    x_test_out = to_ctrgcn_tensor(x_test, num_person=args.num_person)
    y_train_out = one_hot(y_train_idx, num_class)
    y_test_out = one_hot(y_test_idx, num_class)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        x_train=x_train_out,
        y_train=y_train_out,
        x_test=x_test_out,
        y_test=y_test_out,
    )

    print('[DONE] Saved:', args.output)
    print('x_train:', x_train_out.shape, 'y_train:', y_train_out.shape)
    print('x_test :', x_test_out.shape, 'y_test :', y_test_out.shape)


if __name__ == '__main__':
    main()
