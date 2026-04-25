# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CTR-GCN (Channel-wise Topology Refinement Graph Convolution Network) is a skeleton-based action recognition framework. This repo extends the original ICCV 2021 implementation with custom COCO-17 skeleton support, flexible data loading, and a real-time video inference pipeline for fall/activity detection.

## Common Commands

### Training

```bash
# Train on custom COCO-17 dataset (8-class activity recognition)
PYTHONPATH=./torchlight python main.py \
  --config config/custom/coco17_8class_v3.yaml \
  --work-dir work_dir/custom/coco17_8class_v3/ctrgcn_joint \
  --device 0

# Train bone modality (differences between connected joints)
python main.py --config config/custom/coco17_8class_v3.yaml \
  --train_feeder_args bone=True --test_feeder_args bone=True \
  --work-dir work_dir/custom/coco17_8class_v3/ctrgcn_bone --device 0

# Train velocity modality (frame-to-frame temporal differences)
python main.py --config config/custom/coco17_8class_v3.yaml \
  --train_feeder_args vel=True --test_feeder_args vel=True \
  --work-dir work_dir/custom/coco17_8class_v3/ctrgcn_vel --device 0
```

### Testing

```bash
# Evaluate a saved checkpoint
python main.py --config work_dir/custom/coco17_8class_v3/ctrgcn_joint/config.yaml \
  --work-dir work_dir/custom/coco17_8class_v3/ctrgcn_joint \
  --phase test --save-score True \
  --weights work_dir/custom/coco17_8class_v3/ctrgcn_joint/runs-44-55748.pt \
  --device 0
```

### Inference on Video

```bash
# infe_v4.py — preferred: rolling window + per-track temporal aggregation
python infe_v4.py --config infe_v4.yaml --input video.mp4 --output-dir ./results --device 0

# inference.py — simpler frame-level variant
python inference.py --config inference.yaml --video video.mp4 --output-dir ./results --device cuda:0
```

### Multi-Modality Ensemble

```bash
python ensemble.py --datasets ntu120/xsub \
  --joint-dir work_dir/ntu120/csub/ctrgcn \
  --bone-dir work_dir/ntu120/csub/ctrgcn_bone \
  --joint-motion-dir work_dir/ntu120/csub/ctrgcn_motion \
  --bone-motion-dir work_dir/ntu120/csub/ctrgcn_bone_motion
```

## Architecture

### Data Flow

```
NPZ file → Feeder → (N, C, T, V, M) tensors → CTR-GCN Model → class logits
```

- **N**: batch size, **C**: channels (3: x, y, confidence), **T**: temporal frames (36 for COCO), **V**: skeleton joints (17 COCO / 25 NTU), **M**: persons per sample (1 COCO / 2 NTU)

### Model: `model/ctrgcn.py`

The `Model` class stacks 10 `TCN_GCN_unit` blocks. Each unit combines:
- **CTRGC (unit_gcn)**: learns a per-sample, per-channel dynamic adjacency matrix refined on top of the static spatial graph `A`. Processes 3 graph subsets: self-links, inward (parent→child), outward (child→parent).
- **MultiScale_TemporalConv**: 4 dilations + max-pool + 1×1 branch for multi-scale temporal modeling.

Downsampling (stride=2) happens at layers 5 and 8. After layer 10: global average pool → FC → logits.

The model forward reshapes `(N, C, T, V, M)` → `(N*M, C, T, V)` for processing, then reshapes back.

### Graph: `graph/coco17.py`, `graph/ntu_rgb_d.py`

Each graph class builds an adjacency tensor `A` of shape `(3, V, V)` via `graph/tools.py`. The 3 subsets are [self-links, inward edges, outward edges]. This is passed directly to `Model.__init__` and stored for use in each CTRGC layer.

### Feeders: `feeders/feeder_custom.py`, `feeders/feeder_coco17.py`, `feeders/feeder_ntu.py`

Load NPZ files. `feeder_custom.py` auto-detects input shape:
- **5D `(N, C, T, V, M)`**: used directly (preferred)
- **4D `(N, T, V, C)`**: auto-transposed and expanded
- **3D `(N, T, V*C*M)`**: reshaped using `num_person`

`valid_crop_resize` bilinearly interpolates each sample to exactly `window_size` frames (no padding). `p_interval` controls what fraction of the sample to crop: random in `[0.5, 1.0]` for training, fixed `0.95` center-crop for testing.

Bone pairs for COCO-17 and NTU-25 are in `feeders/bone_pairs.py` (1-indexed; converted to 0-indexed at load time).

### Inference Pipeline: `infe_v4.py`

Uses YOLO11m-pose for keypoint detection + ByteTrack for person tracking. Maintains a rolling `deque` buffer per `track_id`. Each inference step:
1. Fills buffer to `window_size` frames
2. Normalizes keypoints relative to **median hip position** (joints 11+12) across valid frames — more robust than first-frame centering
3. Runs CTR-GCN forward on each track's buffer
4. Aggregates overlapping window predictions by weighted average

## Config Structure

Configs are YAML files in `config/{dataset}/`. Key fields:

```yaml
feeder: feeders.feeder_custom.Feeder
train_feeder_args:
  data_path: data/le2i_ofsyn/.../coco17_8class_v3.npz
  split: train
  window_size: 36       # must match model's expected T
  p_interval: [0.5, 1]  # random crop ratio for augmentation
  num_point: 17
  num_person: 1
  bone: False           # enable bone modality
  vel: False            # enable velocity modality

model: model.ctrgcn.Model
model_args:
  num_class: 8
  num_point: 17
  num_person: 1
  graph: graph.coco17.Graph
  graph_args:
    labeling_mode: spatial

base_lr: 0.1
lr_decay_rate: 0.1
step: [35, 55]          # epochs for LR decay
warm_up_epoch: 5
```

Saved checkpoints are named `runs-{epoch}-{global_step}.pt`. `main.py` saves `config.yaml` into `work_dir` at start, so `--config work_dir/.../config.yaml` reproduces training settings at test time.

## Dataset Format

NPZ files with keys: `x_train`, `y_train`, `x_test`, `y_test`.

- `x_*`: float32 array, shape `(N, C, T, V, M)` or compatible format
- `y_*`: int array of class indices, shape `(N,)`

Custom dataset lives in `data/le2i_ofsyn/combine/csv/10_to_8_100/npz/`. NTU data goes in `data/ntu/` and `data/ntu120/`.
