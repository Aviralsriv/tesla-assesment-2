# 🚗 Multimodal Self-Supervised Vehicle Log Model

> Contrastive pretraining on video/LiDAR/CAN/GPS/HD-Map using DINOv2-style losses + EMA teacher.

---

## Architecture

```
Raw Logs (video / LiDAR / CAN / GPS / HD-Map)
          │
          ▼
  ┌──────────────────────────────────────┐
  │  VehicleLogDataset  (pipeline.py)    │
  │  • Open3D BEV projection (200×200m)  │
  │  • LiDAR voxelization               │
  │  • CAN / GPS tensor sequences        │
  │  • PyG HeteroData lane graph         │
  └─────────────────┬────────────────────┘
                    │  DualViewAugment
          ┌─────────┴──────────┐
        View 1              View 2
          │                    │
  ┌───────▼────────────────────▼───────┐
  │       Parallel Modality Encoders    │
  │  BEV   → DINOv2 ViT-B/14  →  768  │
  │  LiDAR → PointNet          → 1024  │
  │  CAN   → Bi-LSTM           →  512  │
  │  GPS   → Transformer       →  512  │
  │  Map   → GCNConv (3-layer) →  512  │
  └─────────────┬──────────────────────┘
                │  ProjectionHead (MLP, BN, GELU)
                ▼  1024-dim L2-normed
  ┌─────────────────────────────────────┐
  │  FusionTransformer                  │
  │  5 tokens × 1024 → cross-attn      │
  │  → flatten → Linear → 2048-dim      │
  └──────────────┬──────────────────────┘
                 │
  ┌──────────────▼──────────────────────┐
  │  Contrastive Losses                  │
  │  • NT-Xent  per modality            │
  │  • NT-Xent  fused embedding         │
  │  • DINO-style vs EMA teacher        │
  └──────────────────────────────────────┘
                 │  pretrain 100 epochs
                 ▼
  ┌──────────────────────────────────────┐
  │  Downstream Heads                    │
  │  • LinearProbe  → AUROC > 0.9       │
  │  • TrajectoryHead → ADE / FDE       │
  └──────────────────────────────────────┘
```

---

## File Structure

```
vehicle_ssl/
├── requirements.txt
├── train.py            ← main training entry point
├── eval.py             ← linear probe + trajectory eval
├── pipeline.py         ← Dataset, DataLoader, collate_fn
├── models/
│   ├── encoders.py     ← 5 modality encoders + ProjectionHead
│   ├── fusion.py       ← FusionTransformer + MultiModalEncoder + EMA
│   └── losses.py       ← NT-Xent, Sinkhorn-Knopp, DINO loss
└── utils/
    ├── bev.py          ← Open3D BEV projection + voxelization
    └── augment.py      ← Multi-crop, point dropout, CAN/GPS jitter
```

---

## Quick Start

### 1. Install

```bash
conda create -n vssl python=3.11
conda activate vssl
pip install -r requirements.txt

# PyG (match your CUDA version)
pip install torch-geometric torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
```

### 2. Prepare Data

HDF5 format (one file, N segment groups):
```
logs.h5
├── segment_0/
│   ├── lidar        (N_pts, 4)   float32
│   ├── camera       (T, H, W, 3) uint8
│   ├── can          (T, 50)      float32
│   ├── gps          (T, 3)       float32  [lat, lon, vel]
│   ├── map_nodes    (M, 8)       float32
│   └── map_edges    (2, E)       int64
└── segment_1/ ...
```

### 3. Pretrain

```bash
python train.py \
    --data_root /data/logs.h5 \
    --epochs 100 \
    --batch_size 256 \
    --lr 1e-4 \
    --amp \
    --wandb
```

### 4. Evaluate

```bash
# Linear probe (accident classification)
python eval.py \
    --ckpt checkpoints/ckpt_epoch99.pt \
    --data_root /data/logs.h5 \
    --task probe \
    --num_classes 2

# Trajectory prediction
python eval.py \
    --ckpt checkpoints/ckpt_epoch99.pt \
    --data_root /data/logs.h5 \
    --task trajectory
```

---

## Key Hyperparameters

| Param | Default | Notes |
|---|---|---|
| `lr` | `1e-4` | AdamW base LR (cosine decay) |
| `batch_size` | `256` | Per-GPU |
| `temperature` | `0.1` | NT-Xent τ |
| `ema_momentum` | `0.996` | Teacher EMA |
| `epochs` | `100` | ~5% linear warmup |
| `weight_decay` | `0.04` | BN/bias excluded |
| `amp` | `True` | Mixed precision |
| `freeze_dino` | `False` | Fine-tune DINOv2 |

---

## Expected Results (after 100 epochs)

| Task | Metric | Target |
|---|---|---|
| Accident prediction | AUROC | > 0.90 |
| Driver behaviour | AUROC | > 0.85 |
| Trajectory (30-step) | ADE | < 1.5 m |
| Trajectory (30-step) | FDE | < 3.0 m |
