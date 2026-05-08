"""
Demo Inference Script — Tesla Multimodal SSL Model
Loads trained checkpoint, runs on a synthetic sample, shows embeddings & similarity.
"""
import torch
import torch.nn as nn
import numpy as np
import time
import os
import sys

print("=" * 60)
print("  Tesla Multimodal SSL Model — Demo Inference")
print("  Author: Aviral Srivastava")
print("=" * 60)

# ── Setup ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT   = "checkpoints/ckpt_last.pt"

print(f"\n[INFO] Device : {DEVICE}")
print(f"[INFO] CUDA   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[INFO] GPU    : {torch.cuda.get_device_name(0)}")
    print(f"[INFO] VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Load Model ───────────────────────────────────────────────────────────────
print(f"\n[STEP 1] Loading model from: {CKPT}")
t0 = time.time()

from models.fusion import MultiModalEncoder

model = MultiModalEncoder(freeze_dino=True).to(DEVICE)
model.eval()

ckpt = torch.load(CKPT, map_location=DEVICE)
model.load_state_dict(ckpt["student"])

print(f"[OK]    Model loaded in {time.time()-t0:.2f}s")
print(f"[INFO]  Checkpoint epoch : {ckpt['epoch']}")
print(f"[INFO]  Checkpoint step  : {ckpt['step']}")

# Count params
total_params = sum(p.numel() for p in model.parameters())
trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[INFO]  Total params     : {total_params/1e6:.1f}M")
print(f"[INFO]  Trainable params : {trainable/1e6:.1f}M")

# ── Create Synthetic Inputs ───────────────────────────────────────────────────
print(f"\n[STEP 2] Creating synthetic sensor inputs...")

def make_sample(batch_size=1):
    """Synthetic vehicle sensor data — same shape as real data."""
    bev   = torch.randn(batch_size, 3, 224, 224).to(DEVICE)    # Camera BEV
    lidar = torch.randn(batch_size, 16384, 3).to(DEVICE)        # LiDAR pts
    can   = torch.randn(batch_size, 8, 50).to(DEVICE)            # CAN signals (T=8, input_dim=50)
    gps   = torch.randn(batch_size, 50, 3).to(DEVICE)           # GPS trace
    try:
        from torch_geometric.data import Data as PyGData
        map_data = PyGData(
            x=torch.randn(32, 8).to(DEVICE),
            edge_index=torch.randint(0, 32, (2, 60)).to(DEVICE),
            batch=torch.zeros(32, dtype=torch.long).to(DEVICE)
        )
    except ImportError:
        map_data = {
            'x':          torch.randn(batch_size, 32, 4).to(DEVICE),
            'edge_index': torch.randint(0, 32, (batch_size, 2, 60)).to(DEVICE),
        }
    return {'bev': bev, 'lidar': lidar, 'can': can, 'gps': gps, 'map': map_data}

sample_A = make_sample()  # Segment A — e.g. highway driving
sample_B = make_sample()  # Segment B — e.g. urban driving
sample_C = make_sample()  # Segment A' — augmented version of A (should be similar)
# Make sample_C closer to A
sample_C['bev']   = sample_A['bev']   + torch.randn_like(sample_A['bev']) * 0.01
sample_C['lidar'] = sample_A['lidar'] + torch.randn_like(sample_A['lidar']) * 0.01
sample_C['can']   = sample_A['can']   + torch.randn_like(sample_A['can']) * 0.01
sample_C['gps']   = sample_A['gps']   + torch.randn_like(sample_A['gps']) * 0.001

print("[OK]    Inputs ready:")
print(f"        BEV   : {sample_A['bev'].shape}")
print(f"        LiDAR : {sample_A['lidar'].shape}")
print(f"        CAN   : {sample_A['can'].shape}")
print(f"        GPS   : {sample_A['gps'].shape}")

# ── Run Inference ─────────────────────────────────────────────────────────────
print(f"\n[STEP 3] Running inference...")

with torch.no_grad():
    t_inf = time.time()

    out_A = model(sample_A)
    out_B = model(sample_B)
    out_C = model(sample_C)

    inf_time = time.time() - t_inf

# ── Extract & Normalize Embeddings ────────────────────────────────────────────
z_A = nn.functional.normalize(out_A['fused'], dim=-1)  # (1, 2048)
z_B = nn.functional.normalize(out_B['fused'], dim=-1)
z_C = nn.functional.normalize(out_C['fused'], dim=-1)

print(f"[OK]    Inference time : {inf_time:.3f}s")
print(f"[INFO]  Embedding shape: {z_A.shape}  (2048-dim fused representation)")

# ── Similarity Analysis ───────────────────────────────────────────────────────
print(f"\n[STEP 4] Cosine similarity analysis...")

def safe_sim(a, b):
    dot = (a * b).sum().item()
    if not (dot == dot):  # nan check
        return 0.0
    return dot

sim_AC = safe_sim(z_A, z_C)
sim_AB = safe_sim(z_A, z_B)
sim_BC = safe_sim(z_B, z_C)

print()
print("  +---------------------------------------------+")
print("  |        EMBEDDING SIMILARITY RESULTS         |")
print("  +---------------------------------------------+")
print(f"  |  Segment A vs A' (same+noise): {sim_AC:+.4f}       |")
print(f"  |  Segment A vs B (different) : {sim_AB:+.4f}       |")
print(f"  |  Segment B vs C (different) : {sim_BC:+.4f}       |")
print("  +---------------------------------------------+")

# ── Per-Modality Embeddings ───────────────────────────────────────────────────
print(f"\n[STEP 5] Per-modality embedding norms...")
for key in ['bev', 'lidar', 'can', 'gps']:
    if key in out_A:
        norm = out_A[key].norm(dim=-1).mean().item()
        dim  = out_A[key].shape[-1]
        print(f"  {key.upper():<6} : dim={dim:<5} | L2 norm={norm:.4f}")

# ── VRAM Usage ────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    used  = torch.cuda.memory_allocated() / 1e9
    resrv = torch.cuda.memory_reserved() / 1e9
    print(f"\n[INFO]  GPU Memory used    : {used:.2f} GB")
    print(f"[INFO]  GPU Memory reserved: {resrv:.2f} GB")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  INFERENCE COMPLETE")
print(f"  Model: MultiModalEncoder (DINOv2 + PointNet + BiLSTM + Transformer + GCN)")
print(f"  Checkpoint: Epoch {ckpt['epoch']} | Step {ckpt['step']}")
print(f"  Output: 2048-dim fused scene embedding")
print(f"  Device: {DEVICE}")
print("=" * 60)
