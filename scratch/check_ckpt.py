import torch
import os

ckpt_path = r'c:\Users\faiza\OneDrive\Desktop\Tesla Model\checkpoints\ckpt_last.pt'
if os.path.exists(ckpt_path):
    try:
        # Loading on CPU since we only need the metadata
        ckpt = torch.load(ckpt_path, map_location='cpu')
        print(f"Epoch: {ckpt.get('epoch', 'N/A')}")
        print(f"Step: {ckpt.get('step', 'N/A')}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
else:
    print("Checkpoint file not found.")
