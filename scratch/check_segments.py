import h5py
with h5py.File(r'd:\Tesla Model\logs_256.h5', 'r') as f:
    print(f"Total segments: {len(f.keys())}")
