import h5py
import numpy as np

def check_nans(h5_path):
    with h5py.File(h5_path, 'r') as f:
        for k in f.keys():
            seg = f[k]
            for d in seg.keys():
                data = seg[d][:]
                if np.isnan(data).any():
                    print(f"NaN found in {k}/{d}")
                if np.isinf(data).any():
                    print(f"Inf found in {k}/{d}")

if __name__ == "__main__":
    check_nans('logs.h5')
    print("Check complete.")
