import h5py
import numpy as np
from tqdm import tqdm

def expand_h5(input_path, output_path, target_count=256):
    with h5py.File(input_path, 'r') as fin:
        src_keys = sorted(fin.keys())
        src_count = len(src_keys)
        print(f"Source has {src_count} segments.")
        
        with h5py.File(output_path, 'w') as fout:
            for i in tqdm(range(target_count), desc="Expanding"):
                # Cycle through source segments
                src_key = src_keys[i % src_count]
                src_group = fin[src_key]
                
                dst_group = fout.create_group(f"segment_{i}")
                for dset_name in src_group.keys():
                    data = src_group[dset_name][:]
                    # Add a tiny bit of noise to lidar/can/gps to make them "different"
                    if dset_name in ['lidar', 'can', 'gps'] and i >= src_count:
                        pass # No noise added as requested
                    
                    dst_group.create_dataset(dset_name, data=data)

if __name__ == "__main__":
    expand_h5('logs.h5', 'logs_256_clean.h5', target_count=256)
    print("Done! logs_256_clean.h5 created with 256 segments.")
