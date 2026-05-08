import os
import h5py
import numpy as np
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

def convert_nuscenes_to_h5(nusc_path, out_path, seq_len=50):
    nusc = NuScenes(version='v1.0-mini', dataroot=nusc_path, verbose=True)
    
    with h5py.File(out_path, 'w') as f:
        for i, scene in enumerate(tqdm(nusc.scene, desc="Processing scenes")):
            group = f.create_group(f"segment_{i}")
            
            # Collect sample tokens
            sample_tokens = []
            curr_token = scene['first_sample_token']
            while curr_token != '':
                sample_tokens.append(curr_token)
                curr_token = nusc.get('sample', curr_token)['next']
            
            # LiDAR (take first)
            sample = nusc.get('sample', sample_tokens[0])
            lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            pcl_path = os.path.join(nusc_path, lidar_data['filename'])
            pc = LidarPointCloud.from_file(pcl_path)
            group.create_dataset('lidar', data=pc.points.T.astype(np.float32))
            
            # Camera (dummy)
            group.create_dataset('camera', data=np.zeros((seq_len, 224, 224, 3), dtype=np.uint8))
            
            # CAN & GPS with padding/truncating to seq_len
            can_seq = np.zeros((seq_len, 50), dtype=np.float32)
            gps_seq = np.zeros((seq_len, 3), dtype=np.float32)
            
            for j, st in enumerate(sample_tokens[:seq_len]):
                s = nusc.get('sample', st)
                sd = nusc.get('sample_data', s['data']['LIDAR_TOP'])
                ep = nusc.get('ego_pose', sd['ego_pose_token'])
                
                can_seq[j, 0:3] = ep['translation']
                can_seq[j, 3:7] = ep['rotation']
                gps_seq[j, 0:2] = ep['translation'][:2]
            
            group.create_dataset('can', data=can_seq)
            group.create_dataset('gps', data=gps_seq)
            
            # Map
            group.create_dataset('map_nodes', data=np.zeros((10, 8), dtype=np.float32))
            group.create_dataset('map_edges', data=np.zeros((2, 20), dtype=np.int64))

    print(f"Finished! Saved to {out_path}")

if __name__ == "__main__":
    convert_nuscenes_to_h5(
        nusc_path=r"C:\Users\faiza\OneDrive\Desktop\Tesla Model\Dataset",
        out_path=r"C:\Users\faiza\OneDrive\Desktop\Tesla Model\logs.h5"
    )
