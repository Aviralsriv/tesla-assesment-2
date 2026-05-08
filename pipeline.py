import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from utils.bev import project_to_bev, voxelize_lidar
from utils.augment import DualViewAugment
try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
try:
    import torch_geometric.data as pyg_data
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
class VehicleLogDataset(Dataset):
    def __init__(self,
                 data_root: str,
                 seq_len: int = 50,
                 augment: bool = True,
                 bev_grid: tuple = (200, 200),
                 max_pts: int = 16384):
        self.data_root = data_root
        self.seq_len   = seq_len
        self.augment   = augment
        self.bev_grid  = bev_grid
        self.max_pts   = max_pts
        self.aug_fn    = DualViewAugment() if augment else None
        if data_root.endswith('.h5'):
            assert HDF5_AVAILABLE, "pip install h5py"
            self.hdf5_path = data_root
            with h5py.File(data_root, 'r') as f:
                self.segment_keys = sorted(f.keys())
        else:
            self.hdf5_path    = None
            self.segment_keys = sorted([
                d for d in os.listdir(data_root)
                if os.path.isdir(os.path.join(data_root, d))
            ])
    def __len__(self) -> int:
        return len(self.segment_keys)
    def _load_hdf5(self, key: str) -> dict:
        with h5py.File(self.hdf5_path, 'r') as f:
            seg = f[key]
            lidar  = seg['lidar'][:]
            camera = seg['camera'][:]
            can    = seg['can'][:self.seq_len]
            gps    = seg['gps'][:self.seq_len]
            mn     = seg['map_nodes'][:]
            me     = seg['map_edges'][:]
        return dict(lidar=lidar, camera=camera, can=can, gps=gps,
                    map_nodes=mn, map_edges=me)
    def _load_csv_dir(self, key: str) -> dict:
        seg_dir = os.path.join(self.data_root, key)
        lidar   = np.load(os.path.join(seg_dir, 'lidar.npy'))
        camera  = np.load(os.path.join(seg_dir, 'camera.npy'))
        can_df  = pd.read_csv(os.path.join(seg_dir, 'can.csv'))
        gps_df  = pd.read_csv(os.path.join(seg_dir, 'gps.csv'))
        mn      = np.load(os.path.join(seg_dir, 'map_nodes.npy'))
        me      = np.load(os.path.join(seg_dir, 'map_edges.npy'))
        return dict(
            lidar=lidar, camera=camera,
            can=can_df.values[:self.seq_len].astype(np.float32),
            gps=gps_df[['lat','lon','vel']].values[:self.seq_len].astype(np.float32),
            map_nodes=mn, map_edges=me,
        )
    def _preprocess(self, raw: dict) -> dict:
        lidar_pts = raw['lidar'].astype(np.float32)
        if len(lidar_pts) > self.max_pts:
            idx       = np.random.choice(len(lidar_pts), self.max_pts, replace=False)
            lidar_pts = lidar_pts[idx]
        bev = project_to_bev(lidar_pts, grid_size=self.bev_grid)
        lidar_tensor = torch.from_numpy(lidar_pts[:, :3])
        can = torch.from_numpy(raw['can'])
        gps = torch.from_numpy(raw['gps'])
        if PYG_AVAILABLE:
            mn = torch.from_numpy(raw['map_nodes'].astype(np.float32))
            me = torch.from_numpy(raw['map_edges'].astype(np.int64))
            map_data = pyg_data.Data(x=mn, edge_index=me)
        else:
            map_data = {
                'x': torch.from_numpy(raw['map_nodes'].astype(np.float32)),
                'edge_index': torch.from_numpy(raw['map_edges'].astype(np.int64)),
            }
        return dict(bev=bev, lidar=lidar_tensor, can=can, gps=gps, map=map_data)
    def __getitem__(self, idx: int) -> dict:
        key = self.segment_keys[idx]
        raw = self._load_hdf5(key) if self.hdf5_path else self._load_csv_dir(key)
        sample = self._preprocess(raw)
        if self.augment:
            view1, view2 = self.aug_fn(sample)
            return {'view1': view1, 'view2': view2, 'key': key}
        return sample
def ssl_collate_fn(batch: list) -> dict:
    def collate_view(view_key: str) -> dict:
        views = [item[view_key] for item in batch]
        out   = {}
        for mod in ['bev', 'lidar', 'can', 'gps']:
            if mod in views[0]:
                out[mod] = torch.stack([v[mod] for v in views])
        if PYG_AVAILABLE:
            from torch_geometric.data import Batch
            out['map'] = Batch.from_data_list([v['map'] for v in views])
        else:
            out['map'] = {
                'x':          torch.stack([v['map']['x'] for v in views]),
                'edge_index': torch.stack([v['map']['edge_index'] for v in views]),
            }
        return out
    return {
        'view1': collate_view('view1'),
        'view2': collate_view('view2'),
    }
def build_dataloader(data_root: str,
                     batch_size: int = 256,
                     num_workers: int = 8,
                     augment: bool = True) -> DataLoader:
    dataset = VehicleLogDataset(data_root, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=augment,
        num_workers=num_workers,
        collate_fn=ssl_collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )