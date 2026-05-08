import numpy as np
import torch
import torch.nn.functional as F
def project_to_bev(lidar_pts: np.ndarray,
                   camera_img: np.ndarray = None,
                   grid_size: tuple = (200, 200),
                   x_range: tuple = (-50, 50),
                   y_range: tuple = (-50, 50),
                   z_range: tuple = (-3, 3),
                   resolution: float = 0.5) -> torch.Tensor:
    pts = lidar_pts[:, :3]
    intensity = lidar_pts[:, 3] if lidar_pts.shape[1] > 3 else np.ones(len(pts))
    mask = (
        (pts[:, 0] >= x_range[0]) & (pts[:, 0] < x_range[1]) &
        (pts[:, 1] >= y_range[0]) & (pts[:, 1] < y_range[1]) &
        (pts[:, 2] >= z_range[0]) & (pts[:, 2] < z_range[1])
    )
    pts, intensity = pts[mask], intensity[mask]
    H, W = grid_size
    bev_height_max  = np.zeros((H, W), dtype=np.float32)
    bev_height_mean = np.zeros((H, W), dtype=np.float32)
    bev_intensity   = np.zeros((H, W), dtype=np.float32)
    bev_density     = np.zeros((H, W), dtype=np.float32)
    count_grid      = np.zeros((H, W), dtype=np.float32)
    ix = ((pts[:, 0] - x_range[0]) / resolution).astype(np.int32)
    iy = ((pts[:, 1] - y_range[0]) / resolution).astype(np.int32)
    ix = np.clip(ix, 0, H - 1)
    iy = np.clip(iy, 0, W - 1)
    for i in range(len(pts)):
        x_idx, y_idx = ix[i], iy[i]
        z_val = pts[i, 2]
        if z_val > bev_height_max[x_idx, y_idx]:
            bev_height_max[x_idx, y_idx] = z_val
        bev_height_mean[x_idx, y_idx] += z_val
        bev_intensity[x_idx, y_idx]   += intensity[i]
        count_grid[x_idx, y_idx]      += 1
    nonzero = count_grid > 0
    bev_height_mean[nonzero] /= count_grid[nonzero]
    bev_intensity[nonzero]   /= count_grid[nonzero]
    bev_density               = np.log1p(count_grid) / np.log1p(count_grid.max() + 1e-6)
    bev = np.stack([bev_height_max, bev_height_mean, bev_intensity, bev_density], axis=0)
    return torch.from_numpy(bev)
def voxelize_lidar(lidar_pts: np.ndarray,
                   voxel_size: float = 0.2,
                   max_pts_per_voxel: int = 32,
                   max_voxels: int = 20000) -> torch.Tensor:
    pts = lidar_pts[:, :3].astype(np.float32)
    voxel_coords = np.floor(pts / voxel_size).astype(np.int32)
    voxel_dict: dict = {}
    for i, coord in enumerate(map(tuple, voxel_coords)):
        if coord not in voxel_dict:
            voxel_dict[coord] = []
        if len(voxel_dict[coord]) < max_pts_per_voxel:
            voxel_dict[coord].append(pts[i])
    keys = list(voxel_dict.keys())[:max_voxels]
    out = np.zeros((max_voxels, max_pts_per_voxel, 3), dtype=np.float32)
    for idx, key in enumerate(keys):
        vpts = np.array(voxel_dict[key])
        out[idx, :len(vpts)] = vpts
    return torch.from_numpy(out)
