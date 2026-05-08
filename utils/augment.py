import random
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
class BEVAugment(nn.Module):
    def __init__(self, img_size: int = 224, is_global: bool = True):
        super().__init__()
        scale = (0.4, 1.0) if is_global else (0.05, 0.4)
        self.transforms = T.Compose([
            T.RandomResizedCrop(img_size, scale=scale, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] > 3:
            x = x[:3]
        elif x.shape[0] < 3:
            x = x.repeat(3 // x.shape[0] + 1, 1, 1)[:3]
        return self.transforms(x)
class MultiCropAugment(nn.Module):
    def __init__(self, n_local: int = 6):
        super().__init__()
        self.global_aug = BEVAugment(224, is_global=True)
        self.local_aug  = BEVAugment(96,  is_global=False)
        self.n_local    = n_local
    def forward(self, x: torch.Tensor) -> list:
        views  = [self.global_aug(x), self.global_aug(x)]
        views += [self.local_aug(x) for _ in range(self.n_local)]
        return views
class LiDARPointDropout(nn.Module):
    def __init__(self, drop_ratio: float = 0.3, jitter_std: float = 0.02):
        super().__init__()
        self.drop_ratio = drop_ratio
        self.jitter_std = jitter_std
    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        N = pts.shape[0]
        keep = int(N * (1 - self.drop_ratio))
        idx  = torch.randperm(N)[:keep]
        pts  = pts[idx]
        pts  = pts + torch.randn_like(pts) * self.jitter_std
        return pts
class CANAugment(nn.Module):
    def __init__(self, noise_std: float = 0.01, mask_ratio: float = 0.1):
        super().__init__()
        self.noise_std  = noise_std
        self.mask_ratio = mask_ratio
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + torch.randn_like(x) * self.noise_std
        T = x.shape[0]
        n_mask = int(T * self.mask_ratio)
        if n_mask > 0:
            idx = torch.randperm(T)[:n_mask]
            x[idx] = 0.0
        return x
class GPSAugment(nn.Module):
    def __init__(self, coord_std: float = 0.0001, vel_std: float = 0.5):
        super().__init__()
        self.coord_std = coord_std
        self.vel_std   = vel_std
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.zeros_like(x)
        noise[:, :2] = torch.randn_like(x[:, :2]) * self.coord_std
        noise[:, 2]  = torch.randn_like(x[:, 2])  * self.vel_std
        return x + noise
class DualViewAugment(nn.Module):
    def __init__(self):
        super().__init__()
        self.bev_aug   = MultiCropAugment(n_local=6)
        self.lidar_aug = LiDARPointDropout()
        self.can_aug   = CANAugment()
        self.gps_aug   = GPSAugment()
    def _augment_one(self, sample: dict) -> dict:
        out = {}
        bev_views  = self.bev_aug(sample['bev'])
        out['bev'] = bev_views[0]
        out['bev_local_views'] = bev_views[2:]
        out['lidar'] = self.lidar_aug(sample['lidar'])
        out['can']   = self.can_aug(sample['can'])
        out['gps']   = self.gps_aug(sample['gps'])
        out['map']   = sample['map']
        return out
    def forward(self, sample: dict) -> tuple:
        return self._augment_one(sample), self._augment_one(sample)