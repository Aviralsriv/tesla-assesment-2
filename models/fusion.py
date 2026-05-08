import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoders import (
    VideoBEVEncoder, PointNetEncoder, CANEncoder,
    GPSEncoder, MapGNNEncoder, ProjectionHead,
)
MODALITY_DIMS = {
    'bev':   768,
    'lidar': 1024,
    'can':   512,
    'gps':   512,
    'map':   512,
}
PROJ_DIM  = 1024
FUSED_DIM = 2048
class FusionTransformer(nn.Module):
    def __init__(self, in_dim: int = PROJ_DIM, out_dim: int = FUSED_DIM,
                 nhead: int = 8, num_layers: int = 2):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=in_dim, nhead=nhead,
            dim_feedforward=in_dim * 4,
            dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_proj    = nn.Linear(in_dim * 5, out_dim)
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.transformer(tokens)
        x = x.flatten(1)
        return self.out_proj(x)
class MultiModalEncoder(nn.Module):
    def __init__(self, freeze_dino: bool = False):
        super().__init__()
        self.encoders = nn.ModuleDict({
            'bev':   VideoBEVEncoder(freeze_backbone=freeze_dino),
            'lidar': PointNetEncoder(),
            'can':   CANEncoder(),
            'gps':   GPSEncoder(),
            'map':   MapGNNEncoder(),
        })
        self.projectors = nn.ModuleDict({
            mod: ProjectionHead(in_dim=dim, out_dim=PROJ_DIM)
            for mod, dim in MODALITY_DIMS.items()
        })
        self.fusion = FusionTransformer(in_dim=PROJ_DIM, out_dim=FUSED_DIM)
        self.task_head = None
    def encode_modality(self, name: str, data) -> torch.Tensor:
        raw  = self.encoders[name](data)
        return self.projectors[name](raw)
    def forward(self, batch: dict) -> dict:
        proj_feats = {}
        for mod in MODALITY_DIMS:
            if mod in batch:
                proj_feats[mod] = self.encode_modality(mod, batch[mod])
        tokens = torch.stack(list(proj_feats.values()), dim=1)
        fused  = self.fusion(tokens)
        return {'proj_feats': proj_feats, 'fused': fused}
def build_teacher(student: MultiModalEncoder) -> MultiModalEncoder:
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher
@torch.no_grad()
def update_ema(student: nn.Module, teacher: nn.Module, momentum: float = 0.996):
    for s_param, t_param in zip(student.parameters(), teacher.parameters()):
        t_param.data.mul_(momentum).add_(s_param.data, alpha=1.0 - momentum)
