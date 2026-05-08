import torch
import torch.nn as nn
import torch.nn.functional as F
class VideoBEVEncoder(nn.Module):
    def __init__(self, freeze_backbone: bool = False):
        super().__init__()
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vitb14',
            pretrained=True, trust_repo=True
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
class TNet(nn.Module):
    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, k * k),
        )
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.eye_(self.fc[-1].bias.view(k, k))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x).max(dim=-1)[0]
        return self.fc(x).view(-1, self.k, self.k)
class PointNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tnet3  = TNet(k=3)
        self.tnet64 = TNet(k=64)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(2, 1)
        t3  = self.tnet3(x)
        x   = torch.bmm(t3, x)
        x   = self.mlp1(x)
        t64 = self.tnet64(x)
        x   = torch.bmm(t64, x)
        x   = self.mlp2(x)
        return x.max(dim=-1)[0]
class CANEncoder(nn.Module):
    def __init__(self, input_dim: int = 50, hidden_dim: int = 256,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_dim = hidden_dim * 2
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h, _) = self.lstm(x)
        return torch.cat([h[-2], h[-1]], dim=-1)
class GPSEncoder(nn.Module):
    def __init__(self, input_dim: int = 3, model_dim: int = 128,
                 nhead: int = 4, num_layers: int = 4, out_dim: int = 512):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nhead,
            dim_feedforward=model_dim * 4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj    = nn.Linear(model_dim, out_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.out_proj(x)
class MapGNNEncoder(nn.Module):
    def __init__(self, node_feat_dim: int = 8, hidden_dim: int = 128,
                 out_dim: int = 512, num_layers: int = 3):
        super().__init__()
        try:
            from torch_geometric.nn import GCNConv
        except ImportError:
            raise ImportError("pip install torch_geometric")
        layers = []
        in_dim = node_feat_dim
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            layers.append(GCNConv(in_dim, hidden_dim if not is_last else out_dim))
            in_dim = hidden_dim
        self.convs = nn.ModuleList(layers)
    def forward(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        from torch_geometric.nn import global_mean_pool
        return global_mean_pool(x, batch)
class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 2048), nn.LayerNorm(2048), nn.GELU(),
            nn.Linear(2048, 2048), nn.LayerNorm(2048), nn.GELU(),
            nn.Linear(2048, out_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)
