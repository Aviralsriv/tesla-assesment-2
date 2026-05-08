import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.tau = temperature
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        N = z1.size(0)
        z   = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.T) / self.tau
        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, -1e4)
        labels = torch.cat([
            torch.arange(N, 2 * N, device=z.device),
            torch.arange(0, N,     device=z.device),
        ])
        return F.cross_entropy(sim, labels)
@torch.no_grad()
def sinkhorn(scores: torch.Tensor,
             n_iters: int = 3,
             eps: float = 0.05) -> torch.Tensor:
    Q = torch.exp(scores / eps).T
    Q /= Q.sum()
    N, K = scores.shape
    for _ in range(n_iters):
        Q /= Q.sum(dim=1, keepdim=True) * K
        Q /= Q.sum(dim=0, keepdim=True) * N
    return Q.T
class SwAVLoss(nn.Module):
    def __init__(self, n_prototypes: int = 3000, feat_dim: int = 1024,
                 temperature: float = 0.1):
        super().__init__()
        self.tau        = temperature
        self.prototypes = nn.Linear(feat_dim, n_prototypes, bias=False)
        nn.init.normal_(self.prototypes.weight, std=0.01)
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.prototypes.weight.copy_(
                F.normalize(self.prototypes.weight, dim=1)
            )
        scores1 = self.prototypes(z1)
        scores2 = self.prototypes(z2)
        q1 = sinkhorn(scores1.detach())
        q2 = sinkhorn(scores2.detach())
        p1 = F.softmax(scores1 / self.tau, dim=-1)
        p2 = F.softmax(scores2 / self.tau, dim=-1)
        loss = -0.5 * (
            (q1 * torch.log(p2 + 1e-8)).sum(dim=-1).mean() +
            (q2 * torch.log(p1 + 1e-8)).sum(dim=-1).mean()
        )
        return loss
class DINOLoss(nn.Module):
    def __init__(self, out_dim: int = 1024,
                 student_temp: float = 0.1,
                 teacher_temp: float = 0.04,
                 center_ema: float = 0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_ema   = center_ema
        self.register_buffer('center', torch.zeros(1, out_dim))
    @torch.no_grad()
    def update_center(self, teacher_out: torch.Tensor):
        batch_center = teacher_out.mean(dim=0, keepdim=True)
        self.center  = self.center * self.center_ema + batch_center * (1 - self.center_ema)
    def forward(self, student_out: torch.Tensor,
                teacher_out: torch.Tensor) -> torch.Tensor:
        student_probs = F.log_softmax(student_out / self.student_temp, dim=-1)
        teacher_probs = F.softmax(
            (teacher_out - self.center) / self.teacher_temp, dim=-1
        )
        self.update_center(teacher_out)
        return -(teacher_probs * student_probs).sum(dim=-1).mean()
class TotalSSLLoss(nn.Module):
    def __init__(self, modalities: list = None,
                 temperature: float = 0.1,
                 dino_out_dim: int = 2048):
        super().__init__()
        self.modalities  = modalities or ['bev', 'lidar', 'can', 'gps', 'map']
        self.nt_xent     = NTXentLoss(temperature=temperature)
        self.dino_loss   = DINOLoss(out_dim=dino_out_dim)
    def forward(self,
                out1: dict, out2: dict,
                teacher_out2: dict) -> dict:
        losses = {}
        for mod in self.modalities:
            if mod in out1['proj_feats'] and mod in out2['proj_feats']:
                losses[f'ntxent_{mod}'] = self.nt_xent(
                    out1['proj_feats'][mod],
                    out2['proj_feats'][mod],
                )
        losses['ntxent_fused'] = self.nt_xent(out1['fused'], out2['fused'])
        losses['dino'] = self.dino_loss(out1['fused'], teacher_out2['fused'])
        per_mod_w = 1.0 / max(len(self.modalities), 1)
        total = (
            sum(v * per_mod_w for k, v in losses.items() if k.startswith('ntxent_') and k != 'ntxent_fused') +
            losses['ntxent_fused'] * 1.0 +
            losses['dino']          * 0.5
        )
        losses['total'] = total
        return losses