import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import numpy as np
from models.fusion import MultiModalEncoder
from pipeline import VehicleLogDataset, ssl_collate_fn
@torch.no_grad()
def extract_features(model: MultiModalEncoder,
                     loader: DataLoader,
                     device: torch.device,
                     return_fused: bool = True) -> tuple:
    model.eval()
    all_feats, all_labels = [], []
    for batch in loader:
        sample  = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in batch.items() if k != 'label'}
        labels  = batch.get('label')
        out     = model(sample)
        feats   = out['fused'] if return_fused else torch.cat(
            list(out['proj_feats'].values()), dim=-1)
        all_feats.append(feats.cpu().numpy())
        if labels is not None:
            all_labels.append(labels.numpy())
    X = np.concatenate(all_feats)
    y = np.concatenate(all_labels) if all_labels else None
    return X, y
class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
class TrajectoryHead(nn.Module):
    def __init__(self, in_dim: int = 2048, hidden_dim: int = 512,
                 T_pred: int = 30):
        super().__init__()
        self.T_pred = T_pred
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, T_pred * 2),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, self.T_pred, 2)
def ade(pred: torch.Tensor, gt: torch.Tensor) -> float:
    return (pred - gt).norm(dim=-1).mean().item()
def fde(pred: torch.Tensor, gt: torch.Tensor) -> float:
    return (pred[:, -1] - gt[:, -1]).norm(dim=-1).mean().item()
def eval_linear_probe(train_X: np.ndarray, train_y: np.ndarray,
                      val_X:   np.ndarray, val_y:   np.ndarray,
                      num_classes: int, epochs: int = 10,
                      device: torch.device = torch.device('cpu')) -> float:
    in_dim = train_X.shape[-1]
    probe  = LinearProbe(in_dim, num_classes).to(device)
    opt    = torch.optim.Adam(probe.parameters(), lr=1e-3)
    X_t = torch.from_numpy(train_X).float().to(device)
    y_t = torch.from_numpy(train_y).long().to(device)
    X_v = torch.from_numpy(val_X).float().to(device)
    y_v = val_y
    for ep in range(epochs):
        probe.train()
        logits = probe(X_t)
        loss   = F.cross_entropy(logits, y_t)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 5 == 0:
            print(f'  probe epoch {ep+1}/{epochs}  loss={loss.item():.4f}')
    probe.eval()
    with torch.no_grad():
        probs = torch.softmax(probe(X_v), dim=-1).cpu().numpy()
    if num_classes == 2:
        auroc = roc_auc_score(y_v, probs[:, 1])
    else:
        auroc = roc_auc_score(y_v, probs, multi_class='ovr')
    print(f'  Linear probe AUROC: {auroc:.4f}')
    return auroc
def eval_trajectory(model: MultiModalEncoder,
                    traj_head: TrajectoryHead,
                    val_loader: DataLoader,
                    device: torch.device) -> dict:
    model.eval(); traj_head.eval()
    all_ade, all_fde = [], []
    with torch.no_grad():
        for batch in val_loader:
            sample = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in batch.items() if k not in ('label', 'traj_gt')}
            gt = batch['traj_gt'].to(device)
            out   = model(sample)
            pred  = traj_head(out['fused'])
            all_ade.append(ade(pred, gt))
            all_fde.append(fde(pred, gt))
    metrics = {'ADE': np.mean(all_ade), 'FDE': np.mean(all_fde)}
    print(f"  Trajectory → ADE={metrics['ADE']:.3f}m  FDE={metrics['FDE']:.3f}m")
    return metrics
def parse_args():
    p = argparse.ArgumentParser('Vehicle SSL Evaluation')
    p.add_argument('--ckpt',        type=str, required=True)
    p.add_argument('--data_root',   type=str, required=True)
    p.add_argument('--task',        type=str, default='probe',
                   choices=['probe', 'trajectory', 'both'])
    p.add_argument('--num_classes', type=int, default=2)
    p.add_argument('--probe_epochs',type=int, default=10)
    p.add_argument('--batch_size',  type=int, default=64)
    p.add_argument('--num_workers', type=int, default=4)
    return p.parse_args()
def main():
    cfg    = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiModalEncoder().to(device)
    ckpt  = torch.load(cfg.ckpt, map_location=device)
    model.load_state_dict(ckpt['student'])
    for p in model.parameters():
        p.requires_grad_(False)
    print(f'Loaded checkpoint: {cfg.ckpt}')
    train_ds = VehicleLogDataset(cfg.data_root, augment=False)
    n_train  = int(0.8 * len(train_ds))
    n_val    = len(train_ds) - n_train
    train_ds, val_ds = torch.utils.data.random_split(
        train_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    def make_loader(ds):
        return DataLoader(ds, batch_size=cfg.batch_size,
                          num_workers=cfg.num_workers,
                          collate_fn=ssl_collate_fn)
    if cfg.task in ('probe', 'both'):
        print('\n── Linear Probe ─────────────────────────────────────────')
        train_X, train_y = extract_features(model, make_loader(train_ds), device)
        val_X,   val_y   = extract_features(model, make_loader(val_ds),   device)
        if train_y is not None:
            eval_linear_probe(train_X, train_y, val_X, val_y,
                              cfg.num_classes, cfg.probe_epochs, device)
        else:
            print('  No labels found in dataset — skipping probe.')
    if cfg.task in ('trajectory', 'both'):
        print('\n── Trajectory Evaluation ────────────────────────────────')
        traj_head = TrajectoryHead(in_dim=2048).to(device)
        eval_trajectory(model, traj_head, make_loader(val_ds), device)
if __name__ == '__main__':
    main()