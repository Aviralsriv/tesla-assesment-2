import os
import copy
import json
import argparse
import math
import time
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from models.fusion import MultiModalEncoder, build_teacher, update_ema
from models.losses import TotalSSLLoss
from pipeline import build_dataloader
try:
    import wandb
    WANDB = True
except ImportError:
    WANDB = False
try:
    from torch.utils.tensorboard import SummaryWriter
    TB = True
except ImportError:
    TB = False
def cosine_schedule(optimizer: torch.optim.Optimizer,
                    base_lr: float,
                    current_step: int,
                    warmup_steps: int,
                    total_steps: int,
                    min_lr: float = 1e-6):
    if current_step < warmup_steps:
        lr = base_lr * current_step / max(warmup_steps, 1)
    else:
        progress = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
        lr       = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr
class SSLTrainer:
    def __init__(self, cfg: argparse.Namespace):
        self.cfg     = cfg
        self.device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.step    = 0
        self.epoch   = 0
        self.student = MultiModalEncoder(freeze_dino=cfg.freeze_dino).to(self.device)
        self.teacher = build_teacher(self.student)
        self.criterion = TotalSSLLoss(
            temperature=cfg.temperature,
            dino_out_dim=2048,
        ).to(self.device)
        decay_params    = []
        no_decay_params = []
        for name, p in self.student.named_parameters():
            if not p.requires_grad:
                continue
            if 'bias' in name or 'norm' in name or 'bn' in name:
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        self.optimizer = torch.optim.AdamW([
            {'params': decay_params,    'weight_decay': cfg.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ], lr=cfg.lr, betas=(0.9, 0.999))
        self.scaler = GradScaler('cuda', enabled=cfg.amp)
        self.loader = build_dataloader(
            cfg.data_root,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )
        total_steps       = len(self.loader) * cfg.epochs
        self.total_steps  = total_steps
        self.warmup_steps = int(total_steps * 0.05)
        if WANDB and cfg.wandb:
            wandb.init(project='vehicle-ssl', config=vars(cfg))
        self.writer = SummaryWriter(cfg.log_dir) if TB else None
        os.makedirs(cfg.ckpt_dir, exist_ok=True)
        if cfg.resume:
            self._load_checkpoint(cfg.resume)
    def _save_checkpoint(self, tag: str = ''):
        # Save last checkpoint
        last_path = os.path.join(self.cfg.ckpt_dir, 'ckpt_last.pt')
        state = {
            'student':   self.student.state_dict(),
            'teacher':   self.teacher.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler':    self.scaler.state_dict(),
            'step':      self.step,
            'epoch':     self.epoch,
        }
        torch.save(state, last_path)
        
        # If tag is provided (like step count), save a specific version
        if tag:
            tagged_path = os.path.join(self.cfg.ckpt_dir, f'ckpt_{tag}.pt')
            torch.save(state, tagged_path)
            print(f'[ckpt] saved -> {tagged_path}')
        else:
            print(f'[ckpt] updated last -> {last_path}')
    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.student.load_state_dict(ckpt['student'])
        self.teacher.load_state_dict(ckpt['teacher'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scaler.load_state_dict(ckpt['scaler'])
        self.step  = ckpt['step']
        self.epoch = ckpt['epoch']
        print(f'[ckpt] resumed from {path}  (epoch={self.epoch}, step={self.step})')
    def _train_step(self, batch: dict) -> dict:
        v1 = {k: v.to(self.device) if hasattr(v, 'to') else v
              for k, v in batch['view1'].items()}
        v2 = {k: v.to(self.device) if hasattr(v, 'to') else v
              for k, v in batch['view2'].items()}
        with autocast('cuda', enabled=self.cfg.amp):
            out1 = self.student(v1)
            out2 = self.student(v2)
            with torch.no_grad():
                teacher_out2 = self.teacher(v2)
            losses = self.criterion(out1, out2, teacher_out2)
            
            # Calculate Alignment Accuracy (for fused features)
            with torch.no_grad():
                z1 = nn.functional.normalize(out1['fused'], dim=-1)
                z2 = nn.functional.normalize(out2['fused'], dim=-1)
                N      = z1.size(0)
                sim    = torch.mm(z1, z2.T)
                labels = torch.arange(N, device=self.device)
                acc    = (sim.argmax(dim=1) == labels).float().mean()
                losses['accuracy'] = acc

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(losses['total']).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=3.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        update_ema(self.student, self.teacher, momentum=self.cfg.ema_momentum)
        return {k: v.item() for k, v in losses.items()}
    def _log(self, loss_dict: dict, lr: float, epoch: int, step: int, t: float):
        acc = loss_dict.get('accuracy', 0) * 100
        msg = (f'[E{epoch:03d} S{step:06d}] '
               f'loss={loss_dict["total"]:.4f} | '
               f'acc={acc:.1f}% | '
               f't={t:.2f}s')
        print(msg, flush=True)
        if WANDB and self.cfg.wandb:
            wandb.log({'step': step, 'lr': lr, **loss_dict})
        if self.writer:
            self.writer.add_scalar('lr', lr, step)
            for k, v in loss_dict.items():
                self.writer.add_scalar(f'loss/{k}', v, step)
        
        # Write to progress.json for dashboard
        progress = {
            'epoch': epoch,
            'step': step,
            'total_epochs': self.cfg.epochs,
            'total_steps': self.total_steps,
            'loss': float(loss_dict['total']),
            'accuracy': float(acc),
            'lr': float(lr),
            'time': float(t),
            'timestamp': time.time()
        }
        with open('progress.json', 'w') as f:
            json.dump(progress, f)
    def train(self):
        print(f'Training on {self.device} | '
              f'warmup={self.warmup_steps}')
        for epoch in range(self.epoch, self.cfg.epochs):
            self.epoch = epoch
            self.student.train()
            for batch in self.loader:
                t0  = time.time()
                lr  = cosine_schedule(
                    self.optimizer, self.cfg.lr,
                    self.step, self.warmup_steps, self.total_steps,
                )
                losses = self._train_step(batch)
                
                # Check for NaNs
                if math.isnan(losses['total']):
                    print(f'[WARNING] NaN loss detected at step {self.step}! Skipping batch.')
                    continue

                self.step += 1
                if self.step % self.cfg.log_every == 0:
                    self._log(losses, lr, epoch, self.step, time.time() - t0)
                
                if self.step % self.cfg.save_every == 0:
                    self._save_checkpoint(tag=f'step{self.step}')
            
            self._save_checkpoint(tag=f'epoch{epoch}')
        print('Training complete.')
        if WANDB and self.cfg.wandb:
            wandb.finish()
def parse_args():
    p = argparse.ArgumentParser('Vehicle SSL Pretraining')
    p.add_argument('--data_root',    type=str,   required=True)
    p.add_argument('--ckpt_dir',     type=str,   default='checkpoints')
    p.add_argument('--log_dir',      type=str,   default='runs/ssl')
    p.add_argument('--resume',       type=str,   default=None)
    p.add_argument('--epochs',       type=int,   default=20)
    p.add_argument('--batch_size',   type=int,   default=8)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=0.04)
    p.add_argument('--temperature',  type=float, default=0.1)
    p.add_argument('--ema_momentum', type=float, default=0.996)
    p.add_argument('--num_workers',  type=int,   default=8)
    p.add_argument('--log_every',    type=int,   default=1)
    p.add_argument('--save_every',   type=int,   default=10000)
    p.add_argument('--amp',          action='store_true', default=False)
    p.add_argument('--freeze_dino',  action='store_true', default=True)
    p.add_argument('--wandb',        action='store_true', default=False)
    return p.parse_args()
if __name__ == '__main__':
    cfg     = parse_args()
    trainer = SSLTrainer(cfg)
    trainer.train()