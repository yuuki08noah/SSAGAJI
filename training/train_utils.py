"""학습 공통 유틸리티."""

import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from config import train_cfg


def set_seed(seed: int = train_cfg.seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float = train_cfg.weight_decay):
    # Backbone은 낮은 LR, 헤드는 높은 LR
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    return AdamW([
        {"params": backbone_params, "lr": lr * 0.1},
        {"params": head_params, "lr": lr},
    ], weight_decay=weight_decay)


def build_scheduler(optimizer, num_training_steps: int, warmup_ratio: float = train_cfg.warmup_ratio):
    warmup_steps = int(num_training_steps * warmup_ratio)
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingWarmRestarts(optimizer, T_0=num_training_steps - warmup_steps)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


class RMSELoss(torch.nn.Module):
    """회귀 손실: RMSE (점수 스케일과 직관적으로 대응)."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = torch.mean((pred - target) ** 2)
        return torch.sqrt(mse + self.eps)


class EarlyStopper:
    def __init__(self, patience: int = train_cfg.patience, mode: str = "min"):
        self.patience = patience
        self.mode = mode
        self.best = float("inf") if mode == "min" else float("-inf")
        self.counter = 0

    def __call__(self, value: float) -> bool:
        improved = (self.mode == "min" and value < self.best) or \
                   (self.mode == "max" and value > self.best)
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience  # True → 학습 중단


def save_checkpoint(model, optimizer, epoch: int, val_loss: float, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_loss": val_loss,
    }, path / "checkpoint.pt")
    model.save_pretrained(str(path)) if hasattr(model, "save_pretrained") else None


def load_checkpoint(model, path: Path, device: torch.device):
    ckpt = torch.load(path / "checkpoint.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return ckpt["epoch"], ckpt["val_loss"]
