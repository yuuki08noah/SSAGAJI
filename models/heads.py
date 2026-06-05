"""
순수 PyTorch 헤드 모듈 — transformers 의존성 없음.
Mock 테스트에서 직접 import 가능하다.
"""

import torch
import torch.nn as nn


class EmotionClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_emotions: int = 9, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_emotions)

    def forward(self, x: torch.Tensor):
        return self.classifier(self.dropout(x))


class RegressionHead(nn.Module):
    """1~5점 회귀 헤드. Sigmoid → [1, 5] 스케일링."""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        self.scale = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x).squeeze(-1)
        return self.scale(raw) * 4.0 + 1.0  # [1, 5]


class TextRegressionHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.scale = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x).squeeze(-1)
        return self.scale(raw) * 4.0 + 1.0
