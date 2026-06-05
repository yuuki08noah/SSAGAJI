"""
오디오 모델: Wav2Vec2 / HuBERT 백본 + 회귀 헤드.

Stage 1: 감정 분류 태스크로 사전학습 (분노/냉소 등 무례함 관련 노드 강화)
Stage 2: 1~5점 회귀 태스크로 파인튜닝
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from config import audio_cfg
from models.heads import EmotionClassificationHead, RegressionHead


class AudioBackbone(nn.Module):
    """Wav2Vec2 또는 HuBERT 백본. 최상위 트랜스포머 레이어 출력을 반환한다."""

    def __init__(self, model_name: str = audio_cfg.backbone, freeze_feature_extractor: bool = True):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size

        # CNN 특징 추출기는 동결 (학습 불필요한 저수준 필터)
        if freeze_feature_extractor and hasattr(self.model, "feature_extractor"):
            for param in self.model.feature_extractor.parameters():
                param.requires_grad = False

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor = None):
        outputs = self.model(input_values=input_values, attention_mask=attention_mask)
        # (B, T, H) → 평균 풀링 → (B, H)
        hidden = outputs.last_hidden_state
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden.mean(dim=1)
        return pooled  # (B, H)



class AudioSsagajiModel(nn.Module):
    """
    전체 오디오 모델.
    stage="emotion"이면 분류 헤드, stage="regression"이면 회귀 헤드를 사용한다.
    """

    def __init__(
        self,
        model_name: str = audio_cfg.backbone,
        num_emotions: int = 9,
        stage: str = "regression",
        freeze_feature_extractor: bool = True,
    ):
        super().__init__()
        self.backbone = AudioBackbone(model_name, freeze_feature_extractor)
        hidden = self.backbone.hidden_size

        self.emotion_head = EmotionClassificationHead(hidden, num_emotions)
        self.regression_head = RegressionHead(hidden)
        self.stage = stage

    def set_stage(self, stage: str):
        """학습 단계를 전환한다: 'emotion' | 'regression'."""
        assert stage in ("emotion", "regression")
        self.stage = stage

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor = None):
        features = self.backbone(input_values, attention_mask)
        if self.stage == "emotion":
            return self.emotion_head(features)
        return self.regression_head(features)

    def get_features(self, input_values: torch.Tensor, attention_mask: torch.Tensor = None):
        """앙상블에서 사용할 중간 표현을 반환한다."""
        return self.backbone(input_values, attention_mask)
