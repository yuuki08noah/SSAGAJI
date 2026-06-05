"""
텍스트 모델: KoBERT / KLUE-BERT 백본 + 회귀 헤드.
종결 어미, 존댓말 여부, 비속어 등 언어적 무례함을 포착한다.
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from config import text_cfg
from models.heads import TextRegressionHead


class TextBackbone(nn.Module):
    """BERT 계열 백본. [CLS] 토큰 벡터를 반환한다."""

    def __init__(self, model_name: str = text_cfg.backbone, freeze_layers: int = 6):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size

        # 하위 레이어 동결 (파인튜닝 안정성 확보)
        if freeze_layers > 0 and hasattr(self.model, "encoder"):
            for i, layer in enumerate(self.model.encoder.layer):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token: (B, H)
        return cls_output



class TextSsagajiModel(nn.Module):
    def __init__(self, model_name: str = text_cfg.backbone, freeze_layers: int = 6):
        super().__init__()
        self.backbone = TextBackbone(model_name, freeze_layers)
        self.regression_head = TextRegressionHead(self.backbone.hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        features = self.backbone(input_ids, attention_mask)
        return self.regression_head(features)

    def get_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.backbone(input_ids, attention_mask)
