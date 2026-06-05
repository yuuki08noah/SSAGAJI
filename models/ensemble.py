"""
앙상블 모델: 오디오 + 텍스트를 결합하여 최종 싸가지 점수를 산출한다.

fusion 방식:
  - "weighted": 고정 가중치 평균 (기본값)
  - "learned": 학습 가능한 가중치 평균
  - "concat": 두 표현을 연결 후 MLP
"""

import torch
import torch.nn as nn
from config import audio_cfg, text_cfg, ensemble_cfg
from models.audio_model import AudioSsagajiModel
from models.text_model import TextSsagajiModel


class LearnedWeightFusion(nn.Module):
    """두 모달리티의 가중치를 학습한다."""

    def __init__(self):
        super().__init__()
        # 초기값: audio=0.6, text=0.4
        self.logits = nn.Parameter(torch.tensor([0.405, 0.0]))  # softmax → [0.6, 0.4]

    def forward(self, audio_score: torch.Tensor, text_score: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.logits, dim=0)
        return weights[0] * audio_score + weights[1] * text_score


class ConcatFusion(nn.Module):
    """두 표현을 연결 후 MLP로 최종 점수를 산출한다."""

    def __init__(self, audio_hidden: int, text_hidden: int):
        super().__init__()
        in_dim = audio_hidden + text_hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )
        self.scale = nn.Sigmoid()

    def forward(self, audio_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([audio_feat, text_feat], dim=-1)
        raw = self.net(combined).squeeze(-1)
        return self.scale(raw) * 4.0 + 1.0


class SsagajiEnsemble(nn.Module):
    """
    최종 앙상블 모델.

    사용 예:
        model = SsagajiEnsemble(fusion="concat")
        score = model(input_values, input_ids, attention_mask)  # (B,)
    """

    def __init__(
        self,
        audio_model_name: str = audio_cfg.backbone,
        text_model_name: str = text_cfg.backbone,
        fusion: str = ensemble_cfg.fusion,
        audio_weight: float = ensemble_cfg.audio_weight,
        text_weight: float = ensemble_cfg.text_weight,
        freeze_audio_feature_extractor: bool = True,
        freeze_text_layers: int = 6,
    ):
        super().__init__()
        self.audio_model = AudioSsagajiModel(
            model_name=audio_model_name,
            stage="regression",
            freeze_feature_extractor=freeze_audio_feature_extractor,
        )
        self.text_model = TextSsagajiModel(
            model_name=text_model_name,
            freeze_layers=freeze_text_layers,
        )
        self.fusion = fusion
        self.audio_weight = audio_weight
        self.text_weight = text_weight

        if fusion == "learned":
            self.fusion_layer = LearnedWeightFusion()
        elif fusion == "concat":
            self.fusion_layer = ConcatFusion(
                audio_hidden=self.audio_model.backbone.hidden_size,
                text_hidden=self.text_model.backbone.hidden_size,
            )

    def forward(
        self,
        input_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.fusion == "weighted":
            audio_score = self.audio_model(input_values, audio_attention_mask)
            text_score = self.text_model(input_ids, attention_mask)
            score = self.audio_weight * audio_score + self.text_weight * text_score
            return score

        elif self.fusion == "learned":
            audio_score = self.audio_model(input_values, audio_attention_mask)
            text_score = self.text_model(input_ids, attention_mask)
            return self.fusion_layer(audio_score, text_score)

        elif self.fusion == "concat":
            audio_feat = self.audio_model.get_features(input_values, audio_attention_mask)
            text_feat = self.text_model.get_features(input_ids, attention_mask)
            return self.fusion_layer(audio_feat, text_feat)

        else:
            raise ValueError(f"Unknown fusion: {self.fusion}")

    def predict_with_breakdown(
        self,
        input_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict:
        """각 모달리티별 점수와 최종 점수를 반환한다 (해석 가능성 향상)."""
        with torch.no_grad():
            audio_score = self.audio_model(input_values)
            text_score = self.text_model(input_ids, attention_mask)
            final = self.forward(input_values, input_ids, attention_mask)
        return {
            "final_score": final.item(),
            "audio_score": audio_score.item(),
            "text_score": text_score.item(),
        }
