"""
싸가지 점수 추론 모듈.
음성 파일 또는 오디오 바이트를 입력받아 1~5점 점수를 반환한다.
"""

import io
import tempfile
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import torchaudio
import whisper
from transformers import AutoFeatureExtractor, AutoTokenizer

from config import audio_cfg, text_cfg, infer_cfg
from models.ensemble import SsagajiEnsemble
from inference.utils import score_to_level


@dataclass
class ScoreResult:
    final_score: float        # 1~5점 최종 싸가지 점수
    audio_score: float        # 음향 기반 점수
    text_score: float         # 언어 기반 점수
    transcript: str           # STT 변환 텍스트
    level: str                # 등급 설명
    confidence: float         # 신뢰도 (0~1)


class SsagajiScorer:
    """
    싸가지 점수 추론기.

    사용 예:
        scorer = SsagajiScorer(model_dir="checkpoints/best_model")
        result = scorer.score("audio.wav")
        print(result.final_score, result.level)
    """

    def __init__(
        self,
        model_dir: str = str(infer_cfg.model_dir),
        device: str = None,
        whisper_size: str = infer_cfg.whisper_model,
    ):
        self.device = torch.device(device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        ))
        print(f"[Scorer] Device: {self.device}")

        # 앙상블 모델 로드
        self.model = SsagajiEnsemble().to(self.device)
        weights_path = Path(model_dir) / "model_weights.pt"
        self.model.load_state_dict(
            torch.load(weights_path, map_location=self.device)
        )
        self.model.eval()

        # 전처리기
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(audio_cfg.backbone)
        self.tokenizer = AutoTokenizer.from_pretrained(text_cfg.backbone)
        self.max_samples = int(audio_cfg.max_duration * audio_cfg.sample_rate)

        # Whisper STT
        print(f"[Scorer] Whisper ({whisper_size}) 로드 중...")
        self.whisper = whisper.load_model(whisper_size, device=str(self.device))

    def _load_audio(self, path: str) -> np.ndarray:
        waveform, sr = torchaudio.load(path)
        waveform = waveform.mean(0)
        if sr != audio_cfg.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, audio_cfg.sample_rate)
        if waveform.shape[0] > self.max_samples:
            waveform = waveform[: self.max_samples]
        return waveform.numpy()

    def _prepare_audio_tensor(self, audio: np.ndarray) -> torch.Tensor:
        inputs = self.feature_extractor(
            audio,
            sampling_rate=audio_cfg.sample_rate,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_samples,
            truncation=True,
        )
        return inputs.input_values.to(self.device)

    def _prepare_text_tensors(self, text: str):
        enc = self.tokenizer(
            text,
            max_length=text_cfg.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return enc.input_ids.to(self.device), enc.attention_mask.to(self.device)

    def _transcribe(self, audio_path: str) -> str:
        result = self.whisper.transcribe(audio_path, language="ko")
        return result.get("text", "").strip()

    @torch.no_grad()
    def score(self, audio_path: str, text: str = None) -> ScoreResult:
        """
        음성 파일 경로를 입력받아 싸가지 점수를 반환한다.
        text가 제공되면 STT를 건너뛴다.
        """
        # STT
        transcript = text if text else self._transcribe(audio_path)

        # 오디오 전처리
        audio = self._load_audio(audio_path)
        input_values = self._prepare_audio_tensor(audio)

        # 텍스트 전처리
        input_ids, attention_mask = self._prepare_text_tensors(transcript)

        # 개별 점수
        audio_score = self.model.audio_model(input_values).squeeze().item()
        text_score = self.model.text_model(input_ids, attention_mask).squeeze().item()
        final_score = self.model(input_values, input_ids, attention_mask).squeeze().item()

        # 점수 클리핑
        audio_score = float(np.clip(audio_score, 1.0, 5.0))
        text_score = float(np.clip(text_score, 1.0, 5.0))
        final_score = float(np.clip(final_score, 1.0, 5.0))

        # 신뢰도: 두 모달리티 점수 차이가 작을수록 높음
        diff = abs(audio_score - text_score)
        confidence = float(np.exp(-diff))

        return ScoreResult(
            final_score=round(final_score, 2),
            audio_score=round(audio_score, 2),
            text_score=round(text_score, 2),
            transcript=transcript,
            level=score_to_level(final_score),
            confidence=round(confidence, 3),
        )

    def score_bytes(self, audio_bytes: bytes, fmt: str = "wav") -> ScoreResult:
        """바이트 스트림으로 직접 추론한다 (API 서버용)."""
        with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        try:
            return self.score(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
