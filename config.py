from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).parent


@dataclass
class AudioConfig:
    # 한국어 감정 분류로 파인튜닝된 XLSR 모델 (Stage 1 사전학습 불필요)
    backbone: str = "jungjongho/wav2vec2-xlsr-korean-speech-emotion-recognition3"
    # 대안: "kresnik/wav2vec2-large-xlsr-korean" (ASR 특화, Stage 1 필요)
    sample_rate: int = 16000
    max_duration: float = 10.0  # seconds
    hidden_size: int = 1024
    dropout: float = 0.1


@dataclass
class TextConfig:
    backbone: str = "klue/bert-base"
    max_length: int = 128
    hidden_size: int = 768
    dropout: float = 0.1


@dataclass
class EnsembleConfig:
    audio_weight: float = 0.7
    text_weight: float = 0.3
    fusion: str = "weighted"  # "weighted" | "learned" | "concat"


@dataclass
class TrainingConfig:
    # Stage 1: 감정 데이터셋 사전학습
    stage1_epochs: int = 10
    stage1_lr: float = 1e-4
    stage1_batch_size: int = 16

    # Stage 2: 설문 데이터 파인튜닝
    stage2_epochs: int = 20
    stage2_lr: float = 2e-5
    stage2_batch_size: int = 8

    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    seed: int = 42

    # 조기 종료
    patience: int = 5


@dataclass
class DataConfig:
    data_dir: Path = BASE_DIR / "data"
    survey_csv: Path = BASE_DIR / "data" / "survey_results.csv"
    emotion_dataset: str = "AIHub_emotion"  # AI Hub 감정 음성
    augmentation_factor: int = 3  # 원본 대비 증강 배수

    # 감정 → 싸가지 연관 가중치 (사전학습용 pseudo-label)
    emotion_to_rudeness: dict = field(default_factory=lambda: {
        "분노": 4.5,
        "조소/냉소": 4.0,
        "혐오": 4.5,
        "경멸": 4.0,
        "슬픔": 2.0,
        "중립": 2.5,
        "행복": 1.5,
        "놀람": 2.5,
        "공포": 2.0,
    })


@dataclass
class InferenceConfig:
    model_dir: Path = BASE_DIR / "checkpoints" / "best_model"
    device: str = "cuda"  # "cuda" | "cpu" | "mps"
    whisper_model: str = "medium"  # Whisper STT 크기


# 전역 설정
audio_cfg = AudioConfig()
text_cfg = TextConfig()
ensemble_cfg = EnsembleConfig()
train_cfg = TrainingConfig()
data_cfg = DataConfig()
infer_cfg = InferenceConfig()
