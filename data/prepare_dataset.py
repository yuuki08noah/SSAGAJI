"""
데이터셋 준비 모듈.

1. AI Hub 감정 음성 데이터셋 → Stage 1 학습용 pseudo-label 생성
2. 설문 CSV → Stage 2 파인튜닝용 데이터셋 생성
3. 학습/검증/테스트 분할
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor, AutoTokenizer

from config import audio_cfg, data_cfg, text_cfg


# ---------------------------------------------------------------------------
# AI Hub 감정 데이터 → 싸가지 pseudo-label 변환
# ---------------------------------------------------------------------------

def load_aihub_emotion_dataset(root_dir: Path) -> pd.DataFrame:
    """
    AI Hub 감정 음성 데이터셋 구조 가정:
        root_dir/
            {emotion_label}/
                *.wav  (+ *.json 메타)
    """
    rows = []
    for wav_path in sorted(root_dir.rglob("*.wav")):
        # 상위 디렉터리명을 감정 라벨로 사용
        emotion = wav_path.parent.name
        rudeness_score = data_cfg.emotion_to_rudeness.get(emotion)
        if rudeness_score is None:
            continue

        # 트랜스크립션 JSON이 있으면 텍스트 추출
        json_path = wav_path.with_suffix(".json")
        text = ""
        if json_path.exists():
            try:
                meta = json.loads(json_path.read_text(encoding="utf-8"))
                text = meta.get("transcription", meta.get("text", ""))
            except Exception:
                pass

        rows.append({
            "audio_path": str(wav_path),
            "text": text,
            "label": rudeness_score,
            "source": "aihub_emotion",
            "emotion": emotion,
        })

    df = pd.DataFrame(rows)
    print(f"[AI Hub] 총 {len(df)}개 샘플 로드 (감정 종류: {df['emotion'].nunique()})")
    return df


def load_survey_dataset(survey_csv: Path) -> pd.DataFrame:
    """
    두 가지 CSV 형식을 모두 지원한다.

    형식 A (generate_sample_data 생성):
        audio_path, text, rater_1, rater_2, ..., rater_N

    형식 B (설문 앱이 실제 저장):
        session_id, submitted_at, age, gender, sentence_01.wav, sentence_02.wav, ...
        → 파일명 컬럼을 참여자별 점수로 사용, pivot해서 형식 A로 변환
    """
    df = pd.read_csv(survey_csv, encoding="utf-8")

    rater_cols = [c for c in df.columns if re.match(r"rater_\d+", c)]
    audio_cols = [c for c in df.columns if c.endswith(".wav") or c.endswith(".mp3")]

    if rater_cols:
        # 형식 A
        scores = df[rater_cols].values.astype(float)
        df["label"] = scores.mean(axis=1)
        df["label_std"] = scores.std(axis=1)
        df["source"] = "survey"

    elif audio_cols:
        # 형식 B: 참여자당 1행 → 음성파일당 1행으로 pivot
        from data.sentences import SENTENCES
        text_map = {f"sentence_{s['id']:02d}.wav": s["text"] for s in SENTENCES}

        audio_dir = Path(__file__).parent.parent / "survey_app" / "static" / "audio"
        rows = []
        for col in audio_cols:
            scores = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(scores) == 0:
                continue
            rows.append({
                "audio_path": str(audio_dir / col),
                "text": text_map.get(col, ""),
                "label": scores.mean(),
                "label_std": scores.std(),
                "source": "survey",
            })
        df = pd.DataFrame(rows)

    else:
        raise ValueError("지원하는 CSV 형식이 아닙니다. rater_N 열 또는 .wav/.mp3 열이 필요합니다.")

    before = len(df)
    df = df[df["label_std"] <= 1.5].copy()
    print(f"[Survey] {before}→{len(df)}개 (표준편차 필터링 제거: {before - len(df)}개)")

    return df[["audio_path", "text", "label", "source"]]


def split_dataset(df: pd.DataFrame, test_size: float = 0.1, val_size: float = 0.1):
    train_val, test = train_test_split(df, test_size=test_size, random_state=42, shuffle=True)
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=val_ratio, random_state=42)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class SsagajiDataset(Dataset):
    """
    멀티모달 데이터셋: 오디오 + 텍스트 → 싸가지 점수(1~5).
    텍스트가 없는 경우 Whisper STT로 보완한다.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        audio_feature_extractor,
        tokenizer,
        max_duration: float = audio_cfg.max_duration,
        max_length: int = text_cfg.max_length,
        use_whisper: bool = False,
        whisper_model=None,
    ):
        self.df = df.reset_index(drop=True)
        self.fe = audio_feature_extractor
        self.tokenizer = tokenizer
        self.max_samples = int(max_duration * audio_cfg.sample_rate)
        self.max_length = max_length
        self.use_whisper = use_whisper
        self.whisper = whisper_model

    def __len__(self):
        return len(self.df)

    def _load_audio(self, path: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != audio_cfg.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, audio_cfg.sample_rate)
        # (1, T) → (T,)
        waveform = waveform.squeeze(0)
        # 길이 패딩/자르기
        if waveform.shape[0] > self.max_samples:
            waveform = waveform[: self.max_samples]
        return waveform

    def _transcribe(self, path: str) -> str:
        if self.whisper is None:
            return ""
        result = self.whisper.transcribe(path, language="ko")
        return result.get("text", "")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row["audio_path"]
        label = float(row["label"])

        # 오디오 처리
        waveform = self._load_audio(audio_path)
        audio_inputs = self.fe(
            waveform.numpy(),
            sampling_rate=audio_cfg.sample_rate,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_samples,
            truncation=True,
        )
        input_values = audio_inputs.input_values.squeeze(0)  # (T,)

        # 텍스트 처리
        text = str(row.get("text", "")) if row.get("text") else ""
        if not text and self.use_whisper:
            text = self._transcribe(audio_path)
        text_inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_values": input_values,
            "input_ids": text_inputs.input_ids.squeeze(0),
            "attention_mask": text_inputs.attention_mask.squeeze(0),
            "label": torch.tensor(label, dtype=torch.float),
        }


def build_dataloaders(
    train_df, val_df, test_df,
    audio_model_name: str = audio_cfg.backbone,
    text_model_name: str = text_cfg.backbone,
    batch_size: int = 8,
):
    fe = AutoFeatureExtractor.from_pretrained(audio_model_name)
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)

    train_ds = SsagajiDataset(train_df, fe, tokenizer)
    val_ds = SsagajiDataset(val_df, fe, tokenizer)
    test_ds = SsagajiDataset(test_df, fe, tokenizer)

    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader
