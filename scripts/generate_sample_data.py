"""
테스트용 더미 데이터 생성 스크립트.
실제 AI Hub 데이터나 설문 데이터 없이도 파이프라인 동작을 검증할 수 있다.

실행:
    python scripts/generate_sample_data.py
"""

import sys
import random
import struct
import wave
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.sentences import SENTENCES

SAMPLE_RATE = 16000
DURATION = 3  # seconds


def generate_sine_wav(path: Path, freq: float = 440.0, duration: float = DURATION):
    """단순 사인파 WAV 파일을 생성한다 (실제 음성 대용)."""
    n_samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n_samples)
    audio = (np.sin(2 * np.pi * freq * t) * 0.3 * 32767).astype(np.int16)

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())


def generate_survey_audio(audio_dir: Path) -> list[Path]:
    """30개의 테스트 음성 파일을 생성한다."""
    paths = []
    freqs = np.linspace(200, 800, len(SENTENCES))  # 문장마다 다른 주파수
    for i, sentence in enumerate(SENTENCES):
        fname = audio_dir / f"sentence_{sentence['id']:02d}.wav"
        generate_sine_wav(fname, freq=freqs[i])
        paths.append(fname)
    print(f"[샘플 오디오] {len(paths)}개 생성 → {audio_dir}")
    return paths


def generate_survey_csv(audio_paths: list[Path], output_path: Path, n_raters: int = 10):
    """가상의 설문 결과 CSV를 생성한다."""
    rows = []
    for i, (path, sentence) in enumerate(zip(audio_paths, SENTENCES)):
        row = {"audio_path": str(path), "text": sentence["text"]}
        expected = sentence.get("expected")

        for r in range(1, n_raters + 1):
            if expected is not None:
                # 예상 점수 근방에서 정규분포 샘플링
                score = int(np.clip(np.round(np.random.normal(expected, 0.7)), 1, 5))
            else:
                score = random.randint(1, 5)
            row[f"rater_{r}"] = score

        rows.append(row)

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[설문 CSV] {len(df)}행 생성 → {output_path}")
    return df


def generate_aihub_emotion_dummy(root_dir: Path, n_per_emotion: int = 20):
    """AI Hub 감정 데이터 구조와 동일한 더미 파일을 생성한다."""
    emotions = ["분노", "조소/냉소", "혐오", "경멸", "슬픔", "중립", "행복", "놀람", "공포"]
    freqs = {
        "분노": 600, "조소/냉소": 500, "혐오": 550, "경멸": 480,
        "슬픔": 300, "중립": 400, "행복": 700, "놀람": 650, "공포": 350,
    }
    import json

    total = 0
    for emotion in emotions:
        emotion_dir = root_dir / emotion
        emotion_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_emotion):
            wav_path = emotion_dir / f"{emotion}_{i:03d}.wav"
            generate_sine_wav(wav_path, freq=freqs[emotion] + random.uniform(-50, 50))
            json_path = wav_path.with_suffix(".json")
            json_path.write_text(
                json.dumps({"transcription": f"[더미 {emotion} 발화 {i}]"}, ensure_ascii=False),
                encoding="utf-8",
            )
            total += 1

    print(f"[AI Hub 더미] {total}개 파일 생성 → {root_dir}")


if __name__ == "__main__":
    base = Path(__file__).parent.parent

    # 설문용 오디오
    audio_dir = base / "survey_app" / "static" / "audio"
    audio_paths = generate_survey_audio(audio_dir)

    # 설문 CSV
    survey_csv = base / "data" / "survey_results.csv"
    generate_survey_csv(audio_paths, survey_csv, n_raters=15)

    # AI Hub 더미
    aihub_dir = base / "data" / "aihub_emotion"
    generate_aihub_emotion_dummy(aihub_dir, n_per_emotion=30)

    print("\n✅ 샘플 데이터 생성 완료!")
    print(f"  오디오: {audio_dir}")
    print(f"  설문 CSV: {survey_csv}")
    print(f"  AI Hub 더미: {aihub_dir}")
    print("\n다음 단계:")
    print("  1. 설문 서버: uvicorn survey_app.app:app --port 7000")
    print("  2. Stage 1: python training/pretrain_emotion.py --aihub_root data/aihub_emotion")
    print("  3. Stage 2: python training/finetune.py --survey_csv data/survey_results.csv --stage1_ckpt checkpoints/stage1")
    print("  4. 추론 API: uvicorn inference.api:app --port 8000")
