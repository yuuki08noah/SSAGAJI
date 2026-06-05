"""
오디오 증강 모듈.
데이터 부족 문제 해결을 위해 피치 변조, 속도 조절, 노이즈 추가 등을 적용한다.
"""

import random
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple
import audiomentations as A


def build_augmentation_pipeline(sample_rate: int = 16000) -> A.Compose:
    """학습용 증강 파이프라인."""
    return A.Compose([
        A.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.4),
        A.TimeStretch(min_rate=0.85, max_rate=1.15, p=0.4),
        A.PitchShift(min_semitones=-2, max_semitones=2, p=0.4),
        A.Shift(min_shift=-0.2, max_shift=0.2, p=0.3),
        A.AddColorNoise(min_snr_db=10, max_snr_db=30, p=0.2),
        A.TanhDistortion(min_distortion=0.0, max_distortion=0.3, p=0.2),
    ])


class AudioAugmentor:
    def __init__(self, sample_rate: int = 16000, seed: int = 42):
        self.sr = sample_rate
        random.seed(seed)
        np.random.seed(seed)
        self.pipeline = build_augmentation_pipeline(sample_rate)

    def augment(self, audio: np.ndarray, n_variants: int = 3) -> list[np.ndarray]:
        """원본 오디오에서 n_variants 개의 증강본을 생성한다."""
        variants = []
        for _ in range(n_variants):
            augmented = self.pipeline(samples=audio.astype(np.float32), sample_rate=self.sr)
            variants.append(augmented)
        return variants

    def augment_file(
        self,
        input_path: Path,
        output_dir: Path,
        n_variants: int = 3,
        label: float = None,
    ) -> list[Tuple[Path, float]]:
        """파일을 읽어 증강 후 저장하고 (경로, 라벨) 목록을 반환한다."""
        audio, _ = librosa.load(str(input_path), sr=self.sr, mono=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        variants = self.augment(audio, n_variants)
        for i, aug_audio in enumerate(variants):
            stem = input_path.stem
            out_path = output_dir / f"{stem}_aug{i}.wav"
            sf.write(str(out_path), aug_audio, self.sr)
            results.append((out_path, label))
        return results

    def augment_dataset(
        self,
        manifest_df,  # columns: [audio_path, label]
        output_dir: Path,
        n_variants: int = 3,
    ):
        """데이터셋 전체에 증강을 적용한다."""
        import pandas as pd
        from tqdm import tqdm

        new_rows = []
        for _, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="Augmenting"):
            results = self.augment_file(
                Path(row["audio_path"]),
                output_dir / "augmented",
                n_variants=n_variants,
                label=row["label"],
            )
            for aug_path, aug_label in results:
                new_row = {k: row[k] for k in manifest_df.columns if k not in ("audio_path", "label")}
                new_row["audio_path"] = str(aug_path)
                new_row["label"] = aug_label
                new_rows.append(new_row)

        aug_df = pd.DataFrame(new_rows)
        combined = pd.concat([manifest_df, aug_df], ignore_index=True)
        return combined
