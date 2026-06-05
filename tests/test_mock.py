"""
Mock 테스트: 실제 모델 가중치 없이 전체 파이프라인 구조를 검증한다.
무거운 사전학습 모델을 로드하지 않고 아키텍처/데이터 흐름을 빠르게 확인한다.

실행:
    python tests/test_mock.py
"""

import sys
import wave
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# 유틸리티: 더미 WAV 생성
# ---------------------------------------------------------------------------

def make_dummy_wav(duration: float = 2.0, sr: int = 16000) -> str:
    n = int(sr * duration)
    audio = (np.sin(2 * np.pi * 440 * np.linspace(0, duration, n)) * 0.3 * 32767).astype(np.int16)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())
    return tmp.name


# ---------------------------------------------------------------------------
# 모델 아키텍처 Mock 테스트
# ---------------------------------------------------------------------------

class TestAudioModelMock(unittest.TestCase):
    """헤드 구조를 transformers 없이 테스트한다 (models/heads.py 사용)."""

    def setUp(self):
        self.B = 2

    def test_regression_head_output_range(self):
        """RegressionHead 출력이 [1, 5] 범위인지 확인한다."""
        from models.heads import RegressionHead
        head = RegressionHead(hidden_size=64)
        x = torch.randn(self.B, 64)
        out = head(x)
        self.assertEqual(out.shape, (self.B,))
        self.assertTrue((out >= 1.0).all() and (out <= 5.0).all(),
                        f"범위 초과: min={out.min():.3f}, max={out.max():.3f}")

    def test_emotion_head_output_shape(self):
        from models.heads import EmotionClassificationHead
        head = EmotionClassificationHead(hidden_size=64, num_emotions=9)
        x = torch.randn(self.B, 64)
        out = head(x)
        self.assertEqual(out.shape, (self.B, 9))

    def test_text_regression_head_range(self):
        from models.heads import TextRegressionHead
        head = TextRegressionHead(hidden_size=64)
        x = torch.randn(self.B, 64)
        out = head(x)
        self.assertTrue((out >= 1.0).all() and (out <= 5.0).all())


class TestRMSELoss(unittest.TestCase):
    def test_perfect_prediction(self):
        from training.train_utils import RMSELoss
        criterion = RMSELoss()
        pred = torch.tensor([1.0, 3.0, 5.0])
        target = torch.tensor([1.0, 3.0, 5.0])
        loss = criterion(pred, target)
        # eps=1e-8 → sqrt(1e-8) ≈ 1e-4, places=3으로 검증
        self.assertAlmostEqual(loss.item(), 0.0, places=3)

    def test_known_rmse(self):
        from training.train_utils import RMSELoss
        criterion = RMSELoss()
        pred = torch.tensor([1.0, 3.0])
        target = torch.tensor([2.0, 4.0])  # 각 오차 1
        loss = criterion(pred, target)
        self.assertAlmostEqual(loss.item(), 1.0, places=3)


class TestEarlyStopper(unittest.TestCase):
    def test_stops_after_patience(self):
        from training.train_utils import EarlyStopper
        stopper = EarlyStopper(patience=3)
        self.assertFalse(stopper(1.0))  # 개선
        self.assertFalse(stopper(1.1))  # 악화 1
        self.assertFalse(stopper(1.2))  # 악화 2
        self.assertTrue(stopper(1.3))   # 악화 3 → 중단

    def test_resets_on_improvement(self):
        from training.train_utils import EarlyStopper
        stopper = EarlyStopper(patience=2)
        stopper(1.0)
        stopper(1.5)
        stopper(0.8)  # 개선 → counter 리셋
        self.assertFalse(stopper(1.0))
        self.assertTrue(stopper(1.0))  # 이후 2번 악화


class TestAugmentation(unittest.TestCase):
    def _get_augmentor(self):
        try:
            from data.augmentation import AudioAugmentor
            return AudioAugmentor(16000, seed=0)
        except ImportError as e:
            self.skipTest(f"librosa/audiomentations 미설치: {e}")

    def test_augment_returns_variants(self):
        """증강 함수가 n_variants개의 배열을 반환하는지 확인한다."""
        augmentor = self._get_augmentor()
        sr = 16000
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr)).astype(np.float32)
        variants = augmentor.augment(audio, n_variants=3)
        self.assertEqual(len(variants), 3)
        for v in variants:
            self.assertEqual(len(v), len(audio))

    def test_augment_preserves_length(self):
        augmentor = self._get_augmentor()
        sr = 16000
        audio = np.zeros(sr, dtype=np.float32)
        variants = augmentor.augment(audio, n_variants=2)
        for v in variants:
            self.assertAlmostEqual(len(v), sr, delta=sr * 0.2)


class TestMetrics(unittest.TestCase):
    def test_compute_all_metrics_keys(self):
        from evaluation.metrics import compute_all_metrics
        preds = np.array([1.0, 2.5, 3.0, 4.5, 5.0])
        targets = np.array([1.5, 2.0, 3.5, 4.0, 4.5])
        metrics = compute_all_metrics(preds, targets)
        for key in ["RMSE", "MAE", "Pearson_r", "Spearman_r", "Accuracy@1", "Accuracy@1±1"]:
            self.assertIn(key, metrics)

    def test_score_range(self):
        from inference.utils import score_to_level
        self.assertEqual(score_to_level(1.0), "매우 공손함")
        self.assertEqual(score_to_level(5.0), "매우 무례함 (싸가지 없음)")
        self.assertEqual(score_to_level(3.0), "보통")


class TestSentences(unittest.TestCase):
    def test_sentence_count(self):
        from data.sentences import SENTENCES
        self.assertEqual(len(SENTENCES), 30)

    def test_unique_ids(self):
        from data.sentences import SENTENCES
        ids = [s["id"] for s in SENTENCES]
        self.assertEqual(len(ids), len(set(ids)))

    def test_get_by_id(self):
        from data.sentences import get_sentence_by_id
        s = get_sentence_by_id(1)
        self.assertIsNotNone(s)
        self.assertIn("text", s)


if __name__ == "__main__":
    print("=" * 60)
    print("싸가지 점수 AI - Mock 테스트")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
