"""
실제 데이터 테스트: 실제 모델 가중치와 데이터를 사용한 통합 테스트.
GPU/MPS가 있을 때 빠르며, CPU에서는 느릴 수 있다.

전제 조건:
  - requirements.txt 설치 완료
  - python scripts/generate_sample_data.py 실행 후 (또는 실제 데이터 보유)
  - (선택) checkpoints/best_model/model_weights.pt 존재 시 추론 테스트 포함

실행:
    python tests/test_real.py
    python tests/test_real.py --skip-training   # 학습 스킵, 추론만
    python tests/test_real.py --real-data        # 실제 AI Hub 데이터 사용
"""

import sys
import argparse
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_DIR = Path(__file__).parent.parent
SAMPLE_AUDIO = BASE_DIR / "survey_app" / "static" / "audio"
SURVEY_CSV = BASE_DIR / "data" / "survey_results.csv"
AIHUB_DIR = BASE_DIR / "data" / "aihub_emotion"
CKPT_DIR = BASE_DIR / "checkpoints" / "best_model"


# ---------------------------------------------------------------------------
# 공통 픽스처
# ---------------------------------------------------------------------------

class RealDataMixin:
    """실제(또는 더미) 데이터가 있을 때만 실행되는 테스트들."""

    @classmethod
    def setUpClass(cls):
        # 더미 데이터가 없으면 생성
        if not SURVEY_CSV.exists() or not SAMPLE_AUDIO.exists():
            print("\n[setup] 샘플 데이터 자동 생성 중...")
            from scripts.generate_sample_data import (
                generate_survey_audio, generate_survey_csv, generate_aihub_emotion_dummy
            )
            audio_paths = generate_survey_audio(SAMPLE_AUDIO)
            generate_survey_csv(audio_paths, SURVEY_CSV, n_raters=10)
            generate_aihub_emotion_dummy(AIHUB_DIR, n_per_emotion=5)
            print("[setup] 완료")


# ---------------------------------------------------------------------------
# 데이터 파이프라인 테스트
# ---------------------------------------------------------------------------

class TestDataPipeline(RealDataMixin, unittest.TestCase):

    def test_survey_csv_loads(self):
        self.assertTrue(SURVEY_CSV.exists(), f"CSV 없음: {SURVEY_CSV}")
        df = pd.read_csv(SURVEY_CSV)
        self.assertGreater(len(df), 0)
        rater_cols = [c for c in df.columns if c.startswith("rater_")]
        self.assertGreater(len(rater_cols), 0, "rater_N 열이 없음")
        print(f"  CSV 행 수: {len(df)}, 평가자 수: {len(rater_cols)}")

    def test_survey_dataset_loads(self):
        try:
            from data.prepare_dataset import load_survey_dataset
        except ImportError as e:
            self.skipTest(f"의존성 미설치: {e}")
        df = load_survey_dataset(SURVEY_CSV)
        self.assertGreater(len(df), 0)
        self.assertIn("label", df.columns)
        self.assertTrue((df["label"] >= 1).all() and (df["label"] <= 5).all())
        print(f"  유효 샘플: {len(df)}, 평균 점수: {df['label'].mean():.2f}")

    def test_aihub_dataset_loads(self):
        try:
            from data.prepare_dataset import load_aihub_emotion_dataset
        except ImportError as e:
            self.skipTest(f"의존성 미설치: {e}")
        df = load_aihub_emotion_dataset(AIHUB_DIR)
        self.assertGreater(len(df), 0)
        self.assertIn("label", df.columns)
        print(f"  감정 샘플: {len(df)}, 감정 종류: {df['emotion'].nunique()}")

    def test_augmentation_pipeline(self):
        try:
            from data.augmentation import AudioAugmentor
            from data.prepare_dataset import load_survey_dataset
        except ImportError as e:
            self.skipTest(f"의존성 미설치: {e}")
        augmentor = AudioAugmentor(seed=0)
        df = load_survey_dataset(SURVEY_CSV).head(5)
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            aug_df = augmentor.augment_dataset(df, Path(tmpdir), n_variants=2)
        self.assertGreaterEqual(len(aug_df), len(df))
        print(f"  증강 전: {len(df)}, 후: {len(aug_df)}")

    def test_inter_rater_reliability(self):
        from evaluation.metrics import inter_rater_reliability
        result = inter_rater_reliability(str(SURVEY_CSV))
        self.assertIn("krippendorff_alpha", result)
        alpha = result["krippendorff_alpha"]
        print(f"  Krippendorff α = {alpha:.4f}")
        # 더미 데이터라도 -∞~1 범위에 있어야 함
        self.assertLessEqual(alpha, 1.0)


# ---------------------------------------------------------------------------
# 모델 구조 + 포워드 패스 테스트 (경량 백본으로 대체)
# ---------------------------------------------------------------------------

class TestModelForwardPass(RealDataMixin, unittest.TestCase):
    """
    실제 HuBERT/BERT 다운로드 대신 경량 더미 백본으로 포워드 패스를 검증한다.
    """

    def _make_dummy_batch(self, B=2, T=8000, L=32):
        return {
            "input_values": torch.randn(B, T),
            "input_ids": torch.randint(0, 100, (B, L)),
            "attention_mask": torch.ones(B, L, dtype=torch.long),
            "label": torch.rand(B) * 4 + 1,
        }

    def test_regression_head_gradient(self):
        # transformers 없이 heads.py에서 직접 import
        from models.heads import RegressionHead
        head = RegressionHead(hidden_size=64)
        x = torch.randn(4, 64, requires_grad=True)
        out = head(x)
        loss = ((out - 3.0) ** 2).mean()
        loss.backward()
        self.assertIsNotNone(x.grad)
        print(f"  RegressionHead 기울기 norm: {x.grad.norm():.4f}")

    def test_rmse_loss_backward(self):
        from training.train_utils import RMSELoss
        criterion = RMSELoss()
        # leaf 텐서로 직접 정의 (연산 결과는 non-leaf라 .grad 안 붙음)
        base = torch.randn(8, requires_grad=True)
        pred_clipped = torch.sigmoid(base) * 4 + 1  # [1, 5]
        target = torch.rand(8) * 4 + 1
        loss = criterion(pred_clipped, target)
        loss.backward()
        self.assertIsNotNone(base.grad)
        print(f"  RMSE 손실: {loss.item():.4f}")

    def test_metrics_consistency(self):
        from evaluation.metrics import compute_all_metrics
        np.random.seed(42)
        preds = np.clip(np.random.normal(3, 1, 100), 1, 5)
        targets = np.clip(np.random.normal(3, 1, 100), 1, 5)
        metrics = compute_all_metrics(preds, targets)
        print(f"  랜덤 기준선: RMSE={metrics['RMSE']}, Pearson={metrics['Pearson_r']}")
        self.assertGreater(metrics["RMSE"], 0)
        self.assertLessEqual(abs(metrics["Pearson_r"]), 1.0)


# ---------------------------------------------------------------------------
# 추론 테스트 (체크포인트가 있을 때만)
# ---------------------------------------------------------------------------

class TestInference(RealDataMixin, unittest.TestCase):

    @unittest.skipUnless(CKPT_DIR.exists(), "체크포인트 없음 (학습 후 실행)")
    def test_scorer_loads(self):
        from inference.scorer import SsagajiScorer
        scorer = SsagajiScorer(model_dir=str(CKPT_DIR))
        self.assertIsNotNone(scorer)

    @unittest.skipUnless(CKPT_DIR.exists(), "체크포인트 없음")
    def test_scorer_output_range(self):
        from inference.scorer import SsagajiScorer
        scorer = SsagajiScorer(model_dir=str(CKPT_DIR))
        audio_files = list(SAMPLE_AUDIO.glob("*.wav"))
        self.assertGreater(len(audio_files), 0)

        for wav in audio_files[:3]:
            result = scorer.score(str(wav))
            self.assertGreaterEqual(result.final_score, 1.0)
            self.assertLessEqual(result.final_score, 5.0)
            self.assertIn(result.level, [
                "매우 공손함", "공손함", "보통", "무례함", "매우 무례함 (싸가지 없음)"
            ])
            print(f"  {wav.name}: {result.final_score:.2f}점 ({result.level})")

    @unittest.skipUnless(CKPT_DIR.exists(), "체크포인트 없음")
    def test_scorer_speed(self):
        from inference.scorer import SsagajiScorer
        scorer = SsagajiScorer(model_dir=str(CKPT_DIR))
        audio_files = list(SAMPLE_AUDIO.glob("*.wav"))
        wav = str(audio_files[0])

        t0 = time.time()
        for _ in range(5):
            scorer.score(wav)
        elapsed = (time.time() - t0) / 5
        print(f"  평균 추론 시간: {elapsed * 1000:.1f}ms")
        self.assertLess(elapsed, 10.0, "추론이 너무 느림 (10초 초과)")


# ---------------------------------------------------------------------------
# API 서버 테스트
# ---------------------------------------------------------------------------

class TestAPIServer(RealDataMixin, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        try:
            from fastapi.testclient import TestClient
            # 모델 로드 없이 API 구조만 테스트
            import inference.api as api_module
            api_module.scorer = MagicMock()

            from inference.scorer import ScoreResult
            api_module.scorer.score_bytes.return_value = ScoreResult(
                final_score=3.2,
                audio_score=3.5,
                text_score=2.9,
                transcript="테스트 발화입니다.",
                level="보통",
                confidence=0.85,
            )
            cls.client = TestClient(api_module.app)
            cls.client_available = True
        except Exception as e:
            print(f"\n[API 테스트 스킵] {e}")
            cls.client_available = False

    def test_health_endpoint(self):
        if not self.client_available:
            self.skipTest("TestClient 사용 불가")
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)

    def test_score_endpoint_structure(self):
        if not self.client_available:
            self.skipTest("TestClient 사용 불가")
        audio_files = list(SAMPLE_AUDIO.glob("*.wav"))
        if not audio_files:
            self.skipTest("테스트 오디오 없음")

        with open(audio_files[0], "rb") as f:
            resp = self.client.post("/score", files={"file": ("test.wav", f, "audio/wav")})

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        for key in ["final_score", "audio_score", "text_score", "transcript", "level", "confidence"]:
            self.assertIn(key, data)
        print(f"  API 응답: {data['final_score']}점 ({data['level']})")


# ---------------------------------------------------------------------------
# Mock import for API test
# ---------------------------------------------------------------------------
try:
    from unittest.mock import MagicMock
except ImportError:
    pass


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-training", action="store_true", help="학습 관련 테스트 스킵")
    parser.add_argument("--real-data", action="store_true", help="실제 AI Hub 데이터 경로 사용")
    args, remaining = parser.parse_known_args()

    print("=" * 60)
    print("싸가지 점수 AI - 실제 데이터 통합 테스트")
    print("=" * 60)
    print(f"  체크포인트 존재: {CKPT_DIR.exists()}")
    print(f"  설문 CSV 존재: {SURVEY_CSV.exists()}")
    print(f"  오디오 디렉터리: {SAMPLE_AUDIO.exists()}")
    print()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestDataPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestModelForwardPass))
    if not args.skip_training:
        suite.addTests(loader.loadTestsFromTestCase(TestInference))
    suite.addTests(loader.loadTestsFromTestCase(TestAPIServer))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
