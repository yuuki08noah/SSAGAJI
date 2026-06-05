#!/usr/bin/env bash
# 전체 파이프라인 실행 스크립트
set -e

echo "========================================"
echo " 싸가지 점수 AI - 전체 파이프라인"
echo "========================================"

# 1. 의존성 설치
echo "[1/3] 의존성 설치..."
pip install -r requirements.txt

# 2. Stage 2: 설문 파인튜닝 (백본이 이미 감정 사전학습됨)
echo "[2/3] Stage 2: 설문 파인튜닝..."
python training/finetune.py \
  --survey_csv "${SURVEY_CSV:-data/survey_results.csv}" \
  --output_dir checkpoints/best_model \
  --fusion concat

# 3. API 서버 실행
echo "[3/3] API 서버 시작 (포트 8000)..."
uvicorn inference.api:app --host 0.0.0.0 --port 8000 --reload
