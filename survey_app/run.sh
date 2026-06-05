#!/usr/bin/env bash
# 설문 웹 서버 실행
set -e
cd "$(dirname "$0")/.."

echo "🎙️ 싸가지 설문 서버 시작 중..."
echo "  주소: http://localhost:7000"
echo "  관리자: http://localhost:7000/admin"
echo ""

uvicorn survey_app.app:app --host 0.0.0.0 --port 7000 --reload
