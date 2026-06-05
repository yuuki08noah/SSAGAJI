"""
FastAPI 추론 서버.

실행:
    uvicorn inference.api:app --host 0.0.0.0 --port 8000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from inference.scorer import SsagajiScorer, ScoreResult
from config import infer_cfg

app = FastAPI(
    title="싸가지 점수 API",
    description="음성 파일을 분석하여 무례함 정도를 1~5점으로 채점합니다.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 scorer (앱 시작 시 1회 로드)
scorer: SsagajiScorer = None


@app.on_event("startup")
async def startup():
    global scorer
    scorer = SsagajiScorer()


class ScoreResponse(BaseModel):
    final_score: float
    audio_score: float
    text_score: float
    transcript: str
    level: str
    confidence: float
    interpretation: str


def build_interpretation(result: ScoreResult) -> str:
    parts = []
    if result.audio_score > result.text_score + 0.5:
        parts.append("음조/억양에서 무례함이 강하게 감지됨")
    elif result.text_score > result.audio_score + 0.5:
        parts.append("단어/어미 선택에서 무례함이 강하게 감지됨")
    else:
        parts.append("음조와 언어 모두에서 유사한 수준의 무례함 감지됨")

    if result.confidence < 0.5:
        parts.append("(두 모달리티 간 불일치 — 결과를 참고 수준으로만 사용할 것)")
    return ". ".join(parts)


@app.post("/score", response_model=ScoreResponse)
async def score_audio(
    file: UploadFile = File(..., description="분석할 음성 파일 (wav/mp3/m4a)"),
    text: Optional[str] = Form(None, description="STT 결과 텍스트 (선택, 없으면 자동 변환)"),
):
    if scorer is None:
        raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않았습니다.")

    allowed = {"wav", "mp3", "m4a", "ogg", "flac"}
    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 형식: {ext}")

    audio_bytes = await file.read()
    try:
        result = scorer.score_bytes(audio_bytes, fmt=ext)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 오류: {e}")

    # 텍스트가 직접 제공된 경우 재추론
    if text and text != result.transcript:
        result = scorer.score(result.transcript, text=text)

    return ScoreResponse(
        final_score=result.final_score,
        audio_score=result.audio_score,
        text_score=result.text_score,
        transcript=result.transcript,
        level=result.level,
        confidence=result.confidence,
        interpretation=build_interpretation(result),
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": scorer is not None}


@app.get("/")
async def root():
    return {
        "service": "싸가지 점수 API",
        "usage": "POST /score 에 음성 파일을 업로드하세요.",
        "docs": "/docs",
    }
