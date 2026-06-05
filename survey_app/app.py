"""
설문 웹 애플리케이션 (FastAPI + Jinja2 + SQLite).

기능:
  - 참여자가 30개의 음성 샘플을 순서대로 청취
  - 각 음성에 대해 1~5점 '싸가지 점수' 입력
  - 문항별 응답 타임스탬프 기록
  - 관리자 페이지에서 집계 통계 확인 및 CSV 다운로드
"""

import csv
import io
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

BASE_DIR = Path(__file__).parent
AUDIO_DIR = BASE_DIR / "static" / "audio"
DB_PATH   = BASE_DIR / "survey.db"

AUDIO_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="싸가지 설문 조사")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ---------------------------------------------------------------------------
# DB 초기화
# ---------------------------------------------------------------------------

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id  TEXT PRIMARY KEY,
                age         TEXT,
                gender      TEXT,
                started_at  TEXT NOT NULL,
                completed_at TEXT,
                completed   INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS ratings (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id    TEXT NOT NULL,
                audio_filename TEXT NOT NULL,
                question_num  INTEGER NOT NULL,
                score         INTEGER NOT NULL,
                answered_at   TEXT NOT NULL,
                UNIQUE(session_id, question_num),
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );
        """)


init_db()


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def get_audio_files() -> list[dict]:
    files = sorted(AUDIO_DIR.glob("*.wav")) + sorted(AUDIO_DIR.glob("*.mp3"))
    return [
        {"id": i + 1, "filename": f.name, "url": f"/static/audio/{f.name}"}
        for i, f in enumerate(files[:30])
    ]


def now() -> str:
    return datetime.now().isoformat()


# ---------------------------------------------------------------------------
# 라우트
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/start")
async def start_survey(
    request: Request,
    age: str = Form(...),
    gender: str = Form(...),
):
    session_id = str(uuid.uuid4())[:8]
    with get_db() as db:
        db.execute(
            "INSERT INTO sessions (session_id, age, gender, started_at) VALUES (?, ?, ?, ?)",
            (session_id, age, gender, now()),
        )
    return RedirectResponse(f"/survey/{session_id}/1", status_code=303)


@app.get("/survey/{session_id}/{question_num}", response_class=HTMLResponse)
async def survey_question(request: Request, session_id: str, question_num: int):
    with get_db() as db:
        session = db.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if not session:
            return RedirectResponse("/")

        audio_files = get_audio_files()
        total = len(audio_files)
        if question_num < 1 or question_num > total:
            return RedirectResponse(f"/survey/{session_id}/1")

        existing = db.execute(
            "SELECT score FROM ratings WHERE session_id = ? AND question_num = ?",
            (session_id, question_num),
        ).fetchone()

    audio = audio_files[question_num - 1]
    return templates.TemplateResponse(request, "question.html", {
        "session_id": session_id,
        "audio": audio,
        "question_num": question_num,
        "total": total,
        "progress": int(question_num / total * 100),
        "existing_rating": existing["score"] if existing else None,
        "prev_num": question_num - 1 if question_num > 1 else None,
        "next_num": question_num + 1 if question_num < total else None,
    })


@app.post("/survey/{session_id}/{question_num}")
async def submit_rating(
    session_id: str,
    question_num: int,
    rating: int = Form(...),
    action: str = Form("next"),
):
    if not (1 <= rating <= 5):
        raise HTTPException(status_code=400, detail="점수는 1~5 사이여야 합니다.")

    audio_files = get_audio_files()
    total = len(audio_files)
    if question_num < 1 or question_num > total:
        raise HTTPException(status_code=400)

    audio_filename = audio_files[question_num - 1]["filename"]

    with get_db() as db:
        session = db.execute(
            "SELECT completed FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if not session:
            raise HTTPException(status_code=404)

        # UPSERT: 이미 답한 문항이면 덮어씀
        db.execute(
            """INSERT INTO ratings (session_id, audio_filename, question_num, score, answered_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(session_id, question_num) DO UPDATE SET
                   score = excluded.score,
                   answered_at = excluded.answered_at""",
            (session_id, audio_filename, question_num, rating, now()),
        )

    if action == "finish" or question_num == total:
        return RedirectResponse(f"/complete/{session_id}", status_code=303)
    return RedirectResponse(f"/survey/{session_id}/{question_num + 1}", status_code=303)


@app.get("/complete/{session_id}", response_class=HTMLResponse)
async def complete(request: Request, session_id: str):
    with get_db() as db:
        session = db.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if not session:
            return RedirectResponse("/")

        audio_files = get_audio_files()
        total = len(audio_files)

        ratings_rows = db.execute(
            "SELECT question_num, score FROM ratings WHERE session_id = ? ORDER BY question_num",
            (session_id,),
        ).fetchall()
        ratings = {str(r["question_num"]): r["score"] for r in ratings_rows}
        answered = len(ratings)

        if answered < total:
            unanswered = [i for i in range(1, total + 1) if str(i) not in ratings]
            return templates.TemplateResponse(request, "incomplete.html", {
                "session_id": session_id,
                "answered": answered,
                "total": total,
                "unanswered": unanswered,
            })

        # 중복 제출 방지
        if not session["completed"]:
            db.execute(
                "UPDATE sessions SET completed = 1, completed_at = ? WHERE session_id = ?",
                (now(), session_id),
            )

    return templates.TemplateResponse(request, "complete.html", {
        "session_id": session_id,
        "ratings": ratings,
        "audio_files": audio_files,
    })


# ---------------------------------------------------------------------------
# 관리자
# ---------------------------------------------------------------------------

@app.get("/admin", response_class=HTMLResponse)
async def admin(request: Request):
    with get_db() as db:
        active    = db.execute("SELECT COUNT(*) FROM sessions WHERE completed = 0").fetchone()[0]
        completed = db.execute("SELECT COUNT(*) FROM sessions WHERE completed = 1").fetchone()[0]

        audio_files = get_audio_files()
        if not audio_files or completed == 0:
            stats = None
        else:
            per_audio = {}
            for af in audio_files:
                rows = db.execute(
                    """SELECT r.score FROM ratings r
                       JOIN sessions s ON r.session_id = s.session_id
                       WHERE r.audio_filename = ? AND s.completed = 1""",
                    (af["filename"],),
                ).fetchall()
                scores = [r["score"] for r in rows]
                if not scores:
                    continue
                mean = round(sum(scores) / len(scores), 2)
                variance = sum((s - mean) ** 2 for s in scores) / len(scores) if len(scores) > 1 else 0
                std = round(variance ** 0.5, 2)
                per_audio[af["filename"]] = {"mean": mean, "std": std, "count": len(scores)}

            stats = {
                "total_respondents": completed,
                "per_audio": per_audio,
                "download_url": "/admin/download",
            }

    return templates.TemplateResponse(request, "admin.html", {
        "stats": stats,
        "active_sessions": active,
        "completed_sessions": completed,
    })


@app.get("/admin/download")
async def download_csv():
    """완료된 설문 결과를 CSV로 내보낸다 (참여자당 1행)."""
    audio_files = get_audio_files()
    fieldnames = ["session_id", "submitted_at", "age", "gender"] + [af["filename"] for af in audio_files]

    with get_db() as db:
        sessions = db.execute(
            "SELECT * FROM sessions WHERE completed = 1 ORDER BY completed_at"
        ).fetchall()

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for s in sessions:
            ratings_rows = db.execute(
                "SELECT audio_filename, score FROM ratings WHERE session_id = ?",
                (s["session_id"],),
            ).fetchall()
            rating_map = {r["audio_filename"]: r["score"] for r in ratings_rows}

            row = {
                "session_id":   s["session_id"],
                "submitted_at": s["completed_at"],
                "age":          s["age"],
                "gender":       s["gender"],
            }
            for af in audio_files:
                row[af["filename"]] = rating_map.get(af["filename"], "")
            writer.writerow(row)

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=survey_results.csv"},
    )


@app.get("/api/stats")
async def api_stats():
    with get_db() as db:
        completed = db.execute("SELECT COUNT(*) FROM sessions WHERE completed = 1").fetchone()[0]
        audio_files = get_audio_files()
        audio_stats = {}
        for af in audio_files:
            rows = db.execute(
                """SELECT r.score FROM ratings r
                   JOIN sessions s ON r.session_id = s.session_id
                   WHERE r.audio_filename = ? AND s.completed = 1""",
                (af["filename"],),
            ).fetchall()
            scores = [r["score"] for r in rows]
            if scores:
                mean = sum(scores) / len(scores)
                variance = sum((s - mean) ** 2 for s in scores) / len(scores)
                audio_stats[af["filename"]] = {"mean": round(mean, 2), "std": round(variance ** 0.5, 2)}

    return {"total": completed, "audio_stats": audio_stats}
