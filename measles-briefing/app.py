"""Measles briefing demo.

Vercel detects this file because it defines a top-level `app = FastAPI()`.
Set the project Root Directory to `measles-briefing`.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from engine import answer_question, corpus_summary, run_briefing

ROOT = Path(__file__).resolve().parent
INDEX_PATH = ROOT / "static" / "index.html"

app = FastAPI(title="Measles briefing")


class AskBody(BaseModel):
    question: str = Field(..., max_length=1000)


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    if not INDEX_PATH.is_file():
        raise HTTPException(status_code=500, detail="UI file is missing")
    return HTMLResponse(INDEX_PATH.read_text(encoding="utf-8"))


@app.get("/api/meta")
def meta() -> dict:
    return corpus_summary()


@app.post("/api/ask")
def ask(body: AskBody) -> dict:
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")
    return answer_question(question)


@app.post("/api/briefing")
def briefing() -> dict:
    answers = run_briefing()
    return {"answers": answers, "mode": answers[0]["mode"] if answers else "mock"}
