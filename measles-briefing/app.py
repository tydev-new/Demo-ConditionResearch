"""Condition research briefing.

Vercel detects this file because it defines a top-level `app = FastAPI()`.
Set the project Root Directory to `measles-briefing`.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from engine import answer_question, drop_index, run_briefing, runtime_meta
import agent
from sources import QUICK_PICKS, load_condition, normalize_condition, public_summary, remember

ROOT = Path(__file__).resolve().parent
INDEX_PATH = ROOT / "static" / "index.html"

app = FastAPI(title="Condition research briefing")


class QueryBody(BaseModel):
    query: str = Field(..., max_length=200)


class LoadBody(BaseModel):
    id: str = Field(..., max_length=80)
    name: str = Field(..., max_length=200)
    mesh_id: str = Field(..., max_length=20)
    synonyms: list[str] = Field(default_factory=list)
    query: str | None = Field(default=None, max_length=200)


class AskBody(BaseModel):
    question: str = Field(..., max_length=1000)
    condition_id: str = Field(..., max_length=80)


class BriefingBody(BaseModel):
    condition_id: str = Field(..., max_length=80)


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    if not INDEX_PATH.is_file():
        raise HTTPException(status_code=500, detail="UI file is missing")
    return HTMLResponse(INDEX_PATH.read_text(encoding="utf-8"))


@app.get("/api/meta")
def meta() -> dict:
    payload = runtime_meta()
    payload["quick_picks"] = QUICK_PICKS
    return payload


@app.post("/api/resolve")
def resolve(body: QueryBody) -> dict:
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Enter a condition name.")
    try:
        return normalize_condition(query)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"MeSH lookup failed: {exc}") from exc


@app.post("/api/conditions/load")
def load(body: LoadBody) -> dict:
    chosen = body.model_dump()
    try:
        loaded = load_condition(chosen)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not gather sources: {exc}") from exc
    document = loaded["document"]
    remember(document, persist_tmp=loaded["origin"] == "fetched")
    drop_index(document["condition"]["id"])
    summary = public_summary(document, loaded["origin"])
    condition = document["condition"]
    summary["mapping"] = {
        "query": body.query or condition["name"],
        "mesh_name": condition["name"],
        "mesh_id": condition.get("mesh_id"),
        "mesh_url": f"https://meshb.nlm.nih.gov/record/ui?ui={condition.get('mesh_id')}",
        "synonyms": list(condition.get("synonyms") or []),
    }
    return summary


@app.post("/api/ask")
def ask(body: AskBody) -> dict:
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")
    try:
        return agent.answer(question, body.condition_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/briefing")
def briefing(body: BriefingBody) -> dict:
    try:
        answers = agent.run_briefing(body.condition_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"answers": answers, "mode": answers[0]["mode"] if answers else "mock"}
