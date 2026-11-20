# Condition research briefing

A small research-gathering and Q&A demo for a health-system strategy team. The folder is still `measles-briefing` so an existing Vercel project can keep that root directory.

Ask about any condition. The app resolves the name with the NLM MeSH descriptor lookup (no API key). An exact label, or a single hit, is selected automatically. Several hits are returned for the user to pick.

Each condition has its own corpus:

1. MedlinePlus health topic (top web-service hit: page URL plus the full summary)
2. ClinicalTrials.gov API v2 (`query.cond`), capped at about 100 studies, recruiting and active first
3. PubMed E-utilities (`name[tiab]` over the last 2 years, top 20 abstracts)

Quick picks (Measles, Heart failure, Type 2 diabetes, Asthma) are committed under `data/corpora/<condition_id>.json`, so those load immediately. Measles keeps the original CDC clinical overview snapshot alongside trials and PubMed. Any other condition is fetched on demand (about 8 seconds per request, under about 25 seconds overall), then cached in memory and `/tmp`. Vercel’s filesystem is read-only except `/tmp`. **Vercel Blob** is the right place for durable per-condition corpora later; this demo does not use it.

Retrieval is a small pure-Python TF-IDF index over that condition only. Top 5 chunks. If the best score is under the threshold in `engine.py`, the answer is exactly: `Not enough evidence in the gathered sources.` Otherwise every sentence cites a chunk id, and sentences without a valid citation are dropped.

**Run briefing** asks three fixed questions with the condition name filled in: standard of care, treatments or vaccines in trials, and organizations involved.

The default answerer is **mock extractive** (no API key). If `OPENAI_API_KEY` is set, the app calls chat completions over plain HTTPS.

This is a briefing demo, not medical advice.

## Run locally

```bash
cd measles-briefing
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000

```bash
pytest
```

## Refresh the quick-pick corpora

```bash
python scripts/fetch_sources.py
```

The script moves the legacy measles snapshot to `data/corpora/measles.json` if needed, then fetches heart failure, type 2 diabetes, and asthma. It does **not** set `User-Agent`. CDC returns HTTP 403 when the client spoofs a browser. NCBI calls stay at or under 3 per second.

## Deploy on Vercel

1. Import this Git repository.
2. Set **Root Directory** to `measles-briefing`.
3. Leave framework detection on. Entrypoint is `app.py` (`app = FastAPI()`). Python **3.12** comes from `.python-version`.
4. No build command. `vercel.json` sets `maxDuration` to **60** (Hobby maximum) on `app.py` and includes `data/**` plus `static/**` so the prebuilt corpora and the page ship with the function.
5. On-demand gathers write only to `/tmp` and process memory. A new instance fetches again.

Optional environment variables:

| Name | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Turns on the HTTPS chat-completions path |
| `OPENAI_BASE_URL` | Defaults to `https://api.openai.com/v1` |
| `OPENAI_MODEL` or `MODEL` | Defaults to `gpt-4o-mini` |
