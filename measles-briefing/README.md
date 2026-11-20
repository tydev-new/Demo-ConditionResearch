# Measles briefing

A small research-gathering and Q&A demo for a health-system strategy team. It answers questions about **measles only**, from three public sources cached in `data/corpus.json`:

1. [CDC Clinical Overview of Measles](https://www.cdc.gov/measles/hcp/clinical-overview/index.html) (page text)
2. [ClinicalTrials.gov API v2](https://clinicaltrials.gov/api/v2/studies?query.cond=measles) (one short chunk per study: title, NCT id, phase, status, sponsor, collaborators, interventions, study link)
3. [PubMed E-utilities](https://pubmed.ncbi.nlm.nih.gov/) (`measles[tiab] AND 2025:2026[dp]`, top 20 abstracts that have text)

Asking a question retrieves the top 5 chunks with a small pure-Python TF-IDF cosine index. If the best score is below the threshold in `engine.py`, the answer is exactly: `Not enough evidence in the gathered sources.` Otherwise the answer is written only from those chunks. Every sentence carries a chunk id such as `[cdc-003]`, and sentences without a valid citation are dropped.

**Run briefing** asks three fixed questions: standard of care, treatments or vaccines in trials, and organizations involved.

The default answerer is a **mock extractive** mode (no API key). It picks the most relevant sentences from the retrieved chunks and cites them. If `OPENAI_API_KEY` is set, the app calls a chat-completions endpoint over plain HTTPS instead.

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

Tests (no network; they use the committed corpus):

```bash
pytest
```

## Refresh the corpus

Fetching happens offline, on purpose. Serverless requests should not depend on CDC, ClinicalTrials.gov, or PubMed being reachable.

```bash
python scripts/fetch_sources.py
```

The script writes `data/corpus.json`. Commit that file when you refresh it.

CDC returns HTTP 200 to Python’s default urllib user-agent and HTTP 403 if the client sends a browser user-agent. The fetch script does **not** set `User-Agent`.

## Deploy on Vercel

1. Import this Git repository.
2. Set **Root Directory** to `measles-briefing`.
3. Leave the framework on automatic detection. Vercel’s Python runtime picks up `requirements.txt`, `.python-version` (`3.12`), and the FastAPI entrypoint `app.py` (`app = FastAPI()`).
4. No build command. `vercel.json` adds `data/**` and `static/**` to the function bundle so the cached corpus and the page are available at runtime. The working directory on Vercel is the project root (`measles-briefing`); the app also resolves files relative to `app.py`.
5. Deploy.

Optional environment variables, if you want real model answers instead of the mock:

| Name | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Turns on the HTTPS chat-completions path |
| `OPENAI_BASE_URL` | Defaults to `https://api.openai.com/v1` (include `/v1`) |
| `OPENAI_MODEL` or `MODEL` | Defaults to `gpt-4o-mini` |

The page shows whether the answer used mock extractive mode or OpenAI. If the key is set but the call fails, the app falls back to the mock and says so.
