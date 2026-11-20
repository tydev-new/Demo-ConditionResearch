"""Tool-calling agent for the Ask step.

The model (OpenAI-compatible chat completions with tools, e.g. via OpenRouter) decides which
evidence to gather: the condition's indexed chunks, live ClinicalTrials.gov, live PubMed, or an
allowlisted public page. Every chunk a tool returns joins the condition's evidence pool, and the
final answer goes through the same per-sentence citation validator as the fixed pipeline. Any
LLM or tool-loop failure falls back to engine.answer_question (fixed retrieve + answer).
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from html.parser import HTMLParser

import engine
import sources

MAX_TOOL_CALLS = 4
TIME_BUDGET = 40.0          # seconds for the whole loop; Vercel maxDuration is 60
TOOL_TIMEOUT = 8            # seconds per outbound HTTP request
LLM_TIMEOUT = 20            # cap per model call (also limited by the remaining budget)
CHUNK_PREVIEW = 700         # characters of each chunk shown to the model
ALLOWED_HOSTS = (
    "medlineplus.gov", "cdc.gov", "nih.gov", "ncbi.nlm.nih.gov",
    "clinicaltrials.gov", "who.int", "fda.gov",
)
TRIAL_STATUSES = {
    "RECRUITING", "NOT_YET_RECRUITING", "ENROLLING_BY_INVITATION", "ACTIVE_NOT_RECRUITING",
    "COMPLETED", "TERMINATED", "SUSPENDED", "WITHDRAWN",
}
TRIAL_PHASES = {"EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4"}

_LOCK = threading.Lock()


class AgentError(RuntimeError):
    pass


# ---------------------------------------------------------------- tool schemas

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_sources",
            "description": "Search the sources already gathered for this condition (MedlinePlus/CDC "
                           "guidance, trial records, PubMed abstracts). Start here.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "What to look for."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_trials",
            "description": "Live ClinicalTrials.gov search scoped to this condition. Use for trials, "
                           "pipeline, phases, recruitment, sponsors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Extra keywords, e.g. a drug; may be empty."},
                    "phase": {"type": "string", "enum": sorted(TRIAL_PHASES)},
                    "status": {"type": "string", "enum": sorted(TRIAL_STATUSES)},
                    "sponsor": {"type": "string", "description": "Lead sponsor or collaborator name."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_pubmed",
            "description": "Live PubMed search scoped to this condition; returns up to 5 recent abstracts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "years": {"type": "integer", "description": "Only papers from the last N years (1-10)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": "Fetch a public page from medlineplus.gov, cdc.gov, nih.gov, clinicaltrials.gov, "
                           "who.int or fda.gov and return its text as chunks.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
]


# ---------------------------------------------------------------- evidence pool

def add_chunks(condition_id: str, chunks: list[dict]) -> list[dict]:
    """Add tool-fetched chunks to the condition's in-memory pool (reused by later questions)."""
    state = engine.get_state(condition_id)
    out = []
    with _LOCK:
        added = False
        for chunk in chunks:
            existing = state["by_id"].get(chunk["id"])
            if existing:
                out.append(existing)
                continue
            chunk["condition_id"] = condition_id
            state["chunks"].append(chunk)
            state["by_id"][chunk["id"]] = chunk
            out.append(chunk)
            added = True
        if added:
            state["index"] = engine.TfidfIndex(state["chunks"])
    return out


def condition_name(condition_id: str) -> str:
    state = engine.get_state(condition_id)
    return (state["document"].get("condition") or {}).get("name") or condition_id


# ---------------------------------------------------------------- tools

def tool_search_sources(condition_id: str, query: str, k: int = 6) -> tuple[list[dict], str]:
    state = engine.get_state(condition_id)
    pool = [(s, c) for s, c in state["index"].search(query, k=engine.CANDIDATE_POOL)
            if c.get("condition_id") == condition_id and s >= engine.SCORE_THRESHOLD * 0.5]
    hits = [c for _s, c in engine.rerank(query, pool, k=k)]
    return hits, f"{len(hits)} chunks"


def tool_search_trials(condition_id: str, query: str = "", phase: str | None = None,
                       status: str | None = None, sponsor: str | None = None) -> tuple[list[dict], str]:
    params = {
        "query.cond": sources._plain_name(condition_name(condition_id)),
        "pageSize": "15",
        "sort": "LastUpdatePostDate:desc",
    }
    if query and query.strip():
        params["query.term"] = query.strip()[:200]
    if status:
        status = status.upper().replace(" ", "_")
        if status not in TRIAL_STATUSES:
            raise ValueError(f"unknown status {status}")
        params["filter.overallStatus"] = status
    if phase:
        phase = phase.upper().replace(" ", "")
        if phase not in TRIAL_PHASES:
            raise ValueError(f"unknown phase {phase}")
        params["filter.advanced"] = f"AREA[Phase]{phase}"
    if sponsor and sponsor.strip():
        params["query.spons"] = sponsor.strip()[:120]
    url = sources.TRIALS_URL + "?" + urllib.parse.urlencode(params)
    payload = sources.fetch_json(url, timeout=TOOL_TIMEOUT)
    chunks = [c for c in (sources._study_chunk(condition_id, s) for s in payload.get("studies") or []) if c]
    chunks = add_chunks(condition_id, chunks)
    return chunks, f"{len(chunks)} trials found"


def tool_search_pubmed(condition_id: str, query: str, years: int | None = None) -> tuple[list[dict], str]:
    name = sources._plain_name(condition_name(condition_id))
    term = f"({name}[tiab]) AND ({query.strip()[:200]})" if query.strip() else f"{name}[tiab]"
    if years:
        years = max(1, min(int(years), 10))
        today = date.today()
        start = date(today.year - years, today.month, min(today.day, 28)).isoformat().replace("-", "/")
        term += f" AND {start}:{today.isoformat().replace('-', '/')}[dp]"
    common = {"tool": "condition-briefing", "email": "demo@example.com"}
    found = sources.fetch_json(sources.PUBMED_SEARCH + "?" + urllib.parse.urlencode(
        {"db": "pubmed", "term": term, "retmax": "5", "retmode": "json", "sort": "relevance", **common}),
        timeout=TOOL_TIMEOUT)
    ids = (found.get("esearchresult") or {}).get("idlist") or []
    if not ids:
        return [], "0 abstracts"
    xml = sources.fetch_bytes(sources.PUBMED_FETCH + "?" + urllib.parse.urlencode(
        {"db": "pubmed", "id": ",".join(ids), "retmode": "xml", **common}),
        timeout=TOOL_TIMEOUT, accept="application/xml")
    chunks = []
    for article in ET.fromstring(xml).findall(".//PubmedArticle"):
        pmid = (article.findtext(".//PMID") or "").strip()
        title_node = article.find(".//ArticleTitle")
        title = " ".join("".join(title_node.itertext()).split()) if title_node is not None else ""
        parts = []
        for node in article.findall(".//Abstract/AbstractText"):
            bit = " ".join("".join(node.itertext()).split())
            if bit:
                label = node.attrib.get("Label")
                parts.append(f"{label}: {bit}" if label else bit)
        if not (pmid and title and parts):
            continue
        year = article.findtext(".//PubDate/Year") or ""
        chunks.append({
            "id": f"{condition_id}-pmid-{pmid}",
            "condition_id": condition_id,
            "source_name": "PubMed",
            "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "text": f"{title.rstrip('.')}{' (' + year + ')' if year else ''}. {' '.join(parts)[:1400]}",
        })
    chunks = add_chunks(condition_id, chunks)
    return chunks, f"{len(chunks)} abstracts"


def host_allowed(url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("https", "http") or not host:
        return False
    return any(host == allowed or host.endswith("." + allowed) for allowed in ALLOWED_HOSTS)


class _TextOnly(HTMLParser):
    SKIP = {"script", "style", "nav", "header", "footer", "noscript", "svg", "form"}

    def __init__(self):
        super().__init__()
        self.depth = 0
        self.parts: list[str] = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.depth += 1
        if tag == "title":
            self._in_title = True
        if tag in {"p", "li", "h1", "h2", "h3", "h4", "br", "tr", "div"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.depth:
            self.depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self.depth:
            self.parts.append(data)


def tool_fetch_page(condition_id: str, url: str) -> tuple[list[dict], str]:
    if not host_allowed(url):
        raise PermissionError(f"host not allowed: {urllib.parse.urlparse(url).hostname}")
    # No browser User-Agent: CDC answers 403 to one.
    request = urllib.request.Request(url, headers={"Accept": "text/html,application/xhtml+xml"})
    with urllib.request.urlopen(request, timeout=TOOL_TIMEOUT) as response:
        final = response.geturl()
        if not host_allowed(final):
            raise PermissionError("redirected to a host that is not allowed")
        raw = response.read(1_500_000).decode("utf-8", errors="replace")
    parser = _TextOnly()
    parser.feed(raw)
    text = re.sub(r"[ \t]+", " ", " ".join(parser.parts))
    text = re.sub(r"\s*\n\s*", "\n", text).strip()
    title = " ".join(parser.title.split()) or (urllib.parse.urlparse(final).hostname or "page")
    host = (urllib.parse.urlparse(final).hostname or "").removeprefix("www.")
    digest = hashlib.sha1(final.encode()).hexdigest()[:8]
    pieces = engine.chunk_text(text)[:8]
    chunks = [{
        "id": f"{condition_id}-web-{digest}-{n:02d}",
        "condition_id": condition_id,
        "source_name": f"{title} ({host})",
        "source_url": final,
        "text": piece,
    } for n, piece in enumerate(pieces, start=1)]
    chunks = add_chunks(condition_id, chunks)
    return chunks, f"{len(chunks)} chunks from {host}"


TOOL_FUNCS = {
    "search_sources": tool_search_sources,
    "search_trials": tool_search_trials,
    "search_pubmed": tool_search_pubmed,
    "fetch_page": tool_fetch_page,
}


def dispatch(condition_id: str, name: str, args: dict) -> tuple[list[dict], str]:
    func = TOOL_FUNCS.get(name)
    if func is None:
        raise ValueError(f"unknown tool {name}")
    allowed = set(func.__code__.co_varnames[1:func.__code__.co_argcount])
    clean = {k: v for k, v in (args or {}).items() if k in allowed and v not in (None, "")}
    return func(condition_id, **clean)


# ---------------------------------------------------------------- LLM client

class OpenAIChatClient:
    """Minimal OpenAI-compatible chat client (OpenRouter works). The key stays in env."""

    def __init__(self):
        config = engine._openai_config()
        if config is None:
            raise AgentError("OPENAI_API_KEY is not set")
        self._key, self.base, self.model = config

    def chat(self, messages: list[dict], tools: list[dict] | None, *, timeout: float,
             tool_choice: str = "auto") -> dict:
        body = {"model": self.model, "temperature": 0, "messages": messages}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice
        request = urllib.request.Request(
            f"{self.base}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise AgentError(f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:300]}") from exc
        try:
            return payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AgentError("model response had no message") from exc


SYSTEM = (
    "You are a research agent writing short condition briefings for a health-system strategy team. "
    "Use the tools to gather evidence, then answer ONLY from chunks returned by the tools. "
    "Start with search_sources. Use search_trials for trials, phases, recruitment or sponsors "
    "(set phase/status filters when the question names them), search_pubmed for recent research, "
    "and fetch_page only for an allowlisted page you have a URL for. You may make at most "
    f"{MAX_TOOL_CALLS} tool calls. Final answer: 2-6 sentences; after every sentence cite the "
    "supporting chunk id(s) in square brackets, e.g. [heart-failure-mplus-010]. Never cite an id "
    "you were not given. State only facts written in the cited chunk text; never add drug names, "
    "numbers or details from your own knowledge, even well-known ones. Guideline/patient-education chunks (MedlinePlus, CDC) describe usual care; "
    "trial sponsors and collaborators are organizations involved in the condition. If the chunks do "
    f"not directly answer the question, or it is unrelated to the condition, reply exactly: {engine.NOT_ENOUGH}"
)


def _render_chunks(chunks: list[dict]) -> str:
    if not chunks:
        return "No results."
    return "\n\n".join(
        f"[{c['id']}] {c['source_name']} ({c['source_url']})\n{c['text'][:CHUNK_PREVIEW]}" for c in chunks
    )


def _args_label(args: dict) -> str:
    return ", ".join(f'{k}={json.dumps(v)}' for k, v in args.items() if v not in (None, ""))


def run_agent(question: str, condition_id: str, *, client=None, budget: float = TIME_BUDGET) -> dict:
    """Agent loop. Raises AgentError (or anything) on failure; caller falls back."""
    started = time.monotonic()
    client = client or OpenAIChatClient()
    name = condition_name(condition_id)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Condition: {name}\nQuestion: {question}"},
    ]
    steps: list[dict] = []
    seen: dict[str, dict] = {}
    calls = 0
    final = None
    while True:
        remaining = budget - (time.monotonic() - started)
        if remaining < 2:
            raise AgentError("time budget exhausted")
        force_final = calls >= MAX_TOOL_CALLS
        message = client.chat(messages, TOOLS, timeout=min(LLM_TIMEOUT, remaining),
                              tool_choice="none" if force_final else "auto")
        tool_calls = message.get("tool_calls") or []
        if not tool_calls or force_final:
            final = (message.get("content") or "").strip()
            break
        messages.append({"role": "assistant", "content": message.get("content") or "", "tool_calls": tool_calls})
        for call in tool_calls:
            fn = (call.get("function") or {})
            tool = fn.get("name") or ""
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            t0 = time.monotonic()
            if calls >= MAX_TOOL_CALLS:
                chunks, summary, content = [], "skipped (tool limit reached)", "Tool limit reached; answer now."
            else:
                calls += 1
                try:
                    chunks, summary = dispatch(condition_id, tool, args)
                    content = _render_chunks(chunks)
                except Exception as exc:  # a tool error is reported to the model, not fatal
                    chunks, summary = [], f"error: {str(exc)[:120]}"
                    content = f"Tool error: {str(exc)[:200]}"
            for chunk in chunks:
                seen[chunk["id"]] = chunk
            steps.append({"tool": tool, "args": args, "result_summary": summary,
                          "ms": int((time.monotonic() - t0) * 1000)})
            messages.append({"role": "tool", "tool_call_id": call.get("id") or tool, "content": content})

    text, citations = engine.enforce_citations(final or "", set(seen))
    if not citations:
        text, citations = engine.NOT_ENOUGH, []
    result = engine._pack(question, text, citations, [], condition_id=condition_id, by_id=seen,
                          mode="agent", model=getattr(client, "model", None), warning=None)
    for source in result["sources"]:
        source.pop("score", None)
    result["agent_steps"] = steps
    result["elapsed_ms"] = int((time.monotonic() - started) * 1000)
    return result


def answer(question: str, condition_id: str, *, client=None, force_mock: bool = False) -> dict:
    """Agent when an LLM is configured (or a client is injected); fixed pipeline otherwise or on failure."""
    started = time.monotonic()
    if not force_mock and (client is not None or engine._openai_config() is not None):
        try:
            return run_agent(question, condition_id, client=client)
        except FileNotFoundError:
            raise
        except Exception as exc:
            result = engine.answer_question(question, condition_id=condition_id, force_mock=force_mock)
            note = f"Agent failed ({str(exc)[:160]}); used the fixed retrieval pipeline."
            result["warning"] = f"{note} {result['warning']}" if result.get("warning") else note
            result["agent_steps"] = []
            result["agent_fallback"] = True
            result["elapsed_ms"] = int((time.monotonic() - started) * 1000)
            return result
    result = engine.answer_question(question, condition_id=condition_id, force_mock=force_mock)
    result["agent_steps"] = []
    result["elapsed_ms"] = int((time.monotonic() - started) * 1000)
    return result


def run_briefing(condition_id: str, *, client=None, force_mock: bool = False) -> list[dict]:
    engine.get_state(condition_id)  # 404 early
    questions = engine.briefing_questions(condition_name(condition_id))
    with ThreadPoolExecutor(max_workers=3) as pool:
        return list(pool.map(lambda q: answer(q, condition_id, client=client, force_mock=force_mock), questions))
