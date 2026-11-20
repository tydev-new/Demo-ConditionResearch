"""TF-IDF retrieval and cited answers over the cached measles corpus.

Pure Python on purpose: the deploy target is a small Vercel Python function,
so this file does not import scikit-learn or an LLM SDK.
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TOP_K = 5
# Cosine on short questions vs ~800-character chunks. Tuned so the measles
# standard-of-care question clears it and an unrelated question does not.
SCORE_THRESHOLD = 0.12
CHUNK_SIZE = 800
CHUNK_OVERLAP = 160

NOT_ENOUGH = "Not enough evidence in the gathered sources."

def briefing_questions(condition_name: str) -> list[str]:
    name = condition_name.strip()
    return [
        f"What is the current standard of care for {name}?",
        f"What treatments or vaccines are in development / in clinical trials for {name}?",
        f"Which key organizations (companies, institutions, government bodies) are involved in {name}?",
    ]


BRIEFING_QUESTIONS = briefing_questions("measles")

SOURCE_ORDER = [
    "CDC Clinical Overview of Measles",
    "ClinicalTrials.gov",
    "PubMed",
]

SOURCE_HOME = {
    "CDC Clinical Overview of Measles": "https://www.cdc.gov/measles/hcp/clinical-overview/index.html",
    "ClinicalTrials.gov": "https://clinicaltrials.gov/search?cond=measles",
    "PubMed": "https://pubmed.ncbi.nlm.nih.gov/?term=measles%5Btiab%5D+AND+2025%3A2026%5Bdp%5D",
}

_STOP = {
    "a", "an", "the", "of", "for", "and", "or", "to", "in", "on", "with", "by",
    "from", "at", "as", "is", "are", "was", "were", "be", "been", "being", "it",
    "this", "that", "these", "those", "what", "which", "who", "whom", "how",
    "when", "where", "why", "do", "does", "did", "can", "could", "should",
    "would", "may", "might", "about", "into", "over", "under", "than", "then",
    "not", "no", "if", "but", "we", "you", "they", "their", "our", "your", "its",
    "also", "such", "using", "use", "used", "per", "via", "within", "without",
    "current", "key", "involved",
}

# Query-side expansions so briefing wording ("organizations", "trials",
# "standard of care") can meet the words the sources actually use.
_EXPAND = {
    "care": ("treatment", "management", "supportive"),
    "standard": ("recommended", "recommendation", "approved"),
    "treatment": ("therapy", "antiviral", "management"),
    "treatments": ("treatment", "therapy", "antiviral", "management"),
    "vaccine": ("vaccination", "mmr"),
    "vaccines": ("vaccine", "vaccination", "mmr"),
    "trial": ("nct", "phase", "recruiting"),
    "trials": ("trial", "nct", "phase", "recruiting"),
    "development": ("phase", "investigational", "recruiting"),
    "organization": ("sponsor", "institution", "university", "agency"),
    "organizations": ("organization", "sponsor", "institution", "university", "agency"),
    "companies": ("company", "sponsor"),
    "company": ("sponsor",),
    "institutions": ("institution", "university", "hospital", "institute"),
    "institution": ("university", "hospital", "institute"),
    "government": ("cdc", "fda", "nih"),
}

_TOKEN = re.compile(r"[a-z0-9]+")
_CITE = re.compile(r"\[([^\]]+)\]")
_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
_SENT = re.compile(r"(?<=[.!?])\s+")

_INDEXES: dict[str, dict] = {}


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into ~`size` character windows with overlap, on a sentence boundary when possible."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            window = text[start:end]
            cut = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
            if cut >= size // 2:
                end = start + cut + 1
            else:
                space = window.rfind(" ")
                if space >= size // 2:
                    end = start + space
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        nxt = end - overlap
        if nxt <= start:
            nxt = end
        else:
            window = text[nxt:end]
            dot = max(window.find(". "), window.find("? "), window.find("! "))
            if dot != -1:
                nxt = nxt + dot + 2
            else:
                while nxt < end and not text[nxt].isspace():
                    nxt += 1
                while nxt < end and text[nxt].isspace():
                    nxt += 1
            if nxt <= start or nxt >= len(text):
                nxt = end
        start = nxt
    return chunks


def tokenize(text: str, *, expand: bool = False) -> list[str]:
    tokens = [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]
    if not expand:
        return tokens
    extra: list[str] = []
    for token in tokens:
        extra.extend(_EXPAND.get(token, ()))
    return tokens + extra


class TfidfIndex:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        docs = [tokenize(c["text"]) for c in chunks]
        self.n = len(docs)
        df: Counter[str] = Counter()
        for doc in docs:
            df.update(set(doc))
        self.idf = {
            term: math.log((1 + self.n) / (1 + freq)) + 1.0 for term, freq in df.items()
        }
        self.doc_vecs = [self._from_weights(dict(Counter(doc))) for doc in docs]
        self.doc_norms = [_norm(vec) for vec in self.doc_vecs]

    def _from_weights(self, weights: dict[str, float]) -> dict[str, float]:
        total = sum(weights.values()) or 1.0
        vec: dict[str, float] = {}
        for term, count in weights.items():
            idf = self.idf.get(term)
            if not idf or count <= 0:
                continue
            vec[term] = (count / total) * idf
        return vec

    def _query_vec(self, question: str) -> dict[str, float]:
        original = tokenize(question)
        weights = {token: 1.0 for token in original}
        for token in original:
            for extra in _EXPAND.get(token, ()):
                weights.setdefault(extra, 0.4)
        return self._from_weights(weights)

    def search(self, question: str, k: int = TOP_K) -> list[tuple[float, dict]]:
        qvec = self._query_vec(question)
        qn = _norm(qvec)
        if qn == 0.0 or not qvec:
            return []
        scored: list[tuple[float, dict]] = []
        for vec, dn, chunk in zip(self.doc_vecs, self.doc_norms, self.chunks):
            if dn == 0.0:
                continue
            dot = sum(weight * vec.get(term, 0.0) for term, weight in qvec.items())
            scored.append((dot / (qn * dn), chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:k]

    def sentence_score(self, sentence: str, question: str) -> float:
        qvec = self._query_vec(question)
        qn = _norm(qvec)
        counts = Counter(tokenize(sentence))
        svec = self._from_weights(dict(counts))
        sn = _norm(svec)
        if qn == 0.0 or sn == 0.0:
            return 0.0
        dot = sum(weight * svec.get(term, 0.0) for term, weight in qvec.items())
        return dot / (qn * sn)


def _norm(vec: dict[str, float]) -> float:
    if not vec:
        return 0.0
    return math.sqrt(sum(value * value for value in vec.values()))


def runtime_meta() -> dict:
    model = os.environ.get("OPENAI_MODEL") or os.environ.get("MODEL") or "gpt-4o-mini"
    return {
        "top_k": TOP_K,
        "threshold": SCORE_THRESHOLD,
        "llm_configured": bool(os.environ.get("OPENAI_API_KEY")),
        "model": model,
        "default_mode": "openai" if os.environ.get("OPENAI_API_KEY") else "mock",
    }


def drop_index(condition_id: str) -> None:
    _INDEXES.pop(condition_id, None)


def get_state(condition_id: str) -> dict:
    if condition_id in _INDEXES:
        return _INDEXES[condition_id]
    from sources import read_document

    document = read_document(condition_id)
    if not document:
        raise FileNotFoundError(
            f"No corpus for {condition_id}. Load the condition or run scripts/fetch_sources.py."
        )
    chunks = [
        chunk
        for chunk in document.get("chunks") or []
        if chunk.get("condition_id") in (None, condition_id)
    ]
    for chunk in chunks:
        chunk["condition_id"] = condition_id
    state = {
        "chunks": chunks,
        "by_id": {chunk["id"]: chunk for chunk in chunks},
        "index": TfidfIndex(chunks),
        "document": document,
    }
    _INDEXES[condition_id] = state
    return state


_ABBR_END = re.compile(r"(?:\bU\.S\.|\bU\.K\.|\be\.g\.|\bi\.e\.|\bDr\.|\bMr\.|\bMs\.|\bvs\.)$")


def _split_sentences(text: str) -> list[str]:
    raw = [part.strip() for part in _SENT.split(text.strip()) if part.strip()]
    merged: list[str] = []
    for part in raw:
        if merged and _ABBR_END.search(merged[-1]):
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged


def _usable_sentence(sentence: str) -> bool:
    text = sentence.strip().lstrip("\"'“”(")
    if not text or not text[0].isupper():
        return False
    lowered = text.lower()
    field = any(lowered.startswith(prefix) for prefix, _cues in _FIELD_CUES)
    min_words = 4 if field else 8
    return len(text) >= 28 and len(text.split()) >= min_words


# Metadata lines inside a trial chunk. The mock leans on them only when the
# question is actually about trials, vaccines, or organizations.
_FIELD_CUES = (
    ("sponsor:", ("organization", "organizations", "company", "companies", "institution", "institutions", "government", "involved")),
    ("collaborators:", ("organization", "organizations", "company", "companies", "institution", "institutions", "government", "involved")),
    ("interventions:", ("treatment", "treatments", "vaccine", "vaccines", "therapy", "development")),
    ("clinical trial:", ("trial", "trials", "vaccine", "vaccines", "treatment", "treatments", "development")),
)


def _field_factor(sentence: str, question_tokens: set[str]) -> float:
    lowered = sentence.lower()
    for prefix, cues in _FIELD_CUES:
        if lowered.startswith(prefix):
            return 1.25 if question_tokens & set(cues) else 0.12
    return 1.0


def _ids_in(sentence: str) -> list[str]:
    found: list[str] = []
    for match in _CITE.finditer(sentence):
        for cid in _ID.findall(match.group(1)):
            if cid not in found:
                found.append(cid)
    return found


def enforce_citations(text: str, allowed: set[str]) -> tuple[str, list[str]]:
    """Keep sentences that cite an allowed chunk id. Drop everything else."""
    kept: list[tuple[str, list[str]]] = []
    for sentence in _split_sentences(text):
        ids = [cid for cid in _ids_in(sentence) if cid in allowed]
        body = _CITE.sub(" ", sentence)
        body = re.sub(r"\s+", " ", body).strip(" \t;")
        if body and ids:
            kept.append((body, ids))
            continue
        if ids and not body and kept:
            prev_body, prev_ids = kept[-1]
            for cid in ids:
                if cid not in prev_ids:
                    prev_ids.append(cid)
            kept[-1] = (prev_body, prev_ids)
    used: list[str] = []
    parts: list[str] = []
    for body, ids in kept:
        body = body.rstrip(" .")
        cite = " ".join(f"[{cid}]" for cid in ids)
        parts.append(f"{body} {cite}.")
        for cid in ids:
            if cid not in used:
                used.append(cid)
    return " ".join(parts), used


def _mock_answer(question: str, retrieved: list[tuple[float, dict]], index: TfidfIndex) -> str:
    """Extractive answer: most relevant sentences from the retrieved chunks, each cited."""
    original = set(tokenize(question))
    candidates: list[tuple[float, int, int, str, str]] = []
    for rank, (_score, chunk) in enumerate(retrieved):
        sentences = [s for s in _split_sentences(chunk["text"]) if _usable_sentence(s)]
        if not sentences and chunk["text"].strip():
            sentences = [chunk["text"].strip()]
        for pos, sentence in enumerate(sentences):
            relevance = index.sentence_score(sentence, question)
            if relevance <= 0:
                continue
            if not (original & set(tokenize(sentence))):
                relevance *= 0.55
            relevance *= _field_factor(sentence, original)
            relevance += 0.01 * (TOP_K - rank)
            candidates.append((relevance, rank, pos, sentence, chunk["id"]))
    candidates.sort(key=lambda item: item[0], reverse=True)

    chosen: list[tuple[int, int, str, str]] = []
    seen: set[str] = set()
    per_chunk: Counter[str] = Counter()
    for _rel, rank, pos, sentence, cid in candidates:
        key = sentence.lower()[:96]
        cap = 3 if rank == 0 else 1
        if key in seen or per_chunk[cid] >= cap:
            continue
        seen.add(key)
        per_chunk[cid] += 1
        chosen.append((rank, pos, sentence, cid))
        if len(chosen) >= 4:
            break
    if not chosen and retrieved:
        chunk = retrieved[0][1]
        snippet = chunk["text"][:420].rstrip()
        return f"{snippet} [{chunk['id']}]"
    chosen.sort(key=lambda item: (item[0], item[1]))
    return " ".join(f"{sentence.rstrip(' .')} [{cid}]." for _r, _p, sentence, cid in chosen)


def _openai_config() -> tuple[str, str, str] | None:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
    model = (os.environ.get("OPENAI_MODEL") or os.environ.get("MODEL") or "gpt-4o-mini").strip()
    return key, base, model


def _call_openai(question: str, chunks: list[dict]) -> tuple[str, str]:
    config = _openai_config()
    if config is None:
        raise RuntimeError("OPENAI_API_KEY is not set")
    key, base, model = config
    context = "\n\n".join(
        f"[{chunk['id']}] {chunk['source_name']} ({chunk['source_url']})\n{chunk['text']}"
        for chunk in chunks
    )
    prompt = (
        "Answer the question using ONLY the source chunks below. "
        "After every sentence, cite the supporting chunk id in square brackets, "
        "for example [cdc-001]. Every sentence must include at least one citation "
        "to a chunk id from the list. Do not state anything the chunks do not support. "
        f"If the chunks do not support an answer, reply with exactly: {NOT_ENOUGH}\n\n"
        f"Chunks:\n{context}\n\nQuestion: {question}"
    )
    payload = json.dumps(
        {
            "model": model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                    "You write short condition research briefings. "
                    "You only use the provided chunks and cite their ids."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    try:
        text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("OpenAI response did not include a message") from exc
    return str(text).strip(), model


def _pack(
    question: str,
    text: str,
    citations: list[str],
    retrieved: list[tuple[float, dict]],
    *,
    condition_id: str,
    by_id: dict,
    mode: str,
    model: str | None,
    warning: str | None,
) -> dict:
    scores = {chunk["id"]: score for score, chunk in retrieved}
    sources = []
    for cid in citations:
        chunk = by_id.get(cid)
        if not chunk:
            continue
        sources.append(
            {
                "id": chunk["id"],
                "source_name": chunk["source_name"],
                "source_url": chunk["source_url"],
                "text": chunk["text"],
                "score": round(scores.get(cid, 0.0), 4),
            }
        )
    return {
        "question": question,
        "text": text,
        "citations": citations,
        "condition_id": condition_id,
        "mode": mode,
        "model": model,
        "warning": warning,
        "sources": sources,
    }


def answer_question(question: str, *, condition_id: str = "measles", force_mock: bool = False) -> dict:
    question = question.strip()
    state = get_state(condition_id)
    index: TfidfIndex = state["index"]
    # Retrieval stays inside this condition's chunks.
    retrieved = [
        (score, chunk)
        for score, chunk in index.search(question, k=TOP_K)
        if chunk.get("condition_id") == condition_id
    ]
    if not retrieved or retrieved[0][0] < SCORE_THRESHOLD:
        return _pack(
            question,
            NOT_ENOUGH,
            [],
            retrieved,
            condition_id=condition_id,
            by_id=state["by_id"],
            mode="mock" if force_mock or _openai_config() is None else "openai",
            model=None,
            warning=None,
        )

    allowed = {chunk["id"] for _score, chunk in retrieved}
    chunks = [chunk for _score, chunk in retrieved]
    warning = None
    model = None
    mode = "mock"
    raw = ""
    if not force_mock and _openai_config() is not None:
        try:
            raw, model = _call_openai(question, chunks)
            mode = "openai"
        except Exception as exc:
            warning = f"OpenAI request failed. Showing a mock extractive answer. ({exc})"
            raw = _mock_answer(question, retrieved, index)
            mode = "mock"
            model = None
    else:
        raw = _mock_answer(question, retrieved, index)

    text, citations = enforce_citations(raw, allowed)
    if not citations:
        text = NOT_ENOUGH
        citations = []
    return _pack(
        question,
        text,
        citations,
        retrieved,
        condition_id=condition_id,
        by_id=state["by_id"],
        mode=mode,
        model=model,
        warning=warning,
    )


def run_briefing(condition_id: str = "measles", *, force_mock: bool = False) -> list[dict]:
    state = get_state(condition_id)
    name = (state["document"].get("condition") or {}).get("name") or condition_id
    return [
        answer_question(question, condition_id=condition_id, force_mock=force_mock)
        for question in briefing_questions(name)
    ]
