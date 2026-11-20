"""MeSH normalization and per-condition source fetch.

Prebuilt corpora live in data/corpora/. Anything gathered at request time is
cached in memory and under /tmp (Vercel allows writes only there).
"""

from __future__ import annotations

import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORPORA_DIR = ROOT / "data" / "corpora"
# Bump when fetch logic changes so warm instances do not serve stale on-demand corpora.
CACHE_VERSION = "v2"
TMP_DIR = Path(f"/tmp/condition-briefing-{CACHE_VERSION}")

MESH_LOOKUP = "https://id.nlm.nih.gov/mesh/lookup/descriptor"
TRIALS_URL = "https://clinicaltrials.gov/api/v2/studies"
PUBMED_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_MESH_TERM = re.compile(r'"([^"]+)"\[MeSH Terms\]', re.I)
PUBMED_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
MEDLINE_URL = "https://wsearch.nlm.nih.gov/ws/query"

ACTIVE_STATUSES = "RECRUITING,NOT_YET_RECRUITING,ENROLLING_BY_INVITATION,ACTIVE_NOT_RECRUITING"
REQUEST_TIMEOUT = 8
FETCH_BUDGET = 24

# Friendly ids for the committed quick picks. Other conditions use the MeSH id.
MESH_FILES = {
    "D008457": "measles",
    "D006333": "heart-failure",
    "D003924": "type-2-diabetes",
    "D001249": "asthma",
}

# Descriptor "contains" is a substring match. These queries are not substrings
# of the preferred MeSH label, so try the label the API actually stores.
QUERY_ALIASES = {
    "type 2 diabetes": ["diabetes mellitus, type 2"],
    "type ii diabetes": ["diabetes mellitus, type 2"],
    "t2dm": ["diabetes mellitus, type 2"],
    "t2d": ["diabetes mellitus, type 2"],
}

QUICK_PICKS = [
    {"label": "Measles", "query": "measles"},
    {"label": "Heart failure", "query": "heart failure"},
    {"label": "Type 2 diabetes", "query": "type 2 diabetes"},
    {"label": "Asthma", "query": "asthma"},
]

_MEMORY: dict[str, dict] = {}


def fetch_json(url: str, timeout: int = REQUEST_TIMEOUT):
    """GET JSON. Does not set a User-Agent (CDC rejects browser UAs)."""
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_bytes(url: str, timeout: int = REQUEST_TIMEOUT, accept: str | None = None) -> bytes:
    headers = {}
    if accept:
        headers["Accept"] = accept
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def condition_id_for(mesh_id: str) -> str:
    return MESH_FILES.get(mesh_id, mesh_id.lower())


def _mesh_hits(label: str, match: str = "contains") -> list[dict]:
    url = MESH_LOOKUP + "?" + urllib.parse.urlencode(
        {"label": label, "match": match, "limit": 5}
    )
    payload = fetch_json(url, timeout=REQUEST_TIMEOUT)
    if not isinstance(payload, list):
        return []
    matches = []
    seen = set()
    for hit in payload:
        resource = str(hit.get("resource") or "")
        mesh_id = resource.rstrip("/").split("/")[-1]
        name = str(hit.get("label") or "").strip()
        if not mesh_id.startswith("D") or not name or mesh_id in seen:
            continue
        seen.add(mesh_id)
        matches.append({"mesh_id": mesh_id, "name": name, "id": condition_id_for(mesh_id)})
    return matches


def _entrez_mesh_labels(text: str) -> list[str]:
    """MeSH descriptor labels NCBI maps a free-text term to, e.g. flu -> Influenza, Human."""
    url = PUBMED_SEARCH + "?" + urllib.parse.urlencode(
        {"db": "mesh", "term": text, "retmode": "json", "retmax": "0",
         "tool": "condition-briefing", "email": "demo@example.com"}
    )
    try:
        payload = fetch_json(url, timeout=REQUEST_TIMEOUT)
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    labels: list[str] = []
    for row in (payload.get("esearchresult") or {}).get("translationset") or []:
        for label in _MESH_TERM.findall(str(row.get("to") or "")):
            if label not in labels:
                labels.append(label)
    return labels


def normalize_condition(query: str) -> dict:
    """Resolve free text to MeSH and echo what it was mapped to, for display:
    query (what the user typed), mesh_name, mesh_id, synonyms, mapped_by."""
    text = " ".join(query.split())
    result = _normalize_condition(text)
    result["query"] = text
    chosen = result.get("chosen")
    if chosen:
        chosen["query"] = text
        chosen["mesh_name"] = chosen["name"]
        chosen.setdefault("synonyms", [])
    return result


def _plain_name(name: str) -> str:
    """Drop MeSH qualifiers: 'Influenza, Human' -> 'Influenza'; keep 'Diabetes Mellitus, Type 2'."""
    head, _, tail = name.partition(", ")
    if tail.lower() in {"human", "humans", "animal", "acute", "chronic"} or not tail:
        return head
    return name


def _normalize_condition(query: str) -> dict:
    """Resolve a condition name with the NLM MeSH descriptor lookup.

    One hit, or a case-insensitive exact label among several, is chosen.
    Otherwise the caller shows the matches and the user picks.
    """
    text = " ".join(query.split())
    if not text:
        return {"status": "none", "chosen": None, "matches": []}
    # Lay terms ("flu", "heart attack") are MeSH entry terms, not descriptor labels, so a
    # label "contains" search misses them (and "flu" hits a drug class instead). NCBI's
    # automatic term mapping translates them to the preferred descriptor first.
    for label in _entrez_mesh_labels(text):
        mapped = _mesh_hits(label, match="exact")
        if len(mapped) == 1:
            hit = mapped[0]
            chosen = {
                "id": condition_id_for(hit["mesh_id"]),
                "name": hit["name"],
                "synonyms": [text] if text.lower() != hit["name"].lower() else [],
                "mesh_id": hit["mesh_id"],
                "last_fetched": None,
            }
            return {"status": "chosen", "chosen": chosen, "matches": mapped}
    matches = _mesh_hits(text)
    if not matches:
        for alias in QUERY_ALIASES.get(text.lower(), []):
            matches = _mesh_hits(alias)
            if matches:
                break
    exact = [item for item in matches if item["name"].lower() == text.lower()]
    chosen_match = matches[0] if len(matches) == 1 else (exact[0] if len(exact) == 1 else None)
    chosen = None
    if chosen_match:
        chosen = {
            "id": condition_id_for(chosen_match["mesh_id"]),
            "name": chosen_match["name"],
            "synonyms": [],
            "mesh_id": chosen_match["mesh_id"],
            "last_fetched": None,
        }
        return {"status": "chosen", "chosen": chosen, "matches": matches}
    if not matches:
        return {"status": "none", "chosen": None, "matches": []}
    return {"status": "choices", "chosen": None, "matches": matches}


def _clean_html(value: str) -> str:
    text = html.unescape(value or "")
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _phase_label(phases) -> str:
    if not phases:
        return "Not reported"
    labels = []
    for phase in phases:
        raw = str(phase)
        if raw == "NA":
            labels.append("Not applicable")
            continue
        labels.append(" ".join(raw.replace("EARLY_", "Early ").replace("PHASE", "Phase ").split()))
    return ", ".join(labels)


def _study_chunk(condition_id: str, study: dict) -> dict | None:
    protocol = study.get("protocolSection") or {}
    ident = protocol.get("identificationModule") or {}
    nct = ident.get("nctId")
    if not nct:
        return None
    title = (ident.get("briefTitle") or ident.get("officialTitle") or "Untitled study").strip()
    status = ((protocol.get("statusModule") or {}).get("overallStatus") or "Not reported").replace("_", " ").title()
    phase = _phase_label((protocol.get("designModule") or {}).get("phases"))
    sponsor_mod = protocol.get("sponsorCollaboratorsModule") or {}
    lead = (sponsor_mod.get("leadSponsor") or {}).get("name") or "Not reported"
    collabs = [c.get("name") for c in (sponsor_mod.get("collaborators") or []) if c.get("name")]
    interventions = []
    for item in (protocol.get("armsInterventionsModule") or {}).get("interventions") or []:
        name = (item or {}).get("name")
        if name and name not in interventions:
            interventions.append(name)
    link = f"https://clinicaltrials.gov/study/{nct}"
    sentences = [
        f"Clinical trial: {title}.",
        f"NCT ID: {nct}.",
        f"Phase: {phase}.",
        f"Status: {status}.",
        f"Sponsor: {lead}.",
    ]
    if collabs:
        sentences.append("Collaborators: " + "; ".join(collabs[:8]) + ".")
    if interventions:
        sentences.append("Interventions: " + "; ".join(interventions[:8]) + ".")
    sentences.append(f"Study link: {link}.")
    return {
        "id": f"{condition_id}-trial-{nct}",
        "condition_id": condition_id,
        "source_name": "ClinicalTrials.gov",
        "source_url": link,
        "text": " ".join(sentences),
    }


def _fetch_trials(condition_id: str, name: str) -> list[dict]:
    def pull(statuses: str | None) -> list[dict]:
        params = {
            "query.cond": name,
            "pageSize": "100",
            "sort": "LastUpdatePostDate:desc",
        }
        if statuses:
            params["filter.overallStatus"] = statuses
        url = TRIALS_URL + "?" + urllib.parse.urlencode(params)
        payload = fetch_json(url, timeout=REQUEST_TIMEOUT)
        chunks = []
        seen = set()
        for study in payload.get("studies") or []:
            chunk = _study_chunk(condition_id, study)
            if not chunk or chunk["id"] in seen:
                continue
            seen.add(chunk["id"])
            chunks.append(chunk)
            if len(chunks) >= 100:
                break
        return chunks

    try:
        chunks = pull(ACTIVE_STATUSES)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError):
        chunks = []
    if len(chunks) < 15:
        chunks = pull(None)
    return chunks[:100]


def _fetch_pubmed(condition_id: str, name: str) -> list[dict]:
    from engine import chunk_text

    today = date.today()
    start = date(today.year - 2, today.month, today.day).isoformat().replace("-", "/")
    end = today.isoformat().replace("-", "/")
    term = f"{name}[tiab] AND {start}:{end}[dp]"
    search_url = PUBMED_SEARCH + "?" + urllib.parse.urlencode(
        {
            "db": "pubmed",
            "term": term,
            "retmax": "30",
            "retmode": "json",
            "sort": "pub_date",
            "tool": "condition-briefing",
            "email": "demo@example.com",
        }
    )
    payload = fetch_json(search_url, timeout=REQUEST_TIMEOUT)
    result = payload.get("esearchresult") or {}
    ids = result.get("idlist") or []
    if not ids:
        return []
    time.sleep(0.4)  # NCBI asks for at most 3 requests/second
    fetch_url = PUBMED_FETCH + "?" + urllib.parse.urlencode(
        {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "xml",
            "tool": "condition-briefing",
            "email": "demo@example.com",
        }
    )
    root = ET.fromstring(fetch_bytes(fetch_url, timeout=REQUEST_TIMEOUT, accept="application/xml"))
    by_id = {}
    for article in root.findall(".//PubmedArticle"):
        pmid = (article.findtext(".//PMID") or "").strip()
        title_node = article.find(".//ArticleTitle")
        title = "".join(title_node.itertext()).strip() if title_node is not None else ""
        paragraphs = []
        for node in article.findall(".//Abstract/AbstractText"):
            bit = "".join(node.itertext()).strip()
            if not bit:
                continue
            label = node.attrib.get("Label")
            paragraphs.append(f"{label}: {bit}" if label else bit)
        abstract = " ".join(paragraphs)
        if pmid and title and abstract:
            by_id[pmid] = (" ".join(title.split()), " ".join(abstract.split()))
    chunks = []
    papers = []
    for pmid in ids:
        pair = by_id.get(pmid)
        if not pair:
            continue
        if pmid not in papers:
            if len(papers) >= 20:
                break
            papers.append(pmid)
        title, abstract = pair
        pieces = chunk_text(abstract, size=700, overlap=120)
        for index, piece in enumerate(pieces, start=1):
            chunk_id = f"{condition_id}-pmid-{pmid}" if len(pieces) == 1 else f"{condition_id}-pmid-{pmid}-{index}"
            chunks.append(
                {
                    "id": chunk_id,
                    "condition_id": condition_id,
                    "source_name": "PubMed",
                    "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "text": f"{title.rstrip('.')}. {piece.lstrip('. ')}",
                }
            )
    return chunks


def _medline_documents(term: str) -> list:
    url = MEDLINE_URL + "?" + urllib.parse.urlencode({"db": "healthTopics", "term": term, "retmax": "5"})
    root = ET.fromstring(fetch_bytes(url, timeout=REQUEST_TIMEOUT, accept="application/xml"))
    return root.findall(".//document")


def _parse_medline(document) -> tuple[str, str, str, list[str]]:
    page = document.attrib.get("url") or ""
    title, summary, alt = "", "", []
    for node in document.findall("content"):
        kind = node.attrib.get("name")
        text = _clean_html("".join(node.itertext()))
        if not text:
            continue
        if kind == "title":
            title = text
        elif kind == "FullSummary":
            summary = text
        elif kind == "altTitle" and text not in alt:
            alt.append(text)
    return page, title, summary, alt


def related_title_ok(title: str, names: list[str]) -> bool:
    """A related MedlinePlus page counts if its title shares more than half of the words
    of the condition name or of any synonym (e.g. 'Asthma in Children' for asthma)."""
    from engine import tokenize

    words = set(tokenize(title))
    for name in names:
        name_words = set(tokenize(name))
        if name_words and len(name_words & words) * 2 > len(name_words):
            return True
    return False


def _fetch_medline(condition_id: str, name: str, synonyms: list[str] | None = None) -> tuple[list[dict], list[str]]:
    """MedlinePlus health topics. Searched with the user's own words and the plain MeSH
    name (qualifiers like ', Human' dropped). The top topic is always kept; up to two
    related topics are kept when their title matches the name or a synonym."""
    from engine import chunk_text

    names = [_plain_name(name)] + [s for s in (synonyms or []) if s]
    terms: list[str] = []
    for term in list(synonyms or []) + [_plain_name(name), name]:
        if term and term.lower() not in {t.lower() for t in terms}:
            terms.append(term)
    documents = []
    for term in terms[:3]:
        documents = _medline_documents(term)
        if documents:
            break
    chunks: list[dict] = []
    synonyms_out: list[str] = []
    used_pages = 0
    for rank, document in enumerate(documents):
        page, title, summary, alt = _parse_medline(document)
        title = title or name
        if rank > 0 and not related_title_ok(title, names + synonyms_out):
            continue
        if rank == 0:
            synonyms_out = alt
        if not summary:
            continue
        for piece in chunk_text(f"{title}. {summary}"):
            chunks.append(
                {
                    "id": f"{condition_id}-mplus-{len(chunks) + 1:03d}",
                    "condition_id": condition_id,
                    "source_name": "MedlinePlus",
                    "source_url": page or "https://medlineplus.gov/",
                    "text": piece,
                }
            )
        used_pages += 1
        if used_pages >= 3:
            break
    return chunks, synonyms_out[:8]


def gather_sources(condition: dict) -> dict:
    """Fetch trials, PubMed, and MedlinePlus. A failed source does not cancel the others."""
    condition_id = condition["id"]
    name = condition["name"]
    jobs = {
        "ClinicalTrials.gov": lambda: _fetch_trials(condition_id, _plain_name(name)),
        "PubMed": lambda: _fetch_pubmed(condition_id, _plain_name(name)),
        "MedlinePlus": lambda: _fetch_medline(condition_id, name, list(condition.get("synonyms") or [])),
    }
    results: dict[str, tuple[list[dict], list[str]]] = {}
    errors: dict[str, str] = {}
    pool = ThreadPoolExecutor(max_workers=3)
    futures = {pool.submit(fn): source for source, fn in jobs.items()}
    try:
        try:
            for future in as_completed(futures, timeout=FETCH_BUDGET):
                source = futures[future]
                try:
                    value = future.result()
                    if source == "MedlinePlus":
                        results[source] = value
                    else:
                        results[source] = (value, [])
                except Exception as exc:
                    errors[source] = str(exc)[:240] or f"{source} failed"
                    results[source] = ([], [])
        except FuturesTimeout:
            pass
        for future, source in futures.items():
            if source not in results:
                errors[source] = "Timed out"
                results[source] = ([], [])
                future.cancel()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    chunks: list[dict] = []
    synonyms = list(condition.get("synonyms") or [])
    status = []
    for source in ("MedlinePlus", "ClinicalTrials.gov", "PubMed"):
        source_chunks, extra = results.get(source, ([], []))
        for item in extra:
            if item not in synonyms and item.lower() != name.lower():
                synonyms.append(item)
        chunks.extend(source_chunks)
        error = errors.get(source)
        status.append(
            {
                "source_name": source,
                "ok": error is None and bool(source_chunks),
                "chunks": len(source_chunks),
                "error": None if source_chunks else (error or "No records returned"),
            }
        )
    record = {
        "id": condition_id,
        "name": name,
        "synonyms": synonyms[:12],
        "mesh_id": condition.get("mesh_id"),
        "last_fetched": date.today().isoformat(),
    }
    return {"condition": record, "chunks": chunks, "sources": status}


def corpus_path(condition_id: str) -> Path:
    return CORPORA_DIR / f"{condition_id}.json"


def tmp_path(condition_id: str) -> Path:
    return TMP_DIR / f"{condition_id}.json"


def read_document(condition_id: str) -> dict | None:
    if condition_id in _MEMORY:
        return _MEMORY[condition_id]
    for path in (corpus_path(condition_id), tmp_path(condition_id)):
        if path.is_file():
            document = json.loads(path.read_text(encoding="utf-8"))
            _MEMORY[condition_id] = document
            return document
    return None


def remember(document: dict, *, persist_tmp: bool) -> None:
    condition_id = document["condition"]["id"]
    _MEMORY[condition_id] = document
    if persist_tmp:
        try:
            TMP_DIR.mkdir(parents=True, exist_ok=True)
            tmp_path(condition_id).write_text(
                json.dumps(document, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass


def load_condition(chosen: dict) -> dict:
    """Return a corpus for a resolved condition. Prebuilt files skip the network."""
    existing = read_document(chosen["id"])
    if existing:
        origin = "prebuilt" if corpus_path(chosen["id"]).is_file() else "cache"
        return {"document": existing, "origin": origin}
    document = gather_sources(chosen)
    remember(document, persist_tmp=True)
    return {"document": document, "origin": "fetched"}


def public_summary(document: dict, origin: str) -> dict:
    condition = document["condition"]
    return {
        "origin": origin,
        "condition": condition,
        "chunk_count": len(document.get("chunks") or []),
        "sources": document.get("sources") or [],
    }
