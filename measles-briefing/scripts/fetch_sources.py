#!/usr/bin/env python3
"""Fetch the three public measles sources and write data/corpus.json.

CDC note: the clinical overview page returns 200 to Python's default urllib
user-agent and 403 if the client spoofs a browser user-agent. This script
does not set User-Agent at all.

Run from anywhere:

    python measles-briefing/scripts/fetch_sources.py
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import chunk_text  # noqa: E402

DATA_PATH = ROOT / "data" / "corpus.json"

CDC_URL = "https://www.cdc.gov/measles/hcp/clinical-overview/index.html"
TRIALS_URL = "https://clinicaltrials.gov/api/v2/studies"
PUBMED_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

CDC_NAME = "CDC Clinical Overview of Measles"
TRIALS_NAME = "ClinicalTrials.gov"
PUBMED_NAME = "PubMed"

PUBMED_TERM = "measles[tiab] AND 2025:2026[dp]"
PUBMED_WANT = 20
SKIP_TAGS = {"script", "style", "noscript", "svg", "iframe"}
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
BLOCK_TAGS = {"p", "div", "li", "tr", "h1", "h2", "h3", "h4", "section", "br", "ul", "ol"}


def fetch_bytes(url: str, accept: str | None = None) -> bytes:
    """GET a URL without setting a User-Agent header."""
    headers = {}
    if accept:
        headers["Accept"] = accept
    request = urllib.request.Request(url, headers=headers)
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if 400 <= exc.code < 500:
                raise RuntimeError(f"HTTP {exc.code} for {url}") from exc
            last_error = exc
        except urllib.error.URLError as exc:
            last_error = exc
        time.sleep(0.8 + attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def _skip_class(class_attr: str) -> bool:
    for token in class_attr.split():
        if token in {"visually-hidden", "sr-only", "download-container"}:
            return True
        if token.startswith(("page-right-rail", "on-this-page", "cdc-icon", "page-share")):
            return True
    return False


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip: list[bool] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {key: value or "" for key, value in attrs}
        skip = tag in SKIP_TAGS or _skip_class(attr.get("class", ""))
        self._skip.append(skip)
        if any(self._skip):
            if tag in VOID_TAGS:
                self._skip.pop()
            return
        if tag in {"h1", "h2"}:
            self.parts.append("\n\n## ")
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")
        if tag in VOID_TAGS:
            self._skip.pop()

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if self._skip:
            self._skip.pop()
        if any(self._skip):
            return
        if tag in {"h1", "h2"}:
            self.parts.append("\n")
        elif tag in {"h3", "button"}:
            self.parts.append(". ")
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if any(self._skip):
            return
        if data:
            self.parts.append(data)


def html_to_sections(html: str) -> list[tuple[str, str]]:
    start = html.find('<div id="content"')
    end = html.find('class="site-footer')
    if start == -1:
        raise RuntimeError("CDC page did not contain the content region")
    fragment = html[start:end if end != -1 else None]
    parser = _TextExtractor()
    parser.feed(fragment)
    raw = "".join(parser.parts)
    sections: list[tuple[str, str]] = []
    pieces = raw.split("\n## ")
    preface = not raw.lstrip().startswith("##")
    for index, part in enumerate(pieces):
        lines = [line.strip() for line in part.splitlines()]
        lines = [line for line in lines if line and line.lower() not in {"download", "view", "print", "share"}]
        if not lines:
            continue
        if index == 0 and preface:
            heading = "Overview"
            body = " ".join(lines)
        else:
            heading = lines[0].strip(" .")
            body = " ".join(lines[1:])
        text = f"{heading}. {body}".strip() if body else heading
        text = " ".join(text.split())
        text = re.sub(r"\s+\.", ".", text)
        text = re.sub(r"\.{2,}", ".", text)
        if len(text) < 80:
            continue
        if heading.lower() in {"earn continuing education (ce)", "resources and tools"}:
            continue
        sections.append((heading, text))
    if not sections:
        raise RuntimeError("CDC page produced no text sections")
    return sections


def cdc_chunks() -> list[dict]:
    html = fetch_bytes(CDC_URL, accept="text/html").decode("utf-8", errors="replace")
    if "vitamin A" not in html and "Vitamin A" not in html:
        raise RuntimeError("CDC page did not include the vitamin A guidance; refusing to cache it")
    chunks: list[dict] = []
    for _heading, text in html_to_sections(html):
        for piece in chunk_text(text):
            if len(piece) < 40:
                continue
            chunks.append(
                {
                    "id": f"cdc-{len(chunks) + 1:03d}",
                    "source_name": CDC_NAME,
                    "source_url": CDC_URL,
                    "text": piece,
                }
            )
    joined = " ".join(chunk["text"] for chunk in chunks)
    if "supportive" not in joined.lower() or "vitamin a" not in joined.lower():
        raise RuntimeError("CDC chunks lost the standard-of-care passage")
    return chunks


def _phase_label(phases: list | None) -> str:
    if not phases:
        return "Not reported"
    labels = []
    for phase in phases:
        raw = str(phase)
        if raw == "NA":
            labels.append("Not applicable")
            continue
        label = raw.replace("EARLY_", "Early ").replace("PHASE", "Phase ")
        labels.append(" ".join(label.split()))
    return ", ".join(labels)


def _status_label(status: str | None) -> str:
    if not status:
        return "Not reported"
    return status.replace("_", " ").title()


def trial_chunks() -> tuple[list[dict], int]:
    studies: list[dict] = []
    page_token = None
    total = None
    while len(studies) < 500:
        params = {
            "query.cond": "measles",
            "pageSize": "100",
            "countTotal": "true",
        }
        if page_token:
            params["pageToken"] = page_token
        url = f"{TRIALS_URL}?{urllib.parse.urlencode(params)}"
        payload = json.loads(fetch_bytes(url, accept="application/json").decode("utf-8"))
        if total is None:
            total = int(payload.get("totalCount") or 0)
        batch = payload.get("studies") or []
        studies.extend(batch)
        page_token = payload.get("nextPageToken")
        if not page_token or not batch:
            break
        time.sleep(0.3)
    chunks = []
    seen: set[str] = set()
    for study in studies:
        protocol = study.get("protocolSection") or {}
        ident = protocol.get("identificationModule") or {}
        nct = ident.get("nctId")
        if not nct or nct in seen:
            continue
        seen.add(nct)
        title = (ident.get("briefTitle") or ident.get("officialTitle") or "Untitled study").strip()
        status = _status_label((protocol.get("statusModule") or {}).get("overallStatus"))
        phase = _phase_label((protocol.get("designModule") or {}).get("phases"))
        sponsor_mod = protocol.get("sponsorCollaboratorsModule") or {}
        lead = (sponsor_mod.get("leadSponsor") or {}).get("name") or "Not reported"
        collabs = []
        for collab in sponsor_mod.get("collaborators") or []:
            name = (collab or {}).get("name")
            if name:
                collabs.append(name)
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
        chunks.append(
            {
                "id": f"trial-{nct}",
                "source_name": TRIALS_NAME,
                "source_url": link,
                "text": " ".join(sentences),
            }
        )
    chunks.sort(key=lambda chunk: chunk["id"])
    return chunks, total or len(chunks)


def pubmed_chunks() -> tuple[list[dict], int]:
    search_params = {
        "db": "pubmed",
        "term": PUBMED_TERM,
        "retmax": "40",
        "retmode": "json",
        "sort": "pub_date",
        "tool": "measles-briefing",
        "email": "demo@example.com",
    }
    search_url = f"{PUBMED_SEARCH}?{urllib.parse.urlencode(search_params)}"
    payload = json.loads(fetch_bytes(search_url, accept="application/json").decode("utf-8"))
    result = payload.get("esearchresult") or {}
    if result.get("errorlist"):
        raise RuntimeError(f"PubMed esearch error: {result['errorlist']}")
    total = int(result.get("count") or 0)
    ids = result.get("idlist") or []
    if not ids:
        raise RuntimeError("PubMed esearch returned no ids")
    time.sleep(0.4)
    fetch_params = {
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "xml",
        "tool": "measles-briefing",
        "email": "demo@example.com",
    }
    fetch_url = f"{PUBMED_FETCH}?{urllib.parse.urlencode(fetch_params)}"
    xml_bytes = fetch_bytes(fetch_url, accept="application/xml")
    root = ET.fromstring(xml_bytes)
    by_id: dict[str, tuple[str, str]] = {}
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
        abstract = " ".join(abstract.split())
        title = " ".join(title.split())
        if pmid and title and abstract:
            by_id[pmid] = (title, abstract)
    chunks: list[dict] = []
    for pmid in ids:
        pair = by_id.get(pmid)
        if not pair:
            continue
        title, abstract = pair
        pieces = chunk_text(abstract, size=700, overlap=120)
        for index, piece in enumerate(pieces, start=1):
            chunk_id = f"pmid-{pmid}" if len(pieces) == 1 else f"pmid-{pmid}-{index}"
            body = piece.lstrip(". ")
            chunks.append(
                {
                    "id": chunk_id,
                    "source_name": PUBMED_NAME,
                    "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "text": f"{title.rstrip('.')}. {body}",
                }
            )
        if sum(1 for chunk in chunks if chunk["source_url"].endswith(f"/{pmid}/")) and len(
            {c["source_url"] for c in chunks}
        ) >= PUBMED_WANT:
            break
    # Cap to the first PUBMED_WANT papers, not the first N chunks.
    kept: list[dict] = []
    papers: list[str] = []
    for chunk in chunks:
        url = chunk["source_url"]
        if url not in papers:
            if len(papers) >= PUBMED_WANT:
                break
            papers.append(url)
        kept.append(chunk)
    if len(papers) < 15:
        raise RuntimeError(f"Only found {len(papers)} PubMed abstracts; expected about {PUBMED_WANT}")
    return kept, total


def main() -> None:
    print("Fetching CDC clinical overview...")
    cdc = cdc_chunks()
    print(f"  CDC chunks: {len(cdc)}")
    print("Fetching ClinicalTrials.gov...")
    trials, trial_total = trial_chunks()
    print(f"  trials reported: {trial_total}; chunks: {len(trials)}")
    print("Fetching PubMed abstracts...")
    papers, pubmed_total = pubmed_chunks()
    print(f"  pubmed reported: {pubmed_total}; chunks: {len(papers)}")
    corpus = cdc + trials + papers
    ids = [chunk["id"] for chunk in corpus]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate chunk ids")
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(corpus, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(corpus)} chunks to {DATA_PATH}")


if __name__ == "__main__":
    main()
