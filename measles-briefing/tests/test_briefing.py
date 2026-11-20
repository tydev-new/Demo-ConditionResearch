"""Corpus-backed checks for the measles briefing flow. No network."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import engine


def test_standard_of_care_is_cited_from_cdc():
    result = engine.answer_question(engine.BRIEFING_QUESTIONS[0], force_mock=True)
    text = result["text"].lower()
    assert result["citations"], result
    assert "not enough evidence" not in text
    assert "vitamin a" in text or "supportive" in text
    ids = {chunk["id"] for chunk in engine.get_state()["chunks"]}
    by_id = engine.get_state()["by_id"]
    for cid in result["citations"]:
        assert cid in ids
        assert f"[{cid}]" in result["text"]
    assert any("cdc.gov" in by_id[cid]["source_url"] for cid in result["citations"])
    assert result["mode"] == "mock"


def test_off_topic_question_has_no_evidence():
    result = engine.answer_question("What is the price of bitcoin?", force_mock=True)
    assert result["text"] == engine.NOT_ENOUGH
    assert result["citations"] == []


def test_briefing_returns_three_answers():
    answers = engine.run_briefing(force_mock=True)
    assert len(answers) == 3
    assert [item["question"] for item in answers] == engine.BRIEFING_QUESTIONS
    for item in answers:
        assert item["text"]
        assert item["mode"] == "mock"
        assert isinstance(item["citations"], list)


def test_sentences_without_a_valid_citation_are_dropped():
    cleaned, cites = engine.enforce_citations(
        "Vitamin A may be given for severe measles [cdc-001]. Bitcoin is a good investment.",
        {"cdc-001"},
    )
    assert "bitcoin" not in cleaned.lower()
    assert cites == ["cdc-001"]
    assert "[cdc-001]" in cleaned
