"""Briefing checks. MeSH HTTP is mocked; answers use the committed corpora."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import engine
import sources


def test_mesh_exact_label_wins(monkeypatch):
    def fake(url, timeout=8):
        assert "label=heart+failure" in url or "heart failure" in url
        return [
            {"resource": "http://id.nlm.nih.gov/mesh/D006333", "label": "Heart Failure"},
            {"resource": "http://id.nlm.nih.gov/mesh/D054144", "label": "Heart Failure, Diastolic"},
        ]

    monkeypatch.setattr(sources, "fetch_json", fake)
    result = sources.normalize_condition("heart failure")
    assert result["status"] == "chosen"
    assert result["chosen"]["mesh_id"] == "D006333"
    assert result["chosen"]["id"] == "heart-failure"


def test_mesh_multiple_matches_wait_for_a_choice(monkeypatch):
    monkeypatch.setattr(
        sources,
        "fetch_json",
        lambda url, timeout=8: [
            {"resource": "http://id.nlm.nih.gov/mesh/D054144", "label": "Heart Failure, Diastolic"},
            {"resource": "http://id.nlm.nih.gov/mesh/D054143", "label": "Heart Failure, Systolic"},
        ],
    )
    result = sources.normalize_condition("heart failure")
    assert result["status"] == "choices"
    assert result["chosen"] is None
    assert {item["mesh_id"] for item in result["matches"]} == {"D054144", "D054143"}


def test_standard_of_care_is_cited_from_cdc():
    result = engine.answer_question(
        "What is the current standard of care for measles?",
        condition_id="measles",
        force_mock=True,
    )
    text = result["text"].lower()
    assert result["citations"], result
    assert "not enough evidence" not in text
    assert "vitamin a" in text or "supportive" in text
    state = engine.get_state("measles")
    for cid in result["citations"]:
        assert cid in state["by_id"]
        assert f"[{cid}]" in result["text"]
        assert state["by_id"][cid]["condition_id"] == "measles"
    assert any("cdc.gov" in state["by_id"][cid]["source_url"] for cid in result["citations"])
    assert result["mode"] == "mock"


def test_heart_failure_standard_of_care_is_cited():
    result = engine.answer_question(
        "What is the current standard of care for heart failure?",
        condition_id="heart-failure",
        force_mock=True,
    )
    assert result["citations"], result["text"]
    assert "not enough evidence" not in result["text"].lower()
    state = engine.get_state("heart-failure")
    for cid in result["citations"]:
        assert state["by_id"][cid]["condition_id"] == "heart-failure"
        assert f"[{cid}]" in result["text"]


def test_off_topic_question_has_no_evidence():
    result = engine.answer_question(
        "What is the price of bitcoin?",
        condition_id="measles",
        force_mock=True,
    )
    assert result["text"] == engine.NOT_ENOUGH
    assert result["citations"] == []


def test_condition_filter_keeps_corpora_apart():
    measles = engine.answer_question(
        "What is the current standard of care for measles?",
        condition_id="measles",
        force_mock=True,
    )
    heart = engine.answer_question(
        "What is the current standard of care for heart failure?",
        condition_id="heart-failure",
        force_mock=True,
    )
    measles_ids = set(engine.get_state("measles")["by_id"])
    heart_ids = set(engine.get_state("heart-failure")["by_id"])
    assert measles["citations"]
    assert heart["citations"]
    assert set(measles["citations"]).issubset(measles_ids)
    assert set(heart["citations"]).isdisjoint(measles_ids)
    assert set(heart["citations"]).issubset(heart_ids)


def test_briefing_returns_three_answers():
    answers = engine.run_briefing("measles", force_mock=True)
    assert len(answers) == 3
    assert [item["question"] for item in answers] == engine.briefing_questions("Measles")
    for item in answers:
        assert item["text"]
        assert item["condition_id"] == "measles"
        assert isinstance(item["citations"], list)


def test_sentences_without_a_valid_citation_are_dropped():
    cleaned, cites = engine.enforce_citations(
        "Vitamin A may be given for severe measles [cdc-001]. Bitcoin is a good investment.",
        {"cdc-001"},
    )
    assert "bitcoin" not in cleaned.lower()
    assert cites == ["cdc-001"]
    assert "[cdc-001]" in cleaned
