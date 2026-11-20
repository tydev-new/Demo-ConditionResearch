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


def test_care_questions_prefer_guideline_chunks_and_trial_questions_prefer_trials():
    from engine import rerank, question_intent

    def chunk(cid, source):
        return {"id": cid, "source_name": source, "text": ""}

    scored = [
        (0.40, chunk("t1", "ClinicalTrials.gov")),
        (0.38, chunk("t2", "ClinicalTrials.gov")),
        (0.36, chunk("t3", "ClinicalTrials.gov")),
        (0.30, chunk("p1", "PubMed")),
        (0.20, chunk("m1", "MedlinePlus")),
        (0.15, chunk("m2", "MedlinePlus")),
        (0.10, chunk("c1", "CDC Clinical Overview of Measles")),
    ]
    care = "What is the current standard of care for heart failure?"
    assert question_intent(care) == "care"
    top = [c["id"] for _s, c in rerank(care, scored, k=5)]
    assert {"m1", "m2", "c1"} <= set(top)

    trials = "What treatments or vaccines are in development / in clinical trials for heart failure?"
    assert question_intent(trials) == "trials"
    assert [c["id"] for _s, c in rerank(trials, scored, k=3)] == ["t1", "t2", "t3"]

    orgs = "Which key organizations (companies, institutions, government bodies) are involved in heart failure?"
    assert question_intent(orgs) == "orgs"
    assert rerank(orgs, scored, k=1)[0][1]["source_name"] == "ClinicalTrials.gov"


def test_heart_failure_care_answer_cites_medlineplus():
    from engine import answer_question

    answer = answer_question("What is the current standard of care for heart failure?",
                             condition_id="heart-failure", force_mock=True)
    assert answer["citations"]
    assert any("-mplus-" in cid for cid in answer["citations"])


def test_lay_term_maps_to_mesh_descriptor_and_medline_matches_synonym(monkeypatch):
    def fake(url, timeout=8):
        if "esearch" in url:
            return {"esearchresult": {"translationset": [
                {"from": "flu", "to": '"influenza, human"[MeSH Terms] OR flu[Text Word]'}]}}
        assert "match=exact" in url
        return [{"resource": "http://id.nlm.nih.gov/mesh/D007251", "label": "Influenza, Human"}]

    monkeypatch.setattr(sources, "fetch_json", fake)
    result = sources.normalize_condition("flu")
    assert result["status"] == "chosen"
    assert result["chosen"]["mesh_id"] == "D007251"
    assert "flu" in result["chosen"]["synonyms"]
    assert sources._plain_name("Influenza, Human") == "Influenza"
    assert sources._plain_name("Diabetes Mellitus, Type 2") == "Diabetes Mellitus, Type 2"
    # Related MedlinePlus pages are matched against synonyms too ("Bird Flu" vs "flu").
    assert sources.related_title_ok("Bird Flu", ["Influenza", "flu"])
    assert not sources.related_title_ok("Heart Health Tests", ["Heart Failure"])


def test_resolve_echoes_query_and_mesh_mapping(monkeypatch):
    monkeypatch.setattr(
        sources,
        "fetch_json",
        lambda url, timeout=8: [{"resource": "http://id.nlm.nih.gov/mesh/D006333", "label": "Heart Failure"}],
    )
    result = sources.normalize_condition("  heart   failure ")
    assert result["query"] == "heart failure"
    chosen = result["chosen"]
    assert (chosen["query"], chosen["mesh_name"], chosen["mesh_id"]) == ("heart failure", "Heart Failure", "D006333")
    assert isinstance(chosen["synonyms"], list)


def test_load_returns_mapping_for_display():
    from fastapi.testclient import TestClient
    import app as app_module

    client = TestClient(app_module.app)
    body = {"id": "heart-failure", "name": "Heart Failure", "mesh_id": "D006333", "query": "Heart failure"}
    mapping = client.post("/api/conditions/load", json=body).json()["mapping"]
    assert mapping["query"] == "Heart failure"
    assert mapping["mesh_id"] == "D006333"
    assert mapping["mesh_url"].endswith("ui=D006333")
    assert "CHF" in mapping["synonyms"]
