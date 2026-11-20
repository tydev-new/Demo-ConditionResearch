"""Agent loop checks with a fake LLM client and mocked HTTP: deterministic, no key needed."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import agent
import engine
import sources


def _call(name, args, cid="c1"):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class FakeClient:
    """Plays back scripted assistant messages; the last one may cite ids it saw."""

    model = "fake-model"

    def __init__(self, script):
        self.script = list(script)
        self.seen_messages = []

    def chat(self, messages, tools, *, timeout, tool_choice="auto"):
        self.seen_messages.append(list(messages))
        step = self.script.pop(0)
        return step(messages) if callable(step) else step


@pytest.fixture(autouse=True)
def fresh_state():
    engine.drop_index("heart-failure")
    yield
    engine.drop_index("heart-failure")


def _fake_trials(url, timeout=8):
    assert "clinicaltrials.gov" in url
    assert "filter.advanced=AREA%5BPhase%5DPHASE3" in url
    assert "filter.overallStatus=RECRUITING" in url
    return {"studies": [{"protocolSection": {
        "identificationModule": {"nctId": "NCT09999999", "briefTitle": "Drug X in Heart Failure"},
        "statusModule": {"overallStatus": "RECRUITING"},
        "designModule": {"phases": ["PHASE3"]},
        "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Acme Pharma"}},
    }}]}


def test_tool_dispatch_trials_adds_chunks_to_pool(monkeypatch):
    monkeypatch.setattr(sources, "fetch_json", _fake_trials)
    chunks, summary = agent.dispatch("heart-failure", "search_trials",
                                     {"query": "", "phase": "PHASE3", "status": "RECRUITING", "bogus": 1})
    assert summary == "1 trials found"
    assert chunks[0]["id"] == "heart-failure-trial-NCT09999999"
    state = engine.get_state("heart-failure")
    assert "heart-failure-trial-NCT09999999" in state["by_id"]
    # Later searches over the pool see the new chunk.
    hits, _ = agent.dispatch("heart-failure", "search_sources", {"query": "Drug X Acme Pharma"})
    assert hits and hits[0]["id"] == "heart-failure-trial-NCT09999999"


def test_search_sources_and_unknown_tool():
    hits, summary = agent.dispatch("heart-failure", "search_sources", {"query": "heart failure treatment"})
    assert hits and summary.endswith("chunks")
    with pytest.raises(ValueError):
        agent.dispatch("heart-failure", "rm_rf", {})


def test_fetch_page_allowlist():
    assert agent.host_allowed("https://www.cdc.gov/flu/treatment/index.html")
    assert agent.host_allowed("https://pubmed.ncbi.nlm.nih.gov/123/")
    assert agent.host_allowed("https://medlineplus.gov/flu.html")
    assert not agent.host_allowed("https://evil.com/?u=cdc.gov")
    assert not agent.host_allowed("https://cdc.gov.evil.com/")
    assert not agent.host_allowed("file:///etc/passwd")
    with pytest.raises(PermissionError):
        agent.tool_fetch_page("heart-failure", "https://example.com/")


def test_agent_cites_tool_fetched_chunks_and_drops_invented_ids(monkeypatch):
    monkeypatch.setattr(sources, "fetch_json", _fake_trials)
    client = FakeClient([
        {"content": None, "tool_calls": [_call("search_trials", {"query": "", "phase": "PHASE3", "status": "RECRUITING"})]},
        {"content": "Drug X is in a recruiting phase 3 trial sponsored by Acme Pharma [heart-failure-trial-NCT09999999]. "
                    "It cures everyone [made-up-001]."},
    ])
    result = agent.answer("Which phase 3 heart failure trials are recruiting?", "heart-failure", client=client)
    assert result["mode"] == "agent"
    assert result["citations"] == ["heart-failure-trial-NCT09999999"]
    assert "cures everyone" not in result["text"]
    assert result["sources"][0]["source_url"] == "https://clinicaltrials.gov/study/NCT09999999"
    step = result["agent_steps"][0]
    assert step["tool"] == "search_trials" and step["result_summary"] == "1 trials found"
    assert step["args"]["phase"] == "PHASE3"


def test_agent_without_valid_citations_says_not_enough():
    client = FakeClient([
        {"content": None, "tool_calls": [_call("search_sources", {"query": "capital of France"})]},
        {"content": "Paris is the capital of France."},
    ])
    result = agent.answer("What is the capital of France?", "heart-failure", client=client)
    assert result["text"] == engine.NOT_ENOUGH
    assert result["citations"] == []


def test_agent_tool_limit_forces_final_answer():
    calls = [{"content": None, "tool_calls": [_call("search_sources", {"query": "heart failure treatment medicines"}, f"c{i}")]} for i in range(4)]

    def final(messages):
        ids = [line[1:line.index("]")] for m in messages if m["role"] == "tool"
               for line in m["content"].splitlines() if line.startswith("[")]
        return {"content": f"Heart failure is treated with medicines [{ids[0]}]."}

    client = FakeClient(calls + [final])
    result = agent.answer("How is heart failure treated?", "heart-failure", client=client)
    assert len(result["agent_steps"]) == agent.MAX_TOOL_CALLS
    assert result["citations"]


def test_fallback_to_fixed_pipeline_on_llm_failure(monkeypatch):
    class Broken:
        model = "broken"

        def chat(self, *a, **k):
            raise agent.AgentError("HTTP 500: upstream")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = agent.answer("What is the current standard of care for heart failure?", "heart-failure", client=Broken())
    assert result["agent_fallback"] is True
    assert result["mode"] == "mock"
    assert "fixed retrieval pipeline" in result["warning"]
    assert result["citations"]


def test_no_key_uses_fixed_pipeline(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = agent.answer("What is the current standard of care for measles?", "measles")
    assert result["mode"] == "mock" and result["agent_steps"] == []
