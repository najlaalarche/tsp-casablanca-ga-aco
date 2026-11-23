"""SMA ACO sur le SDK Antigravity : faux runner (AUCUN appel réel à Gemini, aucun réseau)."""
import asyncio
import json

import numpy as np
import pytest

from agents_aco import antigravity_backend as ag
from agents_aco import manager_aco
from agents_aco.agents_aco import ARB_SYSTEM, SCI_SYSTEM
from agents_aco.manager_aco import FALLBACK_LABEL, failed_agents, rerun_failed, run_agents_aco

GOOD = {"n_ants": 40, "rho": 0.5, "alpha": 1.0, "beta": 3.0, "cycles": 500,
        "justification": "m = n est l'usage classique ; ρ = 0,5 équilibre mémoire et oubli des pistes."}
WEB_OK = GOOD | {"beta": 2.0, "justification": GOOD["justification"] + " Valeurs usuelles [S1].",
                 "cited_sources": ["S1"]}
ARB_OK = {"final": GOOD | {"beta": 2.5}, "basis": ["[S1] vérifiée", "analyse de β"], "convergences": [],
          "divergences": [], "rejected": [], "consensus": "accord"}
HITS = [{"title": f"ACO paper {i}", "href": f"https://example.org/{i}", "body": "rho beta"} for i in range(3)]


class FakeRunner:
    """Imite AntigravityRunner.run : l'agent web appelle lui-même l'outil recherche_web."""

    model = "gemini-3.5-flash-lite"

    def __init__(self, script):
        self.script = script          # {"web"|"sci"|"arb": [réponse, ...] | "timeout" | "error"}
        self.calls = []

    def run(self, system, prompts, schema, tools=None):
        agent = "arb" if system == ARB_SYSTEM else "sci" if system == SCI_SYSTEM else "web"
        self.calls.append(agent)
        plan = self.script[agent]
        if plan == "timeout":
            raise asyncio.TimeoutError()
        if plan == "error":
            raise RuntimeError("503 UNAVAILABLE")
        turns, prompt, answers = [], prompts(None), list(plan)
        while prompt is not None and answers:
            tool_calls = []
            if agent == "web" and tools and not self.script.get("web_no_tool"):
                tools[0]("ant colony optimization parameters")
                tool_calls.append({"name": "recherche_web", "args": {"requete": "ant colony optimization parameters"}})
            a = answers.pop(0)
            turns.append(ag.Turn(a if isinstance(a, dict) else None, a if isinstance(a, str) else json.dumps(a),
                                 tool_calls))
            prompt = prompts(turns)
        return turns


def _matrix(n=40):
    rng = np.random.default_rng(0)
    xy = rng.random((n, 2)) * 5000
    d = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    np.fill_diagonal(d, 0)
    return d


@pytest.fixture(autouse=True)
def tmp_results(monkeypatch, tmp_path):
    monkeypatch.setattr(manager_aco, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(manager_aco, "LAST_MODEL_FILE", tmp_path / "last.txt")
    return tmp_path


def _run(script, **kw):
    r = FakeRunner(script)
    return run_agents_aco(_matrix(), 40, runner=r, search=lambda q, k: HITS, **kw), r


def test_three_antigravity_agents_ok():
    s, r = _run({"web": [WEB_OK], "sci": [GOOD], "arb": [ARB_OK]})
    assert r.calls == ["web", "sci", "arb"]
    for res in (s.web, s.scientific, s.arbiter):
        assert res.model == "Antigravity SDK · gemini-3.5-flash-lite" and res.extra["sdk"] == "Antigravity SDK"
    assert s.web.extra["tool_calls"] == [{"requete": "ant colony optimization parameters", "resultats": 3}]
    assert [x.id for x in s.web.sources if x.cited] == ["S1"] and s.web.web_search_used
    assert s.arbiter.config.beta == 2.5 and failed_agents(s) == []
    assert s.probe == ["moteur : Antigravity SDK · modèle gemini-3.5-flash-lite"]


def test_web_cannot_cite_source_not_returned_by_tool():
    bad = GOOD | {"justification": GOOD["justification"] + " Voir [S9].", "cited_sources": ["S9"]}
    s, _ = _run({"web": [bad, bad], "sci": [GOOD], "arb": [ARB_OK]})
    assert s.web.config is None and "S9" in s.web.error
    assert s.web.extra["tool_calls"]          # appels d'outil conservés même en cas d'échec


def test_web_without_tool_call_is_rejected():
    s, _ = _run({"web": [WEB_OK, WEB_OK], "sci": [GOOD], "arb": [ARB_OK], "web_no_tool": True})
    assert s.web.config is None and "recherche_web" in s.web.error


def test_invalid_json_gets_one_corrective_retry():
    s, _ = _run({"web": [WEB_OK], "sci": ["pas du json", GOOD], "arb": [ARB_OK]})
    assert s.scientific.config is not None


def test_timeout_60s_then_python_fallback_decision():
    s, _ = _run({"web": "error", "sci": [GOOD], "arb": "timeout"})
    assert s.web.config is None and "503" in s.web.error
    arb = s.arbiter
    assert arb.extra["fallback"] and arb.model == "Python (secours)"
    assert arb.config.justification.startswith(FALLBACK_LABEL)
    assert "60 s" in arb.extra["gemini_error"]
    assert failed_agents(s) == ["web", "arbiter"]


def test_rerun_only_failed_agents_antigravity():
    s, _ = _run({"web": "error", "sci": [GOOD], "arb": "error"})
    r2 = FakeRunner({"web": [WEB_OK], "sci": [GOOD], "arb": [ARB_OK]})
    s2 = rerun_failed(s, _matrix(), 40, runner=r2, search=lambda q, k: HITS)
    assert r2.calls == ["web", "arb"] and s2.scientific is s.scientific
    assert not s2.arbiter.extra.get("fallback") and failed_agents(s2) == []


def test_search_tool_limits_and_dedup():
    tool, hits, calls = ag.make_search_tool(lambda q, k: HITS)
    first = json.loads(tool("aco"))
    assert [h["id"] for h in first] == ["S1", "S2", "S3"]
    assert json.loads(tool("aco again")) == []            # mêmes URL : pas de doublon
    for _ in range(ag.MAX_SEARCHES):
        tool("x")
    assert "limite" in tool("encore") and len(hits) == 3


def test_backend_choice(monkeypatch):
    be = manager_aco.make_backend(client=object())
    assert be.kind == "genai"                              # client explicite → google-genai (tests existants)
    be = manager_aco.make_backend(runner=FakeRunner({}))
    assert be.kind == "antigravity"
    monkeypatch.setattr(ag, "available", lambda: False)
    monkeypatch.setattr(manager_aco, "make_client", lambda models=None: "client-genai")
    assert manager_aco.make_backend().kind == "genai"      # SDK absent → repli google-genai


def test_api_key_never_in_runner_repr_or_logs(caplog):
    r = ag.AntigravityRunner("gemini-x", "SECRET-KEY-123")
    assert "SECRET-KEY-123" not in repr(r.__dict__.get("model")) and "SECRET-KEY-123" not in caplog.text
