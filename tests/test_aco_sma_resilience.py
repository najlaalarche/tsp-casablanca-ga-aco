"""SMA ACO face à un Gemini saturé : faux appel qui simule le 503 (aucun réseau, horloge simulée).

- attentes progressives 2 s / 5 s / 10 s puis bascule de modèle, budget de 60 s par agent ;
- décision de secours calculée par Python quand l'arbitre échoue ;
- relance des seuls agents en échec.
"""
import json

import numpy as np
import pytest

from agents.gemini_client import GeminiCallError, GeminiConfigError, GeminiReply
from agents_aco import manager_aco
from agents_aco.agents_aco import ARB_SYSTEM, SCI_SYSTEM, WEB_SYSTEM
from agents_aco.manager_aco import FALLBACK_LABEL, failed_agents, fallback_decision, rerun_failed, run_agents_aco
from agents_aco.resilient import ResilientGemini
from agents.schemas import AgentResult
from agents_aco.schemas_aco import make_config_validator


class ApiError(Exception):
    def __init__(self, code, msg="This model is currently experiencing high demand."):
        super().__init__(f"{code} {msg}")
        self.code = code


class Clock:
    def __init__(self):
        self.t, self.sleeps = 0.0, []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def client(call, models=("m1", "m2"), clock=None):
    clock = clock or Clock()
    return ResilientGemini(call, list(models), sleep=clock.sleep, clock=clock), clock


def ok(text="{}", model="m1"):
    return GeminiReply(text, model, [], False, [])


# ------------------------------------------------------------ client résilient
def test_503_retries_with_progressive_waits_then_succeeds():
    seq = [ApiError(503), ApiError(503), "ok"]

    def call(model, *a):
        r = seq.pop(0)
        if isinstance(r, Exception):
            raise r
        return ok(r, model)

    c, clock = client(call)
    c.start_agent("A")
    assert c.generate("p", "s").text == "ok"
    assert clock.sleeps == [2, 5] and c.model == "m1"


def test_503_three_waits_then_switches_model():
    calls = []

    def call(model, *a):
        calls.append(model)
        if model == "m1":
            raise ApiError(503)
        return ok("ok", model)

    c, clock = client(call)
    c.start_agent("A")
    assert c.generate("p", "s").model == "m2"
    assert calls == ["m1", "m1", "m1", "m1", "m2"] and clock.sleeps == [2, 5, 10]
    assert c.model == "m2" and any("bascule sur m2" in j for j in c.journal)


def test_non_retriable_error_switches_without_waiting():
    def call(model, *a):
        if model == "m1":
            raise ApiError(404, "not found")
        return ok("ok", model)

    c, clock = client(call)
    c.start_agent("A")
    assert c.generate("p", "s").model == "m2" and clock.sleeps == []


def test_budget_of_60_s_per_agent_is_never_exceeded():
    clock = Clock()

    def call(model, prompt, system, schema, search, temp, timeout_s):
        assert timeout_s <= 60
        clock.t += min(8, timeout_s)   # 8 s avant le 503, ou coupé par le délai HTTP (budget restant)
        raise ApiError(503)

    c, _ = client(call, models=("m1", "m2", "m3"), clock=clock)
    c.start_agent("A")
    with pytest.raises(GeminiCallError, match="60 s"):
        c.generate("p", "s")
    assert clock.t <= 60


def test_budget_resets_for_each_agent():
    clock = Clock()
    state = {"fail": True}

    def call(model, *a):
        clock.t += 20
        if state["fail"]:
            raise ApiError(503)
        return ok("ok", model)

    c, _ = client(call, models=("m1",), clock=clock)
    c.start_agent("A")
    with pytest.raises(GeminiCallError):
        c.generate("p", "s")
    state["fail"] = False
    c.start_agent("B")              # nouveau budget de 60 s
    assert c.generate("p", "s").text == "ok"


def test_invalid_key_is_reported_immediately():
    def call(model, *a):
        raise ApiError(400, "API key not valid")

    c, clock = client(call)
    c.start_agent("A")
    with pytest.raises(GeminiConfigError):
        c.generate("p", "s")
    assert clock.sleeps == []


# ------------------------------------------------------------ SMA complet
GOOD = {"n_ants": 40, "rho": 0.5, "alpha": 1.0, "beta": 3.0, "cycles": 500,
        "justification": "m = n est l'usage classique ; ρ = 0,5 équilibre mémoire et oubli des pistes."}
HITS = [{"title": f"ACO paper {i}", "href": f"https://example.org/{i}", "body": "rho beta"} for i in range(3)]
WEB_OK = GOOD | {"beta": 2.0, "cycles": 1000, "n_ants": 30,
                 "justification": GOOD["justification"] + " Valeurs usuelles [S1].", "cited_sources": ["S1"]}
ARB_OK = {"final": GOOD | {"beta": 2.5}, "basis": ["[S1] vérifiée", "analyse de β"], "convergences": [],
          "divergences": [], "rejected": [], "consensus": "accord"}


def _matrix(n=40):
    rng = np.random.default_rng(0)
    xy = rng.random((n, 2)) * 5000
    d = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    np.fill_diagonal(d, 0)
    return d


class FakeGemini:
    """Faux transport : chaque agent (reconnu par son prompt système) répond ou renvoie un 503."""

    def __init__(self, behaviour):
        self.behaviour = behaviour      # {"web": dict | 503, "sci": ..., "arb": ...}
        self.calls = []

    def __call__(self, model, prompt, system, json_schema, use_search, temperature, timeout_s):
        agent = ("arb" if system == ARB_SYSTEM else "sci" if system == SCI_SYSTEM
                 else "web" if system == WEB_SYSTEM or "recherche" in system else "?")
        self.calls.append(agent)
        if "requêtes de moteur de recherche" in prompt:      # étape « requêtes » de l'agent web
            if self.behaviour["web"] == 503:
                raise ApiError(503)
            return ok(json.dumps({"queries": ["a b c d", "e f g h", "i j k l"]}), model)
        b = self.behaviour[agent]
        if b == 503:
            raise ApiError(503)
        return ok(json.dumps(b), model)


@pytest.fixture
def tmp_results(monkeypatch, tmp_path):
    monkeypatch.setattr(manager_aco, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(manager_aco, "LAST_MODEL_FILE", tmp_path / "last.txt")
    return tmp_path


def _run(behaviour, **kw):
    fake = FakeGemini(behaviour)
    c, clock = client(fake, models=("gemini-3.5-flash-lite", "gemini-3.1-flash-lite"))
    s = run_agents_aco(_matrix(), 40, client=c, search=lambda q, k: HITS, pause=0, **kw)
    return s, fake, clock


def test_saturated_gemini_web_and_arbiter_fail_fallback_on_semantic(tmp_results):
    s, fake, clock = _run({"web": 503, "sci": GOOD, "arb": 503})
    assert s.web.config is None and ("aucun modèle" in s.web.error or "budget" in s.web.error)
    assert s.scientific.config is not None
    arb = s.arbiter
    assert arb.config is not None and arb.extra["fallback"] is True
    assert arb.model == "Python (secours)" and arb.raw == ""                     # rien attribué à Gemini
    assert arb.config.justification.startswith(FALLBACK_LABEL)
    assert "proposition valide unique (agent sémantique)" in arb.extra["rule"]
    assert arb.config.params() == make_config_validator(40)(GOOD).params()
    assert arb.extra["gemini_error"]
    assert set(clock.sleeps) <= {2, 5, 10} and 2 in clock.sleeps                 # attentes progressives
    saved = json.loads(open(s.saved_to, encoding="utf-8").read())
    assert saved["arbiter"]["extra"]["fallback"] is True
    assert failed_agents(s) == ["web", "arbiter"]


def test_fallback_rule_by_rule_between_two_valid_proposals():
    v = make_config_validator(40)
    web = AgentResult("w", "m", v(GOOD | {"n_ants": 31, "rho": 0.4, "beta": 2.0, "cycles": 1000}), "")
    sci = AgentResult("s", "m", v(GOOD | {"n_ants": 40, "rho": 0.6, "beta": 5.0, "cycles": 500}), "")
    fb = fallback_decision(40, web, sci, "503")
    p = fb.config.params()
    assert p == {"n_ants": 36, "rho": 0.5, "alpha": 1.0, "beta": 3.5, "cycles": 1000}
    assert {d["parameter"] for d in fb.extra["divergences"]} == {"n_ants", "rho", "beta", "cycles"}
    assert "règle par règle" in fb.extra["rule"]


def test_no_valid_proposal_means_no_decision(tmp_results):
    s, _, _ = _run({"web": 503, "sci": 503, "arb": ARB_OK})
    assert s.arbiter is None and s.web.config is None and s.scientific.config is None
    assert failed_agents(s) == ["web", "scientific", "arbiter"]


def test_arbiter_ok_no_fallback(tmp_results):
    s, _, _ = _run({"web": WEB_OK, "sci": GOOD, "arb": ARB_OK})
    assert s.arbiter.config.beta == 2.5 and not s.arbiter.extra.get("fallback")
    assert failed_agents(s) == []


def test_rerun_only_failed_agents(tmp_results):
    s, _, _ = _run({"web": 503, "sci": GOOD, "arb": 503})
    sci_before = s.scientific
    fake = FakeGemini({"web": WEB_OK, "sci": GOOD, "arb": ARB_OK})   # Gemini de nouveau disponible
    c, _ = client(fake, models=("gemini-3.5-flash-lite",))
    seen = []
    s2 = rerun_failed(s, _matrix(), 40, client=c, search=lambda q, k: HITS, pause=0,
                      on_agent=lambda i, r: seen.append(i))
    assert "sci" not in fake.calls                       # l'agent sémantique valide n'est PAS réinterrogé
    assert set(fake.calls) == {"web", "arb"} and seen == [0, 2]
    assert s2.scientific is sci_before
    assert s2.web.config is not None and not s2.arbiter.extra.get("fallback")
    assert s2.arbiter.config.beta == 2.5 and failed_agents(s2) == []
    assert s2.saved_to != s.saved_to and any("relance" in x for x in s2.probe)


def test_rerun_with_gemini_still_down_keeps_fallback(tmp_results):
    s, _, _ = _run({"web": 503, "sci": GOOD, "arb": 503})
    fake = FakeGemini({"web": 503, "sci": GOOD, "arb": 503})
    c, _ = client(fake, models=("gemini-3.5-flash-lite",))
    s2 = rerun_failed(s, _matrix(), 40, client=c, search=lambda q, k: HITS, pause=0)
    assert "sci" not in fake.calls
    assert s2.arbiter.extra["fallback"] and failed_agents(s2) == ["web", "arbiter"]
