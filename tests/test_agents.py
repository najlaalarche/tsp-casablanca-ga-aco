"""Agents IA : validation JSON, repli sans recherche web, arbitre, repli de modèles.

Faux client Gemini et faux DuckDuckGo (aucun appel réseau), sauf test_live_gemini (TSP40_LIVE_GEMINI=1).
"""
import json
import os

import pytest

from agents import agent_manager, arbiter_agent, semantic_agent, web_agent
from agents import gemini_client as gc
from agents.gemini_client import GeminiClient, GeminiConfigError, GeminiReply, SearchUnavailable
from agents.schemas import SchemaError, compare_configs, extract_json, validate_arbiter, validate_config

GOOD = {"population_size": 100, "selection_method": "tournament", "mutation_rate": 0.05,
        "crossover_rate": 0.8, "generations": 500, "elitism": True,
        "justification": "Population suffisante pour la diversité, tournoi pour la pression de sélection."}


@pytest.fixture(autouse=True)
def tmp_results(monkeypatch, tmp_path):
    monkeypatch.setattr(agent_manager, "RESULTS_DIR", tmp_path)


# ------------------------------------------------------------------ schémas
def test_validate_good_config():
    c = validate_config(GOOD)
    assert c.params()["population_size"] == 100 and c.selection_method == "tournament"


@pytest.mark.parametrize("key,bad", [
    ("population_size", 75), ("population_size", True), ("selection_method", "rank"),
    ("mutation_rate", 0.9), ("mutation_rate", "0.05"), ("crossover_rate", 0.7),
    ("generations", 2000), ("elitism", "yes"), ("justification", "ok"),
])
def test_validate_rejects_out_of_domain(key, bad):
    with pytest.raises(SchemaError):
        validate_config(GOOD | {key: bad})


def test_validate_missing_field():
    d = dict(GOOD)
    del d["generations"]
    with pytest.raises(SchemaError, match="manquants"):
        validate_config(d)


def test_extract_json_variants():
    assert extract_json('```json\n{"a": 1}\n```')["a"] == 1
    assert extract_json('Voici : {"a": {"b": 2}} fin')["a"]["b"] == 2
    with pytest.raises(SchemaError):
        extract_json("pas de json")
    with pytest.raises(SchemaError):
        extract_json("{cassé")


def test_validate_arbiter_and_python_diff():
    final, extra = validate_arbiter({"final": GOOD, "convergences": ["tournoi"], "rejected": [],
                                     "divergences": [{"parameter": "generations", "decision": "500",
                                                      "reason": "budget"}], "consensus": "ok"})
    assert final.generations == 500 and extra["divergences"][0]["parameter"] == "generations"
    with pytest.raises(SchemaError):
        validate_arbiter({"final": GOOD, "divergences": [{"parameter": "x"}]})
    diff = compare_configs(validate_config(GOOD), validate_config(GOOD | {"generations": 1000}))
    assert [d["parameter"] for d in diff if not d["agree"]] == ["generations"]


# ------------------------------------------------------------- faux client
class FakeClient:
    """Imite GeminiClient.generate : file de réponses prédéfinies, mémorise les appels."""

    def __init__(self, replies, search_error=False):
        self.replies = list(replies)
        self.calls = []
        self.model = "fake-model"
        self.probe_log = ["fake-model : OK"]
        self.search_error = search_error

    def select_model(self):
        return self.model

    def generate(self, prompt, system, *, json_schema=None, use_search=False, temperature=0.4):
        self.calls.append({"use_search": use_search, "schema": json_schema is not None, "prompt": prompt})
        if use_search and self.search_error:
            raise SearchUnavailable("Google Search grounding refusé (429)")
        r = self.replies.pop(0)
        return r if isinstance(r, GeminiReply) else GeminiReply(r, self.model, [], False, [])


QUERIES = json.dumps({"queries": ["ga tsp population size", "tournament vs roulette tsp", "ox crossover inversion mutation"]})
CITED = GOOD | {"justification": "Population 100 et tournoi recommandés pour le TSP [S1] ; mutation par inversion efficace [S2].",
                "cited_sources": ["S1", "S2"]}


def fake_search(fail=False):
    """Faux DuckDuckGo : 2 résultats par requête, dont un doublon d'URL commun à toutes."""
    calls = []

    def search(q, n):
        calls.append(q)
        if fail:
            raise ConnectionError("réseau indisponible")
        return [{"title": f"Article {q}", "href": f"https://ex.org/{len(calls)}", "body": f"extrait {q}"},
                {"title": "Doublon commun", "href": "https://ex.org/common", "body": "commun"}]
    search.calls = calls
    return search


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch):
    """Aucun test unitaire n'interroge DuckDuckGo : la recherche par défaut est remplacée."""
    from agents import web_search
    monkeypatch.setattr(web_search, "ddgs_search", fake_search())
    monkeypatch.setattr(web_search, "PAUSE_S", 0)


def test_run_searches_dedupes_and_numbers():
    from agents.web_search import run_searches
    hits, journal = run_searches(["a", "b", "c"], search=fake_search(), pause=0)
    assert [h.id for h in hits] == ["S1", "S2", "S3", "S4"]          # 3 uniques + 1 doublon commun
    assert len({h.url for h in hits}) == len(hits) and len(journal) == 3


def test_run_searches_partial_and_total_failure():
    from agents.web_search import run_searches, WebSearchError
    calls = {"n": 0}

    def flaky(q, n):
        calls["n"] += 1
        if q == "bad":
            raise TimeoutError("lent")
        return [{"title": "T", "href": f"https://x.org/{q}", "body": "b"}]
    hits, journal = run_searches(["ok", "bad"], search=flaky, pause=0)
    assert len(hits) == 1 and "échec" in journal[1] and calls["n"] == 3   # 2 tentatives pour « bad »
    with pytest.raises(WebSearchError):
        run_searches(["x"], search=fake_search(fail=True), pause=0)


def test_web_agent_ddg_sources_cited_and_verified():
    search = fake_search()
    fake = FakeClient([QUERIES, json.dumps(CITED)])
    res = web_agent.run(fake, "ctx", search=search, pause=0)
    assert search.calls == json.loads(QUERIES)["queries"]               # 3 recherches effectuées
    assert res.web_search_used and len(res.sources) == 4
    assert [s.id for s in res.sources if s.cited] == ["S1", "S2"]
    assert all(s.uri.startswith("https://") and s.snippet for s in res.sources)
    assert "[S1]" in fake.calls[1]["prompt"] and "https://ex.org/common" in fake.calls[1]["prompt"]
    assert "DuckDuckGo réelle" in res.note and fake.calls[1]["use_search"] is False


def test_web_agent_rejects_invented_source_then_corrects():
    bad = CITED | {"justification": "Selon [S9], population 100 et tournoi conviennent au TSP.", "cited_sources": ["S9"]}
    fake = FakeClient([QUERIES, json.dumps(bad), json.dumps(CITED)])
    res = web_agent.run(fake, "ctx", search=fake_search(), pause=0)
    assert "inexistantes" in fake.calls[2]["prompt"] and res.config is not None


def test_web_agent_rejects_foreign_url_and_missing_citation():
    url = CITED | {"justification": "Voir https://invente.org/papier [S1] pour la population de 100."}
    nocite = CITED | {"justification": "Population 100 et tournoi, sans aucune citation dans ce texte.",
                      "cited_sources": []}
    for bad in (url, nocite):
        with pytest.raises(SchemaError):
            web_agent.run(FakeClient([QUERIES, json.dumps(bad), json.dumps(bad)]), "ctx",
                          search=fake_search(), pause=0)


def test_queries_validated_with_default_fallback():
    from agents.web_agent import DEFAULT_QUERIES, validate_queries
    assert validate_queries({"queries": ["a b c", "d e f", "g h i", "a b c"]}) == ["a b c", "d e f", "g h i"]
    with pytest.raises(SchemaError):
        validate_queries({"queries": ["seule"]})
    search = fake_search()
    fake = FakeClient([json.dumps({"queries": ["x"]}), json.dumps({"queries": []}), json.dumps(CITED)])
    res = web_agent.run(fake, "ctx", search=search, pause=0)
    assert search.calls == DEFAULT_QUERIES and "par défaut" in res.note


def test_web_agent_never_claims_search_when_ddg_fails():
    fake = FakeClient([QUERIES, json.dumps(GOOD)])
    res = web_agent.run(fake, "ctx", search=fake_search(fail=True), pause=0)
    assert res.web_search_used is False and res.sources == []
    assert "NON EFFECTUÉE" in res.note and "PAS accès au web" in fake.calls[-1]["prompt"]


def test_invalid_json_gets_one_correction():
    fake = FakeClient([json.dumps(GOOD | {"population_size": 75}), json.dumps(GOOD)])
    res = semantic_agent.run(fake, "ctx")
    assert res.config.population_size == 100 and len(fake.calls) == 2
    assert "invalide" in fake.calls[1]["prompt"]


def test_invalid_twice_raises():
    fake = FakeClient([json.dumps(GOOD | {"generations": 7})] * 2)
    with pytest.raises(SchemaError):
        semantic_agent.run(fake, "ctx")


def test_arbiter_gets_cited_sources_or_unverified_flag(synthetic_problem):
    sci = semantic_agent.run(FakeClient([json.dumps(GOOD | {"generations": 1000})]), "ctx")
    arb_json = {"final": GOOD, "convergences": ["population 100"], "rejected": [],
                "divergences": [{"parameter": "generations", "decision": "500", "reason": "budget"}],
                "consensus": "consensus partiel"}
    web_ok = web_agent.run(FakeClient([QUERIES, json.dumps(CITED)]), "ctx", search=fake_search(), pause=0)
    fake = FakeClient([json.dumps(arb_json)])
    arb = arbiter_agent.run(fake, "ctx", web_ok, sci)
    assert "DuckDuckGo réelle" in fake.calls[0]["prompt"] and "[S1]" in fake.calls[0]["prompt"]
    assert [d["parameter"] for d in arb.extra["python_diff"] if not d["agree"]] == ["generations"]
    web_ko = web_agent.run(FakeClient([QUERIES, json.dumps(GOOD)]), "ctx", search=fake_search(fail=True), pause=0)
    fake2 = FakeClient([json.dumps(arb_json)])
    arbiter_agent.run(fake2, "ctx", web_ko, sci)
    assert "SANS recherche web" in fake2.calls[0]["prompt"]


def test_manager_full_pipeline_and_saved(synthetic_problem):
    arb_json = {"final": GOOD, "convergences": [], "rejected": [], "divergences": [], "consensus": "ok"}
    fake = FakeClient([QUERIES, json.dumps(CITED), json.dumps(GOOD), json.dumps(arb_json)])
    s = agent_manager.run_agents(synthetic_problem, client=fake)
    assert s.arbiter.config is not None and s.model == "fake-model"
    raw = open(s.saved_to, encoding="utf-8").read()
    saved = json.loads(raw)
    assert saved["web"]["web_search_used"] is True and saved["decision"] == "en attente"
    assert [x["id"] for x in saved["web"]["sources"] if x["cited"]] == ["S1", "S2"]
    assert "GEMINI_API_KEY" not in raw


def test_manager_survives_agent_failure(synthetic_problem):
    arb_json = {"final": GOOD, "convergences": [], "rejected": [], "divergences": [], "consensus": "ok"}
    fake = FakeClient([QUERIES, "pas du json", "toujours pas", json.dumps(GOOD), json.dumps(arb_json)])
    s = agent_manager.run_agents(synthetic_problem, client=fake)
    assert s.web.error and s.web.config is None
    assert s.scientific.config is not None and s.arbiter.config is not None


# ------------------------------------------------------- client Gemini
def test_missing_key_clear_message(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(gc, "BASE_DIR", tmp_path)  # pas de .env
    with pytest.raises(GeminiConfigError, match="GEMINI_API_KEY"):
        gc.load_api_key()


class _Err(Exception):
    def __init__(self, code, msg="err"):
        super().__init__(msg)
        self.code, self.message = code, msg


class _Resp:
    def __init__(self, text):
        self.text = text
        self.candidates = []


class _Models:
    def __init__(self, behaviour):
        self.behaviour, self.seen = behaviour, []

    def generate_content(self, model, contents, config):
        self.seen.append(model)
        b = self.behaviour.get(model)
        if isinstance(b, Exception):
            raise b
        return _Resp(b or "OK")


def make_client(behaviour, monkeypatch):
    monkeypatch.setattr(gc.time, "sleep", lambda s: None)
    c = GeminiClient(api_key="test-key", model=None)
    c.candidates = ["m1", "m2", "m3"]
    c.client = type("C", (), {"models": _Models(behaviour)})()
    return c


def test_model_fallback_and_dead_list(monkeypatch):
    c = make_client({"m1": _Err(504, "deadline"), "m2": _Err(404, "gone"), "m3": "réponse"}, monkeypatch)
    r = c.generate("p", "s")
    assert r.model == "m3" and c.model == "m3" and {"m1", "m2"} <= c.dead
    c.client.models.seen.clear()
    c.generate("p", "s")
    assert c.client.models.seen == ["m3"]  # les modèles morts ne sont plus essayés


def test_probe_selects_first_responsive(monkeypatch):
    c = make_client({"m1": _Err(503, "busy"), "m2": "OK"}, monkeypatch)
    assert c.select_model() == "m2" and "m1" in c.dead
    assert any("indisponible" in line for line in c.probe_log)


def test_search_429_is_reported_as_unavailable(monkeypatch):
    c = make_client({"m1": _Err(429, "quota")}, monkeypatch)
    with pytest.raises(SearchUnavailable):
        c.generate("p", "s", use_search=True)


def test_invalid_key_is_config_error(monkeypatch):
    c = make_client({"m1": _Err(400, "API key not valid")}, monkeypatch)
    with pytest.raises(GeminiConfigError):
        c.generate("p", "s")


# ------------------------------------------------------------------ réel
@pytest.mark.skipif(os.getenv("TSP40_LIVE_GEMINI") != "1",
                    reason="Appel réel à Gemini : définir TSP40_LIVE_GEMINI=1")
def test_live_gemini(synthetic_problem):
    client = GeminiClient()
    model = client.select_model()
    from agents.schemas import CONFIG_JSON_SCHEMA
    from agents.base import call_json, ALLOWED
    cfg, reply = call_json(client, f"Propose une configuration d'AG pour un TSP de 40 villes. {ALLOWED}",
                           "Réponds en JSON.", validate_config, json_schema=CONFIG_JSON_SCHEMA)
    assert reply.model == model and cfg.population_size in (10, 50, 100)
