"""V2 ACO : évaluation multi-graines, Optuna, KerasTuner (si installé), agents SMA (faux client Gemini).

Matrices SYNTHÉTIQUES et faux Gemini / DuckDuckGo : aucun appel réseau.
"""
import json

import numpy as np
import pytest

from aco import optuna_aco, tuning
from aco.tuning import EvalRecord, evaluate, pareto_front, recommend
from agents.gemini_client import GeminiReply
from agents.schemas import SchemaError
from agents_aco import manager_aco
from agents_aco.schemas_aco import compare_configs, make_arbiter_validator, make_config_validator


def _matrix(n=12, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2)) * 5000
    d = np.linalg.norm(xy[:, None] - xy[None], axis=2) * (1 + 0.1 * rng.random((n, n)))
    np.fill_diagonal(d, 0)
    return d


P = {"n_ants": 10, "rho": 0.5, "alpha": 1.0, "beta": 3.0, "cycles": 100}


# ------------------------------------------------------------ évaluation
def test_evaluate_uses_every_seed_and_is_reproducible():
    d = _matrix()
    a = evaluate(d, P, [1, 2, 3])
    b = evaluate(d, P, [1, 2, 3])
    assert a.seeds == [1, 2, 3] and len(a.distances_km) == 3
    assert a.distances_km == b.distances_km
    assert a.distance_km == pytest.approx(np.mean(a.distances_km))
    assert a.std_distance_km == pytest.approx(np.std(a.distances_km, ddof=1))


def _rec(i, dist, t):
    return EvalRecord(i, dict(P), [1], [dist], [t], [1])


def test_pareto_and_recommendation():
    recs = [_rec(0, 90, 1.0), _rec(1, 88, 3.0), _rec(2, 95, 0.5), _rec(3, 91, 2.0)]  # #3 dominé par #0
    assert [r.number for r in pareto_front(recs)] == [1, 0, 2]
    best, text = recommend(recs)
    assert best.number in (0, 1, 2) and not recs[3].pareto and "Trial #" in text


def test_ants_range_multiple_of_step():
    for n in (40, 60, 80):
        lo, hi = tuning.ants_range(n)
        assert lo == 10 and hi == 2 * n and (hi - lo) % tuning.ANTS_STEP == 0


# ---------------------------------------------------------------- Optuna
def test_optuna_study_small(monkeypatch, tmp_path):
    monkeypatch.setattr(optuna_aco, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(optuna_aco, "suggest", _small_suggest)  # petit espace : test rapide
    res = optuna_aco.OptunaACO(_matrix(10), n_trials=4, seed=1, n_eval_seeds=2).run()
    assert len(res.records) == 4 and res.best is not None
    assert all(r.seeds == [1, 2] for r in res.records)
    data = json.loads((tmp_path / res.saved_to.split("\\")[-1].split("/")[-1]).read_text(encoding="utf-8"))
    assert data["best"]["trial"] == res.best.number and len(data["trials"]) == 4


def _small_suggest(trial, n):
    return {"n_ants": trial.suggest_int("n_ants", 5, 10, step=5),
            "rho": round(trial.suggest_float("rho", 0.1, 0.9), 3),
            "alpha": round(trial.suggest_float("alpha", 0.5, 3), 2),
            "beta": round(trial.suggest_float("beta", 1, 5), 2),
            "cycles": trial.suggest_categorical("cycles", [5, 10])}


def test_optuna_limits():
    with pytest.raises(ValueError):
        optuna_aco.OptunaACO(_matrix(), n_trials=21)
    with pytest.raises(ValueError):
        optuna_aco.OptunaACO(_matrix(), n_trials=5, n_eval_seeds=0)


# ------------------------------------------------------------- KerasTuner
def test_kerastuner_custom_tuner(monkeypatch, tmp_path):
    kt_mod = pytest.importorskip("aco.kerastuner_aco")
    try:
        kt_mod.check_available()
    except kt_mod.KerasTunerUnavailable:
        pytest.skip("KerasTuner non importable (backend Keras absent)")
    monkeypatch.setattr(kt_mod, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(kt_mod, "CYCLE_CHOICES", (5, 10))
    res = kt_mod.KerasTunerACO(_matrix(10), n_trials=2, seed=1, n_eval_seeds=2).run()
    assert len(res.records) == 2 and res.best == min(res.records, key=lambda r: r.distance_km)
    for r in res.records:   # espace de recherche de BC3, 2 graines par trial
        p = r.params
        assert 20 <= p["n_ants"] <= 80 and 0.05 <= p["rho"] <= 0.9 and 0.5 <= p["alpha"] <= 3
        assert 1 <= p["beta"] <= 6 and p["cycles"] in (5, 10) and r.seeds == [1, 2]


# ------------------------------------------------------------ schémas SMA
GOOD = {"n_ants": 40, "rho": 0.5, "alpha": 1, "beta": 3, "cycles": 500,
        "justification": "m = n est l'usage classique ; ρ = 0,5 équilibre mémoire et oubli des pistes."}


def test_validate_good_aco_config():
    c = make_config_validator(40)(GOOD)
    assert c.params() == {"n_ants": 40, "rho": 0.5, "alpha": 1.0, "beta": 3.0, "cycles": 500}


@pytest.mark.parametrize("key,bad", [("n_ants", 2), ("n_ants", 500), ("n_ants", 40.5), ("rho", 0.0),
                                     ("rho", 1.0), ("rho", "0.5"), ("alpha", 0), ("beta", -1),
                                     ("beta", 20), ("cycles", 200), ("n_ants", True), ("justification", "ok")])
def test_validate_rejects_out_of_domain(key, bad):
    with pytest.raises(SchemaError):
        make_config_validator(40)(GOOD | {key: bad})


def test_arbiter_requires_basis():
    v = make_arbiter_validator(40)
    ok = {"final": GOOD, "basis": ["[S1] vérifiée", "temps mesuré"], "convergences": [], "divergences": [],
          "rejected": [], "consensus": "ok"}
    final, extra = v(ok)
    assert final.cycles == 500 and extra["basis"]
    with pytest.raises(SchemaError):
        v(ok | {"basis": []})


def test_python_diff():
    v = make_config_validator(40)
    diff = compare_configs(v(GOOD), v(GOOD | {"beta": 5}))
    assert [d["parameter"] for d in diff if not d["agree"]] == ["beta"]


# --------------------------------------------------- SMA avec faux Gemini
class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.model, self.probe_log = "fake", []

    def generate(self, prompt, system, **kw):
        self.calls.append(prompt)
        return GeminiReply(self.replies.pop(0), "fake", [], False, [])


def test_sma_pipeline_with_fake_gemini(monkeypatch, tmp_path):
    monkeypatch.setattr(manager_aco, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(manager_aco, "LAST_MODEL_FILE", tmp_path / "m.txt")
    hits = [{"title": f"ACO paper {i}", "href": f"https://example.org/{i}", "body": "rho 0.5 beta 2 to 5"}
            for i in range(3)]
    web = GOOD | {"justification": GOOD["justification"] + " Valeurs classiques [S1].", "cited_sources": ["S1"]}
    sci = GOOD | {"beta": 5}
    arb = {"final": GOOD | {"beta": 4}, "basis": ["source [S1] vérifiée", "analyse théorique de β"],
           "convergences": ["ρ = 0,5"], "divergences": [{"parameter": "beta", "decision": "4", "reason": "compromis"}],
           "rejected": [], "consensus": "accord"}
    client = FakeClient([json.dumps({"queries": ["a b c d", "e f g h", "i j k l"]}),
                         json.dumps(web), json.dumps(sci), json.dumps(arb)])
    shown = []
    s = manager_aco.run_agents_aco(_matrix(40), 40, client=client, search=lambda q, k: hits, pause=0,
                                   on_agent=lambda i, r: shown.append(i))
    assert shown == [0, 1, 2]
    assert s.web.web_search_used and [x.id for x in s.web.sources if x.cited] == ["S1"]
    assert s.arbiter.config.beta == 4 and s.arbiter.extra["basis"]
    assert json.loads((tmp_path / s.saved_to.replace("\\", "/").split("/")[-1]).read_text(encoding="utf-8"))
    assert (tmp_path / "m.txt").read_text() == "fake"


def test_sma_rejects_invented_source(monkeypatch, tmp_path):
    monkeypatch.setattr(manager_aco, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(manager_aco, "LAST_MODEL_FILE", tmp_path / "m.txt")
    hits = [{"title": "ACO", "href": "https://example.org/1", "body": "x"}]
    bad = GOOD | {"justification": GOOD["justification"] + " Voir [S9].", "cited_sources": ["S9"]}
    client = FakeClient([json.dumps({"queries": ["a b c d", "e f g h", "i j k l"]}), json.dumps(bad), json.dumps(bad),
                         json.dumps(GOOD), json.dumps({"final": GOOD, "basis": ["théorie"], "convergences": [],
                                                       "divergences": [], "rejected": [], "consensus": "ok"})])
    s = manager_aco.run_agents_aco(_matrix(40), 40, client=client, search=lambda q, k: hits, pause=0)
    assert s.web.config is None and "S9" in s.web.error   # source inventée : rejetée après une relance
    assert s.scientific.config is not None and s.arbiter.config is not None
