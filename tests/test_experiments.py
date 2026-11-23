"""Comparaison des modes : mêmes graines, statistiques exactes, aucune valeur inventée."""
import csv
import json
import statistics
import threading

import pytest

from optimization import experiment_runner as er
from optimization.experiment_runner import (ExperimentRunner, ModeConfig, agents_sources, compute_stats,
                                            optuna_sources)

P_MAN = {"population_size": 10, "selection_method": "tournament", "mutation_rate": 0.05,
         "crossover_rate": 0.8, "generations": 30, "elitism": True}
P_OPT = P_MAN | {"selection_method": "roulette", "mutation_rate": 0.2}
P_AG = P_MAN | {"population_size": 50, "crossover_rate": 0.5}


@pytest.fixture(autouse=True)
def tmp_results(monkeypatch, tmp_path):
    monkeypatch.setattr(er, "RESULTS_DIR", tmp_path)
    return tmp_path


def configs(*modes):
    all_ = {"manuel": ModeConfig("manuel", P_MAN, "test"), "optuna": ModeConfig("optuna", P_OPT, "test"),
            "agents": ModeConfig("agents", P_AG, "test")}
    return {m: all_[m] for m in modes}


def test_same_seeds_for_every_mode_and_interleaved(synthetic_problem):
    order = []
    res = ExperimentRunner(synthetic_problem, configs("manuel", "optuna", "agents"), 4, 500).run(
        on_start=lambda m, rep, s: order.append((m, s)))
    assert res.seeds == [500, 501, 502, 503]
    for m in ("manuel", "optuna", "agents"):
        assert [r.seed for r in res.runs if r.mode == m] == res.seeds
    assert order[:3] == [("manuel", 500), ("optuna", 500), ("agents", 500)]  # entrelacé


def test_stats_match_runs_exactly(synthetic_problem):
    res = ExperimentRunner(synthetic_problem, configs("manuel", "optuna"), 5, 7).run()
    for m in ("manuel", "optuna"):
        d = [r.distance_km for r in res.runs if r.mode == m]
        s = res.stats[m]
        assert s.n == 5 and s.best_km == min(d) and s.worst_km == max(d)
        assert s.mean_km == pytest.approx(statistics.fmean(d))
        assert s.std_km == pytest.approx(statistics.stdev(d))  # n-1
        assert s.mean_time_s == pytest.approx(statistics.fmean(r.time_s for r in res.runs if r.mode == m))
        assert res.reference[m].seed == 7 and res.reference[m] in res.runs


def test_runs_are_reproducible(synthetic_problem):
    a = ExperimentRunner(synthetic_problem, configs("manuel"), 3, 42).run()
    b = ExperimentRunner(synthetic_problem, configs("manuel"), 3, 42).run()
    assert [r.distance_km for r in a.runs] == [r.distance_km for r in b.runs]


def test_missing_mode_is_absent_not_invented(synthetic_problem):
    res = ExperimentRunner(synthetic_problem, configs("manuel"), 2, 1).run()
    assert set(res.stats) == {"manuel"} and all(r.mode == "manuel" for r in res.runs)


def test_single_rep_std_zero(synthetic_problem):
    s = ExperimentRunner(synthetic_problem, configs("manuel"), 1, 1).run().stats["manuel"]
    assert s.std_km == 0.0 and s.best_km == s.worst_km == s.mean_km


def test_stop_keeps_only_complete_repetitions(synthetic_problem):
    big = configs("manuel", "optuna")
    big["optuna"] = ModeConfig("optuna", P_OPT | {"generations": 1000, "population_size": 100}, "lent")
    runner = ExperimentRunner(synthetic_problem, big, 50, 0)
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("r", runner.run()))
    t.start()
    threading.Timer(1.0, runner.stop).start()
    t.join(60)
    res = out["r"]
    assert res.stopped and len(res.seeds) < 50
    for m in ("manuel", "optuna"):
        assert [r.seed for r in res.runs if r.mode == m] == res.seeds  # répétitions complètes seulement


def test_saved_json_and_csv(synthetic_problem):
    res = ExperimentRunner(synthetic_problem, configs("manuel", "agents"), 2, 3).run()
    data = json.loads(open(res.saved_json, encoding="utf-8").read())
    assert data["seeds"] == [3, 4] and set(data["stats"]) == {"manuel", "agents"}
    with open(res.saved_csv, encoding="utf-8-sig") as f:
        rows = list(csv.reader(f, delimiter=";"))
    assert len(rows) == 1 + 4 and rows[0][0] == "mode"


def test_invalid_config_rejected(synthetic_problem):
    with pytest.raises(ValueError):
        ExperimentRunner(synthetic_problem, {"manuel": ModeConfig("manuel", P_MAN | {"mutation_rate": 2}, "x")})
    with pytest.raises(ValueError):
        ExperimentRunner(synthetic_problem, {})


def test_sources_from_saved_files(tmp_results):
    (tmp_results / "optuna_20260101_000000.json").write_text(json.dumps({
        "seed": 42, "trials": [{}] * 3,
        "recommendation": {"trial": 2, "params": {k: v for k, v in P_OPT.items() if k != "elitism"}}}),
        encoding="utf-8")
    (tmp_results / "optuna_20260101_000001.json").write_text(json.dumps({"trials": [], "recommendation": None}),
                                                             encoding="utf-8")
    (tmp_results / "agents_20260101_000000.json").write_text(json.dumps({
        "model": "m", "decision": "modifiée puis acceptée",
        "arbiter": {"config": P_AG | {"justification": "..."}, "error": None},
        "applied_params": P_AG | {"generations": 100}}), encoding="utf-8")
    (tmp_results / "agents_20260101_000001.json").write_text(json.dumps({
        "model": "m", "decision": "en attente", "arbiter": None, "applied_params": None}), encoding="utf-8")
    opt = optuna_sources(tmp_results)
    assert len(opt) == 1 and opt[0].params["elitism"] is True and "trial #2" in opt[0].source
    ag = agents_sources(tmp_results)
    assert [a.params["generations"] for a in ag] == [30, 100]
    assert "arbitre" in ag[0].source and "modification" in ag[1].source


def test_compute_stats_known_values():
    rec = lambda d, t: er.RunRecord("m", 1, 1, d, t, 1, 10, 0.1, 5.0, [0, 1, 0])
    s = compute_stats("m", [rec(100, 1), rec(110, 3), rec(120, 2)])
    assert (s.best_km, s.mean_km, s.worst_km) == (100, 110, 120)
    assert s.std_km == pytest.approx(10.0) and s.mean_time_s == pytest.approx(2.0)
