"""Optimisation Optuna multiobjectif (données synthétiques, petites configurations)."""
import json
import threading

import optuna
import pytest

from config.settings import POPULATION_CHOICES, CROSSOVER_CHOICES, GENERATION_CHOICES
from optimization import optuna_optimizer as oo
from optimization.optuna_optimizer import (OptunaOptimizer, TrialRecord, pareto_front, recommend,
                                           startup_trials)


@pytest.fixture(autouse=True)
def fast_space(monkeypatch, tmp_path):
    """Espace réduit (générations courtes) pour des tests rapides ; résultats dans tmp."""
    monkeypatch.setattr(oo, "GENERATION_CHOICES", (5, 20, 40))
    monkeypatch.setattr(oo, "RESULTS_DIR", tmp_path)


def rec(n, d, t):
    return TrialRecord(n, {"population_size": 10, "selection_method": "tournament",
                           "mutation_rate": 0.05, "crossover_rate": 0.8, "generations": 100}, d, t, 1)


def test_pareto_front():
    rs = [rec(0, 100, 5), rec(1, 90, 9), rec(2, 110, 1), rec(3, 105, 6), rec(4, 100, 5.5)]
    assert [r.number for r in pareto_front(rs)] == [1, 0, 2]  # 3 et 4 dominés par 0


def test_recommend_normalized_compromise():
    rs = [rec(0, 100, 1.0), rec(1, 80, 20.0), rec(2, 82, 3.0)]
    r = recommend(pareto_front(rs), rs)
    assert r.trial.number == 2  # quasi aussi court que #1, presque aussi rapide que #0
    assert "km" in r.explanation and "s" in r.explanation
    # invariance d'échelle : exprimer le temps en ms ne change pas le choix
    rs_ms = [rec(x.number, x.distance_km, x.time_s * 1000) for x in rs]
    assert recommend(pareto_front(rs_ms), rs_ms).trial.number == 2


def test_recommend_single_and_empty():
    assert recommend([], []) is None
    rs = [rec(0, 50, 1)]
    assert "seule solution" in recommend(rs, rs).explanation


def test_startup_trials():
    assert startup_trials(10) == 3          # TPE guide 7 trials sur 10
    assert startup_trials(3) == 2
    assert startup_trials(100) == 10


def test_search_space_respected(synthetic_problem):
    seen = []
    opt = OptunaOptimizer(synthetic_problem, 12, seed=1)
    res = opt.run(on_trial_start=lambda n, p: seen.append(p))
    assert len(res.trials) == 12 and len(seen) == 12
    for p in seen:
        assert p["population_size"] in POPULATION_CHOICES
        assert p["selection_method"] in ("tournament", "roulette")
        assert 0.01 <= p["mutation_rate"] <= 0.20
        assert p["mutation_rate"] == round(p["mutation_rate"], 4)  # réapplicable tel quel dans l'IHM
        assert p["crossover_rate"] in CROSSOVER_CHOICES
        assert p["generations"] in (5, 20, 40)
        assert p["seed"] == 1 and p["elitism"] is True  # même graine pour tous les trials


def test_study_is_multiobjective_tpe(synthetic_problem, monkeypatch):
    captured = {}
    real = optuna.create_study

    def spy(**kw):
        captured.update(kw)
        return real(**kw)
    monkeypatch.setattr(oo.optuna, "create_study", spy)
    OptunaOptimizer(synthetic_problem, 3, seed=0).run()
    assert captured["directions"] == ["minimize", "minimize"]
    assert isinstance(captured["sampler"], optuna.samplers.TPESampler)


def test_result_consistent_and_saved(synthetic_problem):
    res = OptunaOptimizer(synthetic_problem, 6, seed=3).run()
    assert res.recommendation is not None
    assert res.recommendation.trial in res.pareto
    assert all(t.pareto == (t in res.pareto) for t in res.trials)
    data = json.loads(open(res.saved_to, encoding="utf-8").read())
    assert len(data["trials"]) == 6 and data["recommendation"]["trial"] == res.recommendation.trial.number


def test_same_seed_reproducibility(synthetic_problem):
    """Ce qui est garanti avec une graine fixe :
    - les trials aléatoires initiaux sont identiques ;
    - une même configuration donne toujours exactement la même distance.
    Les trials guidés par TPE peuvent différer, car ils dépendent du temps d'exécution
    mesuré (temps réel, légèrement variable d'une exécution à l'autre)."""
    a = OptunaOptimizer(synthetic_problem, 6, seed=7).run()
    b = OptunaOptimizer(synthetic_problem, 6, seed=7).run()
    k = startup_trials(6)
    assert [t.params for t in a.trials[:k]] == [t.params for t in b.trials[:k]]
    by_params = {}
    for t in a.trials + b.trials:
        key = json.dumps(t.params, sort_keys=True)
        assert by_params.setdefault(key, t.distance_km) == t.distance_km


def test_stop_excludes_partial_trial(synthetic_problem, monkeypatch):
    monkeypatch.setattr(oo, "GENERATION_CHOICES", (100000,))  # trials très longs
    opt = OptunaOptimizer(synthetic_problem, 50, seed=0)
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("r", opt.run()))
    t.start()
    threading.Timer(0.5, opt.stop).start()
    t.join(30)
    assert not t.is_alive()
    assert out["r"].stopped and out["r"].trials == []


def test_invalid_trials(synthetic_problem):
    with pytest.raises(ValueError):
        OptunaOptimizer(synthetic_problem, 0)


# ------------------------------------------------- évaluation multi-graines
def test_each_trial_evaluated_on_same_k_seeds(synthetic_problem):
    from core.genetic_algorithm import GeneticAlgorithm, GAParams
    import statistics
    res = OptunaOptimizer(synthetic_problem, 4, seed=10, n_eval_seeds=3).run()
    assert res.eval_seeds == [10, 11, 12]
    for t in res.trials:
        assert t.seeds == [10, 11, 12] and len(t.distances_km) == 3 and len(t.times_s) == 3
        # la valeur transmise à Optuna est bien la moyenne des exécutions réelles
        assert t.distance_km == pytest.approx(statistics.fmean(t.distances_km))
        assert t.time_s == pytest.approx(statistics.fmean(t.times_s))
        assert t.std_distance_km == pytest.approx(statistics.stdev(t.distances_km))
        # et chaque distance correspond à une vraie exécution de l'AG avec cette graine
        for s, d in zip(t.seeds, t.distances_km):
            r = GeneticAlgorithm(synthetic_problem, GAParams(**t.params, seed=s)).run()
            assert r.best_distance_m / 1000 == pytest.approx(d)


def test_default_is_three_seeds_and_saved(synthetic_problem):
    res = OptunaOptimizer(synthetic_problem, 2, seed=5).run()
    assert res.eval_seeds == [5, 6, 7]
    data = json.loads(open(res.saved_to, encoding="utf-8").read())
    assert data["eval_seeds"] == [5, 6, 7] and len(data["trials"][0]["distances_km"]) == 3
    assert "3 graines" in res.recommendation.explanation


def test_single_seed_still_supported(synthetic_problem):
    res = OptunaOptimizer(synthetic_problem, 2, seed=5, n_eval_seeds=1).run()
    assert all(len(t.distances_km) == 1 and t.std_distance_km == 0.0 for t in res.trials)
    with pytest.raises(ValueError):
        OptunaOptimizer(synthetic_problem, 2, n_eval_seeds=0)
