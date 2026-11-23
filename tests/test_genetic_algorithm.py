"""Moteur AG sur données synthétiques (jamais confondues avec le vrai problème)."""
import threading

import numpy as np
import pytest

from core.genetic_algorithm import GeneticAlgorithm, GAParams, RunControl
from core.validation import is_valid_tour


def optimal_circle_length(problem):
    n = problem.n
    fwd = [0, *range(1, n), 0]
    bwd = [0, *range(n - 1, 0, -1), 0]
    return min(problem.tour_length(t[1:-1]) for t in (fwd, bwd))


def test_tour_length_includes_return(synthetic_problem):
    p = synthetic_problem
    ind = list(range(1, p.n))
    manual = sum(p.d[a, b] for a, b in zip([0, *ind], [*ind, 0]))
    assert p.tour_length(ind) == pytest.approx(manual)
    assert p.population_lengths(np.array([ind]))[0] == pytest.approx(manual)


@pytest.mark.parametrize("selection", ["tournament", "roulette"])
def test_result_valid_and_start_fixed(synthetic_problem, selection):
    r = GeneticAlgorithm(synthetic_problem, GAParams(
        population_size=30, generations=60, selection_method=selection, seed=1)).run()
    assert is_valid_tour(r.best_tour, synthetic_problem.n, 0)
    assert r.best_tour[0] == r.best_tour[-1] == 0
    assert r.best_distance_m == pytest.approx(synthetic_problem.tour_length(r.best_tour[1:-1]))
    assert len(r.history_best) == 60


def test_elitism_monotonic(synthetic_problem):
    r = GeneticAlgorithm(synthetic_problem, GAParams(
        population_size=20, generations=100, elitism=True, mutation_rate=0.3, seed=3)).run()
    assert all(b <= a + 1e-9 for a, b in zip(r.history_best, r.history_best[1:]))
    # avec élitisme, le meilleur de chaque génération = meilleur global
    assert np.allclose(r.history_gen_best, r.history_best)


def test_convergence_to_optimum(synthetic_problem):
    r = GeneticAlgorithm(synthetic_problem, GAParams(
        population_size=100, generations=400, mutation_rate=0.2, seed=7)).run()
    assert r.best_distance_m <= optimal_circle_length(synthetic_problem) * 1.02
    assert r.improvement_pct > 0


def test_reproducible_with_seed(synthetic_problem):
    a = GeneticAlgorithm(synthetic_problem, GAParams(population_size=20, generations=50, seed=11)).run()
    b = GeneticAlgorithm(synthetic_problem, GAParams(population_size=20, generations=50, seed=11)).run()
    assert a.best_tour == b.best_tour and a.history_best == b.history_best


def test_invalid_params():
    with pytest.raises(ValueError):
        GAParams(mutation_rate=1.5).validate()
    with pytest.raises(ValueError):
        GAParams(selection_method="rank").validate()


def test_stop_control(synthetic_problem):
    ctl = RunControl()
    seen = []

    def cb(gen, _):
        seen.append(gen)
        if gen == 10:
            ctl.stop()

    r = GeneticAlgorithm(synthetic_problem, GAParams(generations=1000, seed=0)).run(cb, ctl)
    assert r.stopped and r.generations_run == 10
    assert is_valid_tour(r.best_tour, synthetic_problem.n, 0)


def test_step_mode(synthetic_problem):
    ctl = RunControl()
    ctl.step()  # démarre en pas à pas : une génération par appel
    gens = []
    t = threading.Thread(target=lambda: GeneticAlgorithm(
        synthetic_problem, GAParams(generations=50, seed=0)).run(lambda g, s: gens.append(g), ctl))
    t.start()
    t.join(0.5)
    assert gens == [1]
    ctl.step()
    t.join(0.5)
    assert gens == [1, 2]
    ctl.resume()
    t.join(10)
    assert gens[-1] == 50
