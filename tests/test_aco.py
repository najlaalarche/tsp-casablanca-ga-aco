"""Tests du moteur ACO (Ant System). Matrices SYNTHÉTIQUES : test du moteur uniquement."""
import numpy as np
import pytest

from aco.ant_system import AntSystem, ACOParams
from core.genetic_algorithm import RunControl
from core.validation import is_valid_tour


def _matrix(n=15, seed=0, asym=True):
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2)) * 5000
    d = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    if asym:
        d = d * (1 + 0.1 * rng.random((n, n)))
    np.fill_diagonal(d, 0)
    return d


@pytest.mark.parametrize("n,start,ants", [(5, 0, 3), (15, 0, None), (15, 7, 40), (30, 3, 8)])
def test_constructed_tours_are_valid_permutations(n, start, ants):
    aco = AntSystem(_matrix(n), ACOParams(n_ants=ants, cycles=1, seed=1), start=start)
    for _ in range(5):
        tours = aco.construct_tours()
        assert tours.shape == (aco.m, n + 1)
        for t in tours.tolist():
            assert is_valid_tour(t, n, start)
            assert t[0] == t[-1] == start
            assert len(set(t[:-1])) == n  # aucun doublon


def test_run_result_valid_and_lengths_consistent():
    d = _matrix(20)
    r = AntSystem(d, ACOParams(cycles=30, seed=3)).run()
    assert is_valid_tour(r.best_tour, 20, 0)
    t = r.best_tour
    assert r.best_distance_m == pytest.approx(sum(d[a, b] for a, b in zip(t[:-1], t[1:])))
    assert len(r.history_best) == 30 and r.cycles_run == 30
    assert all(a >= b for a, b in zip(r.history_best, r.history_best[1:]))  # meilleur global monotone
    assert r.history_best[r.best_cycle - 1] == r.best_distance_m
    assert r.n_ants == 20  # par défaut = nombre de points


def test_seed_reproducible():
    d = _matrix(15)
    a = AntSystem(d, ACOParams(cycles=20, seed=5)).run()
    b = AntSystem(d, ACOParams(cycles=20, seed=5)).run()
    assert a.best_tour == b.best_tour and a.history_best == b.history_best


def test_evaporation_factor():
    aco = AntSystem(_matrix(10), ACOParams(rho=0.3, tau0=2.0, cycles=1))
    off = ~np.eye(10, dtype=bool)
    assert np.allclose(aco.tau[off], 2.0)
    aco.evaporate()
    assert np.allclose(aco.tau[off], 2.0 * 0.7)
    aco.evaporate()
    assert np.allclose(aco.tau[off], 2.0 * 0.7 * 0.7)


def test_deposit_q_over_length_on_directed_arcs():
    d = _matrix(6)
    aco = AntSystem(d, ACOParams(tau0=1.0, q=2.0, cycles=1))
    tour = np.array([[0, 2, 4, 1, 5, 3, 0]])
    L = aco.tours_lengths(tour)
    before = aco.tau.copy()
    aco.deposit(tour, L)
    added = aco.tau - before
    expected = 2.0 / (L[0] / 1000)
    for a, b in zip(tour[0, :-1], tour[0, 1:]):
        assert added[a, b] == pytest.approx(expected)
    assert np.count_nonzero(added) == 6  # arcs orientés du tour seulement (matrice asymétrique)


def test_heuristic_prefers_close_points():
    """β très grand + phéromone uniforme : la fourmi suit le plus proche voisin."""
    d = _matrix(12, asym=False)
    aco = AntSystem(d, ACOParams(beta=60, alpha=1, n_ants=5, tau0=1.0, cycles=1, seed=0))
    nn = aco.nearest_neighbour_tour()
    assert all(t == nn for t in aco.construct_tours().tolist())


def test_improves_over_first_cycle():
    r = AntSystem(_matrix(25), ACOParams(cycles=100, seed=2)).run()
    assert r.best_distance_m <= r.initial_best_m


def test_zero_distance_between_distinct_points_is_handled():
    d = _matrix(8)
    d[2, 3] = d[3, 2] = 0.0
    r = AntSystem(d, ACOParams(cycles=10, seed=0)).run()
    assert is_valid_tour(r.best_tour, 8, 0) and np.isfinite(r.best_distance_m)


def test_stop_before_first_cycle_still_returns_valid_tour():
    c = RunControl()
    c.stop()
    r = AntSystem(_matrix(10), ACOParams(cycles=50)).run(control=c)
    assert r.stopped and r.cycles_run == 0 and is_valid_tour(r.best_tour, 10, 0)


def test_callback_called_each_cycle():
    seen = []
    AntSystem(_matrix(8), ACOParams(cycles=12)).run(on_cycle=lambda c, s: seen.append(c))
    assert seen == list(range(1, 13))


@pytest.mark.parametrize("kw", [
    {"rho": 0.0}, {"rho": 1.0}, {"rho": -0.1}, {"rho": 1.5}, {"rho": float("nan")},
    {"alpha": 0.0}, {"alpha": -1.0}, {"beta": 0.0}, {"beta": -2.0},
    {"n_ants": 0}, {"n_ants": 2.5}, {"cycles": 0}, {"cycles": 10.0}, {"q": 0.0}, {"tau0": 0.0},
])
def test_parameter_bounds_rejected(kw):
    with pytest.raises(ValueError):
        ACOParams(**kw).validate()


@pytest.mark.parametrize("kw", [{"rho": 0.01}, {"rho": 0.99}, {"alpha": 0.1}, {"beta": 10},
                                {"n_ants": 1}, {"cycles": 1}])
def test_parameter_bounds_accepted(kw):
    ACOParams(**kw).validate()


def test_invalid_matrix_rejected():
    with pytest.raises(ValueError):
        AntSystem(np.ones((3, 4)), ACOParams())
    d = _matrix(5)
    d[1, 2] = np.nan
    with pytest.raises(ValueError):
        AntSystem(d, ACOParams())
