"""Opérateurs génétiques : les permutations restent valides."""
import numpy as np

from core import operators as ops
from core.fitness import fitness
from core.validation import is_valid_individual

GENES = np.arange(1, 40)


def test_random_population_valid():
    rng = np.random.default_rng(0)
    pop = ops.random_population(GENES, 30, rng)
    assert pop.shape == (30, 39)
    assert all(is_valid_individual(ind, GENES) for ind in pop)


def test_order_crossover_preserves_permutation():
    rng = np.random.default_rng(1)
    for _ in range(500):
        p1, p2 = rng.permutation(GENES), rng.permutation(GENES)
        child = ops.order_crossover(p1, p2, rng)
        assert is_valid_individual(child, GENES)


def test_order_crossover_keeps_parent_segment():
    rng = np.random.default_rng(2)
    p1, p2 = GENES.copy(), GENES[::-1].copy()
    child = ops.order_crossover(p1, p2, rng)
    # au moins deux gènes de p1 restent à leur position (segment copié)
    assert (child == p1).sum() >= 2


def test_mutations_preserve_permutation():
    rng = np.random.default_rng(3)
    for op in (ops.inversion_mutation, ops.swap_mutation):
        for _ in range(300):
            ind = rng.permutation(GENES)
            out = op(ind, rng)
            assert is_valid_individual(out, GENES)
            assert not np.array_equal(out, ind)


def test_selection_indices_in_range():
    rng = np.random.default_rng(4)
    fit = np.array([0.1, 0.5, 0.2, 0.9])
    for _ in range(100):
        assert 0 <= ops.tournament_selection(fit, rng) < 4
        assert 0 <= ops.roulette_selection(fit, rng) < 4


def test_tournament_favours_best():
    rng = np.random.default_rng(5)
    fit = np.array([0.01, 0.02, 0.03, 1.0])
    picks = [ops.tournament_selection(fit, rng, k=3) for _ in range(500)]
    assert picks.count(3) > 300


def test_fitness_monotonic():
    assert fitness(0) == 1.0
    assert fitness(50_000) > fitness(100_000) > 0
    assert np.isclose(fitness(100_000), 1 / 101)
