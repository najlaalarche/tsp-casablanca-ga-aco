"""Opérateurs génétiques préservant les permutations."""
from __future__ import annotations
import numpy as np


# ---------- Initialisation ----------
def random_population(genes: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    return np.array([rng.permutation(genes) for _ in range(size)], dtype=np.int64)


# ---------- Sélection ----------
def tournament_selection(fit: np.ndarray, rng: np.random.Generator, k: int = 3) -> int:
    contenders = rng.choice(len(fit), size=min(k, len(fit)), replace=False)
    return int(contenders[np.argmax(fit[contenders])])


def roulette_selection(fit: np.ndarray, rng: np.random.Generator) -> int:
    p = fit / fit.sum()
    return int(rng.choice(len(fit), p=p))


# ---------- Croisement ----------
def order_crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Order Crossover (OX) : segment de p1 conservé, le reste dans l'ordre de p2."""
    n = len(p1)
    a, b = sorted(rng.choice(n, size=2, replace=False))
    child = np.full(n, -1, dtype=p1.dtype)
    child[a:b + 1] = p1[a:b + 1]
    taken = set(child[a:b + 1].tolist())
    fill = [g for g in np.roll(p2, -(b + 1)) if g not in taken]
    positions = [(b + 1 + i) % n for i in range(n - (b - a + 1))]
    child[positions] = fill
    return child


# ---------- Mutation ----------
def inversion_mutation(ind: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    a, b = sorted(rng.choice(len(ind), size=2, replace=False))
    out = ind.copy()
    out[a:b + 1] = out[a:b + 1][::-1]
    return out


def swap_mutation(ind: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    a, b = rng.choice(len(ind), size=2, replace=False)
    out = ind.copy()
    out[a], out[b] = out[b], out[a]
    return out


MUTATIONS = {"inversion": inversion_mutation, "swap": swap_mutation}
