"""Vérifications de validité des solutions TSP."""
from __future__ import annotations


def is_valid_tour(tour: list[int], n: int, start: int) -> bool:
    """Tournée fermée : départ fixe en tête et en queue, chaque point exactement une fois."""
    return (len(tour) == n + 1 and tour[0] == start and tour[-1] == start
            and sorted(tour[:-1]) == list(range(n)))


def is_valid_individual(ind, genes) -> bool:
    return len(ind) == len(genes) and sorted(map(int, ind)) == sorted(map(int, genes))
