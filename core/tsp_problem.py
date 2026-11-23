"""Modèle du TSP fermé à départ fixe sur matrice routière (asymétrique)."""
from __future__ import annotations
import numpy as np


class TSPProblem:
    """Un individu = permutation des points hors départ.

    La tournée complète est [start] + individu + [start] : le départ ne fait
    jamais partie du génome, il ne peut donc pas être déplacé par l'AG.
    """

    def __init__(self, distances_m: np.ndarray, start: int = 0):
        d = np.asarray(distances_m, dtype=float)
        if d.ndim != 2 or d.shape[0] != d.shape[1]:
            raise ValueError("La matrice de distances doit être carrée.")
        if not np.all(np.isfinite(d)):
            raise ValueError("La matrice contient des distances manquantes.")
        self.d = d
        self.n = d.shape[0]
        self.start = start
        self.genes = np.array([i for i in range(self.n) if i != start], dtype=np.int64)

    def full_tour(self, individual) -> list[int]:
        return [self.start, *map(int, individual), self.start]

    def tour_length(self, individual) -> float:
        """Distance totale en mètres, retour au départ inclus."""
        ind = np.asarray(individual)
        return float(self.d[self.start, ind[0]] + self.d[ind[:-1], ind[1:]].sum()
                     + self.d[ind[-1], self.start])

    def population_lengths(self, pop: np.ndarray) -> np.ndarray:
        """Évaluation vectorisée d'une population (shape: P x (n-1))."""
        return (self.d[self.start, pop[:, 0]] + self.d[pop[:, :-1], pop[:, 1:]].sum(axis=1)
                + self.d[pop[:, -1], self.start])

    def leg_lengths(self, individual) -> list[float]:
        t = self.full_tour(individual)
        return [float(self.d[a, b]) for a, b in zip(t[:-1], t[1:])]
