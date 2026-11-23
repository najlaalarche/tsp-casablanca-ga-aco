"""Fonction fitness pour un problème de minimisation."""
import numpy as np


def fitness(distance_m):
    """Fitness = 1 / (1 + distance en km). Strictement positive, décroissante, stable."""
    return 1.0 / (1.0 + np.asarray(distance_m, dtype=float) / 1000.0)
