import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.tsp_problem import TSPProblem  # noqa: E402


@pytest.fixture
def synthetic_problem():
    """Données SYNTHÉTIQUES (cercle de 12 points, matrice asymétrique) : test du moteur uniquement."""
    n = 12
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xy = np.c_[np.cos(ang), np.sin(ang)] * 1000
    d = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    d = d + np.triu(np.full((n, n), 5.0), 1)  # légère asymétrie
    np.fill_diagonal(d, 0)
    return TSPProblem(d, start=0)
