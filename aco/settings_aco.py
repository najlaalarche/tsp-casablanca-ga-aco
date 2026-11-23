"""Paramètres de l'application ACO (les paramètres AG de config/settings.py ne sont pas modifiés)."""
from config.settings import INPUT_DIR

CITY_LABEL = "Casablanca – Sidi Maarouf"
SIZE_CHOICES = (40, 60, 80)
CYCLE_CHOICES = (100, 500, 1000)
DEFAULTS = {"rho": 0.5, "alpha": 1.0, "beta": 3.0, "cycles": 500, "seed": 42}


def points_file(n: int):
    """Fichier des N points : data/input/points_tsp{N}.csv (aucun point n'est généré)."""
    return INPUT_DIR / f"points_tsp{n}.csv"
