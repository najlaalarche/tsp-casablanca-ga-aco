"""Paramètres globaux de l'application TSP 40."""
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = DATA_DIR / "results"
POINTS_FILE = INPUT_DIR / "points_tsp40.csv"

for _d in (CACHE_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

try:  # .env optionnel (serveur OSRM personnel, futures clés API)
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

OSRM_BASE_URL = os.getenv("OSRM_BASE_URL", "https://router.project-osrm.org").rstrip("/")
OSRM_PROFILE = os.getenv("OSRM_PROFILE", "driving")
OSRM_TIMEOUT_S = 30
OSRM_MAX_RETRIES = 3

EXPECTED_POINTS = 40
START_POINT_ID = 0  # Gare Casa Oasis

# Choix proposés dans le quadrant 1 (cahier des charges §4)
POPULATION_CHOICES = (10, 50, 100)
SELECTION_CHOICES = {"Tournoi": "tournament", "Roue biaisée": "roulette"}
MUTATION_CHOICES = (0.01, 0.05, 0.10, 0.20)
CROSSOVER_CHOICES = (0.2, 0.5, 0.8)
GENERATION_CHOICES = (100, 500, 1000)

# Thème sombre
COLORS = {
    "bg": "#14161c", "panel": "#1d2029", "panel2": "#252935", "fg": "#e6e8ee",
    "muted": "#9aa1b2", "accent": "#4f8cff", "accent2": "#2ec4a0",
    "warn": "#f2b84b", "danger": "#ef5b5b", "start": "#ff5c8a", "border": "#323746",
}
