"""Client OSRM (Table + Route) avec délais, tentatives limitées et erreurs explicites."""
from __future__ import annotations
import logging
import time
import requests

from config.settings import OSRM_BASE_URL, OSRM_PROFILE, OSRM_TIMEOUT_S, OSRM_MAX_RETRIES

log = logging.getLogger(__name__)
HEADERS = {"User-Agent": "TSP40-academic-app/1.0"}


class OSRMError(RuntimeError):
    pass


def _coords(points) -> str:
    return ";".join(f"{p.lon:.7f},{p.lat:.7f}" for p in points)


def _get(url: str, params: dict) -> dict:
    last = None
    for attempt in range(1, OSRM_MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=OSRM_TIMEOUT_S)
            if r.status_code == 429:
                raise OSRMError("Serveur OSRM saturé (HTTP 429).")
            r.raise_for_status()
            data = r.json()
            if data.get("code") != "Ok":
                raise OSRMError(f"OSRM a répondu : {data.get('code')} - {data.get('message', '')}")
            return data
        except (requests.RequestException, OSRMError, ValueError) as e:
            last = e
            log.warning("OSRM tentative %d/%d échouée : %s", attempt, OSRM_MAX_RETRIES, e)
            if attempt < OSRM_MAX_RETRIES:
                time.sleep(1.5 * attempt)
    raise OSRMError(
        f"Service OSRM indisponible ({OSRM_BASE_URL}). Vérifiez la connexion Internet.\nDétail : {last}")


def table(points):
    """Matrices complètes distances (m) et durées (s), potentiellement asymétriques."""
    url = f"{OSRM_BASE_URL}/table/v1/{OSRM_PROFILE}/{_coords(points)}"
    data = _get(url, {"annotations": "distance,duration"})
    return data["distances"], data["durations"]


def route(points_in_order) -> dict:
    """Géométrie routière réelle (GeoJSON) + distances par étape, en une seule requête."""
    url = f"{OSRM_BASE_URL}/route/v1/{OSRM_PROFILE}/{_coords(points_in_order)}"
    # continue_straight=false : demi-tour autorisé aux points de passage, comme dans
    # OSRM Table. Sinon le trajet tracé est plus long que la distance optimisée.
    data = _get(url, {"overview": "full", "geometries": "geojson", "steps": "false",
                      "continue_straight": "false"})
    rt = data["routes"][0]
    return {
        "geometry": rt["geometry"]["coordinates"],  # [[lon, lat], ...]
        "distance_m": rt["distance"],
        "duration_s": rt["duration"],
        "legs": [{"distance_m": leg["distance"], "duration_s": leg["duration"]} for leg in rt["legs"]],
    }
