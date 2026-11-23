"""Construction (une seule fois) et mise en cache de la matrice routière OSRM."""
from __future__ import annotations
import logging
from dataclasses import dataclass
import numpy as np

from config.settings import OSRM_BASE_URL, OSRM_PROFILE
from services import cache_manager, osrm_service

log = logging.getLogger(__name__)


@dataclass
class DistanceMatrix:
    distances_m: np.ndarray  # [i, j] = distance routière de i vers j (asymétrique)
    durations_s: np.ndarray
    source: str              # "cache" ou "osrm"
    cache_file: str


def _key(points) -> str:
    return cache_manager.fingerprint(OSRM_BASE_URL, OSRM_PROFILE,
                                     [(p.id, p.lat, p.lon) for p in points])


def get_matrix(points, force_refresh: bool = False) -> DistanceMatrix:
    name = f"osrm_matrix_{_key(points)}.json"
    cached = None if force_refresh else cache_manager.load(name)
    if cached and cached.get("ids") == [p.id for p in points]:
        log.info("Matrice OSRM chargée depuis le cache %s", name)
        return DistanceMatrix(np.array(cached["distances_m"], float),
                              np.array(cached["durations_s"], float), "cache", name)

    log.info("Appel OSRM Table pour %d points", len(points))
    dist, dur = osrm_service.table(points)
    missing = [(i, j) for i, row in enumerate(dist) for j, v in enumerate(row) if v is None]
    if missing:
        raise osrm_service.OSRMError(
            f"{len(missing)} liaisons routières introuvables (ex. {missing[:3]}).")
    d = np.array(dist, float)
    t = np.array(dur, float)
    cache_manager.save(name, {"ids": [p.id for p in points], "server": OSRM_BASE_URL,
                              "profile": OSRM_PROFILE, "distances_m": d.tolist(),
                              "durations_s": t.tolist()})
    return DistanceMatrix(d, t, "osrm", name)


def get_route(points_in_order) -> dict:
    """Itinéraire routier réel pour un ordre de visite (mis en cache par ordre)."""
    name = "osrm_route_" + cache_manager.fingerprint(
        OSRM_BASE_URL, OSRM_PROFILE, "continue_straight=false",
        [(p.id, p.lat, p.lon) for p in points_in_order]) + ".json"
    cached = cache_manager.load(name)
    if cached:
        return cached
    data = osrm_service.route(points_in_order)
    cache_manager.save(name, data)
    return data
