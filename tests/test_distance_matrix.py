"""Matrice OSRM, cache, gestion d'erreur et cartes (OSRM simulé : aucun appel réseau)."""
import json

import numpy as np
import pytest

from config.settings import POINTS_FILE
from services import cache_manager, distance_matrix, osrm_service
from services.data_loader import load_points
from visualization import folium_map
from visualization.cartesian_map import local_projection


@pytest.fixture
def pts():
    return load_points(POINTS_FILE)


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_manager, "CACHE_DIR", tmp_path)
    return tmp_path


def fake_table(points):
    n = len(points)
    d = [[0.0 if i == j else 1000.0 + 10 * i + j for j in range(n)] for i in range(n)]
    return d, [[v / 10 for v in row] for row in d]


def test_matrix_called_once_then_cached(pts, tmp_cache, monkeypatch):
    calls = []
    monkeypatch.setattr(osrm_service, "table", lambda p: calls.append(1) or fake_table(p))
    m1 = distance_matrix.get_matrix(pts)
    m2 = distance_matrix.get_matrix(pts)
    assert len(calls) == 1
    assert m1.source == "osrm" and m2.source == "cache"
    assert m1.distances_m.shape == (40, 40)
    assert np.array_equal(m1.distances_m, m2.distances_m)
    assert m1.distances_m[1, 2] != m1.distances_m[2, 1]  # asymétrie conservée


def test_missing_route_raises(pts, tmp_cache, monkeypatch):
    def broken(p):
        d, t = fake_table(p)
        d[3][7] = None
        return d, t
    monkeypatch.setattr(osrm_service, "table", broken)
    with pytest.raises(osrm_service.OSRMError, match="introuvables"):
        distance_matrix.get_matrix(pts)


def test_osrm_unreachable_clear_error(monkeypatch):
    import requests
    monkeypatch.setattr(osrm_service, "OSRM_MAX_RETRIES", 2)
    monkeypatch.setattr(osrm_service.time, "sleep", lambda s: None)

    def boom(*a, **k):
        raise requests.ConnectionError("no network")
    monkeypatch.setattr(osrm_service.requests, "get", boom)
    with pytest.raises(osrm_service.OSRMError, match="indisponible"):
        osrm_service._get("http://x", {})


def test_route_request_allows_uturns(monkeypatch):
    seen = {}

    def fake_get(url, params):
        seen.update(params)
        return {"routes": [{"geometry": {"coordinates": [[0, 0], [1, 1]]}, "distance": 5.0,
                            "duration": 1.0, "legs": [{"distance": 5.0, "duration": 1.0}]}]}
    monkeypatch.setattr(osrm_service, "_get", fake_get)
    osrm_service.route([])
    assert seen["continue_straight"] == "false"


def test_real_cached_matrix_if_present(pts):
    """Vérifie la matrice OSRM réelle si elle a déjà été téléchargée par l'application."""
    name = f"osrm_matrix_{distance_matrix._key(pts)}.json"
    data = cache_manager.load(name)
    if data is None:
        pytest.skip("Matrice réelle pas encore en cache")
    d = np.array(data["distances_m"])
    assert d.shape == (40, 40) and np.all(np.isfinite(d)) and np.all(np.diag(d) == 0)
    assert d[d > 0].min() > 50  # points distincts


def test_maps_generated(pts, tmp_path):
    html = folium_map.initial_map(pts, 0, tmp_path / "init.html").read_text(encoding="utf-8")
    assert "Gare Casa Oasis" in html and "leaflet" in html.lower()
    tour = [p.id for p in pts] + [0]
    route = {"geometry": [[p.lon, p.lat] for p in pts], "distance_m": 1234.0, "duration_s": 60.0,
             "legs": [{"distance_m": 10.0, "duration_s": 1.0}] * 40}
    html = folium_map.route_map(pts, tour, route, tmp_path / "route.html").read_text(encoding="utf-8")
    assert "polyline" in html.lower() and "1.23 km" in html


def test_local_projection(pts):
    xy = local_projection(pts, pts[0])
    assert xy[0] == (0.0, 0.0)
    x39, y39 = xy[39]
    assert -16 < y39 < -12 and -3 < x39 < 2  # ZI Oulad Saleh ~13,6 km au sud du départ
