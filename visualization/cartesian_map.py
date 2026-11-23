"""Projection locale GPS -> plan X/Y métrique (km) pour le quadrant 3."""
from __future__ import annotations
import math

R_EARTH_KM = 6371.0088


def local_projection(points, origin=None):
    """Projection équirectangulaire locale centrée sur `origin` (point de départ par défaut).

    Sur une emprise d'environ 15 km, l'erreur est négligeable (< 0,1 %) : X = Est, Y = Nord, en km.
    """
    o = origin or points[0]
    lat0 = math.radians(o.lat)
    xy = {}
    for p in points:
        x = math.radians(p.lon - o.lon) * math.cos(lat0) * R_EARTH_KM
        y = math.radians(p.lat - o.lat) * R_EARTH_KM
        xy[p.id] = (x, y)
    return xy
