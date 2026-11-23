"""Cartes Folium de l'application ACO, toujours cadrées sur les points.

Réutilise les briques de visualization/folium_map.py (non modifié, utilisé par l'AG) et ajoute un
cadrage forcé : fit_bounds au chargement (plusieurs fois, tant que la fenêtre WebView2 incrustée prend
sa taille définitive), à chaque redimensionnement, et à la demande de l'hôte Tkinter (commande FIT).
"""
from __future__ import annotations
from pathlib import Path

import folium

from config.settings import CACHE_DIR
from visualization.folium_map import (POINT_COLOR, ROUTE_COLOR, START_COLOR, _base_map, _number_icon,
                                      _popup)


def _bounds(points):
    return [[min(p.lat for p in points), min(p.lon for p in points)],
            [max(p.lat for p in points), max(p.lon for p in points)]]


def _add_fit(m: folium.Map, points, top_pad: int = 20) -> None:
    """top_pad : marge haute (px) réservée à la légende pour qu'elle ne masque aucun point."""
    js = f"""
    document.addEventListener('DOMContentLoaded', function() {{
      var map = {m.get_name()}, b = {_bounds(points)}, el = map.getContainer();
      function fit() {{
        map.invalidateSize({{animate: false}});
        // tant que le conteneur n'a pas sa vraie taille, fitBounds donnerait un zoom trop faible
        if (el.clientWidth > 50 && el.clientHeight > 50)
          {{
            // cadre large : les points sont au centre, la légende (coin haut-droit) tient à côté ;
            // cadre étroit : on réserve la marge haute pour qu'elle ne masque aucun point.
            var pad = (el.clientWidth / el.clientHeight > 1.6) ? 20 : {top_pad};
            map.fitBounds(b, {{paddingTopLeft: [20, pad], paddingBottomRight: [20, 20], animate: false}});
          }}
      }}
      window.__tspRefit = fit;   // commande FIT envoyée par l'hôte Tkinter (redimensionnement, run)
      new ResizeObserver(fit).observe(el);
      window.addEventListener('resize', fit);
      window.addEventListener('load', fit);
      [0, 150, 400, 900, 1800, 3500].forEach(function(t) {{ setTimeout(fit, t); }});
    }});
    """
    m.get_root().script.add_child(folium.Element(js))


def initial_map(points, start_id: int, out: Path) -> Path:
    m = _base_map(points)
    for p in points:
        s = p.id == start_id
        folium.Marker([p.lat, p.lon], icon=_number_icon("D" if s else str(p.id), START_COLOR if s else POINT_COLOR,
                                                        28 if s else 20),
                      z_index_offset=1000 if s else 0, tooltip=f"#{p.id} {p.name}" + (" (départ)" if s else ""),
                      popup=_popup(p)).add_to(m)
    _add_fit(m, points)
    m.save(str(out))
    return out


def route_map(points, tour: list[int], route: dict, out: Path | None = None) -> Path:
    by_id = {p.id: p for p in points}
    m = _base_map(points)
    folium.PolyLine([[lat, lon] for lon, lat in route["geometry"]], color=ROUTE_COLOR, weight=4, opacity=0.85,
                    tooltip=f"Trajet routier OSRM : {route['distance_m'] / 1000:.2f} km").add_to(m)
    for rank, pid in enumerate(tour[:-1]):
        p, leg, nxt = by_id[pid], route["legs"][rank], by_id[tour[rank + 1]]
        extra = (f"<hr style='margin:4px 0'>Étape {rank} → {rank + 1} : vers #{nxt.id} {nxt.name}"
                 f"<br>{leg['distance_m'] / 1000:.2f} km, {leg['duration_s'] / 60:.1f} min")
        s = rank == 0
        folium.Marker([p.lat, p.lon], icon=_number_icon("D" if s else str(rank), START_COLOR if s else POINT_COLOR,
                                                        28 if s else 20),
                      z_index_offset=1000 if s else 0, tooltip=f"Ordre {rank} : #{p.id} {p.name}",
                      popup=_popup(p, extra)).add_to(m)
    # en haut à droite : ne masque ni les points (cadrés au centre) ni le zoom (en haut à gauche)
    legend = (f"<div style='position:fixed;top:10px;right:10px;z-index:9999;background:#1d2029;"
              f"color:#e6e8ee;padding:6px 9px;border-radius:6px;font:11px sans-serif;opacity:.88'>"
              f"<b>Parcours optimisé (ACO)</b><br>Distance routière : {route['distance_m'] / 1000:.2f} km"
              f"<br>Durée estimée : {route['duration_s'] / 60:.0f} min<br>"
              f"Étiquette = rang de visite, D = départ/retour</div>")
    m.get_root().html.add_child(folium.Element(legend))
    _add_fit(m, points, top_pad=85)
    out = out or CACHE_DIR / "aco_map_route.html"
    m.save(str(out))
    return out
