"""Cartes Folium : points initiaux et parcours routier réel (géométrie OSRM Route)."""
from __future__ import annotations
from pathlib import Path
import folium

from config.settings import CACHE_DIR

START_COLOR = "#ff5c8a"
POINT_COLOR = "#4f8cff"
ROUTE_COLOR = "#2ec4a0"


def _center(points):
    return [sum(p.lat for p in points) / len(points), sum(p.lon for p in points) / len(points)]


def _number_icon(label: str, color: str, size: int = 20) -> folium.DivIcon:
    return folium.DivIcon(
        icon_size=(size, size), icon_anchor=(size // 2, size // 2),
        html=(f'<div style="width:{size}px;height:{size}px;border-radius:50%;background:{color};'
              f'color:#fff;font:bold 10px sans-serif;display:flex;align-items:center;'
              f'justify-content:center;border:2px solid #fff;box-shadow:0 0 3px #000">{label}</div>'))


def _popup(p, extra: str = "") -> folium.Popup:
    html = (f"<b>#{p.id} — {p.name}</b><br>Zone : {p.zone}<br>"
            f"Lat : {p.lat:.7f}<br>Lon : {p.lon:.7f}{extra}")
    return folium.Popup(html, max_width=280)


def _base_map(points) -> folium.Map:
    # Tuiles OpenStreetMap : la page doit être servie en http:// (serveur local du
    # processus map_host) car tile.openstreetmap.org bloque les requêtes sans Referer (file://).
    m = folium.Map(location=_center(points), zoom_start=12, tiles="OpenStreetMap",
                   control_scale=True, zoomSnap=0.25, zoomDelta=0.5)
    bounds = [[min(p.lat for p in points), min(p.lon for p in points)],
              [max(p.lat for p in points), max(p.lon for p in points)]]
    m.fit_bounds(bounds)
    # La fenêtre WebView2 change de taille quand elle est incrustée dans Tkinter :
    # on recalcule la taille Leaflet et on recadre tant que l'utilisateur n'a pas navigué.
    name = m.get_name()
    js = f"""
    document.addEventListener('DOMContentLoaded', function() {{
      var map = {name}, touched = false, b = {bounds}, el = map.getContainer();
      ['mousedown', 'wheel', 'touchstart', 'keydown'].forEach(function(ev) {{
        el.addEventListener(ev, function() {{ touched = true; }}, {{passive: true}});
      }});
      function refit() {{
        map.invalidateSize();
        // animate:false : une animation lancée pendant que la fenêtre est hors écran
        // (requestAnimationFrame suspendu) resterait bloquée.
        if (!touched) map.fitBounds(b, {{padding: [18, 18], animate: false}});
      }}
      window.__tspRefit = refit;  // appelé aussi par l'hôte Tkinter après chaque redimensionnement
      new ResizeObserver(refit).observe(el);
      setTimeout(refit, 50);
    }});
    """
    m.get_root().script.add_child(folium.Element(js))
    return m


def initial_map(points, start_id: int, out: Path | None = None) -> Path:
    m = _base_map(points)
    for p in points:
        is_start = p.id == start_id
        folium.Marker([p.lat, p.lon],
                      icon=_number_icon("D" if is_start else str(p.id),
                                        START_COLOR if is_start else POINT_COLOR,
                                        28 if is_start else 20),
                      z_index_offset=1000 if is_start else 0,
                      tooltip=f"#{p.id} {p.name}" + (" (départ)" if is_start else ""),
                      popup=_popup(p)).add_to(m)
    out = out or CACHE_DIR / "map_initial.html"
    m.save(str(out))
    return out


def route_map(points, tour: list[int], route: dict, out: Path | None = None,
              algo: str = "AG") -> Path:
    """tour : ordre de visite fermé [start, ..., start] ; route : réponse OSRM Route.

    algo : nom de l'algorithme affiché dans la légende (« AG » par défaut, « ACO »...)."""
    by_id = {p.id: p for p in points}
    m = _base_map(points)
    coords = [[lat, lon] for lon, lat in route["geometry"]]
    folium.PolyLine(coords, color=ROUTE_COLOR, weight=4, opacity=0.85,
                    tooltip=f"Trajet routier OSRM : {route['distance_m'] / 1000:.2f} km").add_to(m)
    legs = route["legs"]
    for rank, pid in enumerate(tour[:-1]):
        p = by_id[pid]
        leg = legs[rank]
        nxt = by_id[tour[rank + 1]]
        extra = (f"<hr style='margin:4px 0'>Étape {rank} → {rank + 1} : vers #{nxt.id} {nxt.name}"
                 f"<br>{leg['distance_m'] / 1000:.2f} km, {leg['duration_s'] / 60:.1f} min")
        is_start = rank == 0
        folium.Marker([p.lat, p.lon],
                      icon=_number_icon("D" if is_start else str(rank),
                                        START_COLOR if is_start else POINT_COLOR,
                                        28 if is_start else 20),
                      z_index_offset=1000 if is_start else 0,
                      tooltip=f"Ordre {rank} : #{p.id} {p.name}",
                      popup=_popup(p, extra)).add_to(m)
    legend = (f"<div style='position:fixed;bottom:18px;left:12px;z-index:9999;background:#1d2029;"
              f"color:#e6e8ee;padding:8px 10px;border-radius:6px;font:12px sans-serif;opacity:.92'>"
              f"<b>Parcours optimisé ({algo})</b><br>Distance routière : {route['distance_m'] / 1000:.2f} km"
              f"<br>Durée estimée : {route['duration_s'] / 60:.0f} min<br>"
              f"Étiquette = rang de visite, D = départ/retour</div>")
    m.get_root().html.add_child(folium.Element(legend))
    out = out or CACHE_DIR / "map_route.html"
    m.save(str(out))
    return out
