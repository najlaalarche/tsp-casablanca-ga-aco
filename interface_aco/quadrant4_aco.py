"""Quadrant 4 (ACO) : même carte intégrée que l'AG, avec recadrage forcé après chaque chargement."""
from __future__ import annotations

from interface.quadrant_4 import Quadrant4

FIT_DELAYS_MS = (800, 2000, 4000)   # la page et la fenêtre WebView2 prennent leur taille progressivement


class Quadrant4ACO(Quadrant4):
    def show(self, html, caption: str):
        super().show(html, caption)
        for t in FIT_DELAYS_MS:
            self.after(t, self.fit)

    def fit(self):
        """Demande à Leaflet de recadrer sur les points (window.__tspRefit, défini par folium_map_aco)."""
        self.browser._send("FIT")
