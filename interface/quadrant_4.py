"""Quadrant 4 : carte Folium/Leaflet réelle intégrée (Edge WebView2) + trajet OSRM."""
from __future__ import annotations
import webbrowser
from pathlib import Path
from tkinter import ttk

from config.settings import COLORS as C
from interface.map_embed import EmbeddedBrowser


class Quadrant4(ttk.Frame):
    def __init__(self, master, **kw):
        super().__init__(master, padding=10, **kw)
        head = ttk.Frame(self)
        head.pack(fill="x")
        ttk.Label(head, text="4 · Carte routière réelle (Folium · OSRM)", style="H.TLabel").pack(side="left")
        self.btn_ext = ttk.Button(head, text="Ouvrir dans le navigateur", command=self._open_ext,
                                  state="disabled")
        self.btn_ext.pack(side="right")
        self.status = ttk.Label(self, text="Démarrage du navigateur embarqué…", style="Muted.TLabel")
        self.status.pack(fill="x", pady=(2, 4))
        self.browser = EmbeddedBrowser(self, on_status=self._on_status, bg=C["panel2"])
        self.browser.pack(fill="both", expand=True)
        self.current: Path | None = None

    def _on_status(self, msg, ok):
        self.status.configure(text=msg, foreground=C["muted"] if ok else C["danger"])

    def show(self, html: Path, caption: str):
        self.current = html
        self.browser.load(html)
        self.btn_ext.configure(state="normal")
        self.status.configure(text=caption, foreground=C["muted"])

    def _open_ext(self):
        if self.current:
            # en http:// (serveur local) pour que les tuiles OpenStreetMap soient autorisées
            url = self.browser.url_for(self.current) or self.current.resolve().as_uri()
            webbrowser.open(url)

    def close(self):
        self.browser.close()
