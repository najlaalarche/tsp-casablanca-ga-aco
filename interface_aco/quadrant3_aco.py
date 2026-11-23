"""Quadrant 3 (ACO) : plan cartésien qui remplit tout le cadre.

Différences avec le Quadrant3 de l'application AG (non modifié) :
- mise en page « constrained » : la figure occupe tout le cadre, aussi après redimensionnement ;
- axes ajustés aux points avec une marge de 5 %, puis élargis dans UNE seule direction pour que
  1 km en X = 1 km en Y à l'écran (ratio égal) tout en remplissant le cadre.
"""
from __future__ import annotations

from config.settings import COLORS as C
from interface.quadrant_3 import Quadrant3

MARGIN = 0.05


class Quadrant3ACO(Quadrant3):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.fig.set_layout_engine("constrained")
        self.canvas.mpl_connect("draw_event", self._on_draw)

    def _limits(self):
        """Limites (x0, x1, y0, y1) : points + 5 %, puis ratio égal adapté à la taille de l'axe."""
        xs = [v[0] for v in self.xy.values()]
        ys = [v[1] for v in self.xy.values()]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        w, h = max(x1 - x0, 1e-3), max(y1 - y0, 1e-3)
        x0, x1 = x0 - MARGIN * w, x1 + MARGIN * w
        y0, y1 = y0 - MARGIN * h, y1 + MARGIN * h
        w, h = x1 - x0, y1 - y0
        bb = self.ax.get_window_extent()
        ratio = bb.height / bb.width if bb.width > 10 and bb.height > 10 else h / w
        if h / w < ratio:            # cadre plus « haut » que les données : on élargit Y
            extra = (w * ratio - h) / 2
            y0, y1 = y0 - extra, y1 + extra
        else:                        # cadre plus « large » : on élargit X
            extra = (h / ratio - w) / 2
            x0, x1 = x0 - extra, x1 + extra
        return x0, x1, y0, y1

    def _apply_limits(self) -> bool:
        if not self.xy:
            return False
        new = self._limits()
        cur = (*self.ax.get_xlim(), *self.ax.get_ylim())
        tol = 1e-3 * max(new[1] - new[0], new[3] - new[2])
        if all(abs(a - b) <= tol for a, b in zip(new, cur)):
            return False
        self.ax.set_xlim(new[0], new[1])
        self.ax.set_ylim(new[2], new[3])
        return True

    def _on_draw(self, _event):
        # Après chaque dessin (dont les redimensionnements), on recale le ratio sur la taille réelle.
        if self._apply_limits():
            self.canvas.draw_idle()

    def _redraw(self):
        ax = self.ax
        ax.clear()
        if not self.points:
            self.canvas.draw_idle()
            return
        if self.tour:
            xs = [self.xy[i][0] for i in self.tour]
            ys = [self.xy[i][1] for i in self.tour]
            ax.plot(xs, ys, color=C["accent2"], lw=1.4, zorder=1, label="Parcours (ordre de visite)")
            ax.plot(xs[-2:], ys[-2:], color=C["warn"], lw=1.8, ls="--", zorder=2, label="Retour au départ")
            for a, b in zip(self.tour[:-1], self.tour[1:]):
                (x0, y0), (x1, y1) = self.xy[a], self.xy[b]
                ax.annotate("", xy=((x0 + 2 * x1) / 3, (y0 + 2 * y1) / 3), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle="-|>", color=C["accent2"], lw=0, mutation_scale=9))
            if self.legs_m and "selected" in self.v_legs.state():
                for (a, b), d in zip(zip(self.tour[:-1], self.tour[1:]), self.legs_m):
                    (x0, y0), (x1, y1) = self.xy[a], self.xy[b]
                    ax.text((x0 + x1) / 2, (y0 + y1) / 2, f"{d / 1000:.1f}", fontsize=6.5,
                            color=C["muted"], ha="center", va="center")
        others = [p for p in self.points if p.id != self.start]
        ax.scatter([self.xy[p.id][0] for p in others], [self.xy[p.id][1] for p in others],
                   s=26, color=C["accent"], zorder=3, label="Points")
        sx, sy = self.xy[self.start]
        ax.scatter([sx], [sy], s=110, marker="*", color=C["start"], zorder=4, label=f"Départ #{self.start}")
        for p in self.points:
            x, y = self.xy[p.id]
            ax.annotate(str(p.id), (x, y), xytext=(3, 3), textcoords="offset points", fontsize=7, color=C["fg"])
        ax.set_xlabel("X — Est (km)")
        ax.set_ylabel("Y — Nord (km)")
        ax.set_aspect("auto")          # le ratio égal est garanti par les limites (_limits)
        ax.legend(loc="best", fontsize=7, framealpha=0.85)
        self._apply_limits()
        self.canvas.draw_idle()
