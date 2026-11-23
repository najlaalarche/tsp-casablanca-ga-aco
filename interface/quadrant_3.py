"""Quadrant 3 : visualisation cartésienne (projection locale en km) du parcours."""
from __future__ import annotations
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from config.settings import COLORS as C
from visualization.cartesian_map import local_projection


class Quadrant3(ttk.Frame):
    def __init__(self, master, **kw):
        super().__init__(master, padding=10, **kw)
        head = ttk.Frame(self)
        head.pack(fill="x")
        ttk.Label(head, text="3 · Plan cartésien (projection locale, km)", style="H.TLabel").pack(side="left")
        self.v_legs = ttk.Checkbutton(head, text="Distances par étape", command=self._redraw)
        self.v_legs.state(["!alternate", "!selected"])
        self.v_legs.pack(side="right")
        self.fig = Figure(figsize=(5, 3.5), dpi=96)
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        tb_frame = ttk.Frame(self)
        tb_frame.pack(fill="x", side="bottom")
        self.toolbar = NavigationToolbar2Tk(self.canvas, tb_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="left")
        self.canvas.get_tk_widget().configure(bg=C["panel"], highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.points, self.xy, self.tour, self.legs_m, self.start = [], {}, None, None, 0

    def set_points(self, points, start_id):
        self.points, self.start = points, start_id
        self.xy = local_projection(points, next(p for p in points if p.id == start_id))
        self.tour = None
        self._redraw()

    def set_tour(self, tour, legs_m=None):
        self.tour, self.legs_m = tour, legs_m
        self._redraw()

    def _redraw(self):
        ax = self.ax
        ax.clear()
        if not self.points:
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
        ax.scatter([sx], [sy], s=110, marker="*", color=C["start"], zorder=4,
                   label=f"Départ #{self.start}")
        for p in self.points:
            x, y = self.xy[p.id]
            ax.annotate(str(p.id), (x, y), xytext=(3, 3), textcoords="offset points",
                        fontsize=7, color=C["fg"])
        ax.set_xlabel("X — Est (km)")
        ax.set_ylabel("Y — Nord (km)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="lower left", fontsize=7)
        self.fig.tight_layout()
        self.canvas.draw_idle()
