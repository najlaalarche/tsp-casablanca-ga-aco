"""Quadrant 2 : courbes de convergence et indicateurs de performance."""
from __future__ import annotations
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from config.settings import COLORS as C


class Quadrant2(ttk.Frame):
    KPIS = [("dist", "Distance"), ("time", "Temps"), ("bestgen", "Gén. du meilleur"),
            ("impr", "Amélioration"), ("fit", "Fitness")]

    def __init__(self, master, **kw):
        super().__init__(master, padding=(10, 6), **kw)
        ttk.Label(self, text="2 · Analyse des performances", style="H.TLabel").pack(anchor="w")
        kp = ttk.Frame(self)
        kp.pack(fill="x", pady=(2, 2))
        self.kpi = {}
        for i, (key, label) in enumerate(self.KPIS):
            kp.columnconfigure(i, weight=1)
            box = ttk.Frame(kp)
            box.grid(row=0, column=i, sticky="ew")
            ttk.Label(box, text=label, style="Muted.TLabel").pack(anchor="w")
            v = ttk.Label(box, text="—", style="KPI.TLabel")
            v.pack(anchor="w")
            self.kpi[key] = v

        self.fig = Figure(figsize=(5, 3.2), dpi=96)
        self.ax1 = self.fig.add_subplot(1, 2, 1)
        self.ax2 = self.fig.add_subplot(1, 2, 2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().configure(bg=C["panel"], highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.reset()

    def reset(self):
        for ax in (self.ax1, self.ax2):
            ax.clear()
        self.ax1.set_title("Convergence : meilleure distance")
        self.ax1.set_xlabel("Génération")
        self.ax1.set_ylabel("Distance (km)")
        self.ax2.set_title("Distance par génération")
        self.ax2.set_xlabel("Génération")
        self.ax2.set_ylabel("Distance (km)")
        (self.l_best,) = self.ax1.plot([], [], color=C["accent2"], lw=2, label="Meilleure globale")
        (self.l_gen,) = self.ax2.plot([], [], color=C["accent"], lw=1, label="Meilleur de la génération")
        (self.l_mean,) = self.ax2.plot([], [], color=C["warn"], lw=1, alpha=0.8, label="Moyenne population")
        self.ax1.legend(loc="upper right")
        self.ax2.legend(loc="upper right")
        self.fig.tight_layout()
        self.canvas.draw_idle()
        for v in self.kpi.values():
            v.configure(text="—")

    def update_curves(self, gens, best, gen_best, mean):
        self.l_best.set_data(gens, best)
        self.l_gen.set_data(gens, gen_best)
        self.l_mean.set_data(gens, mean)
        for ax in (self.ax1, self.ax2):
            ax.relim()
            ax.autoscale_view()
        self.canvas.draw_idle()

    def update_kpis(self, dist_km, elapsed_s, best_gen, impr_pct, fit):
        self.kpi["dist"].configure(text=f"{dist_km:.2f} km")
        self.kpi["time"].configure(text=f"{elapsed_s:.2f} s")
        self.kpi["bestgen"].configure(text=f"{best_gen}")
        self.kpi["impr"].configure(text=f"{impr_pct:.1f} %")
        self.kpi["fit"].configure(text=f"{fit:.5f}")
