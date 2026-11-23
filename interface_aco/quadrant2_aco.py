"""Quadrant 2 (ACO) : courbe de convergence et métriques."""
from __future__ import annotations
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from config.settings import COLORS as C


class Quadrant2ACO(ttk.Frame):
    KPIS = [("ok", "Tout livré ?"), ("dist", "Distance parcourue"), ("per", "km / livraison"),
            ("time", "Temps d'exécution"), ("best", "Cycle du meilleur")]

    def __init__(self, master, **kw):
        super().__init__(master, padding=(10, 6), **kw)
        ttk.Label(self, text="2 · Convergence et métriques", style="H.TLabel").pack(anchor="w")
        kp = ttk.Frame(self)
        kp.pack(fill="x", pady=(2, 2))
        self.kpi, self.kpi_lbl = {}, {}
        for i, (key, label) in enumerate(self.KPIS):
            kp.columnconfigure(i, weight=1)
            box = ttk.Frame(kp)
            box.grid(row=0, column=i, sticky="ew")
            self.kpi_lbl[key] = ttk.Label(box, text=label, style="Muted.TLabel")
            self.kpi_lbl[key].pack(anchor="w")
            v = ttk.Label(box, text="—", style="KPI.TLabel")
            v.pack(anchor="w")
            self.kpi[key] = v
        self.detail = ttk.Label(self, text="", style="Muted.TLabel")
        self.detail.pack(anchor="w")

        self.fig = Figure(figsize=(5, 3.2), dpi=96)
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().configure(bg=C["panel"], highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.reset()

    def reset(self, n_deliveries: int | None = None):
        ax = self.ax
        ax.clear()
        ax.set_title("Convergence : meilleure distance par cycle")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Distance routière (km)")
        (self.l_mean,) = ax.plot([], [], color=C["warn"], lw=0.8, alpha=0.6, label="Moyenne des fourmis")
        (self.l_cyc,) = ax.plot([], [], color=C["accent"], lw=0.8, alpha=0.8, label="Meilleure du cycle")
        (self.l_best,) = ax.plot([], [], color=C["accent2"], lw=2.2, label="Meilleure globale")
        ax.legend(loc="upper right")
        self.fig.tight_layout()
        self.canvas.draw_idle()
        for v in self.kpi.values():
            v.configure(text="—", foreground=C["accent2"])
        if n_deliveries is not None:
            self.kpi_lbl["per"].configure(text=f"km / livraison (÷{n_deliveries})")
        self.detail.configure(text="")

    def update_curves(self, cycles, best, cyc_best, mean):
        self.l_best.set_data(cycles, best)
        self.l_cyc.set_data(cycles, cyc_best)
        self.l_mean.set_data(cycles, mean)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()

    def update_kpis(self, dist_km: float, n_deliveries: int, elapsed_s: float, best_cycle: int,
                    all_delivered: bool | None = None):
        self.kpi["dist"].configure(text=f"{dist_km:.2f} km")
        self.kpi["per"].configure(text=f"{dist_km / n_deliveries:.3f} km")
        self.kpi["time"].configure(text=f"{elapsed_s:.2f} s")
        self.kpi["best"].configure(text=f"{best_cycle}")
        if all_delivered is not None:
            self.kpi["ok"].configure(text="Oui" if all_delivered else "Non",
                                     foreground=C["accent2"] if all_delivered else C["danger"])
