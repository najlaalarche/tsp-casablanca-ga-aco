"""Fenêtre de comparaison des 3 modes : 10 répétitions, tableaux (sections 8 et 9), boîtes à moustaches."""
from __future__ import annotations
import logging
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from config.settings import COLORS as C
from optimization.experiment_runner import (ExperimentRunner, ModeConfig, MODES, agents_sources,
                                            optuna_sources)

log = logging.getLogger(__name__)
# Palette catégorielle validée (validate_palette.js, mode sombre, surface #252935) — ordre fixe.
MODE_COLORS = {"manuel": "#3987e5", "optuna": "#d95926", "agents": "#199e70"}
MODE_LABELS = {"manuel": "Manuel", "optuna": "Optuna", "agents": "Multi-agents"}
SEL_FR = {"tournament": "tournoi", "roulette": "roulette"}


def summary(p: dict) -> str:
    return (f"pop {p['population_size']} · {SEL_FR[p['selection_method']]} · mut {p['mutation_rate']:g} · "
            f"crois {p['crossover_rate']} · {p['generations']} gén. · élitisme {'oui' if p['elitism'] else 'non'}")


class ComparisonDialog(tk.Toplevel):
    def __init__(self, master, problem, manual_params: dict, is_busy):
        super().__init__(master)
        self.title("Comparaison des trois modes — répétitions et statistiques")
        self.configure(bg=C["bg"])
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = min(int(sw * 0.92), 1950), min(int(sh * 0.9), 1250)
        self.geometry(f"{w}x{h}+{(sw - w) // 2}+{max(0, (sh - h) // 4)}")
        self.minsize(1050, 680)
        self.problem, self.is_busy = problem, is_busy
        self.q: queue.Queue = queue.Queue()
        self.runner: ExperimentRunner | None = None
        self.running = False
        self.runs = []
        self.result = None
        self.sources = {
            "manuel": [ModeConfig("manuel", manual_params, "Formulaire actuel du quadrant 1")],
            "optuna": optuna_sources(),
            "agents": agents_sources(),
        }
        self._build()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(100, self._pump)

    # ------------------------------------------------------------------ UI
    def _build(self):
        cfg = ttk.Frame(self, padding=(10, 8))
        cfg.pack(fill="x")
        ttk.Label(cfg, text="Configurations comparées (sources réelles uniquement)", style="H.TLabel").grid(
            row=0, column=0, columnspan=4, sticky="w")
        self.v_use, self.v_src, self.src_cb, self.param_lbl = {}, {}, {}, {}
        for i, mode in enumerate(MODES, start=1):
            srcs = self.sources[mode]
            self.v_use[mode] = tk.BooleanVar(value=bool(srcs))
            cb = ttk.Checkbutton(cfg, text=MODE_LABELS[mode], variable=self.v_use[mode],
                                 command=self._update_info)
            cb.grid(row=i, column=0, sticky="w", padx=(0, 10), pady=2)
            tk.Label(cfg, text="■", fg=MODE_COLORS[mode], bg=C["panel"], font=("Segoe UI", 12)).grid(
                row=i, column=1, sticky="w")
            self.v_src[mode] = tk.StringVar()
            combo = ttk.Combobox(cfg, textvariable=self.v_src[mode], state="readonly", width=75,
                                 values=[s.source for s in srcs])
            combo.grid(row=i, column=2, sticky="ew", padx=6)
            combo.bind("<<ComboboxSelected>>", lambda e, m=mode: self._on_source(m))
            self.src_cb[mode] = combo
            self.param_lbl[mode] = ttk.Label(cfg, style="Muted.TLabel")
            self.param_lbl[mode].grid(row=i, column=3, sticky="w")
            if srcs:
                combo.current(0)
                self._on_source(mode)
            else:
                cb.configure(state="disabled")
                self.v_src[mode].set("Aucune source : lancez d'abord "
                                     + ("une étude Optuna (bouton ②)" if mode == "optuna"
                                        else "les agents IA (bouton ③)"))
                self.param_lbl[mode].configure(text="mode exclu (aucune valeur inventée)", foreground=C["warn"])
        cfg.columnconfigure(2, weight=1)

        run = ttk.Frame(self, padding=(10, 4))
        run.pack(fill="x")
        ttk.Label(run, text="Répétitions par mode").pack(side="left")
        self.v_reps = tk.StringVar(value="10")
        sp = ttk.Spinbox(run, from_=1, to=100, textvariable=self.v_reps, width=5, command=self._update_info)
        sp.pack(side="left", padx=(4, 14))
        sp.bind("<KeyRelease>", lambda e: self._update_info())
        ttk.Label(run, text="Graine de base").pack(side="left")
        self.v_seed = tk.StringVar(value="1000")
        e = ttk.Entry(run, textvariable=self.v_seed, width=8)
        e.pack(side="left", padx=(4, 14))
        e.bind("<KeyRelease>", lambda ev: self._update_info())
        self.inputs = [sp, e]
        self.btn_start = ttk.Button(run, text="▶ Lancer la comparaison", style="Accent.TButton", command=self._start)
        self.btn_start.pack(side="left")
        self.btn_stop = ttk.Button(run, text="■ Arrêter", style="Danger.TButton", command=self._stop, state="disabled")
        self.btn_stop.pack(side="left", padx=6)
        self.info = ttk.Label(run, style="Muted.TLabel")
        self.info.pack(side="left", padx=10)

        pr = ttk.Frame(self, padding=(10, 0))
        pr.pack(fill="x")
        self.progress = ttk.Progressbar(pr, maximum=100)
        self.progress.pack(fill="x")
        self.prog_lbl = ttk.Label(pr, style="Muted.TLabel", text="")
        self.prog_lbl.pack(anchor="w", pady=(2, 4))

        bot = ttk.Frame(self, padding=(10, 6))
        bot.pack(side="bottom", fill="x")
        self.saved_lbl = ttk.Label(bot, style="Muted.TLabel", text="")
        self.saved_lbl.pack(side="left")
        ttk.Button(bot, text="Fermer", command=self._close).pack(side="right")

        body = ttk.PanedWindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=10)
        left = ttk.Frame(body, padding=6)
        right = ttk.Frame(body, padding=6)
        body.add(left, weight=1)
        body.add(right, weight=1)

        st = ttk.Style(self)
        st.configure("Treeview", background=C["panel2"], fieldbackground=C["panel2"], foreground=C["fg"],
                     rowheight=25, borderwidth=0)
        st.configure("Treeview.Heading", background=C["panel"], foreground=C["fg"], relief="flat")
        self.lbl_single = ttk.Label(left, text="Section 8 — Exécution unique de référence", style="H.TLabel")
        self.lbl_single.pack(anchor="w")
        self.tv_single = self._table(left, 5)
        self.lbl_stats = ttk.Label(left, text="Section 9 — Statistiques sur les répétitions", style="H.TLabel")
        self.lbl_stats.pack(anchor="w", pady=(10, 0))
        self.tv_stats = self._table(left, 8)
        ttk.Label(left, text="Toutes les valeurs proviennent de simulations réellement exécutées sur la matrice "
                             "OSRM ; écart-type d'échantillon (n−1). Le détail par exécution est enregistré en CSV.",
                  style="Muted.TLabel", wraplength=700, justify="left").pack(anchor="w", pady=(6, 0))

        ttk.Label(right, text="Distribution sur les répétitions", style="H.TLabel").pack(anchor="w")
        self.fig = Figure(figsize=(6, 6), dpi=96)
        self.ax_d = self.fig.add_subplot(2, 1, 1)
        self.ax_t = self.fig.add_subplot(2, 1, 2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().configure(bg=C["panel"], highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.tip = None
        self.canvas.mpl_connect("motion_notify_event", self._hover)
        self._draw()
        self._update_info()

    def _table(self, parent, height):
        cols = ["ind"] + list(MODES)
        tv = ttk.Treeview(parent, columns=cols, show="headings", height=height)
        tv.heading("ind", text="Indicateur")
        tv.column("ind", width=270, minwidth=270, anchor="w")
        for m in MODES:
            tv.heading(m, text=MODE_LABELS[m])
            tv.column(m, width=150, anchor="center")
        tv.pack(fill="x", pady=4)
        return tv

    def _on_source(self, mode):
        idx = self.src_cb[mode].current()
        if idx >= 0:
            self.param_lbl[mode].configure(text=summary(self.sources[mode][idx].params))

    def _selected(self) -> dict[str, ModeConfig]:
        out = {}
        for m in MODES:
            idx = self.src_cb[m].current()
            if self.v_use[m].get() and idx >= 0:
                out[m] = self.sources[m][idx]
        return out

    def _update_info(self):
        try:
            n = int(self.v_reps.get())
            k = len(self._selected())
            seed = int(self.v_seed.get())
            self.info.configure(text=f"{n * k} exécutions · graines {seed} à {seed + n - 1}, "
                                     f"identiques pour chaque mode")
        except ValueError:
            self.info.configure(text="Paramètres invalides")

    # ------------------------------------------------------------ exécution
    def _start(self):
        if self.is_busy():
            messagebox.showwarning("Occupé", "Une simulation ou une optimisation est en cours.", parent=self)
            return
        configs = self._selected()
        if not configs:
            messagebox.showwarning("Comparaison", "Sélectionnez au moins un mode.", parent=self)
            return
        try:
            n, seed = int(self.v_reps.get()), int(self.v_seed.get())
            self.runner = ExperimentRunner(self.problem, configs, n, seed)
        except ValueError as e:
            messagebox.showwarning("Paramètres invalides", str(e), parent=self)
            return
        self.runs, self.result = [], None
        self.running = True
        self._set_controls()
        for tv in (self.tv_single, self.tv_stats):
            tv.delete(*tv.get_children())
        self.saved_lbl.configure(text="")
        self._draw()
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            res = self.runner.run(on_run=lambda r, d, t: self.q.put(("run", r, d, t)),
                                  on_start=lambda m, rep, s: self.q.put(("start", m, rep, s)))
            self.q.put(("done", res))
        except Exception as e:  # noqa: BLE001
            log.exception("Comparaison")
            self.q.put(("error", str(e)))

    def _stop(self):
        if self.runner:
            self.runner.stop()
            self.prog_lbl.configure(text="Arrêt demandé : seules les répétitions complètes seront conservées…")

    def _set_controls(self):
        st = "disabled" if self.running else "normal"
        self.btn_start.configure(state=st)
        for w in self.inputs:
            w.configure(state=st)
        for m in MODES:
            self.src_cb[m].configure(state="disabled" if self.running or not self.sources[m] else "readonly")
        self.btn_stop.configure(state="normal" if self.running else "disabled")

    def _pump(self):
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "start":
                    _, m, rep, seed = msg
                    self.prog_lbl.configure(text=f"Répétition {rep}/{len(self.runner.seeds)} · graine {seed} · "
                                                 f"mode {MODE_LABELS[m]} en cours…")
                elif kind == "run":
                    _, r, done, total = msg
                    self.runs.append(r)
                    self.progress["value"] = 100 * done / total
                    self._draw()
                elif kind == "done":
                    self.running = False
                    self.result = msg[1]
                    self._set_controls()
                    self._show_result()
                elif kind == "error":
                    self.running = False
                    self._set_controls()
                    messagebox.showerror("Erreur de comparaison", msg[1], parent=self)
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(150, self._pump)

    # ------------------------------------------------------------ résultats
    def _show_result(self):
        res = self.result
        modes = [m for m in MODES if m in res.stats]
        n = len(res.seeds)
        state = "interrompue" if res.stopped else "terminée"
        self.prog_lbl.configure(text=f"Comparaison {state} : {n} répétition(s) complète(s) × {len(modes)} mode(s).")
        self.progress["value"] = 100
        cell = lambda m, f: f(m) if m in modes else "—"   # « — » = mode non exécuté, jamais une valeur inventée

        self.tv_single.delete(*self.tv_single.get_children())
        if res.reference:
            seed0 = res.seeds[0]
            self.lbl_single.configure(text=f"Section 8 — Exécution unique de référence (graine {seed0})")
            ref = res.reference
            rows = [
                ("Distance totale (km)", lambda m: f"{ref[m].distance_km:.2f}"),
                ("Temps d'exécution (s)", lambda m: f"{ref[m].time_s:.2f}"),
                ("Nombre de générations", lambda m: f"{ref[m].generations} (meilleur à {ref[m].best_generation})"),
                ("Meilleure fitness 1/(1+km)", lambda m: f"{ref[m].fitness:.5f}"),
                ("Taux d'amélioration (%)", lambda m: f"{ref[m].improvement_pct:.1f}"),
            ]
            for label, f in rows:
                self.tv_single.insert("", "end", values=[label] + [cell(m, f) if m in ref else "—" for m in MODES])

        self.tv_stats.delete(*self.tv_stats.get_children())
        self.lbl_stats.configure(text=f"Section 9 — Statistiques sur {n} répétition(s) par mode")
        s = res.stats
        rows = [
            ("Meilleure distance (km)", lambda m: f"{s[m].best_km:.2f}"),
            ("Distance moyenne (km)", lambda m: f"{s[m].mean_km:.2f}"),
            ("Pire distance (km)", lambda m: f"{s[m].worst_km:.2f}"),
            ("Écart-type distance (km)", lambda m: f"{s[m].std_km:.2f}"),
            ("Temps moyen (s)", lambda m: f"{s[m].mean_time_s:.2f} ± {s[m].std_time_s:.2f}"),
            ("Génération moyenne du meilleur", lambda m: f"{s[m].mean_best_generation:.0f}"),
            ("Amélioration moyenne (%)", lambda m: f"{s[m].mean_improvement_pct:.1f}"),
            ("Meilleure fitness", lambda m: f"{s[m].best_fitness:.5f}"),
        ]
        for label, f in rows:
            self.tv_stats.insert("", "end", values=[label] + [cell(m, f) for m in MODES])
        self.saved_lbl.configure(text=f"Enregistré : {res.saved_json}  ·  détail : {res.saved_csv}")
        self._draw()

    def _draw(self):
        self.tip = None  # ax.clear() supprime l'ancienne info-bulle
        modes = [m for m in MODES if any(r.mode == m for r in self.runs)]
        for ax, attr, title, unit in ((self.ax_d, "distance_km", "Distance totale par exécution", "km"),
                                      (self.ax_t, "time_s", "Temps d'exécution par exécution", "s")):
            ax.clear()
            ax.set_title(title, loc="left")
            ax.set_ylabel(unit)
            ax.grid(axis="x", visible=False)
            ax.grid(axis="y", alpha=0.35, linewidth=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if not modes:
                ax.text(0.5, 0.5, "Aucune exécution pour l'instant", transform=ax.transAxes,
                        ha="center", va="center", color=C["muted"])
                ax.set_xticks([])
                continue
            data = [[getattr(r, attr) for r in self.runs if r.mode == m] for m in modes]
            bp = ax.boxplot(data, widths=0.45, patch_artist=True, showfliers=False,
                            medianprops=dict(color=C["fg"], linewidth=2),
                            whiskerprops=dict(color=C["muted"], linewidth=1.2),
                            capprops=dict(color=C["muted"], linewidth=1.2),
                            boxprops=dict(linewidth=1.2))
            for patch, m in zip(bp["boxes"], modes):
                patch.set_facecolor(MODE_COLORS[m] + "55")
                patch.set_edgecolor(MODE_COLORS[m])
            rng = np.random.default_rng(0)  # dispersion horizontale fixe (lisibilité), sans effet sur les valeurs
            for i, (m, vals) in enumerate(zip(modes, data), start=1):
                xs = i + rng.uniform(-0.12, 0.12, len(vals))
                ax.scatter(xs, vals, s=36, color=MODE_COLORS[m], edgecolor=C["panel2"], linewidth=1.5, zorder=3)
                ax.annotate(f"moy. {np.mean(vals):.2f}", (i + 0.3, np.mean(vals)), fontsize=8,
                            color=C["fg"], va="center")
            ax.set_xticks(range(1, len(modes) + 1), [f"{MODE_LABELS[m]}\n(n={len(d)})" for m, d in zip(modes, data)])
            ax.set_xlim(0.5, len(modes) + 0.9)
            self._points = getattr(self, "_points", {})
            self._points[ax] = [(m, r) for m in modes for r in self.runs if r.mode == m]
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _hover(self, event):
        """Info-bulle : exécution la plus proche du curseur (mode, graine, distance, temps)."""
        ax = event.inaxes
        if ax not in (self.ax_d, self.ax_t) or not self.runs:
            if self.tip is not None:
                self.tip.set_visible(False)
                self.canvas.draw_idle()
            return
        attr = "distance_km" if ax is self.ax_d else "time_s"
        modes = [m for m in MODES if any(r.mode == m for r in self.runs)]
        best, best_d = None, 1e9
        for i, m in enumerate(modes, start=1):
            for r in self.runs:
                if r.mode != m:
                    continue
                px, py = ax.transData.transform((i, getattr(r, attr)))
                d = abs(px - event.x) * 0.3 + abs(py - event.y)
                if d < best_d:
                    best, best_d = (i, r), d
        if best is None or best_d > 25:
            if self.tip is not None:
                self.tip.set_visible(False)
                self.canvas.draw_idle()
            return
        i, r = best
        if self.tip is None or self.tip.axes is not ax:
            if self.tip is not None:
                self.tip.remove()
            self.tip = ax.annotate("", (0, 0), xytext=(12, 12), textcoords="offset points", fontsize=8,
                                   color=C["fg"], zorder=10,
                                   bbox=dict(boxstyle="round,pad=0.4", fc=C["panel"], ec=C["border"]))
        self.tip.xy = (i, getattr(r, attr))
        self.tip.set_text(f"{MODE_LABELS[r.mode]} · répétition {r.rep} · graine {r.seed}\n"
                          f"{r.distance_km:.2f} km · {r.time_s:.2f} s")
        self.tip.set_visible(True)
        self.canvas.draw_idle()

    def _close(self):
        if self.running:
            if not messagebox.askyesno("Comparaison en cours", "Arrêter et fermer ?", parent=self):
                return
            self.runner.stop()
        self.destroy()
