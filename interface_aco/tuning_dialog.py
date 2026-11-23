"""Fenêtre de réglage automatique de l'ACO, partagée par BC2 (Optuna) et BC3 (KerasTuner)."""
from __future__ import annotations
import logging
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from aco.tuning import DEFAULT_EVAL_SEEDS, MAX_TRIALS, fmt_params, space_description
from config.settings import COLORS as C

log = logging.getLogger(__name__)


class TuningDialog(tk.Toplevel):
    COLS = [("n", "#", 40), ("ants", "Fourmis", 65), ("rho", "ρ", 60), ("alpha", "α", 55),
            ("beta", "β", 55), ("cyc", "Cycles", 60), ("dist", "Dist. moy. (km)", 110),
            ("std", "± écart-type", 90), ("time", "Temps moy. (s)", 105), ("mark", "", 80)]

    def __init__(self, master, *, title: str, method: str, n_points: int, seed: int,
                 make_engine: Callable[[int, int, int, dict], object], on_apply, is_busy,
                 intro: str, options: dict[str, tuple[str, list[str]]] | None = None,
                 default_trials: int = 20, space_text: str | None = None):
        """make_engine(n_trials, seed, n_seeds, options) -> objet avec run(on_trial, on_trial_start), stop().
        options : {clé: (libellé, [choix])} affichés en combo box (ex. oracle KerasTuner)."""
        super().__init__(master)
        self.title(title)
        self.configure(bg=C["bg"])
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = min(int(sw * 0.8), 1700), min(int(sh * 0.8), 1050)
        self.geometry(f"{w}x{h}+{(sw - w) // 2}+{max(0, (sh - h) // 3)}")
        self.minsize(950, 600)
        self.method, self.n_points = method, n_points
        self.make_engine, self.on_apply, self.is_busy = make_engine, on_apply, is_busy
        self.q: queue.Queue = queue.Queue()
        self.engine = None
        self.result = None
        self.records = []
        self.running = False
        self.space_text = space_text or space_description(n_points)
        self._build(seed, intro, options or {}, default_trials)
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(100, self._pump)

    # ------------------------------------------------------------------ UI
    def _build(self, seed, intro, options, default_trials):
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")
        self.v_trials = tk.StringVar(value=str(default_trials))
        self.v_nseeds = tk.StringVar(value=str(DEFAULT_EVAL_SEEDS))
        self.v_seed = tk.StringVar(value=str(seed))
        self.inputs = []
        for label, var, width in (("Trials", self.v_trials, 6), ("Graines par trial", self.v_nseeds, 4),
                                  ("Graine de base", self.v_seed, 8)):
            ttk.Label(top, text=label).pack(side="left")
            e = ttk.Entry(top, textvariable=var, width=width)
            e.pack(side="left", padx=(4, 14))
            e.bind("<KeyRelease>", lambda _e: self._update_info())
            self.inputs.append(e)
        self.opt_vars = {}
        for key, (label, choices) in options.items():
            ttk.Label(top, text=label).pack(side="left")
            v = tk.StringVar(value=choices[0])
            cb = ttk.Combobox(top, textvariable=v, values=choices, state="readonly", width=24)
            cb.pack(side="left", padx=(4, 14))
            self.opt_vars[key] = v
            self.inputs.append(cb)
        self.btn_start = ttk.Button(top, text="▶ Démarrer", style="Accent.TButton", command=self._start)
        self.btn_start.pack(side="left")
        self.btn_stop = ttk.Button(top, text="■ Arrêter", style="Danger.TButton", command=self._stop,
                                   state="disabled")
        self.btn_stop.pack(side="left", padx=6)

        pr = ttk.Frame(self, padding=(10, 0))
        pr.pack(fill="x")
        self.info = ttk.Label(pr, style="Muted.TLabel")
        self.info.pack(anchor="w")
        ttk.Label(pr, text=intro + "\nEspace de recherche : " + self.space_text,
                  style="Muted.TLabel", justify="left").pack(anchor="w", pady=(2, 4))
        self.progress = ttk.Progressbar(pr, maximum=100)
        self.progress.pack(fill="x")
        self.prog_lbl = ttk.Label(pr, text="", style="Muted.TLabel")
        self.prog_lbl.pack(anchor="w", pady=(2, 6))
        self._update_info()

        bot = ttk.Frame(self, padding=10)
        bot.pack(fill="x", side="bottom")
        ttk.Label(bot, text="Meilleure configuration", style="H.TLabel").pack(anchor="w")
        self.reco = tk.Text(bot, height=5, wrap="word", bg=C["panel2"], fg=C["fg"], relief="flat",
                            font=("Segoe UI", 10), padx=8, pady=6)
        self.reco.pack(fill="x", pady=4)
        self._set_reco("Lancez la recherche : la meilleure configuration apparaîtra à la fin.")
        btns = ttk.Frame(bot)
        btns.pack(fill="x")
        self.btn_apply_run = ttk.Button(btns, text="✔ Appliquer au quadrant 1 et lancer l'ACO",
                                        style="Accent.TButton", state="disabled",
                                        command=lambda: self._apply(True))
        self.btn_apply_run.pack(side="left")
        self.btn_apply = ttk.Button(btns, text="Appliquer sans lancer", state="disabled",
                                    command=lambda: self._apply(False))
        self.btn_apply.pack(side="left", padx=6)
        self.btn_apply_sel = ttk.Button(btns, text="Appliquer le trial sélectionné", state="disabled",
                                        command=self._apply_selected)
        self.btn_apply_sel.pack(side="left")
        ttk.Button(btns, text="Fermer", command=self._close).pack(side="right")

        mid = ttk.PanedWindow(self, orient="horizontal")
        mid.pack(fill="both", expand=True, padx=10)
        left, right = ttk.Frame(mid, padding=6), ttk.Frame(mid, padding=6)
        mid.add(left, weight=3)
        mid.add(right, weight=2)
        ttk.Label(left, text="Trials", style="H.TLabel").pack(anchor="w")
        st = ttk.Style(self)
        st.configure("Treeview", background=C["panel2"], fieldbackground=C["panel2"],
                     foreground=C["fg"], rowheight=24, borderwidth=0)
        st.configure("Treeview.Heading", background=C["panel"], foreground=C["fg"], relief="flat")
        st.map("Treeview", background=[("selected", C["accent"])])
        tvf = ttk.Frame(left)
        tvf.pack(fill="both", expand=True, pady=4)
        self.tree = ttk.Treeview(tvf, columns=[c[0] for c in self.COLS], show="headings")
        for key, label, w in self.COLS:
            self.tree.heading(key, text=label, command=lambda k=key: self._sort(k))
            self.tree.column(key, width=w, minwidth=w, anchor="center", stretch=key in ("dist", "time"))
        sb = ttk.Scrollbar(tvf, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.tree.tag_configure("pareto", foreground=C["accent2"])
        self.tree.tag_configure("best", background="#1f3b33")
        self.tree.tag_configure("running", foreground=C["warn"])
        self.tree.bind("<<TreeviewSelect>>", lambda e: self.btn_apply_sel.configure(
            state="normal" if self.tree.selection() and not self.running
            and self.tree.selection()[0].startswith("t") else "disabled"))

        ttk.Label(right, text="Trials : distance moyenne / temps moyen", style="H.TLabel").pack(anchor="w")
        self.fig = Figure(figsize=(4.5, 3.4), dpi=96)
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().configure(bg=C["panel"], highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._draw()

    def _read(self):
        n, k, b = int(self.v_trials.get()), int(self.v_nseeds.get()), int(self.v_seed.get())
        if not 1 <= n <= MAX_TRIALS or not 1 <= k <= 10:
            raise ValueError
        return n, k, b

    def _update_info(self):
        try:
            n, k, b = self._read()
            seeds = f"graine {b}" if k == 1 else f"graines {b} à {b + k - 1}"
            self.info.configure(text=f"{n} trial(s) × {k} graine(s) = {n * k} exécutions de l'ACO "
                                     f"({seeds}, identiques pour chaque trial).", foreground=C["muted"])
        except ValueError:
            self.info.configure(text=f"Trials : 1 à {MAX_TRIALS} · graines par trial : 1 à 10 · graine de base : entier.",
                                foreground=C["danger"])

    def _set_reco(self, text):
        self.reco.configure(state="normal")
        self.reco.delete("1.0", "end")
        self.reco.insert("1.0", text)
        self.reco.configure(state="disabled")

    # ------------------------------------------------------------ actions
    def _start(self):
        if self.is_busy():
            messagebox.showwarning("Calcul en cours", "Attendez la fin du calcul en cours : il fausserait "
                                   "la mesure des temps.", parent=self)
            return
        try:
            n, k, b = self._read()
        except ValueError:
            messagebox.showwarning("Paramètres invalides", f"Trials : 1 à {MAX_TRIALS}. Graines par trial : 1 à 10. "
                                   "Graine de base : entier.", parent=self)
            return
        try:
            self.engine = self.make_engine(n, b, k, {key: v.get() for key, v in self.opt_vars.items()})
        except Exception as e:  # noqa: BLE001  (bibliothèque absente, paramètres…)
            messagebox.showerror("Démarrage impossible", str(e), parent=self)
            return
        self.n_trials = n
        self.tree.delete(*self.tree.get_children())
        self.records, self.result = [], None
        self.running = True
        self._set_controls()
        self._set_reco("Recherche en cours…")
        self.progress["value"] = 0
        self._draw()
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            res = self.engine.run(on_trial=lambda r, all_: self.q.put(("trial", r, all_)),
                                  on_trial_start=lambda n, p: self.q.put(("start", n, p)))
            self.q.put(("done", res))
        except Exception as e:  # noqa: BLE001
            log.exception(self.method)
            self.q.put(("error", str(e)))

    def _stop(self):
        if self.engine:
            self.engine.stop()
            self.prog_lbl.configure(text="Arrêt demandé : le trial en cours est interrompu et ignoré…")

    def _set_controls(self):
        st = "disabled" if self.running else "normal"
        for w in self.inputs:
            w.configure(state=("readonly" if isinstance(w, ttk.Combobox) and not self.running else st))
        self.btn_start.configure(state=st)
        self.btn_stop.configure(state="normal" if self.running else "disabled")
        ok = "normal" if (not self.running and self.result and self.result.best) else "disabled"
        self.btn_apply_run.configure(state=ok)
        self.btn_apply.configure(state=ok)

    # ------------------------------------------------------- file messages
    def _pump(self):
        try:
            while True:
                self._handle(self.q.get_nowait())
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(100, self._pump)

    def _row(self, number, p, dist="…", std="…", time_="…", mark=""):
        return (number, p["n_ants"], f"{p['rho']:.3f}", f"{p['alpha']:.2f}", f"{p['beta']:.2f}",
                p["cycles"], dist, std, time_, mark)

    def _handle(self, msg):
        kind = msg[0]
        if kind == "start":
            _, n, p = msg
            self.prog_lbl.configure(text=f"Trial {len(self.records) + 1}/{self.n_trials} en cours : {fmt_params(p)}…")
            iid = f"run{n}"
            if not self.tree.exists(iid):
                self.tree.insert("", "end", iid=iid, tags=("running",), values=self._row(n, p))
                self.tree.see(iid)
        elif kind == "trial":
            _, rec, all_ = msg
            self.records = all_
            if self.tree.exists(f"run{rec.number}"):
                self.tree.delete(f"run{rec.number}")
            self._refresh_rows()
            self.progress["value"] = 100 * len(self.records) / self.n_trials
            self._draw()
        elif kind == "done":
            self.result = msg[1]
            self.records = self.result.records
            self.running = False
            for iid in self.tree.get_children():
                if iid.startswith("run"):
                    self.tree.delete(iid)
            self._refresh_rows()
            self._draw()
            self.progress["value"] = 100
            r = self.result
            self.prog_lbl.configure(text=f"Recherche {'interrompue' if r.stopped else 'terminée'} : "
                                         f"{len(r.records)} trial(s). Résultats : {r.saved_to or '—'}")
            self._set_reco(r.explanation)
            self._set_controls()
        elif kind == "error":
            self.running = False
            self._set_controls()
            self.prog_lbl.configure(text="Erreur : voir le message.")
            messagebox.showerror(f"Erreur {self.method}", msg[1], parent=self)

    def _refresh_rows(self):
        best = self.result.best.number if self.result and self.result.best else None
        for r in self.records:
            iid = f"t{r.number}"
            mark = "★ retenu" if r.number == best else ("Pareto" if r.pareto else "")
            vals = self._row(r.number, r.params, f"{r.distance_km:.2f}",
                             f"{r.std_distance_km:.2f}" if len(r.seeds) > 1 else "—", f"{r.time_s:.2f}", mark)
            tags = (("pareto",) if r.pareto else ()) + (("best",) if r.number == best else ())
            if self.tree.exists(iid):
                self.tree.item(iid, values=vals, tags=tags)
            else:
                self.tree.insert("", "end", iid=iid, values=vals, tags=tags)

    def _sort(self, key):
        idx = [c[0] for c in self.COLS].index(key)

        def k(iid):
            v = self.tree.item(iid, "values")[idx]
            try:
                return (0, float(v))
            except ValueError:
                return (1, str(v))
        for i, iid in enumerate(sorted(self.tree.get_children(), key=k)):
            self.tree.move(iid, "", i)

    def _draw(self):
        ax = self.ax
        ax.clear()
        ax.set_xlabel("Temps moyen d'une exécution (s)")
        ax.set_ylabel("Distance moyenne (km)")
        if self.records:
            other = [r for r in self.records if not r.pareto]
            front = sorted([r for r in self.records if r.pareto], key=lambda r: r.time_s)
            ax.scatter([r.time_s for r in other], [r.distance_km for r in other], s=28,
                       color=C["muted"], alpha=0.8, label="Trial")
            if front:
                ax.plot([r.time_s for r in front], [r.distance_km for r in front], "-o", ms=6,
                        color=C["accent2"], drawstyle="steps-post", label="Front de Pareto")
            for r in self.records:
                ax.annotate(str(r.number), (r.time_s, r.distance_km), xytext=(4, 3),
                            textcoords="offset points", fontsize=7, color=C["fg"])
            if self.result and self.result.best:
                b = self.result.best
                ax.scatter([b.time_s], [b.distance_km], s=180, marker="*", color=C["start"], zorder=5,
                           label=f"Retenu (#{b.number})")
            ax.legend(loc="upper right", fontsize=7)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _apply(self, run: bool):
        if self.result and self.result.best:
            self.on_apply(self.result.best.params, self.result.seeds[0], run, self.result)
            if run:
                self.lower(self.master)

    def _apply_selected(self):
        sel = self.tree.selection()
        if not sel or not sel[0].startswith("t"):
            return
        rec = next(r for r in self.records if r.number == int(sel[0][1:]))
        self.on_apply(rec.params, rec.seeds[0], False, self.result)

    def _close(self):
        if self.running:
            if not messagebox.askyesno("Recherche en cours", "Arrêter la recherche et fermer ?", parent=self):
                return
            self.engine.stop()
        self.destroy()
