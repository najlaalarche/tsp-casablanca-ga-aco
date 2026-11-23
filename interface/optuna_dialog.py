"""Fenêtre du bouton 2 : optimisation multiobjectif Optuna (TPE/MOTPE) en direct."""
from __future__ import annotations
import logging
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from config.settings import COLORS as C
from optimization.optuna_optimizer import DEFAULT_EVAL_SEEDS, OptunaOptimizer, startup_trials

log = logging.getLogger(__name__)
SEL_FR = {"tournament": "Tournoi", "roulette": "Roulette"}


class OptunaDialog(tk.Toplevel):
    COLS = [("n", "#", 40), ("pop", "Pop.", 50), ("sel", "Sélection", 80), ("mut", "Mutation", 85),
            ("cx", "Crois.", 55), ("gen", "Gén.", 55), ("dist", "Dist. moy. (km)", 110),
            ("std", "± écart-type", 90), ("time", "Temps moy. (s)", 105), ("pareto", "Pareto", 60)]

    def __init__(self, master, problem, seed: int, on_apply, is_busy):
        super().__init__(master)
        self.title("Réglage automatique — Optuna multiobjectif (TPE / MOTPE)")
        self.configure(bg=C["bg"])
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = min(int(sw * 0.85), 1900), min(int(sh * 0.85), 1200)
        self.geometry(f"{w}x{h}+{(sw - w) // 2}+{max(0, (sh - h) // 3)}")
        self.minsize(950, 600)
        self.problem, self.on_apply, self.is_busy = problem, on_apply, is_busy
        self.q: queue.Queue = queue.Queue()
        self.opt: OptunaOptimizer | None = None
        self.result = None
        self.records = []
        self.running = False
        self._build(seed)
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(100, self._pump)

    # ------------------------------------------------------------------ UI
    def _build(self, seed):
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text="Trials").pack(side="left")
        self.v_trials = tk.StringVar(value="30")
        self.sp_trials = ttk.Spinbox(top, from_=1, to=500, textvariable=self.v_trials, width=6,
                                     command=self._update_info)
        self.sp_trials.pack(side="left", padx=(4, 14))
        self.sp_trials.bind("<KeyRelease>", lambda e: self._update_info())
        ttk.Label(top, text="Graines par trial").pack(side="left")
        self.v_nseeds = tk.StringVar(value=str(DEFAULT_EVAL_SEEDS))
        self.sp_nseeds = ttk.Spinbox(top, from_=1, to=10, textvariable=self.v_nseeds, width=4,
                                     command=self._update_info)
        self.sp_nseeds.pack(side="left", padx=(4, 14))
        self.sp_nseeds.bind("<KeyRelease>", lambda e: self._update_info())
        ttk.Label(top, text="Graine de base").pack(side="left")
        self.v_seed = tk.StringVar(value=str(seed))
        self.e_seed = ttk.Entry(top, textvariable=self.v_seed, width=8)
        self.e_seed.pack(side="left", padx=(4, 14))
        self.e_seed.bind("<KeyRelease>", lambda e: self._update_info())
        self.btn_start = ttk.Button(top, text="▶ Démarrer l'optimisation", style="Accent.TButton",
                                    command=self._start)
        self.btn_start.pack(side="left")
        self.btn_stop = ttk.Button(top, text="■ Arrêter", style="Danger.TButton", command=self._stop,
                                   state="disabled")
        self.btn_stop.pack(side="left", padx=6)
        self.info = ttk.Label(top, style="Muted.TLabel")
        self.info.pack(side="left", padx=10)
        self._update_info()

        pr = ttk.Frame(self, padding=(10, 0))
        pr.pack(fill="x")
        self.progress = ttk.Progressbar(pr, maximum=100)
        self.progress.pack(fill="x")
        self.prog_lbl = ttk.Label(pr, text="Objectifs minimisés séparément : distance (km) et temps "
                                           "d'exécution de l'AG (s), chacun moyenné sur plusieurs graines. Mêmes graines "
                                           "et même matrice OSRM pour tous les trials.", style="Muted.TLabel")
        self.prog_lbl.pack(anchor="w", pady=(2, 6))

        mid = ttk.PanedWindow(self, orient="horizontal")
        mid.pack(fill="both", expand=True, padx=10)
        left = ttk.Frame(mid, padding=6)
        right = ttk.Frame(mid, padding=6)
        mid.add(left, weight=3)
        mid.add(right, weight=2)

        ttk.Label(left, text="Historique des trials", style="H.TLabel").pack(anchor="w")
        st = ttk.Style(self)
        st.configure("Treeview", background=C["panel2"], fieldbackground=C["panel2"],
                     foreground=C["fg"], rowheight=24, borderwidth=0)
        st.configure("Treeview.Heading", background=C["panel"], foreground=C["fg"], relief="flat")
        st.map("Treeview", background=[("selected", C["accent"])])
        tv_frame = ttk.Frame(left)
        tv_frame.pack(fill="both", expand=True, pady=4)
        self.tree = ttk.Treeview(tv_frame, columns=[c[0] for c in self.COLS], show="headings")
        for key, label, w in self.COLS:
            self.tree.heading(key, text=label, command=lambda k=key: self._sort(k))
            self.tree.column(key, width=w, minwidth=w, anchor="center", stretch=key in ("dist", "time", "sel"))
        sb = ttk.Scrollbar(tv_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.tree.tag_configure("pareto", foreground=C["accent2"])
        self.tree.tag_configure("reco", background="#1f3b33")
        self.tree.tag_configure("running", foreground=C["warn"])

        ttk.Label(right, text="Front de Pareto (distance / temps)", style="H.TLabel").pack(anchor="w")
        self.fig = Figure(figsize=(4.5, 3.4), dpi=96)
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().configure(bg=C["panel"], highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._draw()

        bot = ttk.Frame(self, padding=10)
        bot.pack(fill="x", side="bottom", before=mid)  # réservé en premier : jamais rogné
        ttk.Label(bot, text="Configuration recommandée", style="H.TLabel").pack(anchor="w")
        self.reco = tk.Text(bot, height=5, wrap="word", bg=C["panel2"], fg=C["fg"], relief="flat",
                            font=("Segoe UI", 10), padx=8, pady=6)
        self.reco.pack(fill="x", pady=4)
        self._set_reco("Lancez l'optimisation : la recommandation apparaîtra à la fin des trials.")
        btns = ttk.Frame(bot)
        btns.pack(fill="x")
        self.btn_apply_run = ttk.Button(btns, text="✔ Appliquer la configuration et lancer la simulation",
                                        style="Accent.TButton", state="disabled",
                                        command=lambda: self._apply(run=True))
        self.btn_apply_run.pack(side="left")
        self.btn_apply = ttk.Button(btns, text="Appliquer sans lancer", state="disabled",
                                    command=lambda: self._apply(run=False))
        self.btn_apply.pack(side="left", padx=6)
        self.btn_apply_sel = ttk.Button(btns, text="Appliquer le trial sélectionné", state="disabled",
                                        command=self._apply_selected)
        self.btn_apply_sel.pack(side="left")
        ttk.Button(btns, text="Fermer", command=self._close).pack(side="right")
        self.tree.bind("<<TreeviewSelect>>", lambda e: self.btn_apply_sel.configure(
            state="normal" if self.tree.selection() and not self.running else "disabled"))

    def _update_info(self):
        try:
            n = int(self.v_trials.get())
            k = startup_trials(n)
            g = int(self.v_nseeds.get())
            b = int(self.v_seed.get())
            seeds = f"graines {b}" if g == 1 else f"graines {b} à {b + g - 1}"
            self.info.configure(text=f"{k} trial(s) aléatoire(s) puis {max(0, n - k)} guidé(s) par TPE · "
                                     f"{n * g} exécutions de l'AG ({seeds}, identiques pour chaque trial)")
        except ValueError:
            self.info.configure(text="Nombre de trials invalide")

    def _set_reco(self, text):
        self.reco.configure(state="normal")
        self.reco.delete("1.0", "end")
        self.reco.insert("1.0", text)
        self.reco.configure(state="disabled")

    # ------------------------------------------------------------ actions
    def _start(self):
        if self.is_busy():
            messagebox.showwarning("Simulation en cours",
                                   "Attendez la fin de la simulation : elle fausserait la mesure des temps.",
                                   parent=self)
            return
        try:
            n = int(self.v_trials.get())
            seed = int(self.v_seed.get())
            n_seeds = int(self.v_nseeds.get())
            if not 1 <= n <= 500 or not 1 <= n_seeds <= 10:
                raise ValueError
        except ValueError:
            messagebox.showwarning("Paramètres invalides",
                                   "Trials : 1 à 500. Graines par trial : 1 à 10. Graine de base : entier.", parent=self)
            return
        self.tree.delete(*self.tree.get_children())
        self.records, self.result = [], None
        self.n_trials = n
        self.opt = OptunaOptimizer(self.problem, n, seed, n_seeds)
        self.running = True
        self._set_controls()
        self._set_reco("Optimisation en cours…")
        self.progress["value"] = 0
        self._draw()
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            res = self.opt.run(on_trial=lambda r, all_: self.q.put(("trial", r, list(all_))),
                               on_trial_start=lambda n, p: self.q.put(("start", n, p)))
            self.q.put(("done", res))
        except Exception as e:  # noqa: BLE001
            log.exception("Optuna")
            self.q.put(("error", str(e)))

    def _stop(self):
        if self.opt:
            self.opt.stop()
            self.prog_lbl.configure(text="Arrêt demandé : le trial en cours est interrompu et ignoré…")

    def _set_controls(self):
        st = "disabled" if self.running else "normal"
        for w in (self.btn_start, self.sp_trials, self.sp_nseeds, self.e_seed):
            w.configure(state=st)
        self.btn_stop.configure(state="normal" if self.running else "disabled")
        ok = "normal" if (not self.running and self.result and self.result.recommendation) else "disabled"
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

    def _handle(self, msg):
        kind = msg[0]
        if kind == "start":
            _, n, p = msg
            done = len(self.records)
            self.prog_lbl.configure(
                text=f"Trial {n + 1}/{self.n_trials} en cours : pop {p['population_size']}, "
                     f"{SEL_FR[p['selection_method']]}, mutation {p['mutation_rate']:.3f}, "
                     f"croisement {p['crossover_rate']}, {p['generations']} générations…")
            self.progress["value"] = 100 * done / self.n_trials
            self.tree.insert("", "end", iid=f"run{n}", tags=("running",), values=(
                n, p["population_size"], SEL_FR[p["selection_method"]], f"{p['mutation_rate']:.3f}",
                p["crossover_rate"], p["generations"], "…", "…", "…", ""))
            self.tree.see(f"run{n}")
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
            self.running = False
            for iid in self.tree.get_children():
                if iid.startswith("run"):
                    self.tree.delete(iid)  # trial interrompu
            self._refresh_rows()
            self._draw()
            self.progress["value"] = 100
            r = self.result
            state = "interrompue" if r.stopped else "terminée"
            self.prog_lbl.configure(text=f"Optimisation {state} : {len(r.trials)} trial(s), "
                                         f"{len(r.pareto)} solution(s) non dominée(s). "
                                         f"Historique enregistré : {r.saved_to or '—'}")
            if r.recommendation:
                self._set_reco(r.recommendation.explanation)
            else:
                self._set_reco("Aucun trial terminé : pas de recommandation.")
            self._set_controls()
        elif kind == "error":
            self.running = False
            self._set_controls()
            messagebox.showerror("Erreur Optuna", msg[1], parent=self)

    def _refresh_rows(self):
        reco = self.result.recommendation.trial.number if self.result and self.result.recommendation else None
        for r in self.records:
            iid = f"t{r.number}"
            p = r.params
            vals = (r.number, p["population_size"], SEL_FR[p["selection_method"]],
                    f"{p['mutation_rate']:.3f}", p["crossover_rate"], p["generations"],
                    f"{r.distance_km:.2f}", f"{r.std_distance_km:.2f}" if len(r.seeds) > 1 else "—",
                    f"{r.time_s:.2f}",
                    ("★ reco" if r.number == reco else "★") if r.pareto else "")
            tags = (("pareto",) if r.pareto else ()) + (("reco",) if r.number == reco else ())
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
        items = sorted(self.tree.get_children(), key=k)
        for i, iid in enumerate(items):
            self.tree.move(iid, "", i)

    def _draw(self):
        ax = self.ax
        ax.clear()
        ax.set_xlabel("Temps moyen d'exécution (s)")
        ax.set_ylabel("Distance moyenne (km)")
        if self.records:
            dom = [r for r in self.records if not r.pareto]
            front = sorted([r for r in self.records if r.pareto], key=lambda r: r.time_s)
            ax.scatter([r.time_s for r in dom], [r.distance_km for r in dom], s=28,
                       color=C["muted"], alpha=0.8, label="Dominé")
            ax.plot([r.time_s for r in front], [r.distance_km for r in front], "-o", ms=6,
                    color=C["accent2"], drawstyle="steps-post", label="Front de Pareto")
            for r in self.records:
                ax.annotate(str(r.number), (r.time_s, r.distance_km), xytext=(4, 3),
                            textcoords="offset points", fontsize=7, color=C["fg"])
            if self.result and self.result.recommendation:
                t = self.result.recommendation.trial
                ax.scatter([t.time_s], [t.distance_km], s=180, marker="*", color=C["start"],
                           zorder=5, label=f"Recommandé (#{t.number})")
            ax.legend(loc="upper right", fontsize=7)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _apply(self, run: bool):
        if self.result and self.result.recommendation:
            self.on_apply(self.result.recommendation.trial.params, self.result.seed, run)
            if run:
                self.lower(self.master)

    def _apply_selected(self):
        sel = self.tree.selection()
        if not sel or not sel[0].startswith("t"):
            return
        num = int(sel[0][1:])
        rec = next(r for r in self.records if r.number == num)
        self.on_apply(rec.params, self.result.seed if self.result else int(self.v_seed.get()), False)

    def _close(self):
        if self.running:
            if not messagebox.askyesno("Optimisation en cours", "Arrêter l'optimisation et fermer ?",
                                       parent=self):
                return
            self.opt.stop()
        self.destroy()
