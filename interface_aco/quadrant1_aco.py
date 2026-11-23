"""Quadrant 1 (ACO) : définition du problème, réglages, boutons BC1–BC5, verdicts des agents."""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk

from aco.ant_system import ACOParams
from aco.settings_aco import CITY_LABEL, SIZE_CHOICES, CYCLE_CHOICES, DEFAULTS
from config.settings import COLORS as C

AGENTS = (("Agent web", "recherche documentaire"),
          ("Agent sémantique", "analyse scientifique"),
          ("Agent arbitre", "décision finale"))


class Quadrant1ACO(ttk.Frame):
    def __init__(self, master, callbacks: dict, **kw):
        super().__init__(master, padding=(10, 6), **kw)
        self.cb = callbacks
        self.mode = "manuel"
        for c in (1, 3):
            self.columnconfigure(c, weight=1)

        ttk.Label(self, text="1 · Définition et réglage (colonie de fourmis)", style="H.TLabel").grid(
            row=0, column=0, columnspan=4, sticky="w")

        # --- Problème ---
        ttk.Label(self, text="Ville").grid(row=1, column=0, sticky="w", padx=(0, 6))
        ttk.Label(self, text=CITY_LABEL, foreground=C["accent2"]).grid(row=1, column=1, sticky="w")
        ttk.Label(self, text="Nombre de points").grid(row=1, column=2, sticky="w", padx=(0, 6))
        self.v_n = tk.StringVar(value=str(SIZE_CHOICES[0]))
        self.cb_n = ttk.Combobox(self, textvariable=self.v_n, state="readonly", width=8,
                                 values=[str(v) for v in SIZE_CHOICES])
        self.cb_n.grid(row=1, column=3, sticky="ew", pady=2)
        self.cb_n.bind("<<ComboboxSelected>>", lambda _e: callbacks["size"](int(self.v_n.get())))
        self.problem_lbl = ttk.Label(self, text="Chargement des données…", style="Muted.TLabel",
                                     wraplength=560, justify="left")
        self.problem_lbl.grid(row=2, column=0, columnspan=4, sticky="w", pady=(0, 4))

        # --- Réglages ACO ---
        self.v_ants = tk.StringVar(value=str(SIZE_CHOICES[0]))
        self.v_rho = tk.StringVar(value=str(DEFAULTS["rho"]))
        self.v_alpha = tk.StringVar(value=str(DEFAULTS["alpha"]))
        self.v_beta = tk.StringVar(value=str(DEFAULTS["beta"]))
        self.v_cycles = tk.StringVar(value=str(DEFAULTS["cycles"]))
        self.v_seed = tk.StringVar(value=str(DEFAULTS["seed"]))
        rows = [
            ("Fourmis (déf. = N)", ttk.Entry(self, textvariable=self.v_ants, width=10)),
            ("ρ évaporation ]0;1[", ttk.Entry(self, textvariable=self.v_rho, width=10)),
            ("α phéromone (> 0)", ttk.Entry(self, textvariable=self.v_alpha, width=10)),
            ("β proximité (> 0)", ttk.Entry(self, textvariable=self.v_beta, width=10)),
            ("Cycles", ttk.Combobox(self, textvariable=self.v_cycles, state="readonly", width=8,
                                    values=[str(v) for v in CYCLE_CHOICES])),
            ("Graine aléatoire", ttk.Entry(self, textvariable=self.v_seed, width=10)),
        ]
        self.inputs = []
        for i, (label, w) in enumerate(rows):
            r, c = 3 + i // 2, (i % 2) * 2
            ttk.Label(self, text=label).grid(row=r, column=c, sticky="w", pady=2, padx=(0, 6))
            w.grid(row=r, column=c + 1, sticky="ew", pady=2, padx=(0, 10))
            self.inputs.append(w)
        ttk.Label(self, text="Ant System · η = 1/d (OSRM) · départ fixe #0",
                  style="Muted.TLabel").grid(row=6, column=0, columnspan=2, sticky="w")

        # --- Boutons de commande ---
        bc = ttk.Frame(self)
        bc.grid(row=7, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        for i in range(2):
            bc.columnconfigure(i, weight=1, uniform="bc")
        self.btn_manual = ttk.Button(bc, text="BC1 Réglage manuel", style="ModeOn.TButton",
                                     command=self._manual)
        self.btn_optuna = ttk.Button(bc, text="BC2 Optuna", style="Mode.TButton", state="disabled",
                                     command=callbacks["optuna"])
        self.btn_keras = ttk.Button(bc, text="BC3 KerasTuner", style="Mode.TButton", state="disabled",
                                    command=callbacks["keras"])
        self.btn_sma = ttk.Button(bc, text="BC4 Réglage SMA", style="Mode.TButton", state="disabled",
                                  command=callbacks["sma"])
        self.mode_buttons = {"manuel": self.btn_manual, "optuna": self.btn_optuna,
                             "kerastuner": self.btn_keras, "sma": self.btn_sma}
        for i, b in enumerate(self.mode_buttons.values()):   # grille 2 × 2 : libellés complets
            b.grid(row=i // 2, column=i % 2, sticky="ew", padx=(0 if i % 2 == 0 else 3, 0), pady=(0, 3))
        self.tool_buttons = (self.btn_optuna, self.btn_keras, self.btn_sma)
        self.keras_ok = False      # BC3 actif seulement si KerasTuner s'importe
        self._ready = False
        run = ttk.Frame(self)
        run.grid(row=8, column=0, columnspan=4, sticky="ew", pady=(4, 0))
        run.columnconfigure(0, weight=1)
        self.btn_run = ttk.Button(run, text="▶ BC5 Run ACO", style="Accent.TButton",
                                  command=callbacks["run"], state="disabled")
        self.btn_run.grid(row=0, column=0, sticky="ew")
        self.btn_stop = ttk.Button(run, text="■ Arrêter", style="Danger.TButton",
                                   command=callbacks["stop"], state="disabled")
        self.btn_stop.grid(row=0, column=1, sticky="ew", padx=(3, 0))
        self.btn_compare = ttk.Button(run, text="⚖ Comparer (plus tard)", state="disabled",
                                      command=callbacks["compare"])
        self.btn_compare.grid(row=0, column=2, sticky="ew", padx=(3, 0))
        self.mode_lbl = ttk.Label(self, text="Mode : réglage manuel", style="Muted.TLabel")
        self.keras_lbl = ttk.Label(bc, text="Vérification de KerasTuner…", style="Muted.TLabel",
                                   wraplength=500, justify="left")
        self.keras_lbl.grid(row=2, column=0, columnspan=2, sticky="w")
        bc.bind("<Configure>", lambda e: self.keras_lbl.configure(wraplength=max(200, e.width - 10)))
        self.mode_lbl.grid(row=6, column=2, columnspan=2, sticky="e")
        self._applying = False
        for v in (self.v_ants, self.v_rho, self.v_alpha, self.v_beta, self.v_cycles):
            v.trace_add("write", self._on_edit)

        self.progress = ttk.Progressbar(self, maximum=100, mode="determinate")
        self.progress.grid(row=9, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        self.progress_lbl = ttk.Label(self, text="0 %", style="Muted.TLabel")
        self.progress_lbl.grid(row=10, column=0, columnspan=4, sticky="w")

        # --- Verdicts des agents : résumé court ici, détail dans une fenêtre dédiée ---
        vh = ttk.Frame(self)
        vh.grid(row=11, column=0, columnspan=4, sticky="ew", pady=(6, 2))
        ttk.Label(vh, text="Verdicts des agents", style="H.TLabel").pack(side="left")
        self.btn_verdicts = ttk.Button(vh, text="🔍 Voir les verdicts", command=callbacks["verdicts"])
        self.btn_verdicts.pack(side="right")
        self.btn_rerun = ttk.Button(vh, text="↻ Relancer les agents en échec", state="disabled",
                                    command=callbacks["rerun"])
        self.btn_rerun.pack(side="right", padx=(0, 4))
        ag = ttk.Frame(self)
        ag.grid(row=12, column=0, columnspan=4, sticky="ew")
        self.verdicts = []
        for i, (name, role) in enumerate(AGENTS):
            ag.columnconfigure(i, weight=1, uniform="ag")
            f = tk.Frame(ag, bg=C["panel2"], highlightbackground=C["border"], highlightthickness=1)
            f.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 4, 0))
            tk.Label(f, text=name, bg=C["panel2"], fg=C["fg"], font=("Segoe UI Semibold", 9)).pack(
                anchor="w", padx=6, pady=(4, 0))
            body = tk.Label(f, text="", bg=C["panel2"], fg=C["muted"], font=("Segoe UI", 8),
                            justify="left", anchor="nw", wraplength=150)
            body.pack(fill="x", padx=6, pady=(0, 5))
            f.bind("<Configure>", lambda e, b=body: b.configure(wraplength=max(80, e.width - 14)))
            self.verdicts.append(body)
            self.set_verdict(i, None, "Pas encore consulté (BC4).")

    # ------------------------------------------------------------ verdicts
    @staticmethod
    def _short(text: str, limit: int = 70) -> str:
        text = " ".join(text.split())
        return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"

    def set_verdict(self, index: int, params: dict | None, text: str, error: bool = False,
                    fallback: bool = False):
        """Résumé de 1 à 2 lignes ; le texte complet est dans la fenêtre « Voir les verdicts »."""
        if params:
            line = (f"{params['n_ants']} fourmis · ρ {params['rho']} · α {params['alpha']} · "
                    f"β {params['beta']} · {params['cycles']} cyc.")
            color = C["accent2"]
            if fallback:
                line, color = "Secours (Python) : " + line, C["warn"]
        else:
            line = self._short(text)
            color = C["danger"] if error else C["muted"]
        self.verdicts[index].configure(text=line, fg=color)

    # ------------------------------------------------------------ actions
    MODE_LABELS = {"manuel": "réglage manuel", "optuna": "Optuna", "kerastuner": "KerasTuner",
                   "sma": "SMA (agents)", "sma (modifié)": "SMA modifié à la main"}

    def set_mode(self, mode: str):
        self.mode = mode
        key = "sma" if mode.startswith("sma") else mode
        for k, b in self.mode_buttons.items():
            b.configure(style="ModeOn.TButton" if k == key else "Mode.TButton")
        self.mode_lbl.configure(text=f"Mode : {self.MODE_LABELS.get(mode, mode)}")

    def _on_edit(self, *_):
        # Une retouche à la main d'une configuration importée : SMA → « modifié », autres → manuel.
        if self._applying or self.mode == "manuel":
            return
        self.set_mode("sma (modifié)" if self.mode.startswith("sma") else "manuel")

    def _manual(self):
        self.set_mode("manuel")
        if self.inputs:
            self.inputs[0].focus_set()

    def apply_params(self, p: dict, seed: int | None, mode: str):
        self._applying = True
        try:
            self.v_ants.set(str(p["n_ants"]))
            self.v_rho.set(str(p["rho"]))
            self.v_alpha.set(str(p["alpha"]))
            self.v_beta.set(str(p["beta"]))
            self.v_cycles.set(str(p["cycles"]))
            if seed is not None:
                self.v_seed.set(str(seed))
        finally:
            self._applying = False
        self.set_mode(mode)

    def set_default_ants(self, old_n: int | None, new_n: int):
        """Par défaut, le nombre de fourmis suit le nombre de points (sauf saisie manuelle)."""
        if old_n is None or self.v_ants.get().strip() in ("", str(old_n)):
            self.v_ants.set(str(new_n))

    def read_params(self) -> ACOParams:
        def num(var, name):
            txt = var.get().strip().replace(",", ".")
            try:
                return float(txt)
            except ValueError:
                raise ValueError(f"{name} : « {var.get()} » n'est pas un nombre.") from None

        def integer(var, name, allow_empty=False):
            txt = var.get().strip()
            if allow_empty and not txt:
                return None
            try:
                return int(txt)
            except ValueError:
                raise ValueError(f"{name} : « {txt} » n'est pas un nombre entier.") from None

        p = ACOParams(n_ants=integer(self.v_ants, "Nombre de fourmis"),
                      rho=num(self.v_rho, "ρ (évaporation)"),
                      alpha=num(self.v_alpha, "α (phéromone)"),
                      beta=num(self.v_beta, "β (proximité)"),
                      cycles=integer(self.v_cycles, "Nombre de cycles"),
                      seed=integer(self.v_seed, "Graine", allow_empty=True))
        p.validate()
        return p

    def set_running(self, running: bool):
        st = "disabled" if running else "normal"
        for w in self.inputs:
            w.configure(state=("readonly" if isinstance(w, ttk.Combobox) and not running else st))
        self.cb_n.configure(state="disabled" if running else "readonly")
        self.btn_manual.configure(state=st)
        self.btn_run.configure(state=st)
        for b in self.tool_buttons:
            b.configure(state=st if (b is not self.btn_keras or self.keras_ok) else "disabled")
        self.btn_stop.configure(state="normal" if running else "disabled")

    def set_ready(self, ready: bool):
        self._ready = ready
        st = "normal" if ready else "disabled"
        for b in (self.btn_run, *self.tool_buttons):
            b.configure(state=st if (b is not self.btn_keras or self.keras_ok) else "disabled")

    def set_keras_status(self, ok: bool, text: str):
        self.keras_ok = ok
        self.btn_keras.configure(text="BC3 KerasTuner" if ok else "BC3 KerasTuner (grisé)",
                                 state="normal" if ok and self._ready else "disabled")
        self.keras_lbl.configure(text=text, foreground=C["muted"] if ok else C["warn"])

    def set_progress(self, pct: float, text: str):
        self.progress["value"] = pct
        self.progress_lbl.configure(text=text)
