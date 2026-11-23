"""Quadrant 1 : définition du problème, paramètres AG et contrôles de simulation."""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk

from config.settings import (POPULATION_CHOICES, SELECTION_CHOICES, MUTATION_CHOICES,
                             CROSSOVER_CHOICES, GENERATION_CHOICES)
from core.genetic_algorithm import GAParams


class Quadrant1(ttk.Frame):
    def __init__(self, master, callbacks: dict, **kw):
        super().__init__(master, padding=(10, 6), **kw)
        self.cb = callbacks
        self.mode = "manuel"
        ttk.Label(self, text="1 · Problème & configuration", style="H.TLabel").grid(
            row=0, column=0, columnspan=4, sticky="w")
        self.problem_lbl = ttk.Label(self, text="Chargement des données…", style="Muted.TLabel",
                                     wraplength=560, justify="left")
        self.problem_lbl.grid(row=1, column=0, columnspan=4, sticky="w", pady=(0, 4))

        # --- Modes ---
        modes = ttk.Frame(self)
        modes.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(0, 4))
        for i in range(3):
            modes.columnconfigure(i, weight=1)
        self.btn_manual = ttk.Button(modes, text="① Réglage manuel", style="ModeOn.TButton",
                                     command=lambda: self.set_mode("manuel"))
        self.btn_manual.grid(row=0, column=0, sticky="ew", padx=(0, 3))
        self.btn_optuna = ttk.Button(modes, text="② Réglage Optuna", style="Mode.TButton",
                                     state="disabled", command=callbacks["optuna"])
        self.btn_optuna.grid(row=0, column=1, sticky="ew", padx=3)
        self.btn_agents = ttk.Button(modes, text="③ Agents IA", style="Mode.TButton",
                                     state="disabled", command=callbacks["agents"])
        self.btn_agents.grid(row=0, column=2, sticky="ew", padx=(3, 0))

        # --- Paramètres ---
        self.v_pop = tk.StringVar(value="50")
        self.v_sel = tk.StringVar(value="Tournoi")
        self.v_mut = tk.StringVar(value="0.05")
        self.v_cx = tk.StringVar(value="0.8")
        self.v_gen = tk.StringVar(value="500")
        self.v_elit = tk.BooleanVar(value=True)
        self.v_seed = tk.StringVar(value="42")
        self._applying = False
        for v in (self.v_pop, self.v_sel, self.v_mut, self.v_cx, self.v_gen, self.v_elit, self.v_seed):
            v.trace_add("write", self._on_manual_edit)

        rows = [
            ("Population", ttk.Combobox(self, textvariable=self.v_pop, state="readonly", width=10,
                                        values=[str(v) for v in POPULATION_CHOICES])),
            ("Sélection", ttk.Combobox(self, textvariable=self.v_sel, state="readonly", width=14,
                                       values=list(SELECTION_CHOICES))),
            ("Mutation (libre)", ttk.Combobox(self, textvariable=self.v_mut, width=10,
                                              values=[f"{v:.2f}" for v in MUTATION_CHOICES])),
            ("Croisement", ttk.Combobox(self, textvariable=self.v_cx, state="readonly",
                                                width=10, values=[str(v) for v in CROSSOVER_CHOICES])),
            ("Générations", ttk.Combobox(self, textvariable=self.v_gen, state="readonly", width=10,
                                         values=[str(v) for v in GENERATION_CHOICES])),
            ("Graine", ttk.Entry(self, textvariable=self.v_seed, width=12)),
        ]
        self.inputs = []
        for i, (label, w) in enumerate(rows):
            r, c = 3 + i // 2, (i % 2) * 2
            ttk.Label(self, text=label).grid(row=r, column=c, sticky="w", pady=2, padx=(0, 6))
            w.grid(row=r, column=c + 1, sticky="ew", pady=2, padx=(0, 10))
            self.inputs.append(w)
        cb = ttk.Checkbutton(self, text="Élitisme (2 meilleurs)", variable=self.v_elit)
        cb.grid(row=6, column=0, columnspan=2, sticky="w")
        self.inputs.append(cb)
        ttk.Label(self, text="Croisement OX · mutation inversion · départ fixe",
                  style="Muted.TLabel").grid(row=6, column=2, columnspan=2, sticky="w")

        # --- Contrôles ---
        ctl = ttk.Frame(self)
        ctl.grid(row=8, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        for i in range(5):
            ctl.columnconfigure(i, weight=1)
        self.btn_run = ttk.Button(ctl, text="▶ LANCER LA SIMULATION", style="Accent.TButton",
                                  command=callbacks["run"], state="disabled")
        self.btn_run.grid(row=0, column=0, columnspan=5, sticky="ew", pady=(0, 3))
        self.btn_pause = ttk.Button(ctl, text="⏸ Pause", command=callbacks["pause"], state="disabled")
        self.btn_step = ttk.Button(ctl, text="⏭ Pas à pas", command=callbacks["step"], state="disabled")
        self.btn_stop = ttk.Button(ctl, text="■ Arrêter", style="Danger.TButton",
                                   command=callbacks["stop"], state="disabled")
        self.btn_compare = ttk.Button(ctl, text="⚖ Comparer", command=callbacks["compare"], state="disabled")
        self.btn_export = ttk.Button(ctl, text="Exports (V2)", state="disabled")
        for i, b in enumerate((self.btn_pause, self.btn_step, self.btn_stop, self.btn_compare, self.btn_export)):
            b.grid(row=1, column=i, sticky="ew", padx=(0 if i == 0 else 3, 0))

        self.progress = ttk.Progressbar(self, maximum=100, mode="determinate")
        self.progress.grid(row=9, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        self.progress_lbl = ttk.Label(self, text="0 %", style="Muted.TLabel")
        self.progress_lbl.grid(row=10, column=0, columnspan=4, sticky="w")
        for c in (1, 3):
            self.columnconfigure(c, weight=1)

    # ---------- modes de configuration ----------
    def set_mode(self, mode: str):
        """mode : 'manuel', 'optuna', 'agents' ou 'agents (modifié)' (enregistré avec les résultats)."""
        self.mode = mode
        self.btn_manual.configure(style="ModeOn.TButton" if mode == "manuel" else "Mode.TButton")
        self.btn_optuna.configure(style="ModeOn.TButton" if mode == "optuna" else "Mode.TButton")
        self.btn_agents.configure(style="ModeOn.TButton" if mode.startswith("agents") else "Mode.TButton")

    def _on_manual_edit(self, *_):
        # Une modification à la main d'une configuration importée repasse en mode manuel.
        if not self._applying and self.mode != "manuel":
            self.set_mode("manuel")

    def apply_params(self, params: dict, seed: int | None, mode: str):
        """Remplit le formulaire avec une configuration (ex. recommandation Optuna)."""
        self._applying = True
        try:
            self._fill(params, seed)
        finally:
            self._applying = False
        self.set_mode(mode)

    def _fill(self, params: dict, seed: int | None):
        inv_sel = {v: k for k, v in SELECTION_CHOICES.items()}
        self.v_pop.set(str(params["population_size"]))
        self.v_sel.set(inv_sel[params["selection_method"]])
        self.v_mut.set(f"{params['mutation_rate']:.4f}")
        self.v_cx.set(str(params["crossover_rate"]))
        self.v_gen.set(str(params["generations"]))
        self.v_elit.set(True)
        self.v_seed.set("" if seed is None else str(seed))

    def read_params(self) -> GAParams:
        try:
            mut = float(self.v_mut.get().replace(",", "."))
        except ValueError:
            raise ValueError("Taux de mutation invalide : saisissez un nombre entre 0 et 1.")
        seed_txt = self.v_seed.get().strip()
        try:
            seed = int(seed_txt) if seed_txt else None
        except ValueError:
            raise ValueError("La graine doit être un entier (ou vide pour aléatoire).")
        p = GAParams(population_size=int(self.v_pop.get()),
                     selection_method=SELECTION_CHOICES[self.v_sel.get()],
                     mutation_rate=mut, crossover_rate=float(self.v_cx.get()),
                     generations=int(self.v_gen.get()), elitism=self.v_elit.get(), seed=seed)
        p.validate()
        return p

    def set_running(self, running: bool, paused: bool = False):
        st = "disabled" if running else "normal"
        for w in self.inputs:
            w.configure(state=("readonly" if isinstance(w, ttk.Combobox) and w is not self.inputs[2]
                               and not running else st))
        self.btn_run.configure(state=st)
        self.btn_optuna.configure(state=st)
        self.btn_agents.configure(state=st)
        self.btn_compare.configure(state=st)
        self.btn_manual.configure(state=st)
        self.btn_pause.configure(state="normal" if running else "disabled",
                                 text="▶ Reprise" if paused else "⏸ Pause")
        self.btn_stop.configure(state="normal" if running else "disabled")

    def set_progress(self, pct: float, text: str):
        self.progress["value"] = pct
        self.progress_lbl.configure(text=text)
