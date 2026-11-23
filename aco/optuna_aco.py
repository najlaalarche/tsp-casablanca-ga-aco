"""BC2 — Réglage de l'ACO par Optuna (TPE multiobjectif : distance moyenne et temps moyen).

Pourquoi deux objectifs : à graine égale, le meilleur tour global ne peut que s'améliorer quand on
ajoute des cycles (les premiers cycles sont identiques). Avec la distance seule, Optuna choisirait
toujours 1000 cycles et le maximum de fourmis, sans regarder le coût. Le temps d'exécution rend ce
compromis visible ; la recommandation est le point du front de Pareto le plus proche de l'idéal.
"""
from __future__ import annotations
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import optuna

from aco.ant_system import RunControl
from aco.settings_aco import CYCLE_CHOICES
from aco.tuning import (ALPHA_RANGE, ANTS_STEP, BETA_RANGE, DEFAULT_EVAL_SEEDS, MAX_TRIALS, RHO_RANGE,
                        EvalRecord, Stopped, ants_range, evaluate, recommend, space_description)
from config.settings import RESULTS_DIR
from optimization.optuna_optimizer import startup_trials  # réutilisé de l'application AG

log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class TuningResult:
    method: str
    records: list[EvalRecord]
    best: EvalRecord | None            # configuration recommandée
    explanation: str
    seeds: list[int]
    stopped: bool = False
    saved_to: str | None = None
    info: dict = field(default_factory=dict)


def suggest(trial: optuna.Trial, n_points: int) -> dict:
    lo, hi = ants_range(n_points)
    return {
        "n_ants": trial.suggest_int("n_ants", lo, hi, step=ANTS_STEP),
        # arrondis : la valeur affichée est exactement la valeur testée (reproductible)
        "rho": round(trial.suggest_float("rho", *RHO_RANGE), 3),
        "alpha": round(trial.suggest_float("alpha", *ALPHA_RANGE), 2),
        "beta": round(trial.suggest_float("beta", *BETA_RANGE), 2),
        "cycles": trial.suggest_categorical("cycles", list(CYCLE_CHOICES)),
    }


def save_result(res: TuningResult, prefix: str, n_points: int) -> str | None:
    try:
        path = RESULTS_DIR / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}.json"
        data = {"date": datetime.now().isoformat(timespec="seconds"), "method": res.method,
                "n_points": n_points, "eval_seeds": res.seeds, "stopped": res.stopped,
                "space": space_description(n_points), **res.info,
                "aggregation": "moyenne sur les graines d'évaluation (mêmes graines pour chaque trial)",
                "trials": [r.to_dict() for r in res.records],
                "best": None if res.best is None else {"trial": res.best.number, "params": res.best.params},
                "explanation": res.explanation}
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        return str(path)
    except OSError as e:
        log.warning("Sauvegarde impossible : %s", e)
        return None


class OptunaACO:
    method = "Optuna (TPE multiobjectif)"

    def __init__(self, distances_m, n_trials: int, seed: int = 42,
                 n_eval_seeds: int = DEFAULT_EVAL_SEEDS, start: int = 0):
        if not 1 <= n_trials <= MAX_TRIALS:
            raise ValueError(f"Le nombre de trials doit être compris entre 1 et {MAX_TRIALS}.")
        if n_eval_seeds < 1:
            raise ValueError("Il faut au moins une graine d'évaluation par trial.")
        self.d = distances_m
        self.n = len(distances_m)
        self.n_trials = n_trials
        self.seed = seed
        self.seeds = [seed + i for i in range(n_eval_seeds)]
        self.start = start
        self.n_startup = startup_trials(n_trials)
        self.control = RunControl()
        self._stop = threading.Event()
        self.records: list[EvalRecord] = []

    def stop(self):
        self._stop.set()
        self.control.stop()

    def run(self, on_trial: Callable[[EvalRecord, list[EvalRecord]], None] | None = None,
            on_trial_start: Callable[[int, dict], None] | None = None) -> TuningResult:
        sampler = optuna.samplers.TPESampler(seed=self.seed, n_startup_trials=self.n_startup,
                                             multivariate=True)
        study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler,
                                    study_name="tsp_aco_motpe")

        def objective(trial):
            params = suggest(trial, self.n)
            if on_trial_start:
                on_trial_start(trial.number, params)
            try:
                rec = evaluate(self.d, params, self.seeds, trial.number, self.control, self.start)
            except Stopped:
                raise optuna.TrialPruned("interrompu par l'utilisateur")
            self.records.append(rec)
            recommend(self.records)  # met à jour les marques Pareto
            if on_trial:
                on_trial(rec, list(self.records))
            return rec.distance_km, rec.time_s

        def stop_cb(st, _t):
            if self._stop.is_set():
                st.stop()

        study.optimize(objective, n_trials=self.n_trials, callbacks=[stop_cb])
        reco = recommend(self.records)
        res = TuningResult(self.method, self.records, reco[0] if reco else None,
                           reco[1] if reco else "Aucun trial terminé.", self.seeds,
                           stopped=self._stop.is_set(),
                           info={"sampler": "TPESampler multivariate (MOTPE)",
                                 "objectives": ["distance_km moyenne (min)", "time_s moyen (min)"],
                                 "sampler_seed": self.seed, "n_startup_trials": self.n_startup})
        res.saved_to = save_result(res, "aco_optuna", self.n)
        return res
