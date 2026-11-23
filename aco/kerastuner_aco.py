"""BC3 — Réglage de l'ACO avec le moteur de recherche de KerasTuner (oracle RandomSearch).

Aucun réseau de neurones : un Tuner personnalisé surcharge run_trial, qui lit les hyperparamètres
proposés par l'oracle, exécute l'ACO sur plusieurs graines et renvoie la distance moyenne (objectif
unique, minimisé ; les oracles KerasTuner sont mono-objectif). Le temps moyen est enregistré comme
métrique secondaire.

Keras 3 exige un backend à l'import. KERAS_BACKEND est fixé ICI, avant tout import de keras : valeur du
.env si présente, sinon le premier installé parmi torch (PyTorch CPU), jax, tensorflow. Le backend ne sert
qu'à importer KerasTuner : aucun réseau n'est créé ni entraîné.
"""
from __future__ import annotations
import importlib.util
import os
import threading
from datetime import datetime
from typing import Callable

import config.settings  # noqa: F401  (charge .env, dont un éventuel KERAS_BACKEND)
from aco.ant_system import RunControl
from aco.optuna_aco import TuningResult, save_result
from aco.settings_aco import CYCLE_CHOICES
from aco.tuning import ANTS_STEP, DEFAULT_EVAL_SEEDS, EvalRecord, Stopped, evaluate, fmt_params, recommend
from config.settings import RESULTS_DIR

DEFAULT_TRIALS = 15
MAX_TRIALS = 20
ORACLES = ("RandomSearch",)
BACKENDS = ("torch", "jax", "tensorflow")   # ordre de préférence

# Espace de recherche de BC3
KT_ANTS = (20, 80)
KT_RHO = (0.05, 0.9)
KT_ALPHA = (0.5, 3.0)
KT_BETA = (1.0, 6.0)


def kt_space_description() -> str:
    return (f"fourmis {KT_ANTS[0]}–{KT_ANTS[1]} (pas {ANTS_STEP}) · ρ {KT_RHO[0]}–{KT_RHO[1]} · "
            f"α {KT_ALPHA[0]}–{KT_ALPHA[1]} · β {KT_BETA[0]}–{KT_BETA[1]} · cycles {list(CYCLE_CHOICES)}")


def choose_backend() -> str | None:
    """Fixe KERAS_BACKEND avant l'import de keras : .env, sinon torch → jax → tensorflow installé."""
    current = os.environ.get("KERAS_BACKEND", "").strip()
    if current:
        return current
    for b in BACKENDS:
        if importlib.util.find_spec(b) is not None:
            os.environ["KERAS_BACKEND"] = b
            return b
    return None


class KerasTunerUnavailable(RuntimeError):
    pass


def _import_kt():
    backend = choose_backend()
    try:
        import keras_tuner as kt
    except Exception as e:  # noqa: BLE001  (bibliothèque ou backend Keras absent)
        raise KerasTunerUnavailable(
            "KerasTuner n'est pas utilisable dans ce .venv.\n\n"
            f"Détail : {type(e).__name__} : {e}\n\n"
            f"Backend Keras essayé : {backend or 'aucun installé'}. Keras 3 a besoin de PyTorch, JAX ou "
            "TensorFlow. Installation conseillée :\n"
            ".venv\\Scripts\\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu") from e
    return kt


def check_available() -> str:
    return _import_kt().__version__


class KerasTunerACO:
    method = "KerasTuner (RandomSearch)"

    def __init__(self, distances_m, n_trials: int = DEFAULT_TRIALS, seed: int = 42,
                 n_eval_seeds: int = DEFAULT_EVAL_SEEDS, oracle: str = "RandomSearch", start: int = 0):
        if not 1 <= n_trials <= MAX_TRIALS:
            raise ValueError(f"Le nombre de trials doit être compris entre 1 et {MAX_TRIALS}.")
        if n_eval_seeds < 1:
            raise ValueError("Il faut au moins une graine d'évaluation par trial.")
        if oracle not in ORACLES:
            raise ValueError(f"Oracle inconnu : {oracle}.")
        self.kt = _import_kt()
        self.d, self.n, self.start = distances_m, len(distances_m), start
        self.n_trials, self.seed = n_trials, seed
        self.seeds = [seed + i for i in range(n_eval_seeds)]
        self.control = RunControl()
        self._stop = threading.Event()
        self.records: list[EvalRecord] = []

    def stop(self):
        self._stop.set()
        self.control.stop()

    def run(self, on_trial: Callable | None = None, on_trial_start: Callable | None = None) -> TuningResult:
        kt, outer = self.kt, self
        lo, hi = KT_ANTS

        class ACOTuner(kt.Tuner):
            def run_trial(self, trial, *args, **kwargs):
                if outer._stop.is_set():
                    raise kt.errors.FatalError("Recherche arrêtée par l'utilisateur.")
                hp = trial.hyperparameters
                params = {"n_ants": hp.Int("n_ants", lo, hi, step=ANTS_STEP),
                          "rho": round(hp.Float("rho", *KT_RHO), 3),
                          "alpha": round(hp.Float("alpha", *KT_ALPHA), 2),
                          "beta": round(hp.Float("beta", *KT_BETA), 2),
                          "cycles": hp.Choice("cycles", list(CYCLE_CHOICES))}
                number = len(outer.records)
                if on_trial_start:
                    on_trial_start(number, params)
                try:
                    rec = evaluate(outer.d, params, outer.seeds, number, outer.control, outer.start)
                except Stopped:
                    raise kt.errors.FatalError("Recherche arrêtée par l'utilisateur.")
                rec.extra = {"keras_trial_id": trial.trial_id}
                outer.records.append(rec)
                recommend(outer.records)  # marques Pareto pour le graphique
                if on_trial:
                    on_trial(rec, list(outer.records))
                return {"distance_km": rec.distance_km, "time_s": rec.time_s}

        oracle = kt.oracles.RandomSearchOracle(objective=kt.Objective("distance_km", direction="min"),
                                               max_trials=self.n_trials, seed=self.seed)
        workdir = RESULTS_DIR / "kerastuner_aco"
        tuner = ACOTuner(oracle=oracle, directory=str(workdir),
                         project_name=f"aco_{datetime.now():%Y%m%d_%H%M%S}", overwrite=True)
        try:
            tuner.search(verbose=0)
        except kt.errors.FatalError:
            if not self._stop.is_set():
                raise
        best = min(self.records, key=lambda r: r.distance_km) if self.records else None
        text = "Aucun trial terminé."
        if best:
            text = (f"Trial #{best.number} : {fmt_params(best.params)}.\n"
                    f"Meilleure distance moyenne de la recherche sur {len(best.seeds)} graine(s) {best.seeds} : "
                    f"{best.distance_km:.2f} km (écart-type {best.std_distance_km:.2f} km) en {best.time_s:.2f} s.\n"
                    f"Critère KerasTuner : distance moyenne minimale (objectif unique). Le temps est affiché "
                    f"mais n'intervient pas dans le choix.")
        res = TuningResult(self.method, self.records, best, text, self.seeds, stopped=self._stop.is_set(),
                           info={"oracle": "RandomSearchOracle", "objective": "distance_km moyenne (min)",
                                 "oracle_seed": self.seed, "keras_backend": os.environ.get("KERAS_BACKEND")})
        res.saved_to = save_result(res, "aco_kerastuner", self.n)
        return res
