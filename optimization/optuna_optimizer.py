"""Optimisation multiobjectif des hyperparamètres de l'AG avec Optuna (TPE multiobjectif = MOTPE).

Objectifs, minimisés séparément (jamais additionnés) :
  1. distance totale de la meilleure tournée (km) ;
  2. temps d'exécution de l'AG (s).

Conditions comparables : même matrice routière et mêmes graines d'évaluation pour chaque trial,
même graine pour le sampler. Seuls les hyperparamètres changent.

Évaluation robuste : chaque trial exécute l'AG sur plusieurs graines (3 par défaut) et Optuna
reçoit la MOYENNE de la distance et du temps. Avec une seule graine, l'étude sélectionne la
configuration qui a eu le tirage le plus favorable (biais du gagnant) : la recommandation de
l'étude à 1 graine (110,61 km sur la graine 42) est tombée à 112,22 km en moyenne sur 10 autres
graines, avec un écart-type de 7,35 km.
"""
from __future__ import annotations
import json
import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Callable

import statistics

import optuna

from config.settings import (POPULATION_CHOICES, CROSSOVER_CHOICES, GENERATION_CHOICES,
                             MUTATION_CHOICES, RESULTS_DIR)
from core.genetic_algorithm import GeneticAlgorithm, GAParams, RunControl
from core.tsp_problem import TSPProblem

log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

MUTATION_RANGE = (min(MUTATION_CHOICES), max(MUTATION_CHOICES))  # 0.01 – 0.20
DEFAULT_EVAL_SEEDS = 3


@dataclass
class TrialRecord:
    number: int
    params: dict          # population_size, selection_method, mutation_rate, crossover_rate, generations
    distance_km: float          # MOYENNE sur les graines d'évaluation
    time_s: float               # MOYENNE sur les graines d'évaluation
    best_generation: int        # moyenne (arrondie) de la génération du meilleur
    pareto: bool = False
    seeds: list[int] = field(default_factory=list)
    distances_km: list[float] = field(default_factory=list)   # détail par graine
    times_s: list[float] = field(default_factory=list)
    std_distance_km: float = 0.0                              # écart-type d'échantillon (n-1)


@dataclass
class Recommendation:
    trial: TrialRecord
    explanation: str
    method: str = "Point du front de Pareto le plus proche du point idéal (objectifs normalisés min-max)"


@dataclass
class OptunaResult:
    trials: list[TrialRecord]
    pareto: list[TrialRecord]
    recommendation: Recommendation | None
    seed: int
    n_startup_trials: int
    eval_seeds: list[int] = field(default_factory=list)
    stopped: bool = False
    saved_to: str | None = field(default=None)


def startup_trials(n_trials: int) -> int:
    """Nombre de trials aléatoires initiaux avant que TPE ne guide la recherche.

    La valeur par défaut d'Optuna (10) rendrait une étude de 10 trials purement aléatoire.
    """
    return max(2, min(10, n_trials // 3))


def suggest_params(trial: optuna.Trial, seed: int) -> GAParams:
    return GAParams(
        population_size=trial.suggest_categorical("population_size", list(POPULATION_CHOICES)),
        selection_method=trial.suggest_categorical("selection_method", ["tournament", "roulette"]),
        # arrondi à 4 décimales : valeur affichée = valeur testée, donc simulation reproductible à l'identique
        mutation_rate=round(trial.suggest_float("mutation_rate", *MUTATION_RANGE, log=True), 4),
        crossover_rate=trial.suggest_categorical("crossover_rate", list(CROSSOVER_CHOICES)),
        generations=trial.suggest_categorical("generations", list(GENERATION_CHOICES)),
        elitism=True, seed=seed,
    )


def pareto_front(records: list[TrialRecord]) -> list[TrialRecord]:
    """Solutions non dominées (minimisation distance et temps)."""
    front = []
    for a in records:
        dominated = any(
            (b.distance_km <= a.distance_km and b.time_s <= a.time_s)
            and (b.distance_km < a.distance_km or b.time_s < a.time_s)
            for b in records)
        if not dominated:
            front.append(a)
    return sorted(front, key=lambda r: r.distance_km)


def recommend(front: list[TrialRecord], all_records: list[TrialRecord]) -> Recommendation | None:
    """Compromis : point du front le plus proche de l'idéal (meilleure distance, meilleur temps).

    Les deux objectifs sont normalisés min-max sur l'ensemble des trials, pour qu'aucune
    unité (km ou s) ne domine l'autre. Poids égaux, distance euclidienne au point (0, 0).
    """
    if not front:
        return None
    dmin = min(r.distance_km for r in all_records)
    dmax = max(r.distance_km for r in all_records)
    tmin = min(r.time_s for r in all_records)
    tmax = max(r.time_s for r in all_records)

    def norm(v, lo, hi):
        return 0.0 if hi - lo < 1e-12 else (v - lo) / (hi - lo)

    def score(r):
        return (norm(r.distance_km, dmin, dmax) ** 2 + norm(r.time_s, tmin, tmax) ** 2) ** 0.5

    best = min(front, key=score)
    best_dist = min(front, key=lambda r: r.distance_km)
    fastest = min(front, key=lambda r: r.time_s)
    p = best.params
    sel = "tournoi" if p["selection_method"] == "tournament" else "roulette"
    lines = [
        f"Trial #{best.number} : population {p['population_size']}, sélection {sel}, "
        f"mutation {p['mutation_rate']:.3f}, croisement {p['crossover_rate']}, "
        f"{p['generations']} générations.",
        (f"Résultat moyen sur {len(best.seeds)} graines {best.seeds} : {best.distance_km:.2f} km "
         f"(écart-type {best.std_distance_km:.2f} km, de {min(best.distances_km):.2f} à "
         f"{max(best.distances_km):.2f} km) en {best.time_s:.2f} s."
         if len(best.seeds) > 1 else f"Résultat : {best.distance_km:.2f} km en {best.time_s:.2f} s."),
    ]
    if len(front) == 1:
        lines.append("C'est la seule solution non dominée : elle est à la fois la plus courte "
                     "et la plus rapide parmi les trials.")
    else:
        if best is best_dist:
            lines.append(f"C'est la meilleure distance du front. La solution la plus rapide "
                         f"(#{fastest.number}) gagne {best.time_s - fastest.time_s:.2f} s mais "
                         f"allonge le trajet de {fastest.distance_km - best.distance_km:.2f} km.")
        elif best is fastest:
            lines.append(f"C'est la solution la plus rapide du front. La meilleure distance "
                         f"(#{best_dist.number}) gagne {best.distance_km - best_dist.distance_km:.2f} km "
                         f"mais demande {best_dist.time_s - best.time_s:.2f} s de plus.")
        else:
            lines.append(
                f"Par rapport à la meilleure distance (#{best_dist.number}, {best_dist.distance_km:.2f} km), "
                f"elle coûte +{best.distance_km - best_dist.distance_km:.2f} km mais économise "
                f"{best_dist.time_s - best.time_s:.2f} s. Par rapport à la plus rapide "
                f"(#{fastest.number}, {fastest.time_s:.2f} s), elle gagne "
                f"{fastest.distance_km - best.distance_km:.2f} km pour {best.time_s - fastest.time_s:.2f} s de plus.")
    lines.append("Critère : distance et temps normalisés (min-max sur tous les trials), poids égaux, "
                 "point du front le plus proche de l'idéal. Les km et les secondes ne sont jamais additionnés bruts.")
    return Recommendation(best, "\n".join(lines))


class OptunaOptimizer:
    def __init__(self, problem: TSPProblem, n_trials: int, seed: int = 42,
                 n_eval_seeds: int = DEFAULT_EVAL_SEEDS):
        if n_trials < 1:
            raise ValueError("Le nombre de trials doit être >= 1.")
        if n_eval_seeds < 1:
            raise ValueError("Au moins une graine d'évaluation par trial.")
        self.problem = problem
        self.n_trials = n_trials
        self.seed = seed
        # Mêmes graines pour TOUS les trials : les configurations sont comparées sur les mêmes tirages.
        self.eval_seeds = [seed + i for i in range(n_eval_seeds)]
        self.n_startup = startup_trials(n_trials)
        self._stop = threading.Event()
        self._control = RunControl()
        self.records: list[TrialRecord] = []

    def stop(self):
        """Arrête l'étude ; le trial en cours est interrompu et exclu des résultats."""
        self._stop.set()
        self._control.stop()

    def run(self, on_trial: Callable[[TrialRecord, list[TrialRecord]], None] | None = None,
            on_trial_start: Callable[[int, dict], None] | None = None) -> OptunaResult:
        sampler = optuna.samplers.TPESampler(seed=self.seed, n_startup_trials=self.n_startup,
                                             multivariate=True)
        study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler,
                                    study_name="tsp40_ga_motpe")

        def objective(trial: optuna.Trial):
            params = suggest_params(trial, self.seed)
            if on_trial_start:
                on_trial_start(trial.number, asdict(params))
            dists, times, gens = [], [], []
            for s in self.eval_seeds:
                params.seed = s
                r = GeneticAlgorithm(self.problem, params).run(control=self._control)
                if r.stopped:
                    raise optuna.TrialPruned("Trial interrompu par l'utilisateur")
                dists.append(r.best_distance_m / 1000)
                times.append(r.elapsed_s)
                gens.append(r.best_generation)
            rec = TrialRecord(trial.number, {k: asdict(params)[k] for k in (
                "population_size", "selection_method", "mutation_rate", "crossover_rate", "generations")},
                statistics.fmean(dists), statistics.fmean(times), round(statistics.fmean(gens)),
                seeds=list(self.eval_seeds), distances_km=dists, times_s=times,
                std_distance_km=statistics.stdev(dists) if len(dists) > 1 else 0.0)
            self.records.append(rec)
            front_ids = {x.number for x in pareto_front(self.records)}
            for x in self.records:
                x.pareto = x.number in front_ids
            if on_trial:
                on_trial(rec, self.records)
            return rec.distance_km, rec.time_s

        def stop_cb(study_, _trial):
            if self._stop.is_set():
                study_.stop()

        study.optimize(objective, n_trials=self.n_trials, callbacks=[stop_cb])
        front = pareto_front(self.records)
        result = OptunaResult(self.records, front, recommend(front, self.records) if self.records else None,
                              self.seed, self.n_startup, eval_seeds=list(self.eval_seeds),
                              stopped=self._stop.is_set())
        result.saved_to = self._save(result)
        return result

    def _save(self, res: OptunaResult) -> str | None:
        try:
            path = RESULTS_DIR / f"optuna_{datetime.now():%Y%m%d_%H%M%S}.json"
            data = {"date": datetime.now().isoformat(timespec="seconds"), "sampler": "TPESampler (MOTPE)",
                    "objectives": ["distance_km (min)", "time_s (min)"], "seed": res.seed,
                    "n_startup_trials": res.n_startup_trials, "stopped": res.stopped,
                    "eval_seeds": res.eval_seeds,
                    "aggregation": "moyenne de la distance et du temps sur les graines d'évaluation",
                    "trials": [asdict(t) for t in res.trials],
                    "pareto": [t.number for t in res.pareto],
                    "recommendation": None if not res.recommendation else {
                        "trial": res.recommendation.trial.number,
                        "params": res.recommendation.trial.params,
                        "explanation": res.recommendation.explanation,
                        "method": res.recommendation.method}}
            path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            return str(path)
        except OSError as e:
            log.warning("Sauvegarde Optuna impossible : %s", e)
            return None
