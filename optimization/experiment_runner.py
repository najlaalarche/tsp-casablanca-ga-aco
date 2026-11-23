"""Comparaison des trois modes (sections 8 et 9) : répétitions contrôlées et statistiques.

Règles :
- chaque mode utilise une configuration RÉELLE : formulaire manuel, recommandation d'une étude
  Optuna enregistrée, configuration d'une session d'agents enregistrée ;
- même matrice OSRM et mêmes graines pour tous les modes (graine i identique pour les 3 modes) ;
- exécutions entrelacées (graine 1 : manuel, Optuna, agents ; graine 2 : …) pour que les
  variations de charge machine pèsent de la même façon sur les temps des trois modes ;
- aucune valeur n'est inventée : un mode sans configuration disponible est absent du tableau.
"""
from __future__ import annotations
import csv
import json
import logging
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable

from config.settings import RESULTS_DIR
from core.genetic_algorithm import GeneticAlgorithm, GAParams, RunControl
from core.tsp_problem import TSPProblem
from core.validation import is_valid_tour

log = logging.getLogger(__name__)
MODES = ("manuel", "optuna", "agents")
GA_KEYS = ("population_size", "selection_method", "mutation_rate", "crossover_rate", "generations", "elitism")


@dataclass
class ModeConfig:
    mode: str
    params: dict          # paramètres AG (sans graine)
    source: str           # provenance lisible (fichier, trial, décision…)


@dataclass
class RunRecord:
    mode: str
    rep: int
    seed: int
    distance_km: float
    time_s: float
    best_generation: int
    generations: int
    fitness: float
    improvement_pct: float
    tour: list[int]


@dataclass
class ModeStats:
    mode: str
    n: int
    best_km: float
    mean_km: float
    worst_km: float
    std_km: float          # écart-type d'échantillon (ddof = 1)
    mean_time_s: float
    std_time_s: float
    mean_best_generation: float
    mean_improvement_pct: float
    best_fitness: float


@dataclass
class ComparisonResult:
    configs: dict[str, ModeConfig]
    seeds: list[int]
    runs: list[RunRecord]
    stats: dict[str, ModeStats]
    reference: dict[str, RunRecord]   # exécution unique de référence (1re graine) — section 8
    stopped: bool = False
    saved_json: str | None = None
    saved_csv: str | None = None
    date: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


# ---------------------------------------------------------------- sources
def clean_params(p: dict) -> dict:
    out = {k: p[k] for k in GA_KEYS if k in p}
    out.setdefault("elitism", True)
    missing = [k for k in GA_KEYS if k not in out]
    if missing:
        raise ValueError(f"Paramètres incomplets : {missing}")
    return out


def optuna_sources(results_dir: Path = RESULTS_DIR) -> list[ModeConfig]:
    """Recommandations des études Optuna enregistrées (plus récente d'abord)."""
    out = []
    for f in sorted(results_dir.glob("optuna_*.json"), reverse=True):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            reco = d.get("recommendation")
            if not reco:
                continue
            seeds = d.get("eval_seeds") or [d.get("seed")]
            evals = (f"moyenne sur {len(seeds)} graines {seeds}" if len(seeds) > 1
                     else f"1 seule graine ({seeds[0]})")
            out.append(ModeConfig("optuna", clean_params(reco["params"]),
                                  f"{f.name} — trial #{reco['trial']} recommandé "
                                  f"({len(d.get('trials', []))} trials, {evals})"))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
            log.warning("Étude Optuna illisible %s : %s", f.name, e)
    return out


def agents_sources(results_dir: Path = RESULTS_DIR) -> list[ModeConfig]:
    """Configurations des sessions d'agents : finale de l'arbitre, et appliquée si différente."""
    out = []
    for f in sorted(results_dir.glob("agents_*.json"), reverse=True):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            arb = d.get("arbiter") or {}
            cfg = arb.get("config")
            if cfg and not arb.get("error"):
                out.append(ModeConfig("agents", clean_params(cfg),
                                      f"{f.name} — finale de l'arbitre ({d.get('model')}, "
                                      f"décision humaine : {d.get('decision')})"))
            applied = d.get("applied_params")
            if applied and (not cfg or clean_params(applied) != clean_params(cfg)):
                out.append(ModeConfig("agents", clean_params(applied),
                                      f"{f.name} — appliquée après modification humaine"))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
            log.warning("Session agents illisible %s : %s", f.name, e)
    return out


# ------------------------------------------------------------- statistiques
def compute_stats(mode: str, runs: list[RunRecord]) -> ModeStats:
    d = [r.distance_km for r in runs]
    t = [r.time_s for r in runs]
    return ModeStats(
        mode=mode, n=len(runs), best_km=min(d), mean_km=statistics.fmean(d), worst_km=max(d),
        std_km=statistics.stdev(d) if len(d) > 1 else 0.0,
        mean_time_s=statistics.fmean(t), std_time_s=statistics.stdev(t) if len(t) > 1 else 0.0,
        mean_best_generation=statistics.fmean(r.best_generation for r in runs),
        mean_improvement_pct=statistics.fmean(r.improvement_pct for r in runs),
        best_fitness=max(r.fitness for r in runs))


# ------------------------------------------------------------------ runner
class ExperimentRunner:
    def __init__(self, problem: TSPProblem, configs: dict[str, ModeConfig], n_reps: int = 10,
                 base_seed: int = 1000):
        if n_reps < 1:
            raise ValueError("Au moins une répétition.")
        if not configs:
            raise ValueError("Aucun mode à comparer.")
        for c in configs.values():
            GAParams(**c.params).validate()
        self.problem, self.configs = problem, configs
        self.seeds = [base_seed + i for i in range(n_reps)]
        self.control = RunControl()

    @property
    def total_runs(self) -> int:
        return len(self.seeds) * len(self.configs)

    def stop(self):
        self.control.stop()

    def run(self, on_run: Callable[[RunRecord, int, int], None] | None = None,
            on_start: Callable[[str, int, int], None] | None = None) -> ComparisonResult:
        runs: list[RunRecord] = []
        done, stopped = 0, False
        for rep, seed in enumerate(self.seeds, start=1):
            for mode in MODES:
                if mode not in self.configs:
                    continue
                if on_start:
                    on_start(mode, rep, seed)
                params = GAParams(**self.configs[mode].params, seed=seed)
                r = GeneticAlgorithm(self.problem, params).run(control=self.control)
                if r.stopped:
                    stopped = True
                    break
                assert is_valid_tour(r.best_tour, self.problem.n, self.problem.start)
                rec = RunRecord(mode, rep, seed, r.best_distance_m / 1000, r.elapsed_s, r.best_generation,
                                r.generations_run, r.best_fitness, r.improvement_pct, r.best_tour)
                runs.append(rec)
                done += 1
                if on_run:
                    on_run(rec, done, self.total_runs)
            if stopped:
                break
        # Statistiques uniquement sur des répétitions complètes (mêmes graines pour chaque mode).
        complete = {r.seed for r in runs if all(
            any(x.seed == r.seed and x.mode == m for x in runs) for m in self.configs)}
        kept = [r for r in runs if r.seed in complete]
        stats = {m: compute_stats(m, [r for r in kept if r.mode == m])
                 for m in self.configs if any(r.mode == m for r in kept)}
        ref = {m: next(r for r in kept if r.mode == m and r.seed == self.seeds[0])
               for m in stats if any(r.seed == self.seeds[0] for r in kept)}
        res = ComparisonResult(self.configs, [s for s in self.seeds if s in complete], kept, stats, ref,
                               stopped=stopped)
        self._save(res)
        return res

    def _save(self, res: ComparisonResult):
        stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
        try:
            jpath = RESULTS_DIR / f"comparison_{stamp}.json"
            jpath.write_text(json.dumps({
                "date": res.date, "seeds": res.seeds, "stopped": res.stopped,
                "configs": {m: asdict(c) for m, c in res.configs.items()},
                "stats": {m: asdict(s) for m, s in res.stats.items()},
                "reference_single_run": {m: asdict(r) for m, r in res.reference.items()},
                "runs": [asdict(r) for r in res.runs]}, ensure_ascii=False, indent=1), encoding="utf-8")
            res.saved_json = str(jpath)
            cpath = RESULTS_DIR / f"comparison_{stamp}_runs.csv"
            with cpath.open("w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f, delimiter=";")
                w.writerow(["mode", "repetition", "graine", "distance_km", "temps_s", "generation_meilleur",
                            "generations", "fitness", "amelioration_pct", "ordre_visite"])
                for r in res.runs:
                    w.writerow([r.mode, r.rep, r.seed, f"{r.distance_km:.4f}", f"{r.time_s:.4f}",
                                r.best_generation, r.generations, f"{r.fitness:.6f}",
                                f"{r.improvement_pct:.2f}", "-".join(map(str, r.tour))])
            res.saved_csv = str(cpath)
        except OSError as e:
            log.warning("Sauvegarde comparaison impossible : %s", e)
