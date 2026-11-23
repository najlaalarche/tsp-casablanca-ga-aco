"""Briques communes au réglage automatique de l'ACO (Optuna, KerasTuner, comparaison).

- Espace de recherche : fourmis, ρ, α, β, cycles.
- Évaluation robuste : une configuration est exécutée sur PLUSIEURS graines (les mêmes pour toutes
  les configurations) ; on retient la moyenne et l'écart-type de la distance et du temps.
  Avec une seule graine, le réglage choisirait la configuration qui a eu le tirage le plus
  favorable (biais du gagnant).
- Front de Pareto distance / temps et recommandation de compromis (même méthode que l'AG).
"""
from __future__ import annotations
import statistics
from dataclasses import dataclass, field

from aco.ant_system import AntSystem, ACOParams, RunControl
from aco.settings_aco import CYCLE_CHOICES

DEFAULT_EVAL_SEEDS = 2
MAX_TRIALS = 20
RHO_RANGE = (0.05, 0.95)
ALPHA_RANGE = (0.5, 5.0)
BETA_RANGE = (1.0, 10.0)
ANTS_STEP = 5
PARAM_KEYS = ("n_ants", "rho", "alpha", "beta", "cycles")


def ants_range(n_points: int) -> tuple[int, int]:
    """De 10 fourmis à 2 × N (pas de 5)."""
    hi = max(10 + ANTS_STEP, 2 * n_points)
    return 10, 10 + ((hi - 10) // ANTS_STEP) * ANTS_STEP


def space_description(n_points: int) -> str:
    lo, hi = ants_range(n_points)
    return (f"fourmis {lo}–{hi} (pas {ANTS_STEP}) · ρ {RHO_RANGE[0]}–{RHO_RANGE[1]} · "
            f"α {ALPHA_RANGE[0]}–{ALPHA_RANGE[1]} · β {BETA_RANGE[0]}–{BETA_RANGE[1]} · "
            f"cycles {list(CYCLE_CHOICES)}")


class Stopped(Exception):
    """Évaluation interrompue par l'utilisateur."""


@dataclass
class EvalRecord:
    number: int
    params: dict                       # n_ants, rho, alpha, beta, cycles
    seeds: list[int]
    distances_km: list[float]
    times_s: list[float]
    best_cycles: list[int]
    pareto: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def distance_km(self) -> float:
        return statistics.fmean(self.distances_km)

    @property
    def std_distance_km(self) -> float:
        return statistics.stdev(self.distances_km) if len(self.distances_km) > 1 else 0.0

    @property
    def time_s(self) -> float:
        return statistics.fmean(self.times_s)

    def to_dict(self) -> dict:
        return {"number": self.number, "params": self.params, "seeds": self.seeds,
                "distance_km_mean": self.distance_km, "distance_km_std": self.std_distance_km,
                "time_s_mean": self.time_s, "distances_km": self.distances_km,
                "times_s": self.times_s, "best_cycles": self.best_cycles, "pareto": self.pareto,
                **({"extra": self.extra} if self.extra else {})}


def to_aco_params(p: dict, seed: int | None) -> ACOParams:
    return ACOParams(n_ants=int(p["n_ants"]), rho=float(p["rho"]), alpha=float(p["alpha"]),
                     beta=float(p["beta"]), cycles=int(p["cycles"]), seed=seed)


def evaluate(distances_m, params: dict, seeds: list[int], number: int = 0,
             control: RunControl | None = None, start: int = 0) -> EvalRecord:
    """Exécute l'ACO une fois par graine. Lève Stopped si l'utilisateur interrompt."""
    dists, times, best_cycles = [], [], []
    for s in seeds:
        r = AntSystem(distances_m, to_aco_params(params, s), start).run(control=control)
        if r.stopped:
            raise Stopped()
        dists.append(r.best_distance_m / 1000)
        times.append(r.elapsed_s)
        best_cycles.append(r.best_cycle)
    return EvalRecord(number, dict(params), list(seeds), dists, times, best_cycles)


def fmt_params(p: dict) -> str:
    return (f"{p['n_ants']} fourmis, ρ = {p['rho']:.3f}, α = {p['alpha']:.2f}, "
            f"β = {p['beta']:.2f}, {p['cycles']} cycles")


def pareto_front(records: list[EvalRecord]) -> list[EvalRecord]:
    """Solutions non dominées (distance moyenne et temps moyen minimisés)."""
    front = [a for a in records if not any(
        b.distance_km <= a.distance_km and b.time_s <= a.time_s
        and (b.distance_km < a.distance_km or b.time_s < a.time_s) for b in records)]
    return sorted(front, key=lambda r: r.distance_km)


def recommend(records: list[EvalRecord]) -> tuple[EvalRecord, str] | None:
    """Point du front le plus proche du point idéal, objectifs normalisés min-max (poids égaux).

    Les km et les secondes ne sont jamais additionnés bruts.
    """
    if not records:
        return None
    front = pareto_front(records)
    ids = {r.number for r in front}
    for r in records:
        r.pareto = r.number in ids
    dmin, dmax = min(r.distance_km for r in records), max(r.distance_km for r in records)
    tmin, tmax = min(r.time_s for r in records), max(r.time_s for r in records)

    def norm(v, lo, hi):
        return 0.0 if hi - lo < 1e-12 else (v - lo) / (hi - lo)

    best = min(front, key=lambda r: norm(r.distance_km, dmin, dmax) ** 2 + norm(r.time_s, tmin, tmax) ** 2)
    shortest = min(front, key=lambda r: r.distance_km)
    lines = [f"Trial #{best.number} : {fmt_params(best.params)}.",
             f"Moyenne sur {len(best.seeds)} graine(s) {best.seeds} : {best.distance_km:.2f} km "
             f"(écart-type {best.std_distance_km:.2f} km, de {min(best.distances_km):.2f} à "
             f"{max(best.distances_km):.2f} km) en {best.time_s:.2f} s."]
    if best is shortest:
        lines.append("C'est aussi la meilleure distance moyenne de toute l'étude.")
    else:
        lines.append(f"Meilleure distance moyenne de l'étude : trial #{shortest.number} "
                     f"({shortest.distance_km:.2f} km en {shortest.time_s:.2f} s, {fmt_params(shortest.params)}). "
                     f"Le compromis retenu coûte +{best.distance_km - shortest.distance_km:.2f} km mais "
                     f"économise {shortest.time_s - best.time_s:.2f} s par exécution.")
    lines.append(f"{len(front)} solution(s) non dominée(s) sur {len(records)} trial(s). Critère : distance et "
                 "temps normalisés (min-max), poids égaux, point du front le plus proche de l'idéal.")
    return best, "\n".join(lines)
