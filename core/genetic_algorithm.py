"""Moteur d'algorithme génétique, indépendant de l'interface graphique."""
from __future__ import annotations
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Callable

import numpy as np

from core import operators as ops
from core.fitness import fitness
from core.tsp_problem import TSPProblem
from core.validation import is_valid_tour


@dataclass
class GAParams:
    population_size: int = 50
    selection_method: str = "tournament"  # "tournament" | "roulette"
    mutation_rate: float = 0.05
    crossover_rate: float = 0.8
    generations: int = 500
    elitism: bool = True
    elite_size: int = 2
    tournament_k: int = 3
    mutation_operator: str = "inversion"  # "inversion" | "swap"
    seed: int | None = 42

    def validate(self) -> None:
        if self.population_size < 2:
            raise ValueError("La population doit contenir au moins 2 individus.")
        if self.selection_method not in ("tournament", "roulette"):
            raise ValueError("Méthode de sélection inconnue.")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("Le taux de mutation doit être compris entre 0 et 1.")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError("Le taux de croisement doit être compris entre 0 et 1.")
        if self.generations < 1:
            raise ValueError("Le nombre de générations doit être >= 1.")
        if self.mutation_operator not in ops.MUTATIONS:
            raise ValueError("Opérateur de mutation inconnu.")


@dataclass
class GAResult:
    best_tour: list[int]
    best_distance_m: float
    best_fitness: float
    best_generation: int
    initial_best_m: float
    improvement_pct: float
    elapsed_s: float
    generations_run: int
    history_best: list[float] = field(default_factory=list)      # meilleur global (m)
    history_gen_best: list[float] = field(default_factory=list)  # meilleur de la génération (m)
    history_mean: list[float] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    stopped: bool = False


class RunControl:
    """Pause / reprise / arrêt / pas-à-pas, partagé entre le thread AG et l'IHM."""

    def __init__(self):
        self._run = threading.Event()
        self._run.set()
        self._stop = threading.Event()
        self._step = threading.Event()
        self.step_mode = False

    def pause(self): self._run.clear()
    def resume(self): self.step_mode = False; self._run.set()
    def stop(self): self._stop.set(); self._run.set(); self._step.set()
    def step(self): self.step_mode = True; self._run.clear(); self._step.set()

    @property
    def paused(self) -> bool: return not self._run.is_set()
    @property
    def stopped(self) -> bool: return self._stop.is_set()

    def checkpoint(self) -> bool:
        """Bloque si en pause ; renvoie False si arrêt demandé."""
        while not self._run.is_set() and not self._stop.is_set():
            if self._step.wait(timeout=0.1):
                self._step.clear()
                break
        return not self._stop.is_set()


ProgressCallback = Callable[[int, dict], None]


class GeneticAlgorithm:
    def __init__(self, problem: TSPProblem, params: GAParams):
        params.validate()
        self.problem = problem
        self.p = params
        self.rng = np.random.default_rng(params.seed)
        self.mutate = ops.MUTATIONS[params.mutation_operator]

    def _select(self, fit: np.ndarray) -> int:
        if self.p.selection_method == "tournament":
            return ops.tournament_selection(fit, self.rng, self.p.tournament_k)
        return ops.roulette_selection(fit, self.rng)

    def run(self, on_generation: ProgressCallback | None = None,
            control: RunControl | None = None, report_every: int = 1) -> GAResult:
        p, prob, rng = self.p, self.problem, self.rng
        t0 = time.perf_counter()
        pop = ops.random_population(prob.genes, p.population_size, rng)
        lengths = prob.population_lengths(pop)
        best_idx = int(np.argmin(lengths))
        best_ind, best_len, best_gen = pop[best_idx].copy(), float(lengths[best_idx]), 0
        initial_best = best_len
        hist_best, hist_gen, hist_mean = [], [], []
        n_elite = min(p.elite_size, p.population_size - 1) if p.elitism else 0
        stopped = False
        gen = 0

        for gen in range(1, p.generations + 1):
            if control is not None and not control.checkpoint():
                stopped = True
                gen -= 1
                break
            fit = fitness(lengths)
            order = np.argsort(lengths)
            new_pop = [pop[i].copy() for i in order[:n_elite]]
            while len(new_pop) < p.population_size:
                p1 = pop[self._select(fit)]
                if rng.random() < p.crossover_rate:
                    p2 = pop[self._select(fit)]
                    child = ops.order_crossover(p1, p2, rng)
                else:
                    child = p1.copy()
                if rng.random() < p.mutation_rate:
                    child = self.mutate(child, rng)
                new_pop.append(child)
            pop = np.array(new_pop, dtype=np.int64)
            lengths = prob.population_lengths(pop)

            gi = int(np.argmin(lengths))
            if lengths[gi] < best_len:
                best_len, best_ind, best_gen = float(lengths[gi]), pop[gi].copy(), gen
            hist_best.append(best_len)
            hist_gen.append(float(lengths[gi]))
            hist_mean.append(float(lengths.mean()))

            if on_generation and (gen % report_every == 0 or gen == p.generations
                                  or (control and control.step_mode)):
                on_generation(gen, {
                    "best_m": best_len, "gen_best_m": hist_gen[-1], "mean_m": hist_mean[-1],
                    "best_tour": prob.full_tour(best_ind), "best_generation": best_gen,
                    "elapsed_s": time.perf_counter() - t0,
                })

        tour = prob.full_tour(best_ind)
        assert is_valid_tour(tour, prob.n, prob.start), "Tournée invalide produite"
        return GAResult(
            best_tour=tour, best_distance_m=best_len, best_fitness=float(fitness(best_len)),
            best_generation=best_gen, initial_best_m=initial_best,
            improvement_pct=100.0 * (initial_best - best_len) / initial_best,
            elapsed_s=time.perf_counter() - t0, generations_run=gen,
            history_best=hist_best, history_gen_best=hist_gen, history_mean=hist_mean,
            params=asdict(p), stopped=stopped,
        )
