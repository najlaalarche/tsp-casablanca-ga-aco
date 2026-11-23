"""Ant System (Dorigo, 1996) pour le TSP fermé à départ fixe, vectorisé avec NumPy.

Indépendant de l'interface graphique. La matrice de distances est la matrice ROUTIÈRE
OSRM (asymétrique, en mètres) : les phéromones sont donc portées par les arcs orientés i→j.

Un cycle :
  1. m fourmis partent toutes du point de départ et construisent chacune un tour :
     depuis i, le point j non visité est choisi avec la probabilité
         p_ij = τ_ij^α · η_ij^β / Σ_{l non visité} τ_il^α · η_il^β,   η_ij = 1 / d_ij ;
     après le dernier point, la fourmi revient au départ.
  2. Évaporation : τ ← (1 − ρ) · τ.
  3. Dépôt : chaque fourmi k ajoute Q / L_k sur chacun des arcs de son tour.
  4. Le meilleur tour global est conservé.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field, asdict
from typing import Callable

import numpy as np

from core.genetic_algorithm import RunControl  # pause / arrêt partagés avec l'IHM
from core.validation import is_valid_tour


@dataclass
class ACOParams:
    n_ants: int | None = None      # None = nombre de points
    rho: float = 0.5               # évaporation, strictement dans ]0, 1[
    alpha: float = 1.0             # influence de la phéromone (> 0)
    beta: float = 3.0              # influence de la proximité (> 0)
    cycles: int = 500
    q: float = 1.0                 # quantité déposée : Q / L_k (L_k en km)
    tau0: float | None = None      # None = m / L_nn (recommandation de Dorigo)
    seed: int | None = 42

    def validate(self) -> None:
        if self.n_ants is not None and (not _is_int(self.n_ants) or self.n_ants < 1):
            raise ValueError("Le nombre de fourmis doit être un entier supérieur ou égal à 1.")
        if not _is_num(self.rho) or not 0.0 < self.rho < 1.0:
            raise ValueError("ρ (évaporation) doit être strictement compris entre 0 et 1 "
                             "(ex. 0,5).")
        if not _is_num(self.alpha) or self.alpha <= 0:
            raise ValueError("α (influence de la phéromone) doit être strictement positif.")
        if not _is_num(self.beta) or self.beta <= 0:
            raise ValueError("β (influence de la proximité) doit être strictement positif.")
        if not _is_int(self.cycles) or self.cycles < 1:
            raise ValueError("Le nombre de cycles doit être un entier supérieur ou égal à 1.")
        if not _is_num(self.q) or self.q <= 0:
            raise ValueError("Q doit être strictement positif.")
        if self.tau0 is not None and (not _is_num(self.tau0) or self.tau0 <= 0):
            raise ValueError("τ0 doit être strictement positif.")


def _is_int(v) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, bool)


def _is_num(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) \
        and np.isfinite(v)


@dataclass
class ACOResult:
    best_tour: list[int]           # [start, ..., start]
    best_distance_m: float
    best_cycle: int
    initial_best_m: float          # meilleur tour du cycle 1
    elapsed_s: float
    cycles_run: int
    n_ants: int
    tau0: float
    history_best: list[float] = field(default_factory=list)        # meilleur global (m)
    history_cycle_best: list[float] = field(default_factory=list)  # meilleur du cycle (m)
    history_mean: list[float] = field(default_factory=list)        # moyenne des fourmis (m)
    params: dict = field(default_factory=dict)
    stopped: bool = False


CycleCallback = Callable[[int, dict], None]


class AntSystem:
    def __init__(self, distances_m, params: ACOParams, start: int = 0):
        params.validate()
        d = np.asarray(distances_m, dtype=float)
        if d.ndim != 2 or d.shape[0] != d.shape[1]:
            raise ValueError("La matrice de distances doit être carrée.")
        if not np.all(np.isfinite(d)):
            raise ValueError("La matrice contient des distances manquantes.")
        if d.shape[0] < 3:
            raise ValueError("Il faut au moins 3 points.")
        if not 0 <= start < d.shape[0]:
            raise ValueError("Point de départ hors de la matrice.")
        self.d = d
        self.n = d.shape[0]
        self.start = start
        self.p = params
        self.m = int(params.n_ants) if params.n_ants is not None else self.n
        self.rng = np.random.default_rng(params.seed)
        d_km = d / 1000.0
        off = ~np.eye(self.n, dtype=bool)
        # Deux points distincts à 0 m l'un de l'autre (même accès routier) : on borne à 1 m
        # pour garder η fini. Cela ne modifie pas les longueurs, calculées sur la matrice brute.
        safe = np.where(off, np.maximum(d_km, 1e-3), 1.0)
        self.eta = np.where(off, 1.0 / safe, 0.0)
        self.tau0 = float(params.tau0) if params.tau0 is not None else self.m / (
            self.tour_length(self.nearest_neighbour_tour()) / 1000.0)
        self.tau = np.full((self.n, self.n), self.tau0)
        np.fill_diagonal(self.tau, 0.0)

    # ------------------------------------------------------------ outils
    def tour_length(self, tour) -> float:
        """Longueur (m) d'un tour fermé [start, ..., start]."""
        t = np.asarray(tour)
        return float(self.d[t[:-1], t[1:]].sum())

    def tours_lengths(self, tours: np.ndarray) -> np.ndarray:
        return self.d[tours[:, :-1], tours[:, 1:]].sum(axis=1)

    def nearest_neighbour_tour(self) -> list[int]:
        tour, seen = [self.start], {self.start}
        while len(tour) < self.n:
            row = self.d[tour[-1]].copy()
            row[list(seen)] = np.inf
            j = int(np.argmin(row))
            tour.append(j)
            seen.add(j)
        return tour + [self.start]

    # ------------------------------------------------------- un cycle
    def construct_tours(self) -> np.ndarray:
        """Construit m tours en parallèle. Renvoie un tableau (m, n+1) de tours fermés."""
        m, n, rng = self.m, self.n, self.rng
        weights = (self.tau ** self.p.alpha) * (self.eta ** self.p.beta)
        tours = np.empty((m, n + 1), dtype=np.int64)
        tours[:, 0] = self.start
        tours[:, -1] = self.start
        unvisited = np.ones((m, n), dtype=bool)
        unvisited[:, self.start] = False
        cur = np.full(m, self.start, dtype=np.int64)
        rows = np.arange(m)
        for step in range(1, n):
            w = weights[cur] * unvisited
            total = w.sum(axis=1)
            dead = ~(total > 0)  # sous-dépassement numérique : choix uniforme parmi les non visités
            if dead.any():
                w[dead] = unvisited[dead]
                total = w.sum(axis=1)
            cum = np.cumsum(w, axis=1)
            r = rng.random(m) * cum[:, -1]
            nxt = np.argmax(cum > r[:, None], axis=1)  # premier j de poids > 0 qui dépasse r
            tours[:, step] = nxt
            unvisited[rows, nxt] = False
            cur = nxt
        return tours

    def evaporate(self) -> None:
        self.tau *= (1.0 - self.p.rho)

    def deposit(self, tours: np.ndarray, lengths_m: np.ndarray) -> None:
        amount = self.p.q / (lengths_m / 1000.0)
        frm, to = tours[:, :-1].ravel(), tours[:, 1:].ravel()
        np.add.at(self.tau, (frm, to), np.repeat(amount, self.n))

    # ------------------------------------------------------------ boucle
    def run(self, on_cycle: CycleCallback | None = None,
            control: RunControl | None = None) -> ACOResult:
        t0 = time.perf_counter()
        best_tour, best_len, best_cycle = None, np.inf, 0
        hist_best, hist_cyc, hist_mean = [], [], []
        stopped, cycle = False, 0
        for cycle in range(1, self.p.cycles + 1):
            if control is not None and not control.checkpoint():
                stopped, cycle = True, cycle - 1
                break
            tours = self.construct_tours()
            lengths = self.tours_lengths(tours)
            self.evaporate()
            self.deposit(tours, lengths)
            k = int(np.argmin(lengths))
            improved = lengths[k] < best_len
            if improved:
                best_len, best_tour, best_cycle = float(lengths[k]), tours[k].tolist(), cycle
            hist_best.append(best_len)
            hist_cyc.append(float(lengths[k]))
            hist_mean.append(float(lengths.mean()))
            if on_cycle:
                on_cycle(cycle, {"best_m": best_len, "cycle_best_m": hist_cyc[-1],
                                 "mean_m": hist_mean[-1], "best_cycle": best_cycle,
                                 "best_tour": best_tour if improved else None,
                                 "elapsed_s": time.perf_counter() - t0})
        if best_tour is None:  # arrêt avant le premier cycle
            best_tour = self.nearest_neighbour_tour()
            best_len = self.tour_length(best_tour)
        if not is_valid_tour(best_tour, self.n, self.start):
            raise RuntimeError("Tour invalide produit par l'ACO.")
        params = asdict(self.p)
        params["n_ants"] = self.m
        return ACOResult(best_tour=best_tour, best_distance_m=best_len, best_cycle=best_cycle,
                         initial_best_m=hist_best[0] if hist_best else best_len,
                         elapsed_s=time.perf_counter() - t0, cycles_run=cycle, n_ants=self.m,
                         tau0=self.tau0, history_best=hist_best, history_cycle_best=hist_cyc,
                         history_mean=hist_mean, params=params, stopped=stopped)
