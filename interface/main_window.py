"""Fenêtre principale : en-tête + dashboard à 4 quadrants redimensionnables."""
from __future__ import annotations
import json
import logging
import queue
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import ttk, messagebox

from config.settings import POINTS_FILE, START_POINT_ID, RESULTS_DIR, OSRM_BASE_URL
from core.genetic_algorithm import GeneticAlgorithm, RunControl
from core.tsp_problem import TSPProblem
from interface import theme
from interface.quadrant_1 import Quadrant1
from interface.quadrant_2 import Quadrant2
from interface.quadrant_3 import Quadrant3
from interface.quadrant_4 import Quadrant4
from interface.optuna_dialog import OptunaDialog
from interface.agents_dialog import AgentsDialog
from interface.comparison_dialog import ComparisonDialog
from agents.gemini_client import GeminiConfigError, load_api_key
from services.data_loader import load_points, DataValidationError
from services.distance_matrix import get_matrix, get_route
from services.osrm_service import OSRMError
from visualization import folium_map

log = logging.getLogger(__name__)
REFRESH_MS = 60          # fréquence de lecture de la file de messages
REDRAW_MIN_S = 0.15      # limite de rafraîchissement des graphiques


class MainWindow:
    def __init__(self, root: tk.Tk, autorun: bool = False):
        self.root = root
        self.autorun = autorun
        root.title("TSP 40 — Optimisation par algorithme génétique")
        root.geometry("1400x860")
        root.minsize(1000, 650)
        try:
            root.state("zoomed")  # plein écran fenêtré sous Windows
        except tk.TclError:
            pass
        theme.apply(root)
        self.q: queue.Queue = queue.Queue()
        self.points, self.problem, self.matrix = [], None, None
        self.control: RunControl | None = None
        self.worker: threading.Thread | None = None
        self._hist = ([], [], [], [])
        self._last_draw = 0.0
        self._last_tour = None
        self._last_stats = None
        self.optuna_dlg: OptunaDialog | None = None
        self.agents_dlg: AgentsDialog | None = None
        self.compare_dlg: ComparisonDialog | None = None
        self._agents_session = None
        self._mode = "manuel"

        self._build_header()
        self._build_dashboard()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.after(REFRESH_MS, self._pump)
        root.after(200, self._load_data)

    # ------------------------------------------------------------------ UI
    def _build_header(self):
        h = ttk.Frame(self.root, style="Bg.TFrame", padding=(14, 8))
        h.pack(fill="x")
        ttk.Label(h, text="TSP 40 · Algorithme génétique sur réseau routier réel",
                  style="Header.TLabel").pack(side="left")
        self.status = ttk.Label(h, text="", style="HeaderMuted.TLabel")
        self.status.pack(side="right")

    def _build_dashboard(self):
        outer = ttk.PanedWindow(self.root, orient="vertical")
        outer.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        top = ttk.PanedWindow(outer, orient="horizontal")
        bottom = ttk.PanedWindow(outer, orient="horizontal")
        outer.add(top, weight=1)
        outer.add(bottom, weight=1)
        self.q1 = Quadrant1(top, callbacks={"run": self.start_run, "pause": self.toggle_pause,
                                            "stop": self.stop_run, "step": self.step_run,
                                            "optuna": self.open_optuna, "agents": self.open_agents,
                                            "compare": self.open_compare})
        self.q2 = Quadrant2(top)
        self.q3 = Quadrant3(bottom)
        self.q4 = Quadrant4(bottom)
        top.add(self.q1, weight=2)
        top.add(self.q2, weight=3)
        bottom.add(self.q3, weight=2)
        bottom.add(self.q4, weight=3)

    def set_status(self, text: str, error: bool = False):
        self.status.configure(text=text, foreground="#ef5b5b" if error else "#9aa1b2")
        (log.error if error else log.info)(text)

    # ------------------------------------------------------------ données
    def _load_data(self):
        try:
            self.points = load_points(POINTS_FILE)
        except (DataValidationError, OSError, ValueError) as e:
            messagebox.showerror("Données invalides", str(e))
            self.set_status("Erreur de lecture des points GPS", error=True)
            return
        start = next(p for p in self.points if p.id == START_POINT_ID)
        self.q1.problem_lbl.configure(
            text=f"TSP fermé · {len(self.points)} points validés · départ/retour : "
                 f"#{start.id} {start.name} · objectif : distance routière OSRM minimale")
        self.q3.set_points(self.points, START_POINT_ID)
        try:
            html = folium_map.initial_map(self.points, START_POINT_ID)
            self.q4.show(html, "Carte initiale : 40 points (D = départ). Cliquez un marqueur pour le détail.")
        except Exception as e:  # noqa: BLE001
            self.set_status(f"Carte initiale impossible : {e}", error=True)
        self.set_status("Chargement de la matrice routière OSRM…")
        threading.Thread(target=self._matrix_worker, daemon=True).start()

    def _matrix_worker(self):
        try:
            m = get_matrix(self.points)
            self.q.put(("matrix", m))
        except OSRMError as e:
            self.q.put(("matrix_error", str(e)))
        except Exception as e:  # noqa: BLE001
            log.exception("Matrice")
            self.q.put(("matrix_error", f"Erreur inattendue : {e}"))

    # ---------------------------------------------------------- simulation
    # ---------------------------------------------------------------- Optuna
    def sim_running(self) -> bool:
        return bool(self.worker and self.worker.is_alive())

    def open_optuna(self):
        if self.problem is None:
            return
        if self.optuna_dlg is not None and self.optuna_dlg.winfo_exists():
            self.optuna_dlg.lift()
            return
        try:
            seed = int(self.q1.v_seed.get())
        except ValueError:
            seed = 42
        self.optuna_dlg = OptunaDialog(self.root, self.problem, seed, self.apply_optuna,
                                       lambda: self.sim_running() or self.comparison_running())

    def apply_optuna(self, params: dict, seed: int, run: bool):
        if self.sim_running():
            messagebox.showwarning("Simulation en cours", "Attendez la fin de la simulation.")
            return
        self.q1.apply_params(params, seed, mode="optuna")
        self.set_status("Configuration Optuna appliquée au quadrant 1.")
        if run:
            self.start_run()

    # ----------------------------------------------------------- agents IA
    def open_agents(self):
        if self.problem is None:
            return
        if self.agents_dlg is not None and self.agents_dlg.winfo_exists():
            self.agents_dlg.lift()
            return
        try:
            load_api_key()  # message clair si la clé est absente, sans ouvrir la fenêtre
        except GeminiConfigError as e:
            messagebox.showwarning("Agents IA indisponibles", str(e))
            self.set_status("Agents IA : clé Gemini absente (.env).", error=True)
            return
        self.agents_dlg = AgentsDialog(self.root, self.problem, self.apply_agents)

    def apply_agents(self, params: dict, session, modified: bool):
        if self.sim_running():
            messagebox.showwarning("Simulation en cours", "Attendez la fin de la simulation.")
            return
        seed_txt = self.q1.v_seed.get().strip()
        seed = int(seed_txt) if seed_txt.lstrip("-").isdigit() else None
        self.q1.apply_params(params, seed, mode="agents (modifié)" if modified else "agents")
        self._agents_session = session
        self.set_status(f"Configuration des agents {session.decision} : lancement.")
        self.start_run()

    # ---------------------------------------------------------- comparaison
    def open_compare(self):
        if self.problem is None:
            return
        if self.compare_dlg is not None and self.compare_dlg.winfo_exists():
            self.compare_dlg.lift()
            return
        try:
            p = self.q1.read_params()
        except ValueError as e:
            messagebox.showwarning("Paramètres manuels invalides", str(e))
            return
        manual = {k: getattr(p, k) for k in ("population_size", "selection_method", "mutation_rate",
                                             "crossover_rate", "generations", "elitism")}
        self.compare_dlg = ComparisonDialog(self.root, self.problem, manual, self.busy)

    def busy(self) -> bool:
        return self.sim_running() or self.optuna_running()

    def comparison_running(self) -> bool:
        return bool(self.compare_dlg is not None and self.compare_dlg.winfo_exists()
                    and self.compare_dlg.running)

    def optuna_running(self) -> bool:
        return bool(self.optuna_dlg is not None and self.optuna_dlg.winfo_exists()
                    and self.optuna_dlg.running)

    # ------------------------------------------------------------ simulation
    def start_run(self, step_mode: bool = False):
        if self.problem is None:
            return
        if self.comparison_running():
            messagebox.showwarning("Comparaison en cours",
                                   "Une comparaison est en cours : attendez sa fin (mesure des temps).")
            return
        if self.optuna_running():
            messagebox.showwarning("Optuna en cours",
                                   "Une optimisation Optuna est en cours : attendez sa fin "
                                   "(elle mesure des temps d'exécution).")
            return
        self._mode = self.q1.mode
        try:
            params = self.q1.read_params()
        except ValueError as e:
            messagebox.showwarning("Paramètres invalides", str(e))
            return
        self.control = RunControl()
        if step_mode:
            self.control.step()
        self._hist = ([], [], [], [])
        self._last_tour = None
        self.q2.reset()
        self.q3.set_tour(None)
        self.q1.set_running(True, paused=step_mode)
        self.q1.set_progress(0, "0 %")
        self._params = params
        self.set_status(f"Simulation [{self._mode}] en cours (pop {params.population_size}, "
                        f"{params.generations} générations, {params.selection_method})…")
        self.worker = threading.Thread(target=self._ga_worker, args=(params, self.control), daemon=True)
        self.worker.start()

    def _ga_worker(self, params, control):
        try:
            ga = GeneticAlgorithm(self.problem, params)
            result = ga.run(on_generation=lambda g, s: self.q.put(("gen", g, s)), control=control)
            self.q.put(("done", result))
        except Exception as e:  # noqa: BLE001
            log.exception("AG")
            self.q.put(("ga_error", str(e)))

    def toggle_pause(self):
        if not self.control:
            return
        if self.control.paused:
            self.control.resume()
            self.q1.set_running(True, paused=False)
            self.set_status("Simulation reprise.")
        else:
            self.control.pause()
            self.q1.set_running(True, paused=True)
            self.set_status("Simulation en pause.")

    def step_run(self):
        if self.worker and self.worker.is_alive():
            self.control.step()
            self.q1.set_running(True, paused=True)
            self.set_status("Mode pas à pas : une génération exécutée.")
        else:
            self.start_run(step_mode=True)

    def stop_run(self):
        if self.control:
            self.control.stop()
            self.set_status("Arrêt demandé…")

    # ------------------------------------------------------ boucle messages
    def _pump(self):
        try:
            for _ in range(2000):
                msg = self.q.get_nowait()
                self._handle(msg)
        except queue.Empty:
            pass
        if self._last_stats and time.perf_counter() - self._last_draw > REDRAW_MIN_S:
            self._redraw_live()
        self.root.after(REFRESH_MS, self._pump)

    def _handle(self, msg):
        kind = msg[0]
        if kind == "gen":
            _, g, s = msg
            gens, best, gb, mean = self._hist
            gens.append(g)
            best.append(s["best_m"] / 1000)
            gb.append(s["gen_best_m"] / 1000)
            mean.append(s["mean_m"] / 1000)
            self._last_stats = (g, s)
            if self.control and self.control.step_mode:
                self._redraw_live()
        elif kind == "matrix":
            self.matrix = msg[1]
            self.problem = TSPProblem(self.matrix.distances_m, START_POINT_ID)
            src = "cache local" if self.matrix.source == "cache" else f"OSRM ({OSRM_BASE_URL})"
            self.set_status(f"Matrice routière 40×40 prête — source : {src}.")
            self.q1.btn_run.configure(state="normal")
            self.q1.btn_step.configure(state="normal")
            self.q1.btn_optuna.configure(state="normal")
            self.q1.btn_agents.configure(state="normal")
            self.q1.btn_compare.configure(state="normal")
            if self.autorun:
                self.autorun = False
                self.root.after(500, self.start_run)
        elif kind == "matrix_error":
            self.set_status("OSRM indisponible : simulation impossible.", error=True)
            messagebox.showerror("OSRM indisponible",
                                 msg[1] + "\n\nAucune distance euclidienne n'est substituée aux "
                                          "distances routières. Relancez quand le service est joignable.")
        elif kind == "done":
            self._finish(msg[1])
        elif kind == "ga_error":
            self.q1.set_running(False)
            self.set_status("Erreur pendant la simulation.", error=True)
            messagebox.showerror("Erreur de simulation", msg[1])
        elif kind == "route":
            _, html, route, result = msg
            gap = route["distance_m"] / 1000 - result.best_distance_m / 1000
            self.q4.show(html, f"Trajet routier OSRM : {route['distance_m'] / 1000:.2f} km "
                               f"(matrice : {result.best_distance_m / 1000:.2f} km, écart {gap:+.2f} km) · "
                               f"{route['duration_s'] / 60:.0f} min")
        elif kind == "route_error":
            self.q4.status.configure(text=f"Trajet OSRM indisponible : {msg[1]}", foreground="#ef5b5b")

    def _redraw_live(self):
        g, s = self._last_stats
        self._last_stats = None
        self._last_draw = time.perf_counter()
        self.q2.update_curves(*self._hist)
        best_km = s["best_m"] / 1000
        init = self._hist[1][0] if self._hist[1] else best_km
        self.q2.update_kpis(best_km, s["elapsed_s"], s["best_generation"],
                            100 * (init - best_km) / init, 1 / (1 + best_km))
        if s["best_tour"] != self._last_tour:
            self._last_tour = s["best_tour"]
            self.q3.set_tour(s["best_tour"])
        total = self._params.generations
        self.q1.set_progress(100 * g / total, f"{100 * g / total:.0f} % — génération {g}/{total}")

    def _finish(self, r):
        if self._last_stats:
            self._redraw_live()
        self.q1.set_running(False)
        self.q2.update_kpis(r.best_distance_m / 1000, r.elapsed_s, r.best_generation,
                            r.improvement_pct, r.best_fitness)
        self.q3.set_tour(r.best_tour, self.problem.leg_lengths(r.best_tour[1:-1]))
        state = "interrompue" if r.stopped else "terminée"
        self.set_status(f"Simulation {state} : {r.best_distance_m / 1000:.2f} km en "
                        f"{r.elapsed_s:.2f} s ({r.generations_run} générations). Calcul du trajet routier…")
        self._save_result(r)
        threading.Thread(target=self._route_worker, args=(r,), daemon=True).start()

    def _route_worker(self, r):
        try:
            by_id = {p.id: p for p in self.points}
            ordered = [by_id[i] for i in r.best_tour]
            route = get_route(ordered)
            html = folium_map.route_map(self.points, r.best_tour, route)
            self.q.put(("route", html, route, r))
        except Exception as e:  # noqa: BLE001
            log.exception("Route")
            self.q.put(("route_error", str(e)))

    def _save_result(self, r):
        try:
            by_id = {p.id: p for p in self.points}
            data = {"date": datetime.now().isoformat(timespec="seconds"), "mode": self._mode,
                    "params": r.params, "distance_km": r.best_distance_m / 1000,
                    "elapsed_s": r.elapsed_s, "best_generation": r.best_generation,
                    "generations_run": r.generations_run, "improvement_pct": r.improvement_pct,
                    "fitness": r.best_fitness, "stopped": r.stopped,
                    "tour_ids": r.best_tour, "tour_names": [by_id[i].name for i in r.best_tour],
                    "history_best_km": [d / 1000 for d in r.history_best]}
            if self._mode.startswith("agents") and self._agents_session is not None:
                s = self._agents_session
                data["agents"] = {"session_file": s.saved_to, "model": s.model, "decision": s.decision,
                                  "final_by_arbiter": s.arbiter.config.params() if s.arbiter else None,
                                  "justification": s.arbiter.config.justification if s.arbiter else None}
            path = RESULTS_DIR / f"run_{datetime.now():%Y%m%d_%H%M%S}.json"
            path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError as e:
            log.warning("Sauvegarde du résultat impossible : %s", e)

    def _on_close(self):
        if self.control:
            self.control.stop()
        self.q4.close()
        self.root.destroy()
