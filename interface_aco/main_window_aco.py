"""Fenêtre principale ACO : en-tête + 4 quadrants (réutilise Q3, Q4 et le thème de l'appli AG)."""
from __future__ import annotations
import json
import logging
import queue
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import ttk, messagebox

from aco.ant_system import AntSystem, RunControl
from aco.settings_aco import SIZE_CHOICES, points_file
from config.settings import CACHE_DIR, RESULTS_DIR, OSRM_BASE_URL, START_POINT_ID, BASE_DIR
from core.validation import is_valid_tour
from interface import theme
from interface_aco.quadrant3_aco import Quadrant3ACO
from interface_aco.quadrant4_aco import Quadrant4ACO
from interface_aco.quadrant1_aco import Quadrant1ACO
from interface_aco.quadrant2_aco import Quadrant2ACO
from interface_aco.tools_mixin import ToolsMixin
from interface_aco.scrollable import VScrollFrame
from services.data_loader import load_points, DataValidationError
from services.distance_matrix import get_matrix, get_route
from services.osrm_service import OSRMError
from visualization import folium_map_aco

log = logging.getLogger(__name__)
REFRESH_MS = 60
REDRAW_MIN_S = 0.15
ROUTE_HTML = CACHE_DIR / "aco_map_route.html"


class MainWindowACO(ToolsMixin):
    def __init__(self, root: tk.Tk, autorun: bool = False):
        self.root = root
        self.autorun = autorun
        root.title("TSP · Colonie de fourmis (ACO) · Sidi Maarouf")
        root.geometry("1400x860")
        root.minsize(1000, 650)
        try:
            root.state("zoomed")
        except tk.TclError:
            pass
        theme.apply(root)
        self.q: queue.Queue = queue.Queue()
        self.n: int | None = None
        self.points, self.matrix = [], None
        self._token = 0            # ignore les matrices d'une taille abandonnée
        self.control: RunControl | None = None
        self.worker: threading.Thread | None = None
        self._hist = ([], [], [], [])
        self._last_stats = None
        self._last_draw = 0.0
        self._params = None
        self._pending_tour = None
        self.init_tools()

        self._build_header()
        self._build_dashboard()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.after(REFRESH_MS, self._pump)
        root.after(200, lambda: self.select_size(SIZE_CHOICES[0]))

    # ------------------------------------------------------------------ UI
    def _build_header(self):
        h = ttk.Frame(self.root, style="Bg.TFrame", padding=(14, 8))
        h.pack(fill="x")
        ttk.Label(h, text="TSP · Optimisation par colonie de fourmis (Ant System) sur réseau routier réel",
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
        self.q1_scroll = VScrollFrame(top)
        self.q1 = Quadrant1ACO(self.q1_scroll.inner, callbacks={"run": self.start_run, "stop": self.stop_run,
                                               "size": self.select_size, "optuna": self.open_optuna,
                                               "keras": self.open_keras, "sma": self.open_sma,
                                               "compare": self.open_compare,
                                               "verdicts": self.open_verdicts,
                                               "rerun": self.rerun_failed_agents})
        self.q1.pack(fill="both", expand=True)
        self.q2 = Quadrant2ACO(top)
        self.q3 = Quadrant3ACO(bottom)
        self.q4 = Quadrant4ACO(bottom)
        top.add(self.q1_scroll, weight=2)
        top.add(self.q2, weight=3)
        bottom.add(self.q3, weight=2)
        bottom.add(self.q4, weight=3)

    def set_status(self, text: str, error: bool = False):
        self.status.configure(text=text, foreground="#ef5b5b" if error else "#9aa1b2")
        (log.error if error else log.info)(text)

    # ------------------------------------------------------------ données
    def select_size(self, n: int):
        if n == self.n and self.points:
            return
        if self.tool_busy():
            messagebox.showwarning("Calcul en cours", "Attendez la fin du réglage en cours avant de "
                                   "changer le nombre de points.")
            if self.n is not None:
                self.q1.v_n.set(str(self.n))
            return
        path = points_file(n)
        if not path.exists():
            rel = path.relative_to(BASE_DIR)
            messagebox.showwarning(
                "Fichier absent",
                f"Fichier de {n} points absent : {rel}\n\nAucun point GPS n'est inventé. Ajoutez ce "
                f"fichier (colonnes n, nom, lat, lon ; identifiants 0..{n - 1}, départ n=0) pour "
                f"activer cette taille.")
            self.set_status(f"Fichier de {n} points absent : rien n'est lancé.", error=True)
            if self.n is not None:
                self.q1.v_n.set(str(self.n))
            return
        try:
            points = load_points(path, expected=n, start_id=START_POINT_ID)
        except (DataValidationError, OSError, ValueError) as e:
            messagebox.showerror("Données invalides", f"{path.name} : {e}")
            self.set_status(f"Erreur de lecture de {path.name}", error=True)
            if self.n is not None:
                self.q1.v_n.set(str(self.n))
            return
        self.q1.set_default_ants(self.n, n)
        self.n, self.points, self.matrix = n, points, None
        self.reset_tools_for_size()
        self._token += 1
        self.q1.set_ready(False)
        start = points[0]
        self.q1.problem_lbl.configure(
            text=f"TSP fermé · {n} points validés ({path.name}) · départ/retour : #{start.id} "
                 f"{start.name} · objectif : distance routière OSRM minimale")
        self.q2.reset(n - 1)
        self.q3.set_points(points, START_POINT_ID)
        try:
            html = folium_map_aco.initial_map(points, START_POINT_ID, out=CACHE_DIR / f"aco_map_initial_{n}.html")
            self.q4.show(html, f"Carte initiale : {n} points (D = départ). Cliquez un marqueur pour le détail.")
        except Exception as e:  # noqa: BLE001
            self.set_status(f"Carte initiale impossible : {e}", error=True)
        self.set_status(f"Chargement de la matrice routière OSRM {n}×{n}…")
        threading.Thread(target=self._matrix_worker, args=(points, self._token), daemon=True).start()

    def _matrix_worker(self, points, token):
        try:
            self.q.put(("matrix", token, get_matrix(points)))
        except OSRMError as e:
            self.q.put(("matrix_error", token, str(e)))
        except Exception as e:  # noqa: BLE001
            log.exception("Matrice")
            self.q.put(("matrix_error", token, f"Erreur inattendue : {e}"))

    # ---------------------------------------------------------- simulation
    def start_run(self):
        if self.matrix is None or self.sim_running():
            return
        if self.tool_busy():
            messagebox.showwarning("Calcul en cours", "Un réglage automatique ou une comparaison est en "
                                   "cours : attendez sa fin (il mesure des temps d'exécution).")
            return
        try:
            params = self.q1.read_params()
        except ValueError as e:
            messagebox.showwarning("Paramètres invalides", str(e))
            return
        self._params = params
        self._run_mode = self.q1.mode
        if self._run_mode == "manuel":
            self.remember_manual(params)
        self._hist = ([], [], [], [])
        self._last_stats = None
        self.control = RunControl()
        self.q2.reset(self.n - 1)
        self.q3.set_tour(None)
        self.q1.set_running(True)
        self.q1.set_progress(0, "0 %")
        ants = params.n_ants or self.n
        self.set_status(f"ACO en cours : {self.n} points, {ants} fourmis, ρ={params.rho}, α={params.alpha}, "
                        f"β={params.beta}, {params.cycles} cycles…")
        self.worker = threading.Thread(target=self._aco_worker, args=(params, self.control), daemon=True)
        self.worker.start()

    def _aco_worker(self, params, control):
        try:
            aco = AntSystem(self.matrix.distances_m, params, START_POINT_ID)
            r = aco.run(on_cycle=lambda c, s: self.q.put(("cycle", c, s)), control=control)
            self.q.put(("done", r))
        except Exception as e:  # noqa: BLE001
            log.exception("ACO")
            self.q.put(("aco_error", str(e)))

    def sim_running(self) -> bool:
        return bool(self.worker and self.worker.is_alive())

    def stop_run(self):
        if self.control:
            self.control.stop()
            self.set_status("Arrêt demandé…")

    # ------------------------------------------------------ boucle messages
    def _pump(self):
        try:
            for _ in range(2000):
                self._handle(self.q.get_nowait())
        except queue.Empty:
            pass
        if self._last_stats and time.perf_counter() - self._last_draw > REDRAW_MIN_S:
            self._redraw_live()
        self.root.after(REFRESH_MS, self._pump)

    def _handle(self, msg):
        kind = msg[0]
        if kind == "cycle":
            _, c, s = msg
            cyc, best, cb, mean = self._hist
            cyc.append(c)
            best.append(s["best_m"] / 1000)
            cb.append(s["cycle_best_m"] / 1000)
            mean.append(s["mean_m"] / 1000)
            if s["best_tour"] is not None:
                self._pending_tour = s["best_tour"]
            self._last_stats = (c, s)
        elif kind == "matrix":
            if msg[1] != self._token:
                return
            self.matrix = msg[2]
            src = "cache local" if self.matrix.source == "cache" else f"OSRM ({OSRM_BASE_URL})"
            self.set_status(f"Matrice routière {self.n}×{self.n} prête — source : {src}.")
            self.q1.set_ready(True)
            if self.autorun:
                self.autorun = False
                self.root.after(500, self.start_run)
        elif kind == "matrix_error":
            if msg[1] != self._token:
                return
            self.set_status("OSRM indisponible : simulation impossible.", error=True)
            messagebox.showerror("OSRM indisponible",
                                 msg[2] + "\n\nAucune distance euclidienne n'est substituée aux "
                                          "distances routières. Relancez quand le service est joignable.")
        elif kind == "done":
            self._finish(msg[1])
        elif kind == "aco_error":
            self.q1.set_running(False)
            self.set_status("Erreur pendant la simulation.", error=True)
            messagebox.showerror("Erreur de simulation", msg[1])
        elif kind == "route":
            _, html, route, r = msg
            gap = route["distance_m"] / 1000 - r.best_distance_m / 1000
            self.q4.show(html, f"Trajet routier OSRM : {route['distance_m'] / 1000:.2f} km "
                               f"(matrice : {r.best_distance_m / 1000:.2f} km, écart {gap:+.2f} km) · "
                               f"{route['duration_s'] / 60:.0f} min")
        elif kind == "route_error":
            self.q4.status.configure(text=f"Trajet OSRM indisponible : {msg[1]}", foreground="#ef5b5b")

    def _redraw_live(self):
        c, s = self._last_stats
        self._last_stats = None
        self._last_draw = time.perf_counter()
        self.q2.update_curves(*self._hist)
        self.q2.update_kpis(s["best_m"] / 1000, self.n - 1, s["elapsed_s"], s["best_cycle"])
        tour = getattr(self, "_pending_tour", None)
        if tour is not None:
            self._pending_tour = None
            self.q3.set_tour(tour)
        total = self._params.cycles
        self.q1.set_progress(100 * c / total, f"{100 * c / total:.0f} % — cycle {c}/{total}")

    def _finish(self, r):
        if self._last_stats:
            self._redraw_live()
        self.q1.set_running(False)
        ok = is_valid_tour(r.best_tour, self.n, START_POINT_ID)
        self.q2.update_kpis(r.best_distance_m / 1000, self.n - 1, r.elapsed_s, r.best_cycle, ok)
        self.q2.detail.configure(
            text=f"{self.n - 1} livraisons + retour au départ · {r.n_ants} fourmis · τ0 = {r.tau0:.4g} · "
                 f"1er cycle : {r.initial_best_m / 1000:.2f} km → meilleur : {r.best_distance_m / 1000:.2f} km")
        d = self.matrix.distances_m
        self.q3.set_tour(r.best_tour, [float(d[a, b]) for a, b in zip(r.best_tour[:-1], r.best_tour[1:])])
        if not r.stopped:
            self.q1.set_progress(100, f"100 % — {r.cycles_run} cycles")
        state = "interrompue" if r.stopped else "terminée"
        self.set_status(f"ACO {state} : {r.best_distance_m / 1000:.2f} km en {r.elapsed_s:.2f} s "
                        f"(meilleur au cycle {r.best_cycle}/{r.cycles_run}). Calcul du trajet routier…")
        self._save_result(r, ok)
        threading.Thread(target=self._route_worker, args=(r,), daemon=True).start()

    def _route_worker(self, r):
        try:
            by_id = {p.id: p for p in self.points}
            route = get_route([by_id[i] for i in r.best_tour])
            html = folium_map_aco.route_map(self.points, r.best_tour, route, out=ROUTE_HTML)
            self.q.put(("route", html, route, r))
        except Exception as e:  # noqa: BLE001
            log.exception("Route")
            self.q.put(("route_error", str(e)))

    def _save_result(self, r, ok: bool):
        try:
            by_id = {p.id: p for p in self.points}
            data = {"date": datetime.now().isoformat(timespec="seconds"), "algo": "ACO (Ant System)",
                    "mode": self._run_mode, "n_points": self.n, "params": r.params, "tau0": r.tau0,
                    "all_delivered": ok, "distance_km": r.best_distance_m / 1000,
                    "km_per_delivery": r.best_distance_m / 1000 / (self.n - 1),
                    "elapsed_s": r.elapsed_s, "best_cycle": r.best_cycle, "cycles_run": r.cycles_run,
                    "stopped": r.stopped, "matrix_cache": self.matrix.cache_file,
                    "tour_ids": r.best_tour, "tour_names": [by_id[i].name for i in r.best_tour],
                    "history_best_km": [x / 1000 for x in r.history_best]}
            data.update(self.tool_info_for_run(self._run_mode))
            path = RESULTS_DIR / f"aco_run_{datetime.now():%Y%m%d_%H%M%S}.json"
            path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError as e:
            log.warning("Sauvegarde du résultat impossible : %s", e)

    def _on_close(self):
        if self.control:
            self.control.stop()
        for d in self.dialogs.values():
            eng = getattr(d, "engine", None)
            if eng is not None:
                eng.stop()
        self.q4.close()
        self.root.destroy()
