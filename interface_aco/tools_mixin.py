"""BC2 Optuna, BC3 KerasTuner, BC4 SMA et comparaison, branchés sur la fenêtre ACO."""
from __future__ import annotations
import logging
import queue
import threading
from tkinter import messagebox

from aco.settings_aco import DEFAULTS
from config.settings import START_POINT_ID

log = logging.getLogger(__name__)

METHOD_LABELS = {"manuel": "Manuel", "optuna": "Optuna", "kerastuner": "KerasTuner", "sma": "SMA"}


class ToolsMixin:
    # ------------------------------------------------------------ état
    def init_tools(self):
        self.tool_configs: dict[str, dict] = {}   # "Optuna" -> {"params", "source"} (pour la taille N courante)
        self.dialogs: dict[str, object] = {}
        self.sma_session = None
        self.sma_running = False
        self._run_mode = "manuel"
        self.tq: queue.Queue = queue.Queue()
        self.kt_error: str | None = None
        self.agent_results = [None, None, None]
        self.verdicts_dlg = None
        self.gemini_models: list[str] = []   # modèles qui répondaient au démarrage
        self.root.after(100, self._tools_pump)
        threading.Thread(target=self._check_keras, daemon=True).start()
        threading.Thread(target=self._probe_gemini, daemon=True).start()

    def _check_keras(self):
        try:
            from aco.kerastuner_aco import check_available
            v = check_available()
            import os
            self.tq.put(("kt_status", True, f"KerasTuner {v} prêt · backend Keras "
                         f"{os.environ.get('KERAS_BACKEND')} (oracle RandomSearch, sans réseau de neurones)."))
        except Exception as e:  # noqa: BLE001
            self.kt_error = str(e)
            first = str(e).split("Détail : ")[-1].split("\n")[0][:120]
            self.tq.put(("kt_status", False, f"BC3 grisé : KerasTuner non importable ({first}). Voir README."))

    def reset_tools_for_size(self):
        """Les configurations réglées pour une autre taille ne sont plus comparables."""
        self.tool_configs.clear()
        self.sma_session = None
        self.agent_results = [None, None, None]
        for i in range(3):
            self.q1.set_verdict(i, None, "Pas encore consulté (BC4).")
        self._update_rerun_state()
        self._refresh_verdicts()

    def tool_busy(self) -> bool:
        if self.sma_running:
            return True
        return any(d.winfo_exists() and getattr(d, "running", False) for d in self.dialogs.values())

    def _ready(self) -> bool:
        if self.matrix is None:
            messagebox.showinfo("Matrice non prête", "La matrice routière OSRM n'est pas encore chargée.")
            return False
        if self.sim_running():
            messagebox.showwarning("Simulation en cours", "Attendez la fin de la simulation ACO.")
            return False
        return True

    def _lift(self, key) -> bool:
        d = self.dialogs.get(key)
        if d is not None and d.winfo_exists():
            d.lift()
            return True
        return False

    def manual_config(self) -> dict:
        """Configuration « Manuel » : dernier run manuel, sinon les valeurs par défaut (fourmis = N)."""
        return self.tool_configs.get("Manuel", {}).get("params") or {
            "n_ants": self.n, "rho": DEFAULTS["rho"], "alpha": DEFAULTS["alpha"],
            "beta": DEFAULTS["beta"], "cycles": DEFAULTS["cycles"]}

    def remember_manual(self, params):
        self.tool_configs["Manuel"] = {"params": {"n_ants": params.n_ants or self.n, "rho": params.rho,
                                                  "alpha": params.alpha, "beta": params.beta,
                                                  "cycles": params.cycles}, "source": "quadrant 1"}

    def apply_config(self, key: str, params: dict, seed, run: bool, source: str | None):
        if self.sim_running():
            messagebox.showwarning("Simulation en cours", "Attendez la fin de la simulation.")
            return
        self.q1.apply_params(params, seed, key)
        label = METHOD_LABELS["sma" if key.startswith("sma") else key]
        self.tool_configs[label] = {"params": dict(params), "source": source}
        self.set_status(f"Configuration {label} appliquée au quadrant 1.")
        if run:
            self.start_run()

    def tool_info_for_run(self, mode: str) -> dict:
        if mode.startswith("sma") and self.sma_session is not None:
            s = self.sma_session
            return {"sma": {"session_file": s.saved_to, "model": s.model, "decision": s.decision,
                            "arbiter_params": s.arbiter.config.params() if s.arbiter and s.arbiter.config else None}}
        label = METHOD_LABELS.get(mode)
        if label and label in self.tool_configs and mode != "manuel":
            return {"tuning_source": self.tool_configs[label].get("source")}
        return {}

    # ------------------------------------------------------------ BC2 / BC3
    def _seed(self) -> int:
        try:
            return int(self.q1.v_seed.get())
        except ValueError:
            return DEFAULTS["seed"]

    def open_optuna(self):
        if self._lift("optuna") or not self._ready():
            return
        from aco.optuna_aco import OptunaACO
        from interface_aco.tuning_dialog import TuningDialog
        d = self.matrix.distances_m
        self.dialogs["optuna"] = TuningDialog(
            self.root, title=f"BC2 · Réglage de l'ACO par Optuna ({self.n} points)", method="Optuna",
            n_points=self.n, seed=self._seed(), is_busy=lambda: self.sim_running(),
            make_engine=lambda n, seed, k, _o: OptunaACO(d, n, seed, k, START_POINT_ID),
            on_apply=lambda p, seed, run, res: self.apply_config("optuna", p, seed, run,
                                                                 res.saved_to if res else None),
            intro=("Optuna (TPE multiobjectif) minimise séparément la distance moyenne et le temps moyen. "
                   "Chaque trial exécute l'ACO sur plusieurs graines (les mêmes pour tous les trials).\n"
                   "Retenu : point du front de Pareto le plus proche de l'idéal (objectifs normalisés)."))

    def open_keras(self):
        if self._lift("kerastuner") or not self._ready():
            return
        from aco.kerastuner_aco import (ORACLES, KerasTunerACO, KerasTunerUnavailable, check_available,
                                        kt_space_description)
        try:
            version = check_available()
        except KerasTunerUnavailable as e:
            messagebox.showwarning("KerasTuner indisponible", str(e))
            return
        from interface_aco.tuning_dialog import TuningDialog
        d = self.matrix.distances_m
        self.dialogs["kerastuner"] = TuningDialog(
            self.root, title=f"BC3 · Réglage de l'ACO par KerasTuner {version} ({self.n} points)",
            method="KerasTuner", n_points=self.n, seed=self._seed(), is_busy=lambda: self.sim_running(),
            make_engine=lambda n, seed, k, o: KerasTunerACO(d, n, seed, k, o["oracle"], START_POINT_ID),
            on_apply=lambda p, seed, run, res: self.apply_config("kerastuner", p, seed, run,
                                                                 res.saved_to if res else None),
            options={"oracle": ("Oracle", list(ORACLES))}, default_trials=15,
            space_text=kt_space_description(),
            intro=("KerasTuner : moteur de recherche (oracle) utilisé SANS réseau de neurones — un Tuner "
                   "personnalisé dont run_trial lance l'ACO sur plusieurs graines et renvoie la distance "
                   "moyenne.\nObjectif unique (les oracles KerasTuner sont mono-objectif) : distance moyenne "
                   "minimale ; le temps est mesuré et affiché."))

    # ------------------------------------------------------------ BC4 SMA
    def _probe_gemini(self):
        """Au démarrage : quels modèles Gemini répondent ? (en parallèle, 10 s max, en arrière-plan)"""
        try:
            from agents_aco import antigravity_backend
            if antigravity_backend.available():
                return  # moteur Antigravity : pas de sondage google-genai (aucun appel au démarrage)
            from agents.gemini_client import GeminiClient
            from agents_aco.resilient import probe_models
            self.gemini_models = probe_models(GeminiClient())
            log.info("Modèles Gemini qui répondent au démarrage : %s", self.gemini_models or "aucun")
        except Exception as e:  # noqa: BLE001  (clé absente, réseau…) : le SMA le signalera à l'usage
            log.info("Sondage Gemini impossible : %s", e)

    def _sma_key_ok(self) -> bool:
        from agents.gemini_client import GeminiConfigError, load_api_key
        try:
            load_api_key()
            return True
        except GeminiConfigError as e:
            messagebox.showwarning("Agents indisponibles", str(e))
            self.set_status("SMA : clé Gemini absente (.env).", error=True)
            return False

    def open_sma(self):
        if self.sma_running or not self._ready() or not self._sma_key_ok():
            return
        from agents_aco.manager_aco import run_agents_aco
        self.agent_results = [None, None, None]
        self.sma_session = None
        for i in range(3):
            self.q1.set_verdict(i, None, "En attente…")
        d, n, models = self.matrix.distances_m, self.n, list(self.gemini_models)
        self._start_sma(lambda prog, on_agent: run_agents_aco(d, n, progress=prog, on_agent=on_agent,
                                                              models=models))

    def rerun_failed_agents(self):
        from agents_aco.manager_aco import failed_agents, rerun_failed
        s = self.sma_session
        if self.sma_running or s is None:
            return
        todo = failed_agents(s)
        if not todo:
            messagebox.showinfo("SMA", "Aucun agent en échec : rien à relancer.")
            return
        if not self._ready() or not self._sma_key_ok():
            return
        idx = {"web": 0, "scientific": 1, "arbiter": 2}
        for k in todo:
            self.agent_results[idx[k]] = None
            self.q1.set_verdict(idx[k], None, "Relance en cours…")
        d, n, models = self.matrix.distances_m, self.n, list(self.gemini_models)
        self._start_sma(lambda prog, on_agent: rerun_failed(s, d, n, progress=prog, on_agent=on_agent,
                                                            models=models))

    def _start_sma(self, job):
        self.sma_running = True
        self.q1.set_running(True)
        self.q1.btn_stop.configure(state="disabled")
        self._update_rerun_state()
        self._refresh_verdicts()

        def worker():
            try:
                s = job(lambda m, p: self.tq.put(("sma_progress", m, p)),
                        lambda i, r: self.tq.put(("sma_agent", i, r)))
                self.tq.put(("sma_done", s))
            except Exception as e:  # noqa: BLE001
                log.exception("SMA")
                self.tq.put(("sma_error", str(e)))
        threading.Thread(target=worker, daemon=True).start()

    def can_rerun(self) -> bool:
        from agents_aco.manager_aco import failed_agents
        return bool(not self.sma_running and self.sma_session is not None and failed_agents(self.sma_session))

    def _update_rerun_state(self):
        self.q1.btn_rerun.configure(state="normal" if self.can_rerun() else "disabled")

    def _show_agent(self, i: int, r):
        self.agent_results[i] = r
        if r.config is None:
            self.q1.set_verdict(i, None, f"Échec : {r.error}", error=True)
        else:
            self.q1.set_verdict(i, r.config.params(), "", fallback=bool(r.extra.get("fallback")))
        self._refresh_verdicts()

    def open_verdicts(self):
        from interface_aco.verdicts_dialog import VerdictsDialog
        if self.verdicts_dlg is None or not self.verdicts_dlg.winfo_exists():
            self.verdicts_dlg = VerdictsDialog(self.root, on_rerun=self.rerun_failed_agents)
        self.verdicts_dlg.lift()
        self._refresh_verdicts()

    def _refresh_verdicts(self):
        if self.verdicts_dlg is not None and self.verdicts_dlg.winfo_exists():
            self.verdicts_dlg.refresh(self.agent_results, self.sma_session, self.sma_running, self.can_rerun())

    def _sma_done(self, s):
        self.sma_running = False
        self.q1.set_running(False)
        self.sma_session = s
        self._update_rerun_state()
        self._refresh_verdicts()
        if s.arbiter is None or s.arbiter.config is None:
            s.decision = "aucune décision"
            s.save()
            self.set_status("SMA : pas de décision de l'arbitre (voir les cadres des agents).", error=True)
            messagebox.showwarning("SMA", "Aucun agent n'a produit de proposition valide : pas de décision.\n"
                                          "Rien n'est appliqué. Utilisez « Relancer les agents en échec ».")
            return
        p = s.arbiter.config.params()
        from aco.tuning import fmt_params
        if s.arbiter.extra.get("fallback"):
            who = ("DÉCISION DE SECOURS (arbitre Gemini indisponible), calculée par Python.\n"
                   f"Règle : {s.arbiter.extra['rule']}.\n\nConfiguration : {fmt_params(p)}.")
        else:
            who = f"L'arbitre propose : {fmt_params(p)}."
        ans = messagebox.askyesnocancel(
            "Validation humaine — configuration de l'arbitre",
            f"{who}\n\n"
            f"Oui : appliquer au quadrant 1 et lancer l'ACO.\n"
            f"Non : appliquer sans lancer (vous pouvez modifier les valeurs, puis BC5).\n"
            f"Annuler : refuser la proposition (rien n'est appliqué).")
        if ans is None:
            s.decision = "refusée"
            s.save()
            self._refresh_verdicts()
            self.set_status("Proposition SMA refusée : quadrant 1 inchangé.")
            return
        s.decision = "acceptée et lancée" if ans else "acceptée (sans lancement)"
        s.applied = p
        s.save()
        self._refresh_verdicts()
        self.apply_config("sma", p, None, bool(ans), s.saved_to)

    # ------------------------------------------------------------ comparaison
    def open_compare(self):
        if self._lift("compare") or not self._ready():
            return
        from interface_aco.comparison_dialog_aco import ComparisonDialogACO
        configs = {"Manuel": {"params": self.manual_config(),
                              "source": self.tool_configs.get("Manuel", {}).get("source", "valeurs par défaut")}}
        for label in ("Optuna", "KerasTuner", "SMA"):
            if label in self.tool_configs:
                configs[label] = self.tool_configs[label]
        self.dialogs["compare"] = ComparisonDialogACO(self.root, self.matrix.distances_m, self.n, configs,
                                                      is_busy=lambda: self.sim_running())

    # ------------------------------------------------------------ messages
    def _tools_pump(self):
        try:
            while True:
                msg = self.tq.get_nowait()
                kind = msg[0]
                if kind == "kt_status":
                    self.q1.set_keras_status(msg[1], msg[2])
                elif kind == "sma_progress":
                    self.q1.set_progress(msg[2], f"SMA : {msg[1]}")
                    self.set_status(f"SMA : {msg[1]}")
                elif kind == "sma_agent":
                    self._show_agent(msg[1], msg[2])
                elif kind == "sma_done":
                    self._sma_done(msg[1])
                elif kind == "sma_error":
                    self.sma_running = False
                    self.q1.set_running(False)
                    self._update_rerun_state()
                    self._refresh_verdicts()
                    self.set_status("SMA : erreur.", error=True)
                    messagebox.showerror("Erreur SMA", msg[1])
        except queue.Empty:
            pass
        self.root.after(100, self._tools_pump)
