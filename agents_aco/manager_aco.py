"""Orchestration du SMA ACO : agent web → agent sémantique → arbitre, avec sauvegarde JSON.

- Client résilient (agents_aco/resilient.py) : 503 → attentes 2/5/10 s, puis autres modèles qui
  répondaient au démarrage ; budget de 60 s par agent.
- Si l'arbitre Gemini échoue alors qu'au moins une proposition est valide : décision de secours
  calculée par Python, marquée comme telle, avec la règle appliquée (aucun texte attribué à Gemini).
- rerun_failed : ne réinterroge que les agents en échec (l'arbitre est relancé si besoin).
"""
from __future__ import annotations
import logging
from datetime import datetime
from typing import Callable

from agents.agent_manager import AgentsSession, _safe   # réutilisés de l'application AG
from agents.gemini_client import GeminiClient
from agents.schemas import AgentResult
from agents_aco.agents_aco import (ARB, SCI, WEB, problem_context, run_arbiter, run_scientific, run_web)
from agents_aco.resilient import ResilientGemini, make_call
from agents_aco import antigravity_backend as ag
from agents_aco.schemas_aco import PARAM_KEYS, compare_configs, make_config_validator
from config.settings import CACHE_DIR, RESULTS_DIR

log = logging.getLogger(__name__)

LAST_MODEL_FILE = CACHE_DIR / "gemini_last_model.txt"
DEFAULT_LAST_MODEL = "gemini-3.5-flash-lite"  # dernier modèle ayant répondu (2026-11-23)
FALLBACK_LABEL = "Décision de secours (arbitre Gemini indisponible)"


def last_model() -> str:
    try:
        return LAST_MODEL_FILE.read_text(encoding="utf-8").strip() or DEFAULT_LAST_MODEL
    except OSError:
        return DEFAULT_LAST_MODEL


def make_client(models: list[str] | None = None) -> ResilientGemini:
    """Dernier modèle qui a répondu (ou GEMINI_MODEL du .env), puis les modèles OK au démarrage."""
    gc = GeminiClient()
    first = gc.candidates[0] if len(gc.candidates) == 1 else last_model()
    return ResilientGemini(make_call(gc), [first, *(models or [])])


def _start(client, name):
    if hasattr(client, "start_agent"):
        client.start_agent(name)


def is_fallback(r: AgentResult | None) -> bool:
    return bool(r is not None and r.extra.get("fallback"))


def failed_agents(s: AgentsSession) -> list[str]:
    """Agents à réinterroger : sans proposition valide ; arbitre absent, en échec ou remplacé par le secours."""
    out = [k for k, r in (("web", s.web), ("scientific", s.scientific)) if r is None or r.config is None]
    if s.arbiter is None or s.arbiter.config is None or is_fallback(s.arbiter):
        out.append("arbiter")
    return out


# ------------------------------------------------------------ décision de secours
def fallback_decision(n: int, web: AgentResult, sci: AgentResult, gemini_error: str | None) -> AgentResult | None:
    """Décision calculée par Python à partir des propositions VALIDES (aucun appel à Gemini)."""
    valid = [(lbl, r.config) for lbl, r in (("web", web), ("sémantique", sci)) if r is not None and r.config]
    if not valid:
        return None
    divs = []
    if len(valid) == 1:
        lbl, c = valid[0]
        p = c.params()
        rule = f"proposition valide unique (agent {lbl}) reprise telle quelle"
    else:
        (_, a), (_, b) = valid
        p = {}
        for k in PARAM_KEYS:
            va, vb = getattr(a, k), getattr(b, k)
            if va == vb:
                p[k] = va
                continue
            if k == "cycles":
                p[k], why = max(va, vb), "le plus grand (à graine égale, des cycles en plus ne peuvent pas allonger le meilleur tour)"
            elif k == "n_ants":
                p[k], why = int(round((va + vb) / 2)), "moyenne arrondie à l'entier"
            else:
                p[k], why = round((va + vb) / 2, 3), "moyenne arithmétique"
            divs.append({"parameter": k, "decision": str(p[k]),
                         "reason": f"web = {va} / sémantique = {vb} → {why} (règle Python)"})
        rule = ("arbitrage règle par règle entre les deux propositions valides : valeur commune conservée ; "
                "sinon moyenne pour ρ, α, β, moyenne arrondie pour le nombre de fourmis, "
                "plus grand nombre de cycles")
    text = (f"{FALLBACK_LABEL}. Configuration calculée par Python, sans analyse de Gemini. "
            f"Règle appliquée : {rule}.")
    cfg = make_config_validator(n)({**p, "justification": text})
    basis = [f"proposition valide de l'agent {lbl}" for lbl, _ in valid] + [f"règle Python : {rule}"]
    return AgentResult(ARB, "Python (secours)", cfg, "", note=FALLBACK_LABEL,
                       extra={"fallback": True, "rule": rule, "gemini_error": gemini_error, "basis": basis,
                              "divergences": divs, "python_diff": compare_configs(web.config, sci.config),
                              "convergences": [], "rejected": [], "consensus": ""})


class Backend:
    """Les 3 agents sur un moteur donné : SDK Antigravity (par défaut) ou google-genai (secours)."""

    def __init__(self, kind: str, obj, search=None, pause=None):
        self.kind, self.obj, self.search, self.pause = kind, obj, search, pause
        self.model = getattr(obj, "model", None)
        self.label = "Antigravity SDK" if kind == "antigravity" else "google-genai"

    def start(self, name):
        if self.kind == "genai":
            _start(self.obj, name)

    def web(self, ctx, n):
        if self.kind == "antigravity":
            return ag.run_web_ag(self.obj, ctx, n, search=self.search, pause=self.pause or 0)
        return run_web(self.obj, ctx, n, search=self.search, pause=self.pause)

    def sci(self, ctx, n):
        if self.kind == "antigravity":
            return ag.run_scientific_ag(self.obj, ctx, n)
        return run_scientific(self.obj, ctx, n)

    def arb(self, ctx, n, web, sci):
        if self.kind == "antigravity":
            return ag.run_arbiter_ag(self.obj, ctx, n, web, sci)
        return run_arbiter(self.obj, ctx, n, web, sci)

    @property
    def log(self) -> list[str]:
        if self.kind == "antigravity":
            return [f"moteur : Antigravity SDK · modèle {self.model}"]
        return ["moteur : google-genai (secours)"] + list(getattr(self.obj, "probe_log", [])) +             list(getattr(self.obj, "journal", []))


def make_backend(client=None, runner=None, models=None, search=None, pause=None) -> Backend:
    """client fourni → google-genai ; runner fourni → Antigravity ; sinon Antigravity si le SDK est
    installé, google-genai en secours."""
    if client is not None:
        return Backend("genai", client, search, pause)
    if runner is not None:
        return Backend("antigravity", runner, search, pause)
    if ag.available():
        from agents.gemini_client import load_api_key
        import os
        model = os.getenv("GEMINI_MODEL", "").strip() or last_model()
        return Backend("antigravity", ag.AntigravityRunner(model, load_api_key()), search, pause)
    log.warning("SDK Antigravity indisponible : repli sur google-genai.")
    return Backend("genai", make_client(models), search, pause)


def _arbitrate(be: Backend, ctx, n, web, sci, progress) -> AgentResult | None:
    if not (web.config or sci.config):
        return None
    progress("agent arbitre 3/3 : comparaison et décision…", 70)
    be.start(ARB)
    arb = _safe(ARB, lambda: be.arb(ctx, n, web, sci))
    if arb.config is None:
        fb = fallback_decision(n, web, sci, arb.error)
        if fb is not None:
            progress("arbitre Gemini indisponible : décision de secours calculée par Python…", 90)
            return fb
    return arb


def _finish(be: Backend, web, sci, arb, progress, previous: str | None = None) -> AgentsSession:
    probe = be.log
    if previous:
        probe.insert(0, f"relance des agents en échec de {previous}")
    session = AgentsSession(be.model, web, sci, arb, probe=probe)
    used_gemini = any(r is not None and r.config is not None and not is_fallback(r) for r in (web, sci, arb))
    if session.model and used_gemini:
        try:
            LAST_MODEL_FILE.write_text(session.model, encoding="utf-8")
        except OSError:
            pass
    path, k = RESULTS_DIR / f"aco_agents_{datetime.now():%Y%m%d_%H%M%S}.json", 1
    while path.exists():   # une relance dans la même seconde ne doit pas écraser la session précédente
        k += 1
        path = path.with_name(f"{path.stem.split('-')[0]}-{k}.json")
    session.saved_to = str(path)
    session.save()
    progress("terminé — validation humaine demandée.", 100)
    return session


# ------------------------------------------------------------------ exécution
def run_agents_aco(distances_m, n: int, client=None, progress: Callable[[str, int], None] | None = None,
                   on_agent: Callable[[int, object], None] | None = None, search=None, pause=None,
                   models: list[str] | None = None, runner=None) -> AgentsSession:
    progress = progress or (lambda m, p: None)
    on_agent = on_agent or (lambda i, r: None)
    be = make_backend(client, runner, models, search, pause)
    progress("analyse de la matrice et mesure du temps d'un cycle…", 2)
    ctx = problem_context(distances_m, n)

    progress(f"agent web 1/3 ({be.label}) : recherche DuckDuckGo, synthèse sourcée…", 10)
    be.start(WEB)
    web = _safe(WEB, lambda: be.web(ctx, n))
    on_agent(0, web)
    progress(f"agent sémantique 2/3 ({be.label}) : analyse théorique…", 45)
    be.start(SCI)
    sci = _safe(SCI, lambda: be.sci(ctx, n))
    on_agent(1, sci)
    arb = _arbitrate(be, ctx, n, web, sci, progress)
    if arb is not None:
        on_agent(2, arb)
    return _finish(be, web, sci, arb, progress)


def rerun_failed(session: AgentsSession, distances_m, n: int, client=None,
                 progress: Callable[[str, int], None] | None = None,
                 on_agent: Callable[[int, object], None] | None = None, search=None, pause=None,
                 models: list[str] | None = None, runner=None) -> AgentsSession:
    """Réinterroge UNIQUEMENT les agents en échec. Les propositions valides sont conservées telles quelles.
    L'arbitre est relancé s'il avait échoué (ou était remplacé par le secours), ou si une proposition
    amont vient d'être obtenue."""
    progress = progress or (lambda m, p: None)
    on_agent = on_agent or (lambda i, r: None)
    todo = failed_agents(session)
    if not todo:
        progress("aucun agent en échec.", 100)
        return session
    be = make_backend(client, runner, models, search, pause)
    progress("analyse de la matrice…", 2)
    ctx = problem_context(distances_m, n)
    web, sci, arb = session.web, session.scientific, session.arbiter
    changed = False
    if "web" in todo:
        progress("relance de l'agent web…", 10)
        be.start(WEB)
        web = _safe(WEB, lambda: be.web(ctx, n))
        on_agent(0, web)
        changed |= web.config is not None
    if "scientific" in todo:
        progress("relance de l'agent sémantique…", 45)
        be.start(SCI)
        sci = _safe(SCI, lambda: be.sci(ctx, n))
        on_agent(1, sci)
        changed |= sci.config is not None
    if "arbiter" in todo or changed:
        arb = _arbitrate(be, ctx, n, web, sci, progress)
        if arb is not None:
            on_agent(2, arb)
    return _finish(be, web, sci, arb, progress, previous=session.saved_to)
