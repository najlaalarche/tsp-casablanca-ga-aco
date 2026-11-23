"""Orchestration : Web Research -> Scientifique -> Arbitre, avec suivi et sauvegarde."""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from agents import arbiter_agent, semantic_agent, web_agent
from agents.base import problem_context
from agents.gemini_client import GeminiClient, GeminiConfigError
from agents.schemas import AgentResult, SchemaError
from config.settings import RESULTS_DIR

log = logging.getLogger(__name__)


@dataclass
class AgentsSession:
    model: str | None
    web: AgentResult
    scientific: AgentResult
    arbiter: AgentResult | None
    decision: str = "en attente"          # acceptée / modifiée / refusée
    applied: dict | None = None
    saved_to: str | None = None
    probe: list[str] = field(default_factory=list)  # disponibilité des modèles testés
    started: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> dict:
        return {"date": self.started, "model": self.model, "model_probe": self.probe,
                "decision": self.decision,
                "applied_params": self.applied, "web": self.web.to_dict(),
                "scientific": self.scientific.to_dict(),
                "arbiter": self.arbiter.to_dict() if self.arbiter else None}

    def save(self) -> str | None:
        try:
            path = Path(self.saved_to) if self.saved_to else \
                RESULTS_DIR / f"agents_{datetime.now():%Y%m%d_%H%M%S}.json"
            path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
            self.saved_to = str(path)
            return self.saved_to
        except OSError as e:
            log.warning("Sauvegarde agents impossible : %s", e)
            return None


def _safe(name: str, fn) -> AgentResult:
    try:
        return fn()
    except GeminiConfigError:
        raise
    except (SchemaError, Exception) as e:  # noqa: BLE001
        log.exception("%s", name)
        return AgentResult(name, "", None, "", error=str(e))


def run_agents(problem, client: GeminiClient | None = None,
               progress: Callable[[str, int], None] | None = None) -> AgentsSession:
    """Exécute les trois agents. progress(message, pourcentage)."""
    progress = progress or (lambda m, p: None)
    client = client or GeminiClient()  # GeminiConfigError si clé absente
    ctx = problem_context(problem)
    progress("Sélection d'un modèle Gemini disponible…", 2)
    client.select_model()

    progress("Agent 1/3 — Web Research : requêtes, recherche DuckDuckGo, synthèse sourcée…", 5)
    web = _safe(web_agent.NAME, lambda: web_agent.run(client, ctx))
    progress("Agent 2/3 — Scientifique : analyse théorique…", 40)
    sci = _safe(semantic_agent.NAME, lambda: semantic_agent.run(client, ctx))
    arb = None
    if web.config or sci.config:
        progress("Agent 3/3 — Arbitre : recherche de consensus…", 70)
        arb = _safe(arbiter_agent.NAME, lambda: arbiter_agent.run(client, ctx, web, sci))
    session = AgentsSession(client.model, web, sci, arb, probe=list(client.probe_log))
    session.save()
    progress("Terminé.", 100)
    return session
