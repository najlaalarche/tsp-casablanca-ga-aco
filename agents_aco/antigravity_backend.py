"""SMA ACO avec le SDK officiel Google Antigravity (google-antigravity).

Trois agents Antigravity (Agent + LocalAgentConfig) :
- agent web : outil Python personnalisé « recherche_web » (DuckDuckGo) qu'il appelle lui-même ;
  il ne peut citer que les sources [S#] renvoyées par cet outil (vérifié par Python) ;
- agent sémantique : sans outil, analyse théorique de l'Ant System ;
- arbitre : reçoit les deux propositions et décide.

Sorties JSON (response_schema) validées par Python, une relance corrective dans la même session,
délai de 60 s par agent. La clé GEMINI_API_KEY est lue dans .env et n'est jamais journalisée.
L'exécution réelle passe par AntigravityRunner ; les tests injectent un faux runner (aucun réseau).
"""
from __future__ import annotations
import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Callable

from agents.schemas import AgentResult, SchemaError, Source, extract_json
from agents.web_search import WebHit
from agents import web_search
from agents_aco.agents_aco import (ARB, ARB_PROMPT, ARB_SYSTEM, SCI, SCI_PROMPT, SCI_SYSTEM, WEB, _fmt)
from agents_aco.agents_aco import make_cited_validator
from agents_aco.schemas_aco import (allowed_text, arbiter_json_schema, compare_configs, config_json_schema,
                                    make_arbiter_validator, make_config_validator)

log = logging.getLogger(__name__)
SDK_LABEL = "Antigravity SDK"
AGENT_TIMEOUT_S = 60
MAX_RESULTS_PER_SEARCH = 5
MAX_SEARCHES = 5


def available() -> bool:
    try:
        import google.antigravity  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------------ runner
@dataclass
class Turn:
    data: dict | None          # sortie structurée (dict) si fournie par le SDK
    text: str                  # texte de la réponse
    tool_calls: list[dict]     # [{"name", "args"}]


class AntigravityRunner:
    """Exécute un agent Antigravity : une session, un prompt, relances correctives éventuelles."""

    def __init__(self, model: str, api_key: str, timeout_s: float = AGENT_TIMEOUT_S):
        self.model, self._key, self.timeout_s = model, api_key, timeout_s

    def run(self, system: str, prompts: Callable[[str | None], str | None], schema: dict,
            tools: list[Callable] | None = None) -> list[Turn]:
        """prompts(erreur) renvoie le prompt suivant (None = arrêter). Toujours dans le délai de 60 s."""
        return asyncio.run(asyncio.wait_for(self._session(system, prompts, schema, tools or []),
                                            self.timeout_s))

    async def _session(self, system, prompts, schema, tools) -> list[Turn]:
        from google.antigravity import Agent, CapabilitiesConfig, LocalAgentConfig
        cfg = LocalAgentConfig(
            model=self.model, api_key=self._key, system_instructions=system, tools=tools,
            # aucun outil intégré (fichiers, commandes) : uniquement nos outils Python
            capabilities=CapabilitiesConfig(enabled_tools=[], enable_subagents=False),
            response_schema=schema)
        turns: list[Turn] = []
        async with Agent(cfg) as agent:
            prompt = prompts(None)
            while prompt is not None:
                resp = await agent.chat(prompt)
                calls = [{"name": str(c.name), "args": dict(c.args)} async for c in resp.tool_calls]
                data = await resp.structured_output()
                text = await resp.text()
                if data is not None and not isinstance(data, dict):
                    try:
                        data = json.loads(data) if isinstance(data, str) else dict(data)
                    except (TypeError, ValueError):
                        data = None
                turns.append(Turn(data, text, calls))
                prompt = prompts(turns)
        return turns


# ------------------------------------------------------------ appel validé
@dataclass
class AgentCall:
    validator: Callable
    first_prompt: str
    retries: int = 1
    turns: list[Turn] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    value: object = None

    def next_prompt(self, turns):
        """Valide la dernière réponse ; si invalide, renvoie un prompt correctif (une fois)."""
        if turns is None:
            return self.first_prompt
        last = turns[-1]
        try:
            payload = last.data if last.data is not None else extract_json(last.text)
            self.value = self.validator(payload)
            return None
        except SchemaError as e:
            self.errors.append(str(e))
            if len(self.errors) > self.retries:
                return None
            return (f"Ta réponse précédente est invalide : {e}\n"
                    "Corrige-la et renvoie UNIQUEMENT l'objet JSON demandé.")


def _call(runner, system, prompt, schema, validator, tools=None) -> tuple[object, AgentCall]:
    call = AgentCall(validator, prompt)
    try:
        call.turns = runner.run(system, call.next_prompt, schema, tools)
    except asyncio.TimeoutError:
        raise TimeoutError(f"délai de {AGENT_TIMEOUT_S} s dépassé (Antigravity)") from None
    if call.value is None:
        raise SchemaError("Réponse JSON invalide : " + " | ".join(call.errors or ["aucune réponse"]))
    return call.value, call


def _label(runner) -> str:
    return f"{SDK_LABEL} · {runner.model}"


# ------------------------------------------------------------------ agents
WEB_SYSTEM_AG = ("Tu es un chercheur en optimisation combinatoire. Tu disposes d'un outil recherche_web(requete) "
                 "qui interroge DuckDuckGo et renvoie des sources numérotées [S1], [S2]… Appelle-le toi-même "
                 "(2 à 5 requêtes courtes en anglais), puis propose une configuration de l'Ant System en citant "
                 "EXCLUSIVEMENT les identifiants renvoyés par l'outil. N'invente aucune source, URL ni auteur.")
WEB_PROMPT_AG = """Contexte du problème :
{context}

Mission : utilise l'outil recherche_web pour trouver des sources sur le réglage de l'Ant System / ACO pour le
TSP (nombre de fourmis, ρ, α, β, cycles, stagnation), puis propose UNE configuration.
Valeurs autorisées : {allowed}
Règles (vérifiées par Python) : cite uniquement des identifiants [S#] renvoyés par l'outil, sous la forme [S1] dans
la justification ; aucune URL ; si les sources ne couvrent pas un paramètre, dis-le ; "cited_sources" contient
exactement les identifiants cités. Réponds par un objet JSON : n_ants, rho, alpha, beta, cycles, justification
(français, 5 à 10 phrases), cited_sources."""


def make_search_tool(search=None, pause=0.0):
    """Outil « recherche_web » exposé à l'agent ; mémorise les sources et le journal des appels."""
    search = search or web_search.ddgs_search
    hits: list[WebHit] = []
    log_calls: list[dict] = []
    seen: set[str] = set()

    def recherche_web(requete: str) -> str:
        """Recherche DuckDuckGo. Renvoie une liste JSON de sources {id, titre, url, extrait} à citer par leur id.

        Args:
            requete: requête courte en anglais (4 à 12 mots).
        """
        requete = " ".join(str(requete).split())[:200]
        if len(log_calls) >= MAX_SEARCHES:
            log_calls.append({"requete": requete, "resultats": 0, "erreur": "limite de recherches atteinte"})
            return json.dumps({"erreur": f"limite de {MAX_SEARCHES} recherches atteinte"}, ensure_ascii=False)
        try:
            raw = search(requete, MAX_RESULTS_PER_SEARCH) or []
        except Exception as e:  # noqa: BLE001  (réseau, limitation DuckDuckGo)
            log_calls.append({"requete": requete, "resultats": 0, "erreur": type(e).__name__})
            return json.dumps({"erreur": f"recherche indisponible ({type(e).__name__})"}, ensure_ascii=False)
        new = []
        for r in raw:
            url = (r.get("href") or r.get("url") or "").strip()
            title = (r.get("title") or "").strip()
            if not url.startswith(("http://", "https://")) or url in seen or not title:
                continue
            seen.add(url)
            h = WebHit(f"S{len(hits) + 1}", title, url, " ".join((r.get("body") or "").split())[:400], requete)
            hits.append(h)
            new.append({"id": h.id, "titre": h.title, "url": h.url, "extrait": h.snippet})
        log_calls.append({"requete": requete, "resultats": len(new)})
        return json.dumps(new, ensure_ascii=False)

    return recherche_web, hits, log_calls


def run_web_ag(runner, context: str, n: int, search=None, pause=0.0) -> AgentResult:
    tool, hits, calls = make_search_tool(search, pause)
    schema = config_json_schema(n)
    schema = {**schema, "properties": {**schema["properties"],
                                       "cited_sources": {"type": "array", "items": {"type": "string"}}},
              "required": [*schema["required"], "cited_sources"]}

    def validator(data):
        if not hits:
            raise SchemaError("l'agent n'a obtenu aucune source par recherche_web : appelle l'outil avant de répondre.")
        return make_cited_validator(hits, n)(data)

    try:
        (cfg, cited), call = _call(runner, WEB_SYSTEM_AG, WEB_PROMPT_AG.format(context=context, allowed=allowed_text(n)),
                                   schema, validator, tools=[tool])
    except Exception as e:  # noqa: BLE001
        return AgentResult(WEB, _label(runner), None, "", error=str(e),
                           extra={"sdk": SDK_LABEL, "tool_calls": calls})
    cs = set(cited)
    sources = [Source(h.title, h.url, h.id, h.snippet, h.query, h.id in cs) for h in hits]
    note = (f"{SDK_LABEL} : {len(calls)} appel(s) de l'outil recherche_web, {len(hits)} source(s), "
            f"{len(cited)} citée(s) (vérifié par Python) : {', '.join(cited)}.")
    return AgentResult(WEB, _label(runner), cfg, call.turns[-1].text, sources, True, note,
                       extra={"sdk": SDK_LABEL, "tool_calls": calls, "cited": cited})


def run_scientific_ag(runner, context: str, n: int) -> AgentResult:
    cfg, call = _call(runner, SCI_SYSTEM, SCI_PROMPT.format(context=context, allowed=allowed_text(n)),
                      config_json_schema(n), make_config_validator(n))
    return AgentResult(SCI, _label(runner), cfg, call.turns[-1].text,
                       note=f"Analyse théorique (agent {SDK_LABEL} sans outil).", extra={"sdk": SDK_LABEL})


def run_arbiter_ag(runner, context: str, n: int, web: AgentResult, sci: AgentResult) -> AgentResult:
    diff = compare_configs(web.config, sci.config)
    diff_txt = "\n".join(f"- {d['parameter']}: web={d['web']} / scientifique={d['scientific']} "
                         f"({'accord' if d['agree'] else 'DIVERGENCE'})" for d in diff) or "(non comparable)"
    n_cited = sum(s.cited for s in web.sources)
    web_status = (f"recherche DuckDuckGo réelle via l'outil recherche_web, {len(web.sources)} source(s), {n_cited} "
                  "citée(s), citations vérifiées par Python" if web.web_search_used
                  else "SANS recherche web, sources non vérifiées")
    (final, extra), call = _call(
        runner, ARB_SYSTEM, ARB_PROMPT.format(context=context, web=_fmt(web), sci=_fmt(sci), diff=diff_txt,
                                              web_status=web_status, allowed=allowed_text(n)),
        arbiter_json_schema(n), make_arbiter_validator(n))
    extra["python_diff"] = diff
    extra["sdk"] = SDK_LABEL
    return AgentResult(ARB, _label(runner), final, call.turns[-1].text, note=extra.get("consensus", ""), extra=extra)
