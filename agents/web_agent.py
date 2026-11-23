"""Agent 1 — Web Research : recherche DuckDuckGo réelle (ddgs) puis synthèse Gemini sourcée.

Déroulement :
1. Gemini propose 3 à 5 requêtes de recherche (validées par Python ; requêtes par défaut sinon).
2. Python exécute les requêtes sur DuckDuckGo et collecte titre + URL + extrait, numérotés [S1]…
3. Gemini reçoit UNIQUEMENT ces sources et propose une configuration en les citant.
4. Python vérifie : identifiants cités existants, au moins une citation, aucune URL hors liste.
   Une réponse non conforme est renvoyée une fois au modèle avec le motif, puis rejetée.

Google Search grounding n'est plus utilisé : il est refusé (429) sur le palier gratuit.
"""
from __future__ import annotations
import logging
import re

from agents.base import ALLOWED, call_json
from agents.gemini_client import GeminiClient
from agents.schemas import (AgentResult, CONFIG_JSON_SCHEMA, SchemaError, Source, validate_config)
from agents import web_search
from agents.web_search import WebHit, WebSearchError, run_searches

log = logging.getLogger(__name__)
NAME = "Agent Web Research"
MIN_QUERIES, MAX_QUERIES = 3, 5

DEFAULT_QUERIES = [
    "genetic algorithm traveling salesman problem parameter tuning population size mutation rate",
    "tournament selection vs roulette wheel selection genetic algorithm TSP comparison",
    "order crossover OX inversion mutation TSP genetic algorithm crossover rate",
    "elitism premature convergence genetic algorithm traveling salesman",
]

QUERY_SCHEMA = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"},
                               "minItems": MIN_QUERIES, "maxItems": MAX_QUERIES}},
    "required": ["queries"],
}

QUERY_SYSTEM = ("Tu prépares des recherches web pour un chercheur en optimisation combinatoire. "
                "Tu écris des requêtes courtes et précises, en anglais (littérature plus abondante).")

QUERY_PROMPT = """Contexte du problème :
{context}

Écris entre {lo} et {hi} requêtes de moteur de recherche (en anglais, 4 à 12 mots chacune) pour trouver
des sources sur le réglage d'un algorithme génétique pour ce TSP : taille de population, sélection
tournoi vs roue biaisée, taux de mutation et de croisement, nombre de générations, élitisme,
opérateurs OX et inversion. Une requête par aspect, sans doublon.
Réponds par un objet JSON {{"queries": ["...", "..."]}}."""

SYSTEM = ("Tu es un chercheur en optimisation combinatoire. Tu proposes une configuration d'algorithme "
          "génétique en t'appuyant EXCLUSIVEMENT sur les sources web fournies, que tu cites par leur "
          "identifiant [S1], [S2]… Tu n'inventes aucune source, aucune URL, aucun auteur.")

PROMPT = """Contexte du problème :
{context}

Sources trouvées par une recherche DuckDuckGo réelle (titre, URL, extrait) :
{sources}

Mission : propose UNE configuration de l'algorithme génétique pour ce TSP.
Valeurs autorisées : {allowed}

Règles de citation (vérifiées automatiquement) :
- cite uniquement les identifiants ci-dessus, sous la forme [S1], [S3]… dans la justification ;
- n'écris aucune URL et ne mentionne aucune source absente de la liste ;
- si les extraits ne justifient pas un paramètre, dis-le explicitement (« non couvert par les sources »)
  et donne ton raisonnement sans citation pour ce paramètre ;
- la liste "cited_sources" contient exactement les identifiants utilisés dans la justification.

Réponds par un objet JSON avec les clés population_size, selection_method, mutation_rate,
crossover_rate, generations, elitism, justification (français, 5 à 10 phrases), cited_sources."""

CITED_SCHEMA = {**CONFIG_JSON_SCHEMA,
                "properties": {**CONFIG_JSON_SCHEMA["properties"],
                               "cited_sources": {"type": "array", "items": {"type": "string"}}},
                "required": [*CONFIG_JSON_SCHEMA["required"], "cited_sources"]}

NO_SEARCH_NOTE = ("RECHERCHE WEB NON EFFECTUÉE : DuckDuckGo n'a renvoyé aucun résultat exploitable. "
                  "Proposition fondée sur les connaissances internes du modèle, SANS sources vérifiées.")

CITE_RE = re.compile(r"\[S(\d+)\]")
URL_RE = re.compile(r"https?://\S+")


# ------------------------------------------------------------------ requêtes
def validate_queries(data: dict) -> list[str]:
    qs = data.get("queries") if isinstance(data, dict) else None
    if not isinstance(qs, list):
        raise SchemaError("'queries' doit être une liste.")
    clean, seen = [], set()
    for q in qs:
        if not isinstance(q, str):
            raise SchemaError("Chaque requête doit être un texte.")
        q = " ".join(q.split())
        if 3 <= len(q) <= 200 and q.lower() not in seen:
            seen.add(q.lower())
            clean.append(q)
    if not MIN_QUERIES <= len(clean) <= MAX_QUERIES:
        raise SchemaError(f"{len(clean)} requête(s) valide(s), {MIN_QUERIES} à {MAX_QUERIES} attendues.")
    return clean


def plan_queries(client: GeminiClient, context: str) -> tuple[list[str], str]:
    try:
        qs, _ = call_json(client, QUERY_PROMPT.format(context=context, lo=MIN_QUERIES, hi=MAX_QUERIES),
                          QUERY_SYSTEM, validate_queries, json_schema=QUERY_SCHEMA)
        return qs, "requêtes rédigées par Gemini"
    except Exception as e:  # noqa: BLE001  (pas de blocage : requêtes par défaut documentées)
        log.warning("Requêtes Gemini indisponibles, requêtes par défaut : %s", e)
        return list(DEFAULT_QUERIES), f"requêtes par défaut (Gemini : {e})"


# ------------------------------------------------------------- citations
def make_cited_validator(hits: list[WebHit]):
    valid = {h.id for h in hits}
    urls = {h.url.rstrip("/") for h in hits}

    def validator(data: dict):
        cfg = validate_config(data)
        cited = data.get("cited_sources")
        if not isinstance(cited, list) or not all(isinstance(c, str) for c in cited):
            raise SchemaError("'cited_sources' doit être une liste d'identifiants.")
        cited = [c.strip().strip("[]") for c in cited]
        in_text = {f"S{n}" for n in CITE_RE.findall(cfg.justification)}
        unknown = (set(cited) | in_text) - valid
        if unknown:
            raise SchemaError(f"Sources citées inexistantes : {sorted(unknown)} "
                              f"(seules {sorted(valid, key=lambda x: int(x[1:]))} sont autorisées).")
        if not in_text:
            raise SchemaError("La justification ne cite aucune source [S#].")
        foreign = [u for u in URL_RE.findall(cfg.justification) if u.rstrip("/.,);") not in urls]
        if foreign:
            raise SchemaError(f"URL hors des sources fournies : {foreign[:3]}")
        return cfg, sorted(in_text | set(cited), key=lambda x: int(x[1:]))

    return validator


def format_sources(hits: list[WebHit]) -> str:
    return "\n".join(f"[{h.id}] {h.title}\n     URL : {h.url}\n     Extrait : {h.snippet or '(aucun)'}"
                     for h in hits)


# ------------------------------------------------------------------- agent
def run(client: GeminiClient, context: str, search=None, pause: float | None = None) -> AgentResult:
    """search : fonction (requête, n) -> résultats ; par défaut DuckDuckGo (web_search.ddgs_search)."""
    queries, how = plan_queries(client, context)
    try:
        hits, journal = run_searches(queries, search=search or web_search.ddgs_search, pause=pause)
    except WebSearchError as e:
        log.warning("Recherche DuckDuckGo indisponible : %s", e)
        prompt = (PROMPT.split("Sources trouvées")[0].format(context=context) +
                  "Tu n'as PAS accès au web : ne cite aucune source ni URL.\n"
                  f"Propose UNE configuration. Valeurs autorisées : {ALLOWED}\n"
                  "Réponds en JSON : population_size, selection_method, mutation_rate, crossover_rate, "
                  "generations, elitism, justification.")
        cfg, reply = call_json(client, prompt, SYSTEM.split(" en t'appuyant")[0] + ".",
                               validate_config, json_schema=CONFIG_JSON_SCHEMA)
        return AgentResult(NAME, reply.model, cfg, reply.text, [], False,
                           NO_SEARCH_NOTE + f"\nRequêtes ({how}) : " + " | ".join(queries) + f"\nDétail : {e}",
                           extra={"queries": queries, "search_log": str(e)})

    (cfg, cited), reply = call_json(
        client, PROMPT.format(context=context, sources=format_sources(hits), allowed=ALLOWED),
        SYSTEM, make_cited_validator(hits), json_schema=CITED_SCHEMA)
    cited_set = set(cited)
    sources = [Source(h.title, h.url, h.id, h.snippet, h.query, h.id in cited_set) for h in hits]
    note = (f"Recherche DuckDuckGo réelle : {len(queries)} requête(s) ({how}), {len(hits)} source(s) "
            f"collectée(s), {len(cited)} citée(s) par Gemini : {', '.join(cited)}.")
    return AgentResult(NAME, reply.model, cfg, reply.text, sources, True, note,
                       extra={"queries": queries, "search_log": journal, "cited": cited})
