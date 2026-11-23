"""Les trois agents du SMA, adaptés à l'ACO (Ant System).

Réutilise l'infrastructure des agents de l'application AG : client Gemini (choix du modèle, repli),
recherche DuckDuckGo réelle, appel JSON validé avec une relance corrective, vérification des citations.

1. Agent web : requêtes → DuckDuckGo → configuration citant UNIQUEMENT les sources trouvées [S#].
2. Agent sémantique / scientifique : analyse théorique (rôle de ρ, α, β, m, budget de cycles).
3. Agent arbitre : compare les deux, tranche chaque divergence et dit sur quoi il fonde sa décision.
"""
from __future__ import annotations
import json
import logging
import time

import numpy as np

from agents.base import call_json
from agents.gemini_client import GeminiClient
from agents.schemas import AgentResult, SchemaError, Source
from agents import web_search
from agents.web_agent import CITE_RE, URL_RE, format_sources, validate_queries, MIN_QUERIES, MAX_QUERIES
from agents.web_search import WebHit, WebSearchError, run_searches
from aco.ant_system import AntSystem, ACOParams
from agents_aco.schemas_aco import (allowed_text, arbiter_json_schema, compare_configs, config_json_schema,
                                    make_arbiter_validator, make_config_validator)

log = logging.getLogger(__name__)

WEB, SCI, ARB = "Agent web", "Agent sémantique / scientifique", "Agent arbitre"


# ------------------------------------------------------------------ contexte
def problem_context(distances_m, n: int, start: int = 0) -> str:
    """Description factuelle, calculée sur la matrice OSRM réelle et mesurée sur cette machine."""
    d = np.asarray(distances_m, float) / 1000
    off = d[~np.eye(n, dtype=bool)]
    asym = np.abs(d - d.T)[np.triu_indices(n, 1)]
    t0 = time.perf_counter()
    AntSystem(distances_m, ACOParams(n_ants=n, cycles=20, seed=0), start).run()
    per_cycle = (time.perf_counter() - t0) / 20
    return "\n".join([
        f"- TSP fermé, {n} points GPS réels (Casablanca : Aïn Chock, Sidi Maarouf, Bouskoura), départ et "
        f"retour fixes au point 0 (Gare Casa Oasis), {n - 1} livraisons.",
        f"- Coûts = distances routières OSRM (km), matrice ASYMÉTRIQUE : écart moyen |A→B − B→A| = "
        f"{asym.mean():.2f} km, max {asym.max():.2f} km.",
        f"- Distances entre points : min {off.min():.2f} km, médiane {np.median(off):.2f} km, max {off.max():.2f} km.",
        "- Algorithme : Ant System (Dorigo). Chaque fourmi part du point 0 ; depuis i elle choisit j non visité "
        "avec p_ij ∝ τ_ij^α · η_ij^β, η_ij = 1/d_ij ; retour au point 0. Puis évaporation τ ← (1−ρ)τ et dépôt "
        "Q/L_k (Q = 1, L en km) sur les arcs orientés de chaque fourmi. τ0 = m / L_nn (plus proche voisin). "
        "Pas d'élitisme ni de recherche locale. Le meilleur tour global est conservé.",
        f"- Paramètres à régler : n_ants (m), rho (ρ), alpha (α), beta (β), cycles.",
        f"- Temps mesuré à l'instant sur cette machine : {per_cycle * 1000:.1f} ms par cycle avec {n} fourmis "
        f"(le coût est à peu près proportionnel à fourmis × cycles).",
    ])


# ------------------------------------------------------------------ agent web
DEFAULT_QUERIES = [
    "ant colony optimization TSP parameter setting alpha beta rho",
    "ant system Dorigo number of ants evaporation rate traveling salesman",
    "ACO pheromone evaporation rate stagnation TSP experiments",
    "ant colony optimization parameter tuning beta heuristic influence",
]
QUERY_SCHEMA = {"type": "object",
                "properties": {"queries": {"type": "array", "items": {"type": "string"},
                                           "minItems": MIN_QUERIES, "maxItems": MAX_QUERIES}},
                "required": ["queries"]}
QUERY_PROMPT = """Contexte du problème :
{context}

Écris entre {lo} et {hi} requêtes de moteur de recherche (en anglais, 4 à 12 mots chacune) pour trouver des
sources sur le réglage de l'Ant System / ACO pour le TSP : nombre de fourmis, évaporation ρ, α, β, nombre de
cycles, stagnation. Une requête par aspect, sans doublon. Réponds par un objet JSON {{"queries": ["..."]}}."""
WEB_SYSTEM = ("Tu es un chercheur en optimisation combinatoire. Tu proposes une configuration de l'Ant System "
              "en t'appuyant EXCLUSIVEMENT sur les sources web fournies, que tu cites par leur identifiant "
              "[S1], [S2]… Tu n'inventes aucune source, aucune URL, aucun auteur.")
WEB_PROMPT = """Contexte du problème :
{context}

Sources trouvées par une recherche DuckDuckGo réelle (titre, URL, extrait) :
{sources}

Mission : propose UNE configuration de l'Ant System pour ce TSP.
Valeurs autorisées : {allowed}

Règles de citation (vérifiées automatiquement) :
- cite uniquement les identifiants ci-dessus, sous la forme [S1], [S3]… dans la justification ;
- n'écris aucune URL et ne mentionne aucune source absente de la liste ;
- si les extraits ne couvrent pas un paramètre, dis-le (« non couvert par les sources ») et raisonne sans citation ;
- "cited_sources" contient exactement les identifiants utilisés dans la justification.

Réponds par un objet JSON : n_ants, rho, alpha, beta, cycles, justification (français, 5 à 10 phrases),
cited_sources."""
NO_SEARCH_NOTE = ("RECHERCHE WEB NON EFFECTUÉE : DuckDuckGo n'a renvoyé aucun résultat exploitable. "
                  "Proposition fondée sur les connaissances internes du modèle, SANS sources vérifiées.")


def make_cited_validator(hits: list[WebHit], n: int):
    base = make_config_validator(n)
    valid = {h.id for h in hits}
    urls = {h.url.rstrip("/") for h in hits}

    def validator(data):
        cfg = base(data)
        cited = data.get("cited_sources")
        if not isinstance(cited, list) or not all(isinstance(c, str) for c in cited):
            raise SchemaError("'cited_sources' doit être une liste d'identifiants.")
        cited = [c.strip().strip("[]") for c in cited]
        in_text = {f"S{k}" for k in CITE_RE.findall(cfg.justification)}
        unknown = (set(cited) | in_text) - valid
        if unknown:
            raise SchemaError(f"Sources citées inexistantes : {sorted(unknown)}.")
        if not in_text:
            raise SchemaError("La justification ne cite aucune source [S#].")
        foreign = [u for u in URL_RE.findall(cfg.justification) if u.rstrip("/.,);") not in urls]
        if foreign:
            raise SchemaError(f"URL hors des sources fournies : {foreign[:3]}")
        return cfg, sorted(in_text | set(cited), key=lambda x: int(x[1:]))

    return validator


def run_web(client: GeminiClient, context: str, n: int, search=None, pause=None) -> AgentResult:
    schema = config_json_schema(n)
    try:
        queries, _ = call_json(client, QUERY_PROMPT.format(context=context, lo=MIN_QUERIES, hi=MAX_QUERIES),
                               "Tu prépares des recherches web courtes et précises, en anglais.",
                               validate_queries, json_schema=QUERY_SCHEMA)
        how = "requêtes rédigées par Gemini"
    except Exception as e:  # noqa: BLE001
        log.warning("Requêtes par défaut : %s", e)
        queries, how = list(DEFAULT_QUERIES), "requêtes par défaut"
    try:
        hits, journal = run_searches(queries, search=search or web_search.ddgs_search, pause=pause)
    except WebSearchError as e:
        prompt = (f"Contexte du problème :\n{context}\n\nTu n'as PAS accès au web : ne cite aucune source ni URL.\n"
                  f"Propose UNE configuration de l'Ant System. Valeurs autorisées : {allowed_text(n)}\n"
                  "Réponds en JSON : n_ants, rho, alpha, beta, cycles, justification.")
        cfg, reply = call_json(client, prompt, WEB_SYSTEM.split(" en t'appuyant")[0] + ".",
                               make_config_validator(n), json_schema=schema)
        return AgentResult(WEB, reply.model, cfg, reply.text, [], False,
                           NO_SEARCH_NOTE + f"\nRequêtes ({how}) : " + " | ".join(queries),
                           extra={"queries": queries, "search_log": str(e)})
    cited_schema = {**schema, "properties": {**schema["properties"],
                                             "cited_sources": {"type": "array", "items": {"type": "string"}}},
                    "required": [*schema["required"], "cited_sources"]}
    (cfg, cited), reply = call_json(
        client, WEB_PROMPT.format(context=context, sources=format_sources(hits), allowed=allowed_text(n)),
        WEB_SYSTEM, make_cited_validator(hits, n), json_schema=cited_schema)
    cs = set(cited)
    sources = [Source(h.title, h.url, h.id, h.snippet, h.query, h.id in cs) for h in hits]
    note = (f"Recherche DuckDuckGo réelle : {len(queries)} requête(s) ({how}), {len(hits)} source(s), "
            f"{len(cited)} citée(s) (vérifié par Python) : {', '.join(cited)}.")
    return AgentResult(WEB, reply.model, cfg, reply.text, sources, True, note,
                       extra={"queries": queries, "search_log": journal, "cited": cited})


# --------------------------------------------------------- agent scientifique
SCI_SYSTEM = ("Tu es un spécialiste des métaheuristiques à base de colonies de fourmis. Tu raisonnes à partir des "
              "caractéristiques du problème et des mécanismes de l'algorithme : renforcement, évaporation, "
              "équilibre exploration / exploitation, stagnation, coût de calcul.")
SCI_PROMPT = """Contexte du problème :
{context}

Mission : analyse scientifiquement ce TSP et l'Ant System décrit :
- rôle de ρ (mémoire / oubli, risque de stagnation si les dépôts dominent trop vite) ;
- rôle de α et β (α grand : convergence rapide et stagnation ; β grand : comportement glouton proche du plus
  proche voisin ; β = 0 : recherche aveugle) ;
- nombre de fourmis m (diversité par cycle, coût) et budget de cycles (coût = m × cycles) ;
- particularités : matrice asymétrique, points groupés en quartiers ;
puis propose UNE configuration techniquement justifiée.

Valeurs autorisées : {allowed}

Réponds par un objet JSON avec les clés n_ants, rho, alpha, beta, cycles, justification (français, 4 à 8
phrases, sans référence bibliographique inventée)."""


def run_scientific(client: GeminiClient, context: str, n: int) -> AgentResult:
    cfg, reply = call_json(client, SCI_PROMPT.format(context=context, allowed=allowed_text(n)), SCI_SYSTEM,
                           make_config_validator(n), json_schema=config_json_schema(n))
    return AgentResult(SCI, reply.model, cfg, reply.text, note="Analyse théorique (sans recherche web).")


# ------------------------------------------------------------------ arbitre
ARB_SYSTEM = ("Tu es l'arbitre d'un comité d'experts. Tu compares des propositions de manière critique, tu "
              "identifies convergences et contradictions, tu écartes ce qui n'est pas justifié et tu construis "
              "une configuration finale cohérente. Tu expliques explicitement sur quoi tu fondes ta décision.")
ARB_PROMPT = """Contexte du problème :
{context}

Proposition de l'agent web ({web_status}) :
{web}

Proposition de l'agent sémantique / scientifique :
{sci}

Différences paramètre par paramètre (calculées par Python) :
{diff}

Mission :
1. Liste les points de convergence.
2. Pour chaque divergence, décide (valeur retenue) et explique pourquoi.
3. Écarte explicitement les arguments non justifiés (une proposition « web » sans source vérifiée n'est pas
   étayée par la littérature).
4. Vérifie la cohérence globale (équilibre α/β, ρ vs stagnation, budget fourmis × cycles vs temps mesuré).
5. Dans "basis", indique sur quoi tu fondes ta décision : quelles sources [S#] vérifiées, quels arguments
   théoriques, quelles mesures du contexte (temps, taille, asymétrie). Une entrée par élément.
6. Produis la configuration finale (valeurs autorisées : {allowed}).

Réponds par un objet JSON : {{"final": {{n_ants, rho, alpha, beta, cycles, justification}}, "basis": [..],
"convergences": [..], "divergences": [{{"parameter", "decision", "reason"}}], "rejected": [..],
"consensus": "phrase de synthèse"}}. Textes en français."""


def _fmt(res: AgentResult) -> str:
    if res.config is None:
        return f"(aucune proposition valide : {res.error})"
    d = res.config.params() | {"justification": res.config.justification}
    cited = [s for s in res.sources if s.cited]
    if cited:
        d["sources_citees"] = [f"[{s.id}] {s.title} — {s.snippet[:200]}" for s in cited]
    return json.dumps(d, ensure_ascii=False, indent=1)


def run_arbiter(client: GeminiClient, context: str, n: int, web: AgentResult, sci: AgentResult) -> AgentResult:
    diff = compare_configs(web.config, sci.config)
    diff_txt = "\n".join(f"- {d['parameter']}: web={d['web']} / scientifique={d['scientific']} "
                         f"({'accord' if d['agree'] else 'DIVERGENCE'})" for d in diff) or "(non comparable)"
    n_cited = sum(s.cited for s in web.sources)
    web_status = (f"recherche DuckDuckGo réelle, {len(web.sources)} source(s), {n_cited} citée(s), citations "
                  "vérifiées par Python" if web.web_search_used else "SANS recherche web, sources non vérifiées")
    (final, extra), reply = call_json(
        client, ARB_PROMPT.format(context=context, web=_fmt(web), sci=_fmt(sci), diff=diff_txt,
                                  web_status=web_status, allowed=allowed_text(n)),
        ARB_SYSTEM, make_arbiter_validator(n), json_schema=arbiter_json_schema(n))
    extra["python_diff"] = diff
    return AgentResult(ARB, reply.model, final, reply.text, note=extra.get("consensus", ""), extra=extra)
