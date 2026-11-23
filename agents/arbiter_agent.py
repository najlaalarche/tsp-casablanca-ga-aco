"""Agent 3 — Reasoner / Arbiter : compare les deux propositions et construit un consensus."""
from __future__ import annotations
import json

from agents.base import ALLOWED, call_json
from agents.gemini_client import GeminiClient
from agents.schemas import AgentResult, ARBITER_JSON_SCHEMA, compare_configs, validate_arbiter

NAME = "Agent Arbitre"

SYSTEM = ("Tu es l'arbitre d'un comité d'experts. Tu compares des propositions de manière critique, "
          "tu identifies convergences et contradictions, tu écartes ce qui n'est pas justifié et tu "
          "construis une configuration finale cohérente, recherchant le consensus.")

PROMPT = """Contexte du problème :
{context}

Proposition de l'agent Web Research ({web_status}) :
{web}

Proposition de l'agent Scientifique :
{sci}

Différences paramètre par paramètre (calculées par Python) :
{diff}

Mission :
1. Liste les points de convergence.
2. Pour chaque divergence, décide (valeur retenue) et explique pourquoi.
3. Écarte explicitement les arguments non justifiés ou non vérifiés (en particulier une proposition
   « web » sans source vérifiée ne doit pas être considérée comme étayée par la littérature).
4. Vérifie la cohérence globale (budget population × générations, pression de sélection, diversité).
5. Produis la configuration finale (valeurs autorisées : {allowed}).

Réponds par un objet JSON : {{"final": {{population_size, selection_method, mutation_rate, crossover_rate,
generations, elitism, justification}}, "convergences": [..], "divergences": [{{"parameter", "decision",
"reason"}}], "rejected": [..], "consensus": "phrase de synthèse"}}. Textes en français."""


def _fmt(res: AgentResult) -> str:
    if res.config is None:
        return f"(aucune proposition valide : {res.error})"
    d = res.config.params() | {"justification": res.config.justification}
    cited = [s for s in res.sources if s.cited]
    if cited:
        d["sources_citees"] = [f"[{s.id}] {s.title} — {s.snippet[:200]}" for s in cited]
    return json.dumps(d, ensure_ascii=False, indent=1)


def run(client: GeminiClient, context: str, web: AgentResult, sci: AgentResult) -> AgentResult:
    diff = compare_configs(web.config, sci.config)
    diff_txt = "\n".join(f"- {d['parameter']}: web={d['web']} / scientifique={d['scientific']} "
                         f"({'accord' if d['agree'] else 'DIVERGENCE'})" for d in diff) or "(non comparable)"
    n_cited = sum(s.cited for s in web.sources)
    web_status = (f"recherche DuckDuckGo réelle, {len(web.sources)} source(s) collectée(s), "
                  f"{n_cited} citée(s) — citations vérifiées par Python" if web.web_search_used
                  else "SANS recherche web, sources non vérifiées")
    prompt = PROMPT.format(context=context, web=_fmt(web), sci=_fmt(sci), diff=diff_txt,
                           web_status=web_status, allowed=ALLOWED)
    (final, extra), reply = call_json(client, prompt, SYSTEM, validate_arbiter,
                                      json_schema=ARBITER_JSON_SCHEMA)
    extra["python_diff"] = diff
    return AgentResult(NAME, reply.model, final, reply.text, note=extra.get("consensus", ""), extra=extra)
