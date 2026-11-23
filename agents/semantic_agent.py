"""Agent 2 — Scientific / Semantic : analyse théorique du problème et des interactions d'hyperparamètres."""
from __future__ import annotations

from agents.base import ALLOWED, call_json
from agents.gemini_client import GeminiClient
from agents.schemas import AgentResult, CONFIG_JSON_SCHEMA, validate_config

NAME = "Agent Scientifique"

SYSTEM = ("Tu es un spécialiste des métaheuristiques et de la théorie des algorithmes génétiques. "
          "Tu raisonnes à partir des caractéristiques du problème : taille de l'espace de recherche, "
          "asymétrie des coûts, pression de sélection, diversité, convergence prématurée, budget de calcul.")

PROMPT = """Contexte du problème :
{context}

Mission : analyse scientifiquement ce TSP et l'AG décrit :
- caractéristiques du TSP (taille, asymétrie, structure géographique en grappes) ;
- risques de convergence prématurée (petite population, roue biaisée avec une fitness 1/(1+km) très
  plate, élitisme) ;
- interactions entre hyperparamètres (population × générations = budget ; mutation vs croisement ;
  pression de sélection vs diversité) ;
puis propose UNE configuration techniquement justifiée.

Valeurs autorisées : {allowed}

Réponds par un objet JSON avec les clés population_size, selection_method, mutation_rate,
crossover_rate, generations, elitism, justification (en français, 4 à 8 phrases, sans référence
bibliographique inventée)."""


def run(client: GeminiClient, context: str) -> AgentResult:
    cfg, reply = call_json(client, PROMPT.format(context=context, allowed=ALLOWED), SYSTEM,
                           validate_config, json_schema=CONFIG_JSON_SCHEMA)
    return AgentResult(NAME, reply.model, cfg, reply.text,
                       note="Analyse théorique (sans recherche web).")
