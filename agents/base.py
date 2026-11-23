"""Base commune des agents : contexte du problème, appel JSON validé avec une relance corrective."""
from __future__ import annotations
import logging

import numpy as np

from agents.gemini_client import GeminiClient, GeminiReply
from agents.schemas import SchemaError, extract_json
from config.settings import POPULATION_CHOICES, CROSSOVER_CHOICES, GENERATION_CHOICES, MUTATION_CHOICES

log = logging.getLogger(__name__)

ALLOWED = (f"population_size ∈ {list(POPULATION_CHOICES)} ; selection_method ∈ ['tournament', 'roulette'] ; "
           f"mutation_rate : réel (références {list(MUTATION_CHOICES)}) ; crossover_rate ∈ {list(CROSSOVER_CHOICES)} ; "
           f"generations ∈ {list(GENERATION_CHOICES)} ; elitism : booléen ; justification : texte argumenté.")


def problem_context(problem, points=None) -> str:
    """Description factuelle du problème réel, calculée à partir de la matrice OSRM."""
    d = problem.d / 1000
    off = d[~np.eye(problem.n, dtype=bool)]
    asym = np.abs(d - d.T)[np.triu_indices(problem.n, 1)]
    lines = [
        f"- TSP fermé, {problem.n} points GPS réels (Casablanca : Aïn Chock, Sidi Maarouf, Bouskoura), "
        f"départ et retour fixes au point 0 (Gare Casa Oasis).",
        f"- Coûts = distances routières OSRM (km), matrice ASYMÉTRIQUE : écart moyen |A→B − B→A| = "
        f"{asym.mean():.2f} km, max {asym.max():.2f} km.",
        f"- Distances entre points : min {off.min():.2f} km, médiane {np.median(off):.2f} km, max {off.max():.2f} km.",
        f"- Espace de recherche : {problem.n - 1}! permutations (le départ est exclu du génome).",
        "- AG implémenté : individu = permutation des 39 autres points ; croisement Order Crossover (OX) ; "
        "mutation par inversion (appliquée à un enfant avec probabilité mutation_rate) ; "
        "sélection par tournoi (k=3) ou roue biaisée sur fitness = 1/(1+km) ; élitisme = 2 meilleurs conservés.",
        "- Ordre de grandeur mesuré sur cette machine : population 100 × 1000 générations ≈ 7 à 20 s.",
    ]
    return "\n".join(lines)


def call_json(client: GeminiClient, prompt: str, system: str, validator, *, json_schema=None,
              use_search=False, retries: int = 1) -> tuple[object, GeminiReply]:
    """Appelle Gemini, extrait et valide le JSON ; en cas d'erreur, relance une fois avec le motif."""
    reply = client.generate(prompt, system, json_schema=json_schema, use_search=use_search)
    for attempt in range(retries + 1):
        try:
            return validator(extract_json(reply.text)), reply
        except SchemaError as e:
            log.warning("Réponse JSON invalide (%s) : %s", client.model, e)
            if attempt == retries:
                raise
            fix = (f"{prompt}\n\nTa réponse précédente était invalide : {e}\n"
                   f"Réponse précédente :\n{reply.text[:3000]}\n\n"
                   "Corrige-la et renvoie UNIQUEMENT l'objet JSON valide demandé.")
            prev_sources = reply.sources if use_search else []
            reply = client.generate(fix, system, json_schema=json_schema, use_search=use_search)
            if prev_sources and not reply.sources:
                reply.sources, reply.grounded = prev_sources, True
    raise AssertionError("inaccessible")
