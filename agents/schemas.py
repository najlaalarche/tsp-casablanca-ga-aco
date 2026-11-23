"""Schémas JSON échangés entre agents et validation Python stricte (aucune correction silencieuse)."""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field, asdict

from config.settings import POPULATION_CHOICES, CROSSOVER_CHOICES, GENERATION_CHOICES

SELECTIONS = ("tournament", "roulette")
MUTATION_MIN, MUTATION_MAX = 0.001, 0.5
PARAM_KEYS = ("population_size", "selection_method", "mutation_rate", "crossover_rate",
              "generations", "elitism")


class SchemaError(ValueError):
    pass


# Schéma JSON envoyé à Gemini (structured output) pour les configurations.
CONFIG_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "population_size": {"type": "integer", "enum": list(POPULATION_CHOICES)},
        "selection_method": {"type": "string", "enum": list(SELECTIONS)},
        "mutation_rate": {"type": "number"},
        "crossover_rate": {"type": "number", "enum": list(CROSSOVER_CHOICES)},
        "generations": {"type": "integer", "enum": list(GENERATION_CHOICES)},
        "elitism": {"type": "boolean"},
        "justification": {"type": "string"},
    },
    "required": [*PARAM_KEYS, "justification"],
}

ARBITER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "final": CONFIG_JSON_SCHEMA,
        "convergences": {"type": "array", "items": {"type": "string"}},
        "divergences": {"type": "array", "items": {
            "type": "object",
            "properties": {"parameter": {"type": "string"}, "decision": {"type": "string"},
                           "reason": {"type": "string"}},
            "required": ["parameter", "decision", "reason"]}},
        "rejected": {"type": "array", "items": {"type": "string"}},
        "consensus": {"type": "string"},
    },
    "required": ["final", "convergences", "divergences", "rejected", "consensus"],
}


@dataclass
class GAConfig:
    population_size: int
    selection_method: str
    mutation_rate: float
    crossover_rate: float
    generations: int
    elitism: bool
    justification: str

    def params(self) -> dict:
        return {k: getattr(self, k) for k in PARAM_KEYS}


@dataclass
class Source:
    title: str
    uri: str
    id: str = ""          # « S1 »… (recherche DuckDuckGo)
    snippet: str = ""
    query: str = ""
    cited: bool = False   # citée par Gemini dans sa justification


@dataclass
class AgentResult:
    agent: str
    model: str
    config: GAConfig | None
    raw: str
    sources: list[Source] = field(default_factory=list)
    web_search_used: bool = False
    note: str = ""
    error: str | None = None
    extra: dict = field(default_factory=dict)  # arbitre : convergences, divergences, rejected…

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def extract_json(text: str) -> dict:
    """Extrait le premier objet JSON d'un texte (réponse brute ou bloc ```json)."""
    if not text:
        raise SchemaError("Réponse vide.")
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = m.group(1) if m else None
    if candidate is None:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise SchemaError("Aucun objet JSON trouvé dans la réponse.")
        candidate = text[start:end + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise SchemaError(f"JSON invalide : {e}") from e
    if not isinstance(data, dict):
        raise SchemaError("Le JSON doit être un objet.")
    return data


def validate_config(data: dict) -> GAConfig:
    """Valide une configuration proposée par un agent. Lève SchemaError si elle est hors domaine."""
    if not isinstance(data, dict):
        raise SchemaError("Configuration absente ou mal formée.")
    missing = [k for k in (*PARAM_KEYS, "justification") if k not in data]
    if missing:
        raise SchemaError(f"Champs manquants : {', '.join(missing)}")
    errors = []
    pop = data["population_size"]
    if isinstance(pop, bool) or not isinstance(pop, (int, float)) or int(pop) != pop \
            or int(pop) not in POPULATION_CHOICES:
        errors.append(f"population_size={pop!r} (attendu {list(POPULATION_CHOICES)})")
    sel = data["selection_method"]
    if sel not in SELECTIONS:
        errors.append(f"selection_method={sel!r} (attendu {list(SELECTIONS)})")
    mut = data["mutation_rate"]
    if isinstance(mut, bool) or not isinstance(mut, (int, float)) or not MUTATION_MIN <= mut <= MUTATION_MAX:
        errors.append(f"mutation_rate={mut!r} (attendu entre {MUTATION_MIN} et {MUTATION_MAX})")
    cx = data["crossover_rate"]
    if isinstance(cx, bool) or not isinstance(cx, (int, float)) or \
            not any(abs(cx - c) < 1e-9 for c in CROSSOVER_CHOICES):
        errors.append(f"crossover_rate={cx!r} (attendu {list(CROSSOVER_CHOICES)})")
    gen = data["generations"]
    if isinstance(gen, bool) or not isinstance(gen, (int, float)) or int(gen) != gen \
            or int(gen) not in GENERATION_CHOICES:
        errors.append(f"generations={gen!r} (attendu {list(GENERATION_CHOICES)})")
    if not isinstance(data["elitism"], bool):
        errors.append(f"elitism={data['elitism']!r} (attendu booléen)")
    just = data["justification"]
    if not isinstance(just, str) or len(just.strip()) < 20:
        errors.append("justification absente ou trop courte")
    if errors:
        raise SchemaError("Configuration invalide : " + " ; ".join(errors))
    return GAConfig(int(pop), sel, round(float(mut), 4), float(cx), int(gen), data["elitism"], just.strip())


def validate_arbiter(data: dict) -> tuple[GAConfig, dict]:
    if not isinstance(data, dict) or "final" not in data:
        raise SchemaError("La décision de l'arbitre doit contenir 'final'.")
    final = validate_config(data["final"])
    extra = {}
    for key in ("convergences", "rejected"):
        v = data.get(key, [])
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            raise SchemaError(f"'{key}' doit être une liste de textes.")
        extra[key] = v
    divs = data.get("divergences", [])
    if not isinstance(divs, list) or not all(
            isinstance(d, dict) and {"parameter", "decision", "reason"} <= d.keys() for d in divs):
        raise SchemaError("'divergences' doit être une liste d'objets {parameter, decision, reason}.")
    extra["divergences"] = divs
    cons = data.get("consensus", "")
    if not isinstance(cons, str):
        raise SchemaError("'consensus' doit être un texte.")
    extra["consensus"] = cons
    return final, extra


def compare_configs(a: GAConfig | None, b: GAConfig | None) -> list[dict]:
    """Divergences calculées par Python (indépendamment du discours de l'arbitre)."""
    if a is None or b is None:
        return []
    out = []
    for k in PARAM_KEYS:
        va, vb = getattr(a, k), getattr(b, k)
        same = abs(va - vb) < 1e-9 if isinstance(va, float) and not isinstance(va, bool) else va == vb
        out.append({"parameter": k, "web": va, "scientific": vb, "agree": same})
    return out
