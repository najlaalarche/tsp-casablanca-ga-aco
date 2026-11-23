"""Schémas JSON des agents ACO et validation Python stricte (aucune correction silencieuse)."""
from __future__ import annotations
from dataclasses import dataclass

from agents.schemas import SchemaError
from aco.settings_aco import CYCLE_CHOICES

PARAM_KEYS = ("n_ants", "rho", "alpha", "beta", "cycles")
RHO_MIN, RHO_MAX = 0.01, 0.99      # ρ strictement entre 0 et 1
ALPHA_MAX, BETA_MAX = 10.0, 15.0
MIN_JUSTIFICATION = 40


def ants_bounds(n_points: int) -> tuple[int, int]:
    return 5, max(10, 3 * n_points)


def allowed_text(n_points: int) -> str:
    lo, hi = ants_bounds(n_points)
    return (f"n_ants : entier de {lo} à {hi} (usage courant : m = nombre de points = {n_points}) ; "
            f"rho : réel strictement entre 0 et 1 ({RHO_MIN} à {RHO_MAX}) ; alpha : réel > 0 (≤ {ALPHA_MAX}) ; "
            f"beta : réel > 0 (≤ {BETA_MAX}) ; cycles ∈ {list(CYCLE_CHOICES)} ; "
            f"justification : texte argumenté.")


def config_json_schema(n_points: int) -> dict:
    lo, hi = ants_bounds(n_points)
    return {
        "type": "object",
        "properties": {
            "n_ants": {"type": "integer", "minimum": lo, "maximum": hi},
            "rho": {"type": "number"},
            "alpha": {"type": "number"},
            "beta": {"type": "number"},
            "cycles": {"type": "integer", "enum": list(CYCLE_CHOICES)},
            "justification": {"type": "string"},
        },
        "required": [*PARAM_KEYS, "justification"],
    }


def arbiter_json_schema(n_points: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "final": config_json_schema(n_points),
            "basis": {"type": "array", "items": {"type": "string"}},
            "convergences": {"type": "array", "items": {"type": "string"}},
            "divergences": {"type": "array", "items": {
                "type": "object",
                "properties": {"parameter": {"type": "string"}, "decision": {"type": "string"},
                               "reason": {"type": "string"}},
                "required": ["parameter", "decision", "reason"]}},
            "rejected": {"type": "array", "items": {"type": "string"}},
            "consensus": {"type": "string"},
        },
        "required": ["final", "basis", "convergences", "divergences", "rejected", "consensus"],
    }


@dataclass
class ACOConfig:
    n_ants: int
    rho: float
    alpha: float
    beta: float
    cycles: int
    justification: str

    def params(self) -> dict:
        return {k: getattr(self, k) for k in PARAM_KEYS}


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def make_config_validator(n_points: int):
    lo, hi = ants_bounds(n_points)

    def validate(data: dict) -> ACOConfig:
        if not isinstance(data, dict):
            raise SchemaError("Configuration absente ou mal formée.")
        missing = [k for k in (*PARAM_KEYS, "justification") if k not in data]
        if missing:
            raise SchemaError(f"Champs manquants : {', '.join(missing)}")
        errors = []
        a = data["n_ants"]
        if not _num(a) or int(a) != a or not lo <= int(a) <= hi:
            errors.append(f"n_ants={a!r} (attendu entier de {lo} à {hi})")
        r = data["rho"]
        if not _num(r) or not RHO_MIN <= r <= RHO_MAX:
            errors.append(f"rho={r!r} (attendu strictement entre 0 et 1 : {RHO_MIN} à {RHO_MAX})")
        al = data["alpha"]
        if not _num(al) or not 0 < al <= ALPHA_MAX:
            errors.append(f"alpha={al!r} (attendu > 0 et ≤ {ALPHA_MAX})")
        b = data["beta"]
        if not _num(b) or not 0 < b <= BETA_MAX:
            errors.append(f"beta={b!r} (attendu > 0 et ≤ {BETA_MAX})")
        c = data["cycles"]
        if not _num(c) or int(c) != c or int(c) not in CYCLE_CHOICES:
            errors.append(f"cycles={c!r} (attendu {list(CYCLE_CHOICES)})")
        j = data["justification"]
        if not isinstance(j, str) or len(j.strip()) < MIN_JUSTIFICATION:
            errors.append("justification absente ou trop courte")
        if errors:
            raise SchemaError("Configuration invalide : " + " ; ".join(errors))
        return ACOConfig(int(a), round(float(r), 3), round(float(al), 3), round(float(b), 3), int(c),
                         j.strip())

    return validate


def make_arbiter_validator(n_points: int):
    validate_config = make_config_validator(n_points)

    def validate(data: dict) -> tuple[ACOConfig, dict]:
        if not isinstance(data, dict) or "final" not in data:
            raise SchemaError("La décision de l'arbitre doit contenir 'final'.")
        final = validate_config(data["final"])
        extra = {}
        for key in ("basis", "convergences", "rejected"):
            v = data.get(key, [])
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                raise SchemaError(f"'{key}' doit être une liste de textes.")
            extra[key] = [x.strip() for x in v if x.strip()]
        if not extra["basis"]:
            raise SchemaError("'basis' doit indiquer sur quoi l'arbitre fonde sa décision (au moins un élément).")
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

    return validate


def compare_configs(a: ACOConfig | None, b: ACOConfig | None) -> list[dict]:
    """Divergences calculées par Python, indépendamment du discours de l'arbitre."""
    if a is None or b is None:
        return []
    out = []
    for k in PARAM_KEYS:
        va, vb = getattr(a, k), getattr(b, k)
        out.append({"parameter": k, "web": va, "scientific": vb, "agree": abs(va - vb) < 1e-9})
    return out
