"""Cache JSON local indexé par une empreinte des coordonnées."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from config.settings import CACHE_DIR


def fingerprint(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


def load(name: str) -> dict | None:
    path = CACHE_DIR / name
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
    return None


def save(name: str, data: dict) -> Path:
    path = CACHE_DIR / name
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)
    return path
