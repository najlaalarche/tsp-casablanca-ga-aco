"""Chargement et validation des 40 points GPS (CSV ou Excel)."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from config.settings import EXPECTED_POINTS, START_POINT_ID


class DataValidationError(ValueError):
    pass


@dataclass(frozen=True)
class Point:
    id: int
    name: str
    zone: str
    lat: float
    lon: float


def load_points(path: str | Path, expected: int = EXPECTED_POINTS,
                start_id: int = START_POINT_ID) -> list[Point]:
    path = Path(path)
    if not path.exists():
        raise DataValidationError(f"Fichier introuvable : {path}")
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip().lower() for c in df.columns]
    id_col = "n" if "n" in df.columns else "id"
    name_col = "nom" if "nom" in df.columns else "name"
    for col in (id_col, name_col, "lat", "lon"):
        if col not in df.columns:
            raise DataValidationError(f"Colonne manquante : {col}")
    points = [
        Point(int(r[id_col]), str(r[name_col]).strip(), str(r.get("zone", "")).strip(),
              float(r["lat"]), float(r["lon"]))
        for _, r in df.iterrows()
    ]
    validate_points(points, expected, start_id)
    return sorted(points, key=lambda p: p.id)


def validate_points(points: list[Point], expected: int, start_id: int) -> None:
    if len(points) != expected:
        raise DataValidationError(f"{len(points)} points trouvés, {expected} attendus.")
    ids = [p.id for p in points]
    if len(set(ids)) != len(ids):
        raise DataValidationError("Identifiants en double.")
    if len({(p.lat, p.lon) for p in points}) != len(points):
        raise DataValidationError("Coordonnées GPS en double.")
    for p in points:
        if not (-90 <= p.lat <= 90 and -180 <= p.lon <= 180):
            raise DataValidationError(f"Coordonnées invalides pour le point {p.id} ({p.name}).")
    if start_id not in ids:
        raise DataValidationError(f"Point de départ {start_id} absent.")
    if sorted(ids) != list(range(len(ids))):
        raise DataValidationError("Les identifiants doivent être 0..N-1 (index de la matrice).")
