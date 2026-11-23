"""Import et validation des 40 points GPS réels."""
import pandas as pd
import pytest

from config.settings import POINTS_FILE
from core.validation import is_valid_tour
from services.data_loader import load_points, DataValidationError


def test_load_real_points():
    pts = load_points(POINTS_FILE)
    assert len(pts) == 40
    assert len({p.id for p in pts}) == 40
    assert pts[0].id == 0 and pts[0].name == "Gare Casa Oasis"
    assert pts[0].lat == pytest.approx(33.5547245) and pts[0].lon == pytest.approx(-7.6315443)
    assert all(33 < p.lat < 34 and -8 < p.lon < -7 for p in pts)


def test_load_xlsx(tmp_path):
    df = pd.read_csv(POINTS_FILE, encoding="utf-8-sig")
    xlsx = tmp_path / "pts.xlsx"
    df.to_excel(xlsx, index=False)
    assert len(load_points(xlsx)) == 40


@pytest.mark.parametrize("mutate,msg", [
    (lambda df: df.iloc[:39], "39 points"),
    (lambda df: df.assign(lat=df.lat.where(df.n != 5, 123.0)), "invalides"),
    (lambda df: df.assign(n=df.n.where(df.n != 5, 4)), "double"),
])
def test_invalid_data_rejected(tmp_path, mutate, msg):
    df = mutate(pd.read_csv(POINTS_FILE, encoding="utf-8-sig"))
    f = tmp_path / "bad.csv"
    df.to_csv(f, index=False)
    with pytest.raises(DataValidationError, match=msg):
        load_points(f)


def test_is_valid_tour():
    assert is_valid_tour([0, 2, 1, 3, 0], 4, 0)
    assert not is_valid_tour([0, 2, 2, 3, 0], 4, 0)   # doublon
    assert not is_valid_tour([1, 2, 0, 3, 1], 4, 0)   # mauvais départ
    assert not is_valid_tour([0, 2, 1, 3], 4, 0)      # pas de retour
