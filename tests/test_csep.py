import numpy as np
import pytest

from pituplots import Catalog, Grid
from pituplots.errors import MissingDependency

csep = pytest.importorskip("csep", reason="needs the [csep] extra")

from pituplots.adapters.csep import (from_csep_catalog, from_csep_forecast, from_csep_region,  # noqa: E402
                                     to_csep_catalog, to_csep_forecast)


def catalog():
    t = np.array(["2020-01-01T00:00:00", "2021-06-01T12:00:00"], "datetime64[ms]")
    return Catalog([10.5, 11.2], [44.1, 44.6], [5.0, 12.0], [3.5, 4.1], t)


def grid():
    x, y = np.round(np.arange(10, 11.01, 0.1), 6), np.round(np.arange(44, 44.51, 0.1), 6)
    v = np.arange(50, dtype=float).reshape(5, 10) * 1e-3
    m = np.zeros((5, 10), bool)
    m[0, 0] = True
    return Grid(v, x, y, mask=m)


def test_catalog_round_trip():
    c = catalog()
    back = from_csep_catalog(to_csep_catalog(c, name="x"))
    for k in ("lon", "lat", "depth", "mag"):
        assert np.allclose(getattr(back, k), getattr(c, k))
    assert (back.time == c.time).all()


def test_catalog_needs_columns():
    with pytest.raises(ValueError, match="magnitude"):
        to_csep_catalog(Catalog([1], [2], mag=[3]))


def test_forecast_round_trip_and_region():
    g = grid()
    fc = to_csep_forecast(g, magnitudes=[4.0], name="f")
    assert fc.region.num_nodes == 49 and fc.data.shape == (49, 1)
    back = from_csep_forecast(fc)
    assert np.allclose(back.x_edges, g.x_edges) and np.allclose(back.y_edges, g.y_edges)
    assert back.mask[0, 0] and np.allclose(back.values[~back.mask], g.values[~g.mask])
    assert back.meta["magnitudes"].tolist() == [4.0]
    r = from_csep_region(fc.region)
    assert np.isnan(r.values).all() and (r.mask == back.mask).all()


def test_forecast_checks():
    g = grid()
    with pytest.raises(ValueError, match="bins"):
        to_csep_forecast(g, magnitudes=[4.0, 4.1])
    with pytest.raises(ValueError, match="square"):
        to_csep_forecast(Grid(np.ones((1, 2)), [0, 0.1, 0.2], [0, 0.2]))
    with pytest.raises(ValueError, match="degrees"):
        to_csep_forecast(Grid(np.ones((1, 1)), [0, 1], [0, 1], crs="EPSG:2193"))


def test_missing_extra_is_named(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "csep", None)
    with pytest.raises(MissingDependency, match="csep"):
        to_csep_catalog(catalog())
