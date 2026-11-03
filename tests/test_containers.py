import dataclasses
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from pituplots import Catalog, Grid, Profile, Sites
from pituplots.io import containers


def grid():
    return Grid(np.arange(6.0).reshape(2, 3), [0, 1, 2, 3], [10, 11, 12])


@pytest.mark.parametrize("c", [Catalog, Grid, Sites, Profile])
def test_plain_dataclass(c):
    assert dataclasses.is_dataclass(c)
    assert c.__module__ == containers.__name__


def test_catalog_columns():
    c = Catalog([1, 2], [3, 4], [5, 6], [7, 8], ["2020-01-01", "2020-01-02T03:04:05.5"],
                extra=pd.DataFrame({"ll": [0.1, 0.2]}))
    assert c.lon.dtype == float and c.mag.dtype == float
    assert c.time.dtype == np.dtype("datetime64[ms]")
    assert c.time[1] == np.datetime64("2020-01-02T03:04:05.500")
    assert c.extra["ll"].tolist() == [0.1, 0.2]


def test_catalog_time_from_datetime():
    c = Catalog([1], [2], time=[datetime(2020, 1, 1, 12)])
    assert c.time[0] == np.datetime64("2020-01-01T12:00")


def test_catalog_minimal_and_empty():
    c = Catalog([], [])
    assert c.lon.size == 0 and c.depth is None and c.mag is None and c.extra == {}


def test_catalog_inputs_not_modified_or_shared():
    assert Catalog([1], [2]).extra is not Catalog([1], [2]).extra
    lon = np.array([1.0, 2.0])
    Catalog(lon, [3, 4])
    assert lon.tolist() == [1, 2]


@pytest.mark.parametrize("kw", [
    {"lat": [1]}, {"depth": [1]}, {"mag": [1, 2, 3]}, {"time": ["2020-01-01"]},
    {"extra": {"a": [1]}},
])
def test_catalog_length_mismatch(kw):
    a = {"lon": [1, 2], "lat": [3, 4]} | kw
    with pytest.raises(ValueError, match="rows"):
        Catalog(**a)


def test_catalog_2d_raises():
    with pytest.raises(ValueError, match="1-D"):
        Catalog(np.zeros((2, 2)), [1, 2])


def test_sites():
    s = Sites([1, 2], [3, 4], names=["a", "b"], values=np.zeros((2, 5)))
    assert s.values.shape == (2, 5) and s.names.tolist() == ["a", "b"] and s.meta == {}
    assert Sites([1], [2]).values is None
    with pytest.raises(ValueError, match="names"):
        Sites([1, 2], [3, 4], names=["a"])


def test_grid_valid():
    g = grid()
    assert g.crs == "EPSG:4326" and g.mask is None and g.meta == {}
    assert Grid(np.zeros((2, 3, 4), np.uint8), [0, 1, 2, 3], [0, 1, 2]).values.dtype == np.uint8
    assert Grid(np.zeros((1, 1)), [179.5, 180.5], [0, 1]).x_edges[-1] > 180


@pytest.mark.parametrize("a", [
    (np.zeros(3), [0, 1, 2, 3], [0, 1]),
    (np.zeros((2, 3, 2)), [0, 1, 2, 3], [0, 1, 2]),
    (np.zeros((2, 3)), [0, 1, 2], [0, 1, 2]),
    (np.zeros((2, 3)), [0, 1, 2, 3], [0, 1]),
    (np.zeros((2, 3)), [0, 1, 1, 3], [0, 1, 2]),
    (np.zeros((2, 3)), [3, 2, 1, 0], [0, 1, 2]),
    (np.zeros((2, 3)), [0, 1, 2, 3], [2, 1, 0]),
])
def test_grid_bad_shapes(a):
    with pytest.raises(ValueError):
        Grid(*a)


def test_grid_bad_mask():
    with pytest.raises(ValueError, match="mask"):
        Grid(np.zeros((2, 3)), [0, 1, 2, 3], [0, 1, 2], mask=np.zeros((3, 2)))


def test_grid_masked_array():
    z = np.ma.masked_array([[1, 2]], mask=[[True, False]])
    g = Grid(z, [0, 1, 2], [0, 1])
    assert g.mask.tolist() == [[True, False]]
    assert np.isnan(g.values[0, 0]) and g.values[0, 1] == 2
    assert not np.ma.isMaskedArray(g.values)
    g = Grid(z, [0, 1, 2], [0, 1], mask=[[False, True]])
    assert g.mask.tolist() == [[True, True]]


def test_from_points_center_and_origin():
    x, y = np.meshgrid([1.0, 2.0, 3.0], [10.0, 12.0])
    v = np.arange(6.0).reshape(2, 3)
    g = Grid.from_points(x.ravel(), y.ravel(), v.ravel(), dh=(1, 2))
    assert g.x_edges.tolist() == [0.5, 1.5, 2.5, 3.5] and g.y_edges.tolist() == [9, 11, 13]
    assert g.values.tolist() == v.tolist() and g.mask is None
    g = Grid.from_points(x.ravel(), y.ravel(), v.ravel(), anchor="origin")
    assert g.x_edges.tolist() == [1, 2, 3, 4] and g.y_edges.tolist() == [10, 12, 14]


def test_from_points_unsorted():
    g = Grid.from_points([2, 1, 2, 1], [1, 2, 2, 1], [10, 20, 30, 40])
    assert g.values.tolist() == [[40, 10], [20, 30]]


def test_from_points_gaps_are_masked():
    g = Grid.from_points([0, 0.3, 0.3], [0, 0, 0.1], [1, 2, 3], dh=0.1)
    assert g.values.shape == (2, 4)
    assert g.mask.sum() == 5 and np.isnan(g.values[g.mask]).all()


def test_from_points_float_noise():
    x = np.arange(0, 2, 0.1) + np.random.default_rng(0).uniform(-1e-7, 1e-7, 20)
    g = Grid.from_points(x, np.zeros(20), np.arange(20.0), dh=0.1)
    assert g.values.shape == (1, 20)
    g = Grid.from_points(np.r_[x, x], np.r_[x * 0, x * 0 + 0.1], np.arange(40.0))
    assert g.values.shape == (2, 20) and g.mask is None
    assert np.diff(g.x_edges) == pytest.approx(0.1, abs=1e-6)


def test_from_points_single_axis_needs_dh():
    with pytest.raises(ValueError, match="pass dh"):
        Grid.from_points([1, 2, 3], [0, 0, 0], [1, 2, 3])
    with pytest.raises(ValueError, match="pass dh"):
        Grid.from_points([1], [0], [1])
    assert Grid.from_points([1], [0], [5], dh=0.1).values.shape == (1, 1)


def test_from_points_off_lattice():
    with pytest.raises(ValueError, match="lattice"):
        Grid.from_points([0, 1, 1.5], [0, 0, 0], [1, 2, 3], dh=(1, 1))


def test_from_points_duplicates():
    assert Grid.from_points([0, 0, 1], [0, 0, 0], [1, 1, 2], dh=1).values.tolist() == [[1, 2]]
    v = Grid.from_points([0, 0, 1], [0, 0, 0], [np.nan, np.nan, 2], dh=1).values
    assert np.isnan(v[0, 0])
    with pytest.raises(ValueError, match="repeated"):
        Grid.from_points([0, 0, 1], [0, 0, 0], [1, 2, 3], dh=1)


def test_from_points_exceeding_180():
    g = Grid.from_points([179.5, 180.5], [0, 0], [1, 2], dh=1)
    assert g.x_edges.tolist() == [179, 180, 181]


def test_from_points_dateline_wrap_raises():
    with pytest.raises(ValueError, match="unwrap_lon"):
        Grid.from_points([179.95, -179.95], [0, 0], [1, 2], dh=0.1)


def test_from_points_global_is_not_a_wrap():
    x = np.arange(-179, 180, 2.0)
    g = Grid.from_points(x, x * 0, x, dh=2)
    assert g.x_edges[0] == -180 and g.x_edges[-1] == 180


def test_from_points_bad_args():
    with pytest.raises(ValueError, match="anchor"):
        Grid.from_points([0], [0], [0], dh=1, anchor="corner")
    with pytest.raises(ValueError, match="no points"):
        Grid.from_points([], [], [], dh=1)
    with pytest.raises(ValueError, match="rows"):
        Grid.from_points([0, 1], [0], [0, 1], dh=1)


def test_profile():
    p = Profile(-71, -33, 90, (-50, 300), 100)
    assert p.x_range == (-50, 300)
    assert Profile(0, 0, 0, (0, 1), 0).width == 0


@pytest.mark.parametrize("r, w", [((0, 0), 10), ((5, 1), 10), ((0, 1), -1)])
def test_profile_invalid(r, w):
    with pytest.raises(ValueError):
        Profile(0, 0, 0, r, w)
