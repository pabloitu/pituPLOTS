import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pyproj import Geod
from shapely.geometry import LineString

from pituplots import (Catalog, Grid, Profile, add_profile_map, add_slab, make_profile, plot_section,
                       profiles_from_line, project)
from pituplots.io import read_vector
from pituplots.sections import unproject

G = Geod(ellps="WGS84")


def events(lon0=-71.0, lat0=-33.0, n=9):
    x = lon0 + np.linspace(0, 2, n)
    return Catalog(x, x * 0 + lat0, np.linspace(5, 45, n), 4 + np.arange(n) / 4,
                   extra={"cls": np.array(list("abcabcabc"))[:n]})


def test_make_profile_azimuth():
    p = make_profile((-71, -33), azimuth=90, length=300, width=50)
    assert isinstance(p, Profile) and p.x_range == (0, 300) and p.width == 50
    assert make_profile((0, 0), azimuth=0, length=(-50, 300)).x_range == (-50, 300)


def test_make_profile_p1():
    p = make_profile((-71, -33), p1=(-70, -32))
    az, _, d = G.inv(-71, -33, -70, -32)
    assert p.azimuth == pytest.approx(az) and p.x_range[1] == pytest.approx(d / 1000)
    assert make_profile((0, 0), p1=(1, 0), length=(-5, 5)).x_range == (-5, 5)


@pytest.mark.parametrize("kw", [{}, {"azimuth": 1, "p1": (1, 1)}, {"azimuth": 1}])
def test_make_profile_bad(kw):
    with pytest.raises(ValueError):
        make_profile((0, 0), **kw)


def test_zero_length_raises():
    with pytest.raises(ValueError):
        make_profile((1, 1), p1=(1, 1))
    with pytest.raises(ValueError):
        make_profile((1, 1), azimuth=0, length=0)


@pytest.mark.parametrize("lat0", [0, -33, 60, -60, 80])
@pytest.mark.parametrize("az", [0, 37, 90, 200])
def test_project_is_geodesic(lat0, az):
    p = make_profile((-71, lat0), azimuth=az, length=1000)
    lon, lat, _ = G.fwd(-71, lat0, az, 500e3)
    x, y = project(lon, lat, p)
    assert x == pytest.approx(500, abs=1e-3) and y == pytest.approx(0, abs=1e-3)


def test_project_left_is_positive():
    p = make_profile((0, 0), azimuth=0, length=100)
    x, y = project(-1.0, 0.0, p)
    assert y > 100 and abs(x) < 1e-6


def test_project_arrays_and_round_trip():
    p = make_profile((10, 45), azimuth=30, length=100)
    x, y = project(np.array([10.5, 11]), np.array([45.2, 45.5]), p)
    assert x.shape == (2,)
    lon, lat = unproject(x, y, p)
    assert lon == pytest.approx([10.5, 11]) and lat == pytest.approx([45.2, 45.5])


def test_new_and_given_ax():
    p = make_profile((-71, -33), azimuth=90, length=300, width=50)
    ax = plot_section(events(), p)
    assert not hasattr(ax, "projection") and ax.collections
    _, a = plt.subplots()
    assert plot_section(events(), p, ax=a) is a
    assert a.get_ylim()[0] > a.get_ylim()[1]


def test_input_types_and_errors():
    p = make_profile((-71, -33), azimuth=90, length=300)
    with pytest.raises(ValueError, match="depths"):
        plot_section(Catalog([0], [0]), p)
    with pytest.raises(TypeError):
        plot_section("x", p)
    d = pd.DataFrame({"longitude": [-70.5], "latitude": [-33.0], "depth": [10.0], "mag": [5.0]})
    assert plot_section(d, p).collections


def test_no_edge_of_swath_leak():
    p = make_profile((0, 0), azimuth=90, length=500, width=100)
    c = Catalog([1, 1, 1, 20], [0, 0.4, 0.6, 0], [10, 10, 10, 10], [5, 5, 5, 5])
    ax = plot_section(c, p, legend=False)
    assert ax.collections[0].get_offsets().shape[0] == 2


def test_color_by_and_hollow():
    p = make_profile((-71, -33), azimuth=90, length=300, width=50)
    c = events()
    ax = plot_section(c, p, color_by="cls", colors={"a": "red"}, hollow=c.mag > 5)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["a", "b", "c"]
    fc = [m.get_facecolor() for m in ax.collections]
    assert any(len(f) == 0 for f in fc)
    ax = plot_section(c, p, color_by=c.depth)
    assert ax.collections[0].get_array() is not None
    assert plot_section(c, p, legend=True).artists


# B3 sec

def test_crosses_antimeridian():
    p = make_profile((179.5, -20), azimuth=90, length=300, width=50)
    x, y = project(np.array([179.9, -179.9, 181.0]), np.array([-20.0] * 3), p)
    assert (np.diff(x[:2]) > 0).all() and x[1] == pytest.approx(x[2] - 0.9 * 104.6, abs=1)
    ax = plot_section(Catalog([179.9, -179.9], [-20, -20], [10, 20], [5, 6]), p)
    assert ax.collections[0].get_offsets().shape[0] == 2


def test_zero_width_takes_events_on_line():
    p = make_profile((0, 0), azimuth=90, length=500, width=0)
    c = Catalog([1, 1], [0, 0.1], [10, 10], [5, 6])
    assert plot_section(c, p).collections[0].get_offsets().shape[0] == 1


def test_very_wide_swath():
    p = make_profile((0, 0), azimuth=90, length=500, width=1e5)
    assert plot_section(events(0, 0), p).collections


def test_no_events_in_swath():
    p = make_profile((100, 40), azimuth=90, length=100, width=10)
    ax = plot_section(events(), p)
    assert not ax.collections or ax.collections[0].get_offsets().size == 0
    ax.figure.canvas.draw()


def test_event_exactly_on_line():
    p = make_profile((10, 10), azimuth=30, length=500, width=2)
    lon, lat, _ = G.fwd(10, 10, 30, 100e3)
    ax = plot_section(Catalog([lon], [lat], [5], [5]), p)
    assert ax.collections[0].get_offsets()[0, 0] == pytest.approx(100)


def test_ve_kept_after_resize():
    p = make_profile((-71, -33), azimuth=90, length=300, width=50)
    ax = plot_section(events(), p, ve=2, zmax=100, figsize=(10, 4))
    def ratio():
        ax.figure.canvas.draw()
        b = ax.get_window_extent()
        return (b.height / abs(np.diff(ax.get_ylim())[0])) / (b.width / abs(np.diff(ax.get_xlim())[0]))
    r = ratio()
    assert r == pytest.approx(2, rel=0.02)
    ax.figure.set_size_inches(7, 7)
    assert ratio() == pytest.approx(2, rel=0.02)


def test_nan_depth_hollow_at_zero():
    p = make_profile((0, 0), azimuth=90, length=500, width=100)
    ax = plot_section(Catalog([1, 2], [0, 0], [np.nan, 10], [5, 6]), p)
    h = [m for m in ax.collections if len(m.get_facecolor()) == 0]
    assert len(h) == 1 and h[0].get_offsets()[0, 1] == 0
    ax = plot_section(Catalog([1, 2], [0, 0], [np.nan, 10], [5, 6]), p, nan_depth=33)
    h = [m for m in ax.collections if len(m.get_facecolor()) == 0]
    assert h[0].get_offsets()[0, 1] == 33


def test_empty_catalog_section():
    p = make_profile((0, 0), azimuth=90, length=500)
    plot_section(Catalog([], [], [], []), p).figure.canvas.draw()


# profiles_from_line

LINE = LineString([(-72, -40), (-72.5, -33), (-71, -25)])


def test_profiles_at_latitudes():
    ps = profiles_from_line(LINE, [-35, -30], 100, length=(-50, 500))
    assert len(ps) == 2 and ps[0].lat0 == pytest.approx(-35, abs=0.02)
    assert ps[0].x_range == (-50, 500)


def test_profiles_perpendicular():
    ps = profiles_from_line(LineString([(0, 0), (0, 10)]), [5], 10)
    assert ps[0].azimuth == pytest.approx(90, abs=1e-6)
    ps = profiles_from_line(LineString([(0, 0), (0, 10)]), [5], 10, side="left")
    assert ps[0].azimuth == pytest.approx(270, abs=1e-6)


def test_profiles_toward_point():
    ps = profiles_from_line(LINE, [-35], 100, side=(-60, -35))
    assert 0 < ps[0].azimuth < 180
    ps = profiles_from_line(LINE, [-35], 100, side=(-90, -35))
    assert 180 < ps[0].azimuth < 360


def test_profiles_by_km_and_inputs():
    ps = profiles_from_line(np.asarray(LINE.coords), [100.0], 100, by="km")
    assert len(ps) == 1
    a = LineString([(0, 0), (0, 5)])
    b = LineString([(0, 5), (0, 10)])
    assert profiles_from_line([a, b], [7], 10)[0].lat0 == pytest.approx(7, abs=0.02)


def test_profiles_outside_raises():
    with pytest.raises(ValueError, match="outside"):
        profiles_from_line(LINE, [0], 10)
    with pytest.raises(ValueError, match="outside"):
        profiles_from_line(LINE, [1e6], 10, by="km")
    with pytest.raises(ValueError, match="by"):
        profiles_from_line(LINE, [0], 10, by="x")


def test_profiles_from_trench_fixture(data):
    geoms, _, crs = read_vector(data / "chile/sam_nazca_trench.shp")
    assert crs == "EPSG:4326"
    b = geoms[0].bounds
    ps = profiles_from_line(geoms, [(b[1] + b[3]) / 2], 100, side=(-60, -30))
    assert 30 < ps[0].azimuth < 150


def test_profile_across_dateline_line():
    ps = profiles_from_line(LineString([(179, -30), (-179, -20)]), [-25], 50)
    assert ps[0].lon0 > 179 or ps[0].lon0 < -179


# add_slab, add_profile_map

def slab():
    x = np.arange(-74, -64.9, 0.5)
    y = np.arange(-36, -29.9, 0.5)
    X = np.meshgrid(x[:-1] + 0.25, y[:-1] + 0.25)[0]
    return Grid((X + 74) * 20, x, y)


def test_add_slab():
    p = make_profile((-72, -33), azimuth=90, length=500, width=100)
    ax = plot_section(events(), p)
    t = slab()
    ln = add_slab(ax, p, t, thickness=Grid(t.values * 0 + 40, t.x_edges, t.y_edges), swath=True)
    x, z = ln.get_data()
    assert (np.diff(z[np.isfinite(z)]) > 0).all()
    assert len(ax.lines) >= 4


def test_add_slab_0_360_lookup():
    t = slab()
    u = Grid(t.values, t.x_edges + 360, t.y_edges)
    p = make_profile((-72, -33), azimuth=90, length=300)
    _, ax = plt.subplots()
    assert add_slab(ax, p, u).get_ydata() == pytest.approx(add_slab(ax, p, t).get_ydata(), nan_ok=True)


def test_add_slab_off_grid_is_nan():
    p = make_profile((10, 10), azimuth=90, length=100)
    _, ax = plt.subplots()
    assert np.isnan(add_slab(ax, p, slab()).get_ydata()).all()
    with pytest.raises(TypeError):
        add_slab(ax, p, "x")


def test_add_profile_map():
    p = [make_profile((-72, -33), azimuth=90, length=500), make_profile((-72, -30), azimuth=90, length=500)]
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-80, -60, -40, -25])
    lines = add_profile_map(ax, p, labels=["A", "B"])
    assert len(lines) == 2 and [t.get_text() for t in ax.texts] == ["A", "B"]
    ax.figure.canvas.draw()
    _, a = plt.subplots()
    assert len(add_profile_map(a, p[0])) == 1
    with pytest.raises(ValueError, match="labels"):
        add_profile_map(a, p, labels=["A"])


def test_add_profile_map_crossing():
    p = make_profile((179, -30), azimuth=90, length=500)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    ln = add_profile_map(ax, p)[0]
    assert np.diff(ln.get_xdata()).max() < 2
