"""B3 "Longitude and extent", at the level of the lattice and of raw GeoAxes.

The checks that need make_map (crossing extent -> PlateCarree(180), set_global)
are added with it.
"""
import ast
import json
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import ListedColormap

import pituplots
from pituplots import Catalog, Grid, Sites, extent_from, unwrap_lon
from pituplots.maps import central_lon, seam_split

RED = ListedColormap(["red"])


def draw(ax, z, x, y):
    c = central_lon(ax.projection)
    crs = ccrs.PlateCarree(central_longitude=c)
    for zi, xi in seam_split(z, x, c):
        ax.pcolormesh(xi, y, zi, transform=crs, cmap=RED, vmin=0, vmax=1)


def canvas(proj, extent=None, size=(6, 3), dpi=60):
    fig = plt.figure(figsize=size, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1], projection=proj)
    ax.set_axis_off()
    if extent is None:
        ax.set_global()
    else:
        ax.set_extent(extent, ccrs.PlateCarree())
    return ax


def red(ax):
    ax.figure.canvas.draw()
    a = np.asarray(ax.figure.canvas.buffer_rgba())[..., :3].astype(int)
    return (a[..., 0] > 200) & (a[..., 1] < 60)


# unwrap_lon

@pytest.mark.parametrize("lon, out", [
    ([179.95, -179.95], [179.95, 180.05]),
    ([179.95, 180.05], [179.95, 180.05]),
    ([170, -170, 175], [170, 190, 175]),
    ([163, 179, 181, 183], [163, 179, 181, 183]),
    ([179, 181, -179], [179, 181, 181]),
    ([-5, 355, 5], [-5, -5, 5]),
    ([285, 290], [285, 290]),
    ([-75, -70], [-75, -70]),
    ([-75, 285], [-75, -75]),
    ([1, 2, 3], [1, 2, 3]),
    ([0, 180], [0, 180]),
])
def test_unwrap_cases(lon, out):
    assert unwrap_lon(lon) == pytest.approx(np.array(out, float))


@pytest.mark.parametrize("lon, out", [
    ([180, -180], [180, 180]),
    ([180, -180, 179, -179], [180, 180, 179, 181]),
    ([0, 360], [0, 0]),
    ([0, 360, 1, 359], [0, 0, 1, -1]),
])
def test_unwrap_on_180_and_0_360(lon, out):
    assert unwrap_lon(lon) == pytest.approx(np.array(out, float))


@pytest.mark.parametrize("lon", [
    np.arange(-179, 180, 2.0),
    np.arange(1, 360, 2.0),
    np.arange(-180, 181, 20.0),
    np.arange(0, 361, 20.0),
    np.array([-180.0, 0, 180]),
])
def test_unwrap_global_span_is_untouched(lon):
    assert unwrap_lon(lon).tolist() == lon.tolist()


@pytest.mark.parametrize("seed", range(5))
def test_unwrap_any_convention_gives_the_same_span(seed):
    r = np.random.default_rng(seed)
    base = 150 + np.r_[r.uniform(0, 70, 200), 0, 70]
    turns = r.integers(-1, 2, base.size) * 360
    x = unwrap_lon(base + turns)
    assert np.ptp(x) == pytest.approx(70)
    t = (x - base) / 360
    assert t == pytest.approx(np.rint(t), abs=1e-9)


def test_unwrap_is_idempotent_and_copies():
    lon = np.array([179.95, -179.95])
    x = unwrap_lon(lon)
    assert lon.tolist() == [179.95, -179.95]
    assert unwrap_lon(x).tolist() == x.tolist()
    y = np.array([1.0, 2.0])
    assert unwrap_lon(y) is not y


def test_unwrap_exact_values_survive():
    assert unwrap_lon([-179.95, 179.95])[1] == 179.95


def test_unwrap_shapes_and_nan():
    x = unwrap_lon([[170, np.nan], [-170, 175]])
    assert x.shape == (2, 2) and np.isnan(x[0, 1])
    assert x[1, 0] == 190
    assert unwrap_lon([]).size == 0
    assert np.isnan(unwrap_lon([np.nan])).all()
    assert unwrap_lon(5.0) == 5


def test_cat_nz_stays_continuous(data):
    d = pd.read_csv(data / "new_zealand/cat_nz.csv")
    x = unwrap_lon(d["lon"])
    assert x.min() > 160 and x.max() < 190 and x.max() > 180
    w = ((d["lon"] + 180) % 360 - 180).to_numpy()
    assert w.min() < -170 and unwrap_lon(w) == pytest.approx(x)


# Grid.from_points across the dateline

def lattice(lon):
    return Grid.from_points(lon, np.zeros(len(lon)), [1, 2, 3, 4][:len(lon)], dh=(0.1, 1))


@pytest.mark.parametrize("lon", [
    [179.95, -179.95],
    [179.95, 180.05],
    [-179.95, 179.95],
    [-180.05, -179.95],
])
def test_from_points_one_patch(lon):
    g = lattice(lon)
    assert g.values.shape == (1, 2) and g.mask is None
    assert sorted(g.values.ravel()) == [1, 2]
    assert np.ptp(g.x_edges) == pytest.approx(0.2, abs=1e-6)
    assert g.x_edges[1] % 360 == pytest.approx(180, abs=1e-6)


def test_from_points_conventions_match():
    x = np.array([179.75, 179.85, 179.95, 180.05, 180.15])
    v = np.arange(5.0)
    a = Grid.from_points(x, x * 0, v, dh=(0.1, 1))
    b = Grid.from_points((x + 180) % 360 - 180, x * 0, v, dh=(0.1, 1))
    c = Grid.from_points(np.where(v % 2, x, (x + 180) % 360 - 180), x * 0, v, dh=(0.1, 1))
    for g in (b, c):
        assert g.values.tolist() == a.values.tolist()
        assert g.x_edges == pytest.approx(a.x_edges)


def test_from_points_infers_dh_across_dateline():
    x = np.array([179.5, -179.5, -178.5])
    g = Grid.from_points(np.r_[x, x], np.r_[x * 0, x * 0 + 1], np.arange(6.0))
    assert g.values.shape == (2, 3) and g.mask is None


def test_from_points_input_not_modified():
    x = np.array([179.95, -179.95])
    Grid.from_points(x, [0, 0], [1, 2], dh=0.1)
    assert x.tolist() == [179.95, -179.95]


def test_io_stays_free_of_cartopy():
    root = Path(pituplots.__file__).parent / "io"
    for f in root.glob("*.py"):
        for n in ast.walk(ast.parse(f.read_text())):
            names = [a.name for a in n.names] if isinstance(n, ast.Import) else []
            names += [n.module or ""] if isinstance(n, ast.ImportFrom) else []
            for m in names:
                assert m.partition(".")[0] != "cartopy", f
                assert m in ("pituplots.errors",) or m.partition(".")[0] != "pituplots" \
                    or m.startswith("pituplots.io"), f


# extent_from

def test_extent_points():
    assert extent_from(([0, 10], [0, 20])) == pytest.approx([-0.5, 10.5, -1, 21])
    assert extent_from(([0, 10], [0, 20]), pad=0) == pytest.approx([0, 10, 0, 20])
    assert extent_from(([0, 10], [0, 20], [5, 6])) == pytest.approx([-0.5, 10.5, -1, 21])


@pytest.mark.parametrize("make", [
    lambda x, y: Catalog(x, y),
    lambda x, y: Sites(x, y),
    lambda x, y: (x, y),
    lambda x, y: (x, y, x * 0, x * 0),
    lambda x, y: pd.DataFrame({"longitude": x, "latitude": y}),
])
def test_extent_inputs(make):
    x, y = np.array([0.0, 10]), np.array([0.0, 20])
    assert extent_from(make(x, y), pad=0) == pytest.approx([0, 10, 0, 20])


def test_extent_dataframe_lists_columns():
    with pytest.raises(ValueError, match="foo"):
        extent_from(pd.DataFrame({"foo": [1], "lat": [2]}))


def test_extent_bad_type():
    with pytest.raises(TypeError):
        extent_from("nope")


def test_extent_grid_and_tuple():
    g = Grid(np.zeros((2, 3)), [0, 1, 2, 3], [10, 11, 12])
    assert extent_from(g, pad=0) == pytest.approx([0, 3, 10, 12])
    assert extent_from((g.values, g.x_edges, g.y_edges), pad=0) == pytest.approx([0, 3, 10, 12])


def test_extent_grid_uses_valid_cells():
    m = np.ones((2, 4), bool)
    m[0, 1] = False
    g = Grid(np.zeros((2, 4)), [0, 1, 2, 3, 4], [0, 1, 2], mask=m)
    assert extent_from(g, pad=0) == pytest.approx([1, 2, 0, 1])
    g = Grid(np.zeros((2, 4)), [0, 1, 2, 3, 4], [0, 1, 2], mask=np.ones((2, 4), bool))
    assert extent_from(g, pad=0) == pytest.approx([0, 4, 0, 2])


def test_extent_projected_grid_is_in_degrees():
    g = Grid(np.zeros((1, 1)), [1.5e6, 1.6e6], [5.4e6, 5.5e6], crs="EPSG:2193")
    e = extent_from(g, pad=0)
    assert 171 < e[0] < e[1] < 174 and -42 < e[2] < e[3] < -40


def test_extent_bbox():
    assert extent_from([0, 10, 0, 20], pad=0) == pytest.approx([0, 10, 0, 20])
    assert extent_from([170, -170, 0, 20], pad=0) == pytest.approx([170, -170, 0, 20])
    assert extent_from([-170, 170, 0, 20], pad=0) == pytest.approx([-170, 170, 0, 20])


def test_extent_crosses_dateline():
    e = extent_from(([179.95, -179.95], [0, 1]), pad=0)
    assert e == pytest.approx([179.95, -179.95, 0, 1])
    assert e[0] > e[1]


def test_extent_conventions_match(data):
    x = np.array([165.0, 183.0])
    a = extent_from((x, [0, 1]))
    b = extent_from((x - 360, [0, 1]))
    c = extent_from(((x + 180) % 360 - 180, [0, 1]))
    assert a == pytest.approx(b) and a == pytest.approx(c)
    assert a[0] > a[1] and a[0] == pytest.approx(164.1) and a[1] == pytest.approx(-176.1)


def test_extent_cat_nz_crosses(data):
    d = pd.read_csv(data / "new_zealand/cat_nz.csv")
    e = extent_from(d)
    assert e[0] > e[1] and 150 < e[0] < 165 and -180 < e[1] < -170


def test_extent_0_360_noncrossing():
    assert extent_from(([285, 290], [-40, -30]), pad=0) == pytest.approx([-75, -70, -40, -30])


def test_extent_pad_reaching_180():
    e = extent_from(([170, 179.9], [0, 1]))
    assert e[0] > e[1] and e[1] < -179


def test_extent_on_180():
    assert extent_from(([170, 180], [0, 1]), pad=0) == pytest.approx([170, 180, 0, 1])
    assert extent_from(([-180, -170], [0, 1]), pad=0) == pytest.approx([-180, -170, 0, 1])
    assert extent_from(([180, -180], [0, 1]), pad=0)[0] % 360 == pytest.approx(180)


def test_extent_one_point_has_width():
    e = extent_from(([5], [6]))
    assert e[0] < 5 < e[1] and e[2] < 6 < e[3]


def test_extent_empty_is_none():
    assert extent_from(Catalog([], [])) is None
    assert extent_from(([np.nan], [np.nan])) is None


@pytest.mark.parametrize("edges", [np.arange(-180, 181, 2.0), np.arange(0, 361, 2.0)])
def test_extent_global_grid(edges):
    g = Grid(np.zeros((90, 180)), edges, np.arange(-90, 91, 2.0))
    assert extent_from(g) == [-180, 180, -90, 90]


def test_extent_near_global_catalog():
    x = np.linspace(-179, 179, 50)
    e = extent_from((x, x / 2))
    assert e[:2] == [-180, 180]


def test_extent_latitudes_clamp_at_poles():
    g = Grid(np.zeros((1, 1)), [0, 2], [88, 90])
    e = extent_from(g)
    assert e[3] == 90 and e[2] < 88
    e = extent_from(Grid(np.zeros((1, 1)), [0, 2], [-90, -88]))
    assert e[2] == -90


def test_extent_iceland(data):
    r = np.load(data / "regions/iceland_region.npy")
    e = extent_from((r[:, 0], r[:, 1]), pad=0)
    assert e[0] < e[1] and 60 < e[2] < e[3] < 70


def test_extent_antarctic_trench(data):
    g = json.loads((data / "chile/pb2002_antarctic_trench.geojson").read_text())
    p = np.array(g["features"][0]["geometry"]["coordinates"]).reshape(-1, 2)
    e = extent_from((p[:, 0], p[:, 1]), pad=0)
    assert e[0] < e[1] < 0 and -65 < e[2] < -40


# central_lon

@pytest.mark.parametrize("proj, c", [
    (ccrs.PlateCarree(), 0),
    (ccrs.PlateCarree(central_longitude=180), 180),
    (ccrs.PlateCarree(central_longitude=-70), -70),
    (ccrs.Mercator(), 0),
    (ccrs.Mercator(central_longitude=170), 170),
    (ccrs.LambertConformal(central_longitude=-71), -71),
    (ccrs.epsg(3857), 0),
])
def test_central_lon(proj, c):
    assert central_lon(proj) == c


# seam_split

def test_split_inside_window():
    z = np.arange(6.0).reshape(2, 3)
    p = seam_split(z, [0, 1, 2, 3], 0)
    assert len(p) == 1
    assert p[0][0].tolist() == z.tolist() and p[0][1].tolist() == [0, 1, 2, 3]


def test_split_native_coordinates():
    (z, x), = seam_split(np.ones((1, 2)), [179, 180, 181], 180)
    assert x.tolist() == [-1, 0, 1]
    (z, x), = seam_split(np.ones((1, 2)), [-181, -180, -179], 180)
    assert x.tolist() == [-1, 0, 1]
    (z, x), = seam_split(np.ones((1, 2)), [285, 287, 289], 0)
    assert x.tolist() == [-75, -73, -71]


def test_split_on_edge():
    z = np.array([[1.0, 2, 3, 4]])
    p = seam_split(z, [178, 179, 180, 181, 182], 0)
    assert len(p) == 2
    by = {tuple(x): zi.ravel().tolist() for zi, x in p}
    assert by[(178, 179, 180)] == [1, 2]
    assert by[(-180, -179, -178)] == [3, 4]


def test_split_inside_a_cell():
    p = seam_split(np.array([[1.0, 2, 3]]), [178.5, 179.5, 180.5, 181.5], 0)
    by = {tuple(x): zi.ravel().tolist() for zi, x in p}
    assert by == {(178.5, 179.5, 180): [1, 2], (-180, -179.5, -178.5): [2, 3]}


def test_split_at_0_on_central_180():
    p = seam_split(np.array([[1.0, 2]]), [-1, 0, 1], 180)
    assert sorted(tuple(x) for _, x in p) == [(-180, -179), (179, 180)]


def test_split_global_grid():
    z = np.arange(180.0).reshape(1, 180)
    e = np.arange(-180, 181, 2.0)
    assert len(seam_split(z, e, 0)) == 1
    p = seam_split(z, e, 180)
    assert len(p) == 2
    assert sum(zi.shape[1] for zi, _ in p) == 180


def test_split_keeps_masks_images_and_input():
    z = np.ma.masked_array([[1.0, 2]], mask=[[True, False]])
    x = np.array([179.0, 180, 181])
    p = seam_split(z, x, 0)
    assert [zi.mask.tolist() for zi, _ in p] == [[[True]], [[False]]]
    assert x.tolist() == [179, 180, 181]
    p = seam_split(np.zeros((2, 2, 4), np.uint8), x, 0)
    assert [zi.shape for zi, _ in p] == [(2, 1, 4), (2, 1, 4)]
    assert p[0][0].dtype == np.uint8


def test_split_tiny_noise_at_seam_gives_no_sliver():
    p = seam_split(np.ones((1, 2)), [179, 180 + 1e-12, 181], 0)
    assert all(np.diff(x).min() > 0.5 for _, x in p) and len(p) == 2


# drawing

X = np.arange(175, 186.0)
Y = np.arange(-5, 6.0)
Z = np.ones((10, 10))
PROJ = [ccrs.PlateCarree(), ccrs.PlateCarree(central_longitude=180), ccrs.Mercator(),
        ccrs.Mercator(central_longitude=180)]


@pytest.mark.parametrize("proj", PROJ)
def test_no_band_and_nothing_lost(proj):
    ax = canvas(proj)
    draw(ax, Z, X, Y)
    r = red(ax)
    cols = np.flatnonzero(r.any(axis=0))
    b = ax.get_window_extent()
    px = 10 / 360 * b.width
    assert cols.size == pytest.approx(px, abs=2)
    assert r[r.any(axis=1)].sum(axis=1).max() == pytest.approx(px, abs=2)
    if central_lon(proj) == 0:
        assert cols.min() - b.x0 < 2 and b.x1 - cols.max() < 3
    else:
        assert np.ptp(cols) < 1.2 * px


@pytest.mark.parametrize("proj", PROJ)
@pytest.mark.parametrize("shift", [0, -360])
def test_same_picture_in_any_convention(proj, shift):
    a = canvas(proj)
    draw(a, Z, X, Y)
    b = canvas(proj)
    draw(b, Z, X + shift, Y)
    assert (red(a) == red(b)).all()


def test_contiguous_patch_on_zoomed_map():
    g = lattice([179.95, -179.95])
    ax = canvas(ccrs.PlateCarree(central_longitude=180), [179.5, 180.5, -0.5, 0.5], (3, 3), 50)
    draw(ax, np.ones((1, 2)), g.x_edges, g.y_edges)
    r = red(ax)
    cols = np.flatnonzero(r.any(axis=0))
    assert cols.size == pytest.approx(0.2 * 150, abs=2)
    assert np.diff(cols).max() == 1


@pytest.mark.parametrize("lon0", [-180, 0, 180])
def test_points_on_seam_columns(lon0):
    x = lon0 + np.array([-1.0, 0, 1])
    ax = canvas(ccrs.PlateCarree(), None, (6, 3), 60)
    draw(ax, np.ones((1, 2)), x, np.array([-30.0, 30]))
    assert red(ax).any()


def test_global_grid_fills_the_map():
    z = np.ones((90, 180))
    for c, e in [(0, np.arange(-180, 181, 2.0)), (180, np.arange(-180, 181, 2.0)),
                 (0, np.arange(0, 361, 2.0)), (180, np.arange(0, 361, 2.0))]:
        ax = canvas(ccrs.PlateCarree(central_longitude=c), None, (4, 2), 50)
        draw(ax, z, e, np.arange(-90, 91, 2.0))
        assert red(ax).mean() > 0.99


@pytest.mark.parametrize("proj", [ccrs.PlateCarree(), ccrs.Mercator()])
def test_cell_touching_pole(proj):
    ax = canvas(proj)
    draw(ax, np.ones((1, 4)), np.arange(0, 41, 10.0), np.array([80.0, 90]))
    assert red(ax).any()
    ax = canvas(proj)
    draw(ax, np.ones((1, 4)), np.arange(0, 41, 10.0), np.array([-90.0, -70]))
    assert red(ax).any()


def test_high_latitude_grid_iceland():
    g = Grid(np.ones((4, 4)), np.arange(-24, -15.0, 2), np.arange(63, 67.5, 1.0))
    ax = canvas(ccrs.Mercator(), [-26, -13, 62, 68], (3, 3), 50)
    draw(ax, g.values, g.x_edges, g.y_edges)
    assert red(ax).mean() > 0.2


def test_high_latitude_antarctic_on_mercator():
    x = np.arange(-80, -49, 5.0)
    y = np.arange(-65, -49, 5.0)
    ax = canvas(ccrs.Mercator(), [-85, -45, -67, -47], (3, 3), 50)
    draw(ax, np.ones((3, 6)), x, y)
    assert red(ax).mean() > 0.2
