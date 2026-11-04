import sys
from importlib.util import find_spec

import numpy as np
import pytest

from pituplots import Catalog, Grid, MissingDependency

pv3d = pytest.mark.skipif(find_spec("pyvista") is None, reason="needs pyvista")
GRID = {"crs": "EPSG:32719", "a": 0.973603016434936, "b": -0.23024643732697, "tx": 1069146.66101074,
        "ty": -6149426.80783081}


@pytest.fixture(autouse=True)
def off_screen():
    if find_spec("pyvista"):
        import pyvista as pv

        pv.OFF_SCREEN = True
    yield


def mine():
    r = np.random.default_rng(3)
    n = 30
    return Catalog(r.uniform(0, 1000, n), r.uniform(0, 1000, n), -r.uniform(1.8, 2.4, n), r.uniform(1, 3, n),
                   extra={"strike": np.r_[np.zeros(10), np.full(10, 90.0), np.full(10, np.nan)],
                          "dip": np.r_[np.full(10, 90.0), np.zeros(10), np.full(10, 45.0)],
                          "rake": np.r_[np.zeros(10), np.full(10, 90.0), np.full(10, -90.0)]}, crs="local")


def utm(x, y):
    """Mine grid to UTM 19S, inverting the similarity of GRID."""
    a, b = GRID["a"], GRID["b"]
    m = np.array([[a, b], [-b, a]])
    return np.linalg.solve(m, np.vstack([np.asarray(x) - GRID["tx"], np.asarray(y) - GRID["ty"]]))


# frames


def test_frames():
    from pituplots.scene import frame_xy, from_frame, make_frame

    c = mine()
    assert make_frame(c) == {"kind": "local"}
    f = make_frame(c, GRID)
    assert f["kind"] == "grid" and f["crs"] == "EPSG:32719"
    e, n = utm([100, 900], [200, 800])
    assert np.allclose(frame_xy(f, e, n, "EPSG:32719"), [[100, 900], [200, 800]])
    assert np.allclose(frame_xy(f, [5], [6], "local"), [[5], [6]])
    assert np.allclose(from_frame(f, *frame_xy(f, e, n, "EPSG:32719")), [e, n])
    assert make_frame(Catalog([0], [0], crs="EPSG:32719")) == {"kind": "crs", "crs": "EPSG:32719"}
    assert "equidistant" in make_frame(Catalog([-71, -70], [-33, -32]))["crs"].lower()
    with pytest.raises(ValueError, match="no reference CRS"):
        frame_xy(make_frame(c), [0], [0], "EPSG:4326")
    with pytest.raises(ValueError, match="needs a scene in a local grid"):
        frame_xy(make_frame(Catalog([0], [0], crs="EPSG:32719")), [0], [0], "local")
    with pytest.raises(ValueError, match="needs"):
        make_frame(c, {"crs": "EPSG:32719", "a": 1})
    with pytest.raises(ValueError, match="projected CRS or a grid"):
        make_frame(c, "EPSG:4326")


@pv3d
def test_grid_scene_and_labels():
    from pituplots.scene import plot_catalog_3d, to_scene

    c = mine()
    p = plot_catalog_3d(c, projection=GRID)
    st = p._pituplots
    e, n = utm(c.lon[:3], c.lat[:3])
    q = to_scene(p, e, n, -1000 * c.depth[:3], "EPSG:32719")
    assert np.allclose(q, st["catalogs"][0]["points"][:3], atol=1e-6)
    from pyproj import Transformer

    lon, lat = Transformer.from_crs("EPSG:32719", "EPSG:4326", always_xy=True).transform(e, n)
    assert np.allclose(to_scene(p, lon, lat, -1000 * c.depth[:3], "EPSG:4326"), q, atol=1e-3)
    f = st["picker"].fields((0, 1), st["picker"].hover)
    assert list(f)[:4] == ["lon", "lat", "mag", "depth"] and np.isclose(float(f["lon"]), lon[1], atol=1e-4)


# layers


@pv3d
def test_surface_from_grid():
    from pituplots.layers import add_surface
    from pituplots.scene import plot_catalog_3d

    c = Catalog([-72, -70], [-34, -30], [10, 100], [5, 6])
    p = plot_catalog_3d(c)
    v = np.array([[10.0, 20, 30], [40, 50, np.nan]])
    g = Grid(v, [-72, -71, -70, -69], [-34, -32, -30])
    a = add_surface(p, g, name="slab", crop=None, opacity=0.4)
    m = a.mapper.dataset
    assert p._pituplots["layers"]["slab"]["kind"] == "surface" and a.prop.opacity == pytest.approx(0.4)
    assert m.n_cells == 1 and np.allclose(sorted(m["Depth (km)"]), [10, 20, 40, 50])
    assert np.isclose(m.points[:, 2].min(), -50)
    with pytest.raises(ValueError, match="no cells inside"):
        add_surface(p, Grid(v + 0, [10, 11, 12, 13], [10, 12, 14]), name="far")


@pv3d
def test_dem_drape_and_shapes(tmp_path):
    rio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    from shapely.geometry import LineString, Polygon

    from pituplots.layers import add_dem, add_shapes, drape
    from pituplots.scene import plot_catalog_3d

    c = mine()
    p = plot_catalog_3d(c, projection=GRID)
    e, n = utm([-500, 1500], [-500, 1500])
    w, s_, east, north = e.min() - 500, n.min() - 500, e.max() + 500, n.max() + 500
    nx, ny = 60, 50
    z = np.linspace(2000, 3000, nx)[None, :].repeat(ny, 0).astype("float32")
    f = tmp_path / "dem.tif"
    with rio.open(f, "w", driver="GTiff", width=nx, height=ny, count=1, dtype="float32", crs="EPSG:32719",
                  transform=from_origin(w, north, (east - w) / nx, (north - s_) / ny), nodata=-9999) as d:
        d.write(z, 1)
    a = add_dem(p, f, crop=None)
    assert p._pituplots["layers"]["dem"]["kind"] == "dem" and len(p._pituplots["dems"]) == 1
    assert 1990 < a.mapper.dataset["Elevation (m)"].min() and a.mapper.dataset["Elevation (m)"].max() < 3010
    h = drape(p, [e.min(), e.max()], [n.mean()] * 2, "EPSG:32719")
    assert 2000 < h[0] < h[1] < 3000
    assert np.isnan(drape(p, [0], [0], "EPSG:32719")).all()
    ln = add_shapes(p, [LineString([(100, 100), (900, 900)]), Polygon([(0, 0), (500, 0), (500, 500)])],
                    crs="local", z="dem", name="tunnels", step=50, crop=None)
    m = ln.mapper.dataset
    assert m.n_lines > 2 and np.isfinite(m.points).all() and m.points[:, 2].min() > 1900
    flat = add_shapes(p, [LineString([(100, 100), (900, 900)])], crs="local", z=2500.0, name="flat", crop=None)
    assert np.allclose(flat.mapper.dataset.points[:, 2], 2500)
    with pytest.raises(ValueError, match="no lines inside"):
        add_shapes(p, [LineString([(1e6, 1e6), (1e6 + 1, 1e6)])], crs="local", name="far")


@pv3d
def test_mesh(tmp_path):
    import pyvista as pv

    from pituplots.layers import add_mesh
    from pituplots.scene import plot_catalog_3d

    p = plot_catalog_3d(mine())
    f = tmp_path / "stope.vtk"
    pv.Cube(center=(500, 500, 2000), x_length=100, y_length=50, z_length=20).save(f)
    a = add_mesh(p, f, opacity=0.3)
    b = np.array(a.mapper.dataset.bounds)
    o = p._pituplots["origin"]
    assert np.allclose(b, [450 - o[0], 550 - o[0], 475 - o[1], 525 - o[1], 1990, 2010])
    assert p._pituplots["layers"]["stope"]["kind"] == "mesh"


@pv3d
def test_section_clips_layers():
    import pyvista as pv
    from shapely.geometry import LineString

    from pituplots.layers import add_shapes, add_surface
    from pituplots.scene import add_section, plot_catalog_3d, remove_section

    c = Catalog([-72, -70], [-34, -30], [10, 100], [5, 6])
    p = plot_catalog_3d(c)
    g = Grid(np.arange(40.0).reshape(5, 8), np.linspace(-73, -69, 9), np.linspace(-35, -29, 6))
    a = add_surface(p, g, name="slab", crop=None)
    ln = add_shapes(p, [LineString([(-73, -32), (-69, -32)])], name="line", crop=None)
    full = (a.mapper.dataset.n_cells, ln.mapper.dataset.n_lines)
    add_section(p, center=(-71, -32), crs="EPSG:4326", azimuth=0, width=30)
    s, l2 = pv.wrap(a.mapper.GetInput()), pv.wrap(ln.mapper.GetInput())
    assert 0 < s.n_cells < full[0] and 0 < l2.n_lines < full[1]
    d = np.abs((s.points[np.unique(s.faces.reshape(-1, 5)[:, 1:])][:, :2] - p._pituplots["section"]["center"])
               @ p._pituplots["section"]["normal"])
    assert d.min() <= 30
    add_shapes(p, [LineString([(-73, -31), (-69, -31)])], name="late", crop=None)
    assert pv.wrap(p._pituplots["layers"]["late"]["actor"].mapper.GetInput()).n_lines < 100
    remove_section(p)
    assert pv.wrap(a.mapper.GetInput()).n_cells == full[0]


@pv3d
def test_add_layers_and_section_crs(tmp_path):
    from pituplots.layers import add_layers
    from pituplots.scene import add_section, plot_catalog_3d, to_scene

    c = Catalog([-72, -70], [-34, -30], [10, 100], [5, 6])
    p = plot_catalog_3d(c)
    f = tmp_path / "trench.geojson"
    f.write_text('{"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": '
                 '{"type": "LineString", "coordinates": [[-72.5, -34], [-71.5, -30]]}}]}')
    names = add_layers(p, [{"kind": "shapes", "path": "trench.geojson", "color": "red"}], base=tmp_path)
    assert names == ["trench"] and "trench" in p._pituplots["layers"]
    with pytest.raises(ValueError, match="layer kind"):
        add_layers(p, [{"kind": "volume", "path": "x"}])
    sec = add_section(p, center=(-71, -32), crs="EPSG:4326", azimuth=90, width=10)
    q = to_scene(p, [-71], [-32], 0, "EPSG:4326")[0, :2]
    assert np.allclose(sec["center"], q)


def test_missing_pyvista(monkeypatch):
    from pituplots.layers import add_surface

    monkeypatch.setitem(sys.modules, "pyvista", None)
    with pytest.raises(MissingDependency, match="3d"):
        add_surface(None, None)
