import sys
from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest

from pituplots import Catalog, MissingDependency, PituplotsWarning
from pituplots.scene import (add_section, animate, export_flags, plot_catalog_3d, radius, remove_section, screenshot,
                              show_events, step_section, view_section, window)

pv3d = pytest.mark.skipif(find_spec("pyvista") is None, reason="needs pyvista")


@pytest.fixture(autouse=True)
def off_screen():
    if find_spec("pyvista"):
        import pyvista as pv

        pv.OFF_SCREEN = True
    yield


def mine(n=50):
    r = np.random.default_rng(1)
    t = np.datetime64("2016-01-01") + r.integers(0, 900, n).astype("timedelta64[D]")
    return Catalog(r.normal(500, 300, n), r.normal(800, 300, n), -r.normal(2.2, 0.2, n), r.uniform(1, 3, n), t,
                   {"kind": r.choice(["blast", "shear", "tensile"], n), "id": np.arange(n)}, crs="local")


def glyphs(p, i=-1):
    import pyvista as pv

    return pv.merge([g["mesh"] for g in p._pituplots["catalogs"][i]["parts"]])


def test_radius():
    assert np.isclose(radius(3.0, 3e6), np.cbrt(7 * 10 ** 13.6 / 48e6))
    assert np.isclose(radius(5) / radius(4), 10 ** 0.5)


def test_missing_pyvista(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyvista", None)
    with pytest.raises(MissingDependency, match="3d"):
        plot_catalog_3d(mine())


@pv3d
def test_local_scene():
    c = mine()
    p = plot_catalog_3d(c)
    st = p._pituplots
    assert st["unit"] == "m" and st["crs"] == "local" and st["z_exag"] == 1
    g = glyphs(p)
    assert set(np.unique(g["index"])) == set(range(len(c)))
    xyz, o = c.local_xyz()
    assert st["origin"] == o
    assert np.allclose(g.bounds[4:], [xyz[:, 2].min(), xyz[:, 2].max()], atol=60)
    assert np.allclose(g["value"].min(), c.depth.min()) and np.allclose(g["value"].max(), c.depth.max())


@pv3d
def test_geographic_km_and_exaggeration():
    c = Catalog([-72, -70, -68], [-30, -25, -20], [10, 100, 300], [5, 6, 7])
    p = plot_catalog_3d(c, z_exag=3, color_by="z", size_by=None)
    assert p._pituplots["unit"] == "km"
    z = glyphs(p).points[:, 2]
    assert -910 < z.min() < -890 and np.isclose(glyphs(p)["value"].min(), -300, atol=5)


@pv3d
def test_sizes():
    c = mine()
    e = plot_catalog_3d(c, color_by=None, size=(4, 32), classes=5)._pituplots["catalogs"][0]
    sizes = [g["actor"].prop.point_size for g in e["parts"]]
    assert len(e["parts"]) == 5 and np.allclose(sizes, 4 * 8 ** (np.arange(5) / 4))
    assert e["px"][c.mag.argmin()] == 4 and e["px"][c.mag.argmax()] == 32
    assert sorted(np.concatenate([g["sel"] for g in e["parts"]]).tolist()) == list(range(len(c)))
    e = plot_catalog_3d(c, size_by=None, point_size=9)._pituplots["catalogs"][0]
    assert len(e["parts"]) == 1 and e["parts"][0]["actor"].prop.point_size == 9
    g = glyphs(plot_catalog_3d(c, color_by=None, true_size=True))
    i = c.mag.argmax()
    assert np.isclose(np.ptp(g.extract_points(g["index"] == i).points[:, 0]) / 2, radius(c.mag[i]), rtol=0.05)
    with pytest.raises(ValueError, match="needs size_by"):
        plot_catalog_3d(c, true_size=True, size_by=None)


@pv3d
def test_categories_and_time():
    c = mine()
    p = plot_catalog_3d(c, color_by="kind", colors={"shear": "red"})
    g = glyphs(p)
    i = np.flatnonzero(c.extra["kind"] == "shear")[0]
    assert np.allclose(g["rgb"][g["index"] == i][0], [1, 0, 0])
    assert p.legend is not None
    g = glyphs(plot_catalog_3d(c, color_by="time"))
    assert 2016 <= g["value"].min() and g["value"].max() < 2018.5


@pv3d
def test_nan_events_left_out():
    c = mine(5)
    c.depth[2] = np.nan
    with pytest.warns(PituplotsWarning, match="1 of 5"):
        g = glyphs(plot_catalog_3d(c))
    assert 2 not in set(g["index"])


@pv3d
def test_layers_share_the_scene():
    c = mine()
    p = plot_catalog_3d(c)
    plot_catalog_3d(c[:10], plotter=p, color_by="mag")
    a, b = glyphs(p, 0), glyphs(p, 1)
    assert np.allclose(a.extract_points(a["index"] == 0).center, b.extract_points(b["index"] == 0).center)
    with pytest.raises(ValueError, match="no reference CRS"):
        plot_catalog_3d(Catalog([0], [0], [1]), plotter=p)


@pv3d
def test_config_and_errors(tmp_path):
    c = mine()
    p = plot_catalog_3d(c, config={"scene": {"color_by": "kind", "z_exag": 2, "hover": ["id"], "id": "id"}})
    assert "rgb" in glyphs(p).array_names and p._pituplots["z_exag"] == 2
    p = plot_catalog_3d(c, color_by="mag", config={"scene": {"color_by": "kind"}})
    assert "value" in glyphs(p).array_names
    with pytest.raises(ValueError, match="unknown scene keys"):
        plot_catalog_3d(c, config={"scene": {"colour": 1}})
    with pytest.raises(ValueError, match="view"):
        plot_catalog_3d(c, view="side")
    with pytest.raises(ValueError, match="unit"):
        plot_catalog_3d(c, unit="ft")


@pv3d
def test_sprites_and_screenshot(tmp_path):
    c = mine()
    p = plot_catalog_3d(c, true_size=True, max_glyphs=10, view="map", window_size=(300, 200))
    assert type(p._pituplots["catalogs"][0]["parts"][0]["actor"].mapper).__name__ == "PointGaussianMapper"
    assert p.camera.parallel_projection
    img = screenshot(p, tmp_path / "s.png")
    assert (tmp_path / "s.png").is_file() and img.shape[:2] == (200, 300)
    assert img.std() > 5


# picking


def project(p, i=0):
    import pyvista as pv

    e = p._pituplots["catalogs"][i]
    w, h = p.window_size
    m = pv.array_from_vtkmatrix(p.camera.GetCompositeProjectionTransformMatrix(w / h, -1, 1))
    q = np.c_[e["points"], np.ones(len(e["points"]))] @ m.T
    return (q[:, 0] / q[:, 3] + 1) / 2 * w, (q[:, 1] / q[:, 3] + 1) / 2 * h


@pv3d
@pytest.mark.parametrize("view", ["iso", "map"])
def test_pick_at(view):
    c = mine()
    p = plot_catalog_3d(c, view=view, config={"scene": {"id": "id", "hover": ["mag", "kind"]}})
    p.render()
    pk = p._pituplots["picker"]
    x, y = project(p)
    j = int(np.argmax(c.mag))
    assert pk.at(x[j], y[j]) == (0, j)
    assert pk.at(-100, -100) is None
    pk.show_label(x[j], y[j])
    assert pk.label.GetInput().startswith(f"id {j}\nmag") and "kind" in pk.label.GetInput()


@pv3d
def test_pick_front_event_and_layers():
    c = Catalog([0, 0], [0, 0], [1, 2], [2, 2], extra={"id": np.array([10, 11])}, crs="local")
    p = plot_catalog_3d(c, view="map", size_by=None)
    plot_catalog_3d(Catalog([0], [0], [0.5], [2], crs="local"), plotter=p, size_by=None)
    p.render()
    x, y = project(p)
    assert p._pituplots["picker"].at(x[0], y[0]) == (1, 0)


@pv3d
def test_panel_flags_and_export(tmp_path):
    c = mine()
    c.extra["note"] = np.where(np.arange(len(c)) == 3, "", "x")
    p = plot_catalog_3d(c, config={"scene": {"id": "id", "hover": ["kind"]}})
    pk = p._pituplots["picker"]
    pk.select((0, 3))
    text = pk.panel.GetInput()
    assert text.startswith("id 3\nx") and "kind" in text and "note" not in text and "time" in text
    pk.toggle((0, 3)), pk.toggle((0, 5)), pk.toggle((0, 5)), pk.toggle(None)
    assert pk.flags == [(0, 3)] and "[flagged]" in pk.panel.GetInput()
    assert "flags" in p.actors and "selected" in p.actors
    f = export_flags(p, tmp_path / "f.csv")
    t = pd.read_csv(f)
    assert t.columns.tolist() == ["catalog", "row", "id", "x", "y", "depth", "mag", "time", "kind"]
    assert t["row"].tolist() == [3] and t["id"].tolist() == [3]
    pk.toggle((0, 3))
    assert "flags" not in p.actors
    assert pd.read_csv(export_flags(p, tmp_path / "g.csv")).empty


@pv3d
def test_no_picker():
    p = plot_catalog_3d(mine(), pick=False)
    assert "picker" not in p._pituplots


@pv3d
def test_projection():
    c = Catalog([-71, -70], [-33, -32], [10, 20])
    p = plot_catalog_3d(c, projection="EPSG:32719")
    from pyproj import Transformer

    x, y = Transformer.from_crs("EPSG:4326", "EPSG:32719", always_xy=True).transform(c.lon, c.lat)
    assert np.allclose(p._pituplots["origin"], [(x.min() + x.max()) / 2, (y.min() + y.max()) / 2])
    plot_catalog_3d(Catalog([5e5], [6e6], [1], crs="EPSG:32719"), plotter=p)
    q = plot_catalog_3d(c)
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:32719", always_xy=True).transform(-70.5, -32.5)
    plot_catalog_3d(Catalog([x], [y], [15], crs="EPSG:32719"), plotter=q)
    plot_catalog_3d(Catalog([-70.5], [-32.5], [15]), plotter=q)
    a, b = q._pituplots["catalogs"][1:]
    assert np.allclose(a["points"], b["points"], atol=1e-6)


# filters and time


@pv3d
@pytest.mark.parametrize("mode", [{}, {"true_size": True}, {"true_size": True, "max_glyphs": 1}])
def test_show_events(mode):
    import pyvista as pv

    c = mine()
    p = plot_catalog_3d(c, color_by="mag", **mode)
    keep = c.mag > 2
    assert show_events(p, keep) == keep.sum()
    shown = set()
    for part in p._pituplots["catalogs"][0]["parts"]:
        g = pv.wrap(part["actor"].mapper.GetInput())
        used = np.unique(g.faces.reshape(-1, 4)[:, 1:]) if part["cells"] else slice(None)
        shown |= set(g["index"][used].astype(int)) if g.n_points else set()
        assert g.n_points == 0 or g.GetPointData().GetScalars().GetName() == "value"
    assert shown == set(np.flatnonzero(keep))
    x, y = project(p)
    j = int(np.flatnonzero(~keep)[0])
    p.render()
    assert p._pituplots["picker"].at(x[j], y[j]) != (0, j)
    assert show_events(p, np.zeros(len(c), bool)) == 0
    assert show_events(p) == len(c)
    with pytest.raises(ValueError, match="where has shape"):
        show_events(p, keep[:3])


def test_window():
    t = np.array(["2020-01-01", "2020-01-10", "2020-02-01", "NaT"], "datetime64[ms]")
    assert window(t, "2020-01-10", "9D").tolist() == [False, True, False, False]
    assert window(t, "2020-01-10", "10D").tolist() == [True, True, False, False]
    assert window(t, "2020-01-10").tolist() == [True, True, False, False]
    assert window(t, "2020-02-01", start="2020-01-01").tolist() == [False, True, True, False]


@pv3d
@pytest.mark.parametrize("ext", ["gif", "mp4"])
def test_animate(tmp_path, ext):
    if ext == "mp4":
        pytest.importorskip("imageio_ffmpeg")
    pytest.importorskip("imageio")
    c = mine()
    p = plot_catalog_3d(c, window_size=(160, 128))
    n = animate(p, tmp_path / f"m.{ext}", step="100D", length="200D")
    assert n == len(np.arange(c.time.min() + np.timedelta64(200, "D"), c.time.max() + np.timedelta64(100, "D"),
                              np.timedelta64(100, "D")))
    assert (tmp_path / f"m.{ext}").stat().st_size > 1000
    assert p._pituplots["catalogs"][0]["shown"].all() and "clock" not in p.actors
    with pytest.raises(ValueError, match="needs event times"):
        animate(plot_catalog_3d(Catalog([0, 1], [0, 1], [1, 1], crs="local")), tmp_path / "x.gif", "1D")


# sections


@pv3d
def test_section():
    c = Catalog([-300, -100, 0, 100, 300, 0], [0, 0, 0, 0, 0, 500], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1],
                crs="local")
    p = plot_catalog_3d(c, size_by=None)
    sec = add_section(p, center=(0, 0), azimuth=0, width=150)
    assert np.allclose(sec["normal"], [1, 0]) and sec["shown"] == [4]
    assert p._pituplots["catalogs"][0]["shown"].tolist() == [False, True, True, True, False, True]
    assert "section" in p.actors
    assert show_events(p, c.lat < 100) == 3
    sec = add_section(p, center=(0, -250), azimuth=90, width=50)
    assert sec["shown"] == [5] and p._pituplots["catalogs"][0]["shown"].tolist() == [True] * 5 + [False]
    view_section(p)
    pos, focal, up = p.camera_position
    assert np.allclose(up, (0, 0, 1)) and abs(pos[1] - focal[1]) > 0 and np.isclose(pos[0], focal[0])
    assert p.camera.parallel_projection and not p.actors["section"].GetVisibility()
    remove_section(p)
    assert "section" not in p.actors and p._pituplots["catalogs"][0]["shown"].sum() == 5
    d = add_section(p)
    assert np.isclose(d["width"], 0.05 * np.hypot(600, 500)) and d["azimuth"] == 90


@pv3d
def test_section_widget():
    c = mine()
    p = plot_catalog_3d(c)
    add_section(p, width=100, widget=True)
    w = getattr(p, "widgets", p).plane_widgets[0]
    w.SetNormal(0, 1, 0)
    w.SetOrigin(0, 0, 0)
    w.InvokeEvent("EndInteractionEvent")
    sec = p._pituplots["section"]
    assert np.isclose(sec["azimuth"] % 180, 90) and np.allclose(sec["center"], 0)


@pv3d
def test_step_section_and_wheel():
    c = mine()
    p = plot_catalog_3d(c)
    add_section(p, center=(0, 0), azimuth=90, width=40)
    f0 = np.array(p.camera_position[1])
    sec = step_section(p, 2)
    assert np.allclose(sec["center"], [0, -160]) and np.allclose(np.array(p.camera_position[1]) - f0, [0, -160, 0])
    it = p.iren.interactor
    style = it.GetInteractorStyle()
    it.SetControlKey(1)
    style.InvokeEvent("MouseWheelForwardEvent")
    assert np.allclose(p._pituplots["section"]["center"], [0, -240])
    it.SetControlKey(0)
    style.InvokeEvent("MouseWheelForwardEvent")
    assert np.allclose(p._pituplots["section"]["center"], [0, -240])


@pv3d
def test_graticule_and_legend():
    from pituplots.scene import add_graticule, degrees, to_lonlat

    c = Catalog([-72, -70, -68], [-34, -31, -28], [10, 100, 300], [5, 6, 7], extra={"k": np.array(list("abc"))})
    p = plot_catalog_3d(c, color_by="k")
    assert "graticule" in p.actors and any(k.startswith("graticule_labels") for k in p.actors)
    e = p._pituplots["catalogs"][0]
    lon, lat = to_lonlat(p, e["points"][:, 0], e["points"][:, 1])
    assert np.allclose(lon, c.lon) and np.allclose(lat, c.lat)
    assert add_graticule(p) in (1, 2) and add_graticule(p, step=0.5) == 0.5
    assert degrees(-70.5, 0.5, "WE") == "70.5°W" and degrees(-30, 5, "SN") == "30°S" and degrees(0.25, 0.01, "SN") == "0.25°N"
    box = p.legend
    assert box.GetPosition()[0] < 0.05 and box.GetPosition()[1] > 0.5
    q = plot_catalog_3d(mine())
    assert to_lonlat(q, [0], [0]) is None and "graticule" not in q.actors


@pv3d
def test_colours_survive_filters_and_fade():
    c = mine()
    p = plot_catalog_3d(c, color_by="kind", colors={"shear": "red"}, view="map", size_by=None, point_size=30,
                        window_size=(400, 300), axes=False, colorbar=False)
    a = screenshot(p, None)
    show_events(p, c.mag > 0, alpha=np.full(len(c), 0.5))
    show_events(p, c.mag > 0)
    b = screenshot(p, None)
    assert np.abs(a.astype(int) - b.astype(int)).max() <= 2


@pv3d
def test_fade_and_inset(tmp_path):
    from pituplots.scene import fade, inset

    t = np.array(["2020-01-01", "2020-01-06", "2020-01-11", "NaT"], "datetime64[ms]")
    assert np.allclose(fade(t, "2020-01-11", "10D"), [0.1, 0.55, 1, 0])
    assert np.allclose(fade(t, "2020-01-06", start="2020-01-01", floor=0), [0, 1, 0, 0])
    c = mine()
    p = plot_catalog_3d(c)
    assert "inset" not in p._pituplots
    add_section(p, center=(0, 0), azimuth=30, width=50)
    ren = p._pituplots["inset"]["renderer"]
    assert ren.GetDraw() and p._pituplots["inset"]["band"].n_points == 4
    assert "strike 30" in p._pituplots["inset"]["text"].GetInput()
    remove_section(p)
    assert not ren.GetDraw()
    q = plot_catalog_3d(c, pick=False)
    add_section(q, width=50)
    assert "inset" not in q._pituplots and inset(q).GetDraw()
    n = animate(p, tmp_path / "f.gif", step="200D", length="300D", fading=True)
    assert n > 0 and (tmp_path / "f.gif").stat().st_size > 1000
