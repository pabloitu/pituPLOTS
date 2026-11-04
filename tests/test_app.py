import itertools
import sys
from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest

from pituplots import Catalog, MissingDependency
from pituplots.scene import export_flags

app = pytest.mark.skipif(not all(find_spec(m) for m in ("pyvista", "trame", "trame_vtk", "trame_vuetify")),
                         reason="needs the app extra")
names = (f"test{i}" for i in itertools.count())


def mine(n=40):
    r = np.random.default_rng(2)
    t = np.datetime64("2016-01-01") + r.integers(0, 900, n).astype("timedelta64[D]")
    return Catalog(r.normal(500, 300, n), r.normal(800, 300, n), -r.normal(2.2, 0.2, n), r.uniform(1, 3, n), t,
                   {"kind": r.choice(["blast", "shear"], n), "id": np.arange(n), "note": np.array(["x"] * n)},
                   crs="local")


@pytest.fixture(autouse=True)
def off_screen():
    if find_spec("pyvista"):
        import pyvista as pv

        pv.OFF_SCREEN = True
    yield


def test_missing_trame(monkeypatch):
    from pituplots.app import build_app

    monkeypatch.setitem(sys.modules, "trame", None)
    with pytest.raises(MissingDependency, match="app"):
        build_app(mine())


def test_columns():
    from pituplots.app import columns

    c = mine(50)
    c.extra["many"] = np.array([f"e{i}" for i in range(len(c))])
    assert columns(c) == ["depth", "z", "mag", "time", "kind", "id", "note"]
    assert columns(Catalog([0], [0])) == ["depth", "z"]


@app
def test_state_from_config():
    from pituplots.app import build_app

    cfg = {"scene": {"id": "id", "hover": ["kind"], "color_by": "kind", "true_size": True,
                     "colors": {"shear": "red"}}}
    srv = build_app(mine(), config=cfg, name=next(names))
    s = srv.state
    assert s.color_by == "kind" and s.size_mode == "true size" and s.px == [3, 26]
    assert s.options[:4] == ["depth", "z", "mag", "time"]
    g = srv.plotter._pituplots["catalogs"][0]["parts"][0]["mesh"]
    assert "rgb" in g.array_names
    assert srv.plotter._pituplots["picker"].hover == ["lon", "lat", "mag", "depth", "time", "class", "kind"]


@app
def test_redraw_keeps_camera_and_flags(tmp_path):
    from pituplots.app import build_app

    srv = build_app(mine(), config={"scene": {"id": "id"}}, name=next(names))
    s, ctrl, p = srv.state, srv.controller, srv.plotter
    ctrl.view_update = lambda: None
    ctrl.select((0, 3))
    assert s.event_title == "id 3" and dict(s.event)["kind"] in ("blast", "shear") and not s.flagged
    ctrl.toggle()
    assert s.flagged and len(s.flags) == 1 and s.flags[0]["id"] == "3"
    assert [h["key"] for h in s.flag_headers][:2] == ["row", "id"]
    cam = p.camera_position
    s.color_by, s.size_mode, s.px = "mag", "magnitude", [5, 40]
    ctrl.draw()
    st = p._pituplots
    e = st["catalogs"][0]
    assert len(st["catalogs"]) == 1 and "value" in e["parts"][0]["mesh"].array_names and e["px"].max() == 40
    assert np.allclose(np.array([*p.camera_position], float), np.array([*cam], float))
    assert "flags" in p.actors and "selected" in p.actors
    assert pd.read_csv(export_flags(p, tmp_path / "f.csv"))["id"].tolist() == [3]


@app
def test_main(monkeypatch, tmp_path):
    from trame_server import Server

    from pituplots import app

    f = tmp_path / "c.csv"
    f.write_text("X,Y,Z,ML\n1,2,100,1\n3,4,200,2\n")
    cfg = tmp_path / "c.yaml"
    cfg.write_text('columns: {x: X, y: Y, z: Z, mag: ML}\nz: elevation\nz_unit: m\nscene: {color_by: mag}\n')
    seen = {}
    monkeypatch.setattr(Server, "start", lambda self, **kw: seen.update(kw, title=self.state.color_by))
    app.main([str(f), "--config", str(cfg), "--port", "8123", "--no-browser"])
    assert seen == {"port": 8123, "host": "localhost", "open_browser": False, "title": "mag"}


@app
def test_filters_and_time():
    from pituplots.app import build_app

    c = mine()
    srv = build_app(c, config={"scene": {"id": "id"}}, name=next(names))
    s, ctrl, p = srv.state, srv.controller, srv.plotter
    ctrl.view_update = lambda: None
    shown = lambda: p._pituplots["catalogs"][0]["shown"]  # noqa: E731
    assert s.cat_col == "kind" and s.cat_keep == ["blast", "shear"] and s.shown == f"{len(c)} of {len(c)} events"
    s.cat_keep = ["shear"]
    s.mag_range = [2.0, s.mag_lim[1]]
    ctrl.apply()
    assert np.array_equal(shown(), (c.extra["kind"] == "shear") & (c.mag >= 2 - 1e-9))
    s.cat_keep, s.mag_range = ["blast", "shear"], list(s.mag_lim)
    s.t_mode, s.t_len = "sliding", 100.0
    s.t_end = float(np.datetime64("2017-01-01", "D").astype(int))
    ctrl.apply()
    d = c.time.astype("datetime64[ms]").astype("int64") / 86400e3
    assert np.array_equal(shown(), (d > s.t_end - 100) & (d <= s.t_end)) and s.t_label.endswith("2017-01-01")
    assert "<svg" in s.hist and "fill-opacity" in s.hist
    s.t_mode = "all"
    ctrl.toggle((0, 4))
    s.flagged_only = True
    ctrl.apply()
    assert np.flatnonzero(shown()).tolist() == [4]
    s.flagged_only, s.search = False, "7"
    ctrl.search()
    assert s.event_title == "id 7"
    s.search = "nope"
    ctrl.search()
    assert "no event" in s.message


@app
def test_section_panel():
    from pituplots.app import build_app

    c = mine()
    srv = build_app(c, name=next(names))
    s, ctrl, p = srv.state, srv.controller, srv.plotter
    ctrl.view_update = lambda: None
    assert not s.sec_on and s.unit == "m" and s.sec_width > 0
    s.sec_on, s.sec_width = True, 50.0
    ctrl.cut()
    sec = p._pituplots["section"]
    assert sec["width"] == 50 and s.shown.endswith("in the section")
    s.color_by = "mag"
    ctrl.draw()
    assert p._pituplots["section"]["width"] == 50
    ctrl.face()
    assert p.camera.parallel_projection
    s.sec_on = False
    ctrl.cut()
    assert "section" not in p._pituplots and p._pituplots["catalogs"][0]["shown"].all()


@app
def test_layers_panel(tmp_path):
    import yaml

    from pituplots.app import build_app

    c = Catalog([-72, -70, -71], [-34, -30, -32], [10, 100, 50], [5, 6, 5.5])
    for f, x in (("t", -72.5), ("u", -71.5)):
        (tmp_path / f"{f}.geojson").write_text('{"type": "FeatureCollection", "features": [{"type": "Feature", '
                                               '"properties": {}, "geometry": {"type": "LineString", '
                                               f'"coordinates": [[{x}, -34], [{x + 1}, -30]]}}}}]}}')
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"layers": [{"kind": "shapes", "path": "t.geojson", "opacity": 0.5},
                                              {"kind": "shapes", "path": "u.geojson"}]}))
    srv = build_app(c, config=str(cfg), name=next(names))
    s, ctrl, p = srv.state, srv.controller, srv.plotter
    ctrl.view_update = lambda: None
    assert s.layer_names == ["t", "u"] and s.lo0 == pytest.approx(0.5) and s.lv1
    s.lv0, s.lo1 = False, 0.2
    ctrl.layer()
    assert not p._pituplots["layers"]["t"]["actor"].GetVisibility()
    assert p._pituplots["layers"]["u"]["actor"].prop.opacity == pytest.approx(0.2)
    s.color_by = "mag"
    ctrl.draw()
    assert "t" in p._pituplots["layers"] and not p._pituplots["layers"]["t"]["actor"].GetVisibility()


@app
def test_section_steps():
    from pituplots.app import build_app

    srv = build_app(mine(), name=next(names))
    s, ctrl, p = srv.state, srv.controller, srv.plotter
    ctrl.view_update = lambda: None
    ctrl.step(1)
    assert s.sec_pos == 0
    s.sec_on, s.sec_width = True, 50.0
    ctrl.cut()
    focal = np.array(p.camera_position[1])
    ctrl.step(1)
    ctrl.cut()
    assert s.sec_pos == 100 and np.allclose(np.array(p.camera_position[1]) - focal, [0, -100, 0])
    ctrl.step(-2)
    assert s.sec_pos == -100


@app
def test_hover_and_click_without_panel():
    import pyvista as pv

    from pituplots.app import build_app

    c = Catalog([-72, -70, -71], [-34, -30, -32], [10, 100, 50], [5, 6, 5.5], extra={"class": np.array(["a", "b", "c"])})
    srv = build_app(c, name=next(names))
    s, ctrl, p = srv.state, srv.controller, srv.plotter
    ctrl.view_update = lambda: None
    p.render()
    e = p._pituplots["catalogs"][0]
    w, h = p.window_size
    m = pv.array_from_vtkmatrix(p.camera.GetCompositeProjectionTransformMatrix(w / h, -1, 1))
    q = np.r_[e["points"][1], 1] @ m.T
    ctrl.move({"x": (q[0] / q[3] + 1) / 2 * w, "y": (q[1] / q[3] + 1) / 2 * h})
    assert [ln.split()[0] for ln in s.hover.splitlines()[1:]] == ["lon", "lat", "mag", "depth", "class"]
    ctrl.select((0, 1))
    assert s.event_title == "row 1" and p._pituplots["picker"].panel.GetInput() in ("", None)


@app
def test_minimap():
    from pituplots.app import build_app

    c = Catalog([-72, -70, -71], [-34, -30, -32], [10, 100, 50], [5, 6, 5.5])
    srv = build_app(c, name=next(names))
    s, ctrl = srv.state, srv.controller
    ctrl.view_update = lambda: None
    assert "<path" in s.map_bg and s.map_sec == ""
    s.sec_on, s.sec_width = True, 20.0
    ctrl.cut()
    assert "<polygon" in s.map_sec and s.sec_label.startswith("32.") and "°S" in s.sec_label


@app
def test_years_and_fade():
    from pituplots.app import build_app

    t = np.array(["1600-01-01", "1900-06-01", "2000-01-01", "2020-01-01"], "datetime64[ms]")
    c = Catalog([0, 1, 2, 3], [0, 1, 2, 3], [1, 1, 1, 1], [1, 2, 3, 4], t, crs="local")
    srv = build_app(c, name=next(names))
    s, ctrl, p = srv.state, srv.controller, srv.plotter
    ctrl.view_update = lambda: None
    assert s.t_unit == "yr" and s.t_len == 21 and s.t_step == 4
    s.t_mode, s.t_len, s.t_end = "sliding", 30.0, float(np.datetime64("2020-01-02", "D").astype(int))
    ctrl.apply()
    assert p._pituplots["catalogs"][0]["shown"].tolist() == [False, False, True, True]
    s.t_fade = True
    ctrl.apply()
    import pyvista as pv

    a = [pv.wrap(g["actor"].mapper.GetInput()) for g in p._pituplots["catalogs"][0]["parts"]]
    alpha = {int(r): int(c[3]) for m in a if m.n_points for r, c in zip(m["index"], m["rgba"])}
    assert alpha[2] < alpha[3] and set(alpha) == {2, 3}
    t2 = np.array(["2000-01-01", "2005-01-01"], "datetime64[ms]")
    srv2 = build_app(Catalog([0, 1], [0, 1], [1, 1], [1, 2], t2, crs="local"), name=next(names))
    assert srv2.state.t_unit == "d"
