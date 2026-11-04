import datetime as dt

import numpy as np
import pandas as pd
import pytest

from pituplots import Catalog, MissingDependency, PituplotsWarning, plot_catalog, plot_section
from pituplots.io import load_config, read_catalog, trim_catalog
from pituplots.io.catalogs import decimal_years, from_days

MINE = "id,X,Y,Z,ML,date,frac\n1,100,200,2100,1.0,42377,0.25\n2,300,400,2300,2.0,42378,0.5\n"


def write(p, text):
    p.write_text(text)
    return p


def mine_cfg(**kw):
    return {"columns": {"x": "X", "y": "Y", "z": "Z", "mag": "ML"}, "z": "elevation", "z_unit": "m",
            "time": {"columns": ["date", "frac"], "format": "excel"}, **kw}


# config


def test_config_yaml_and_dict(tmp_path):
    f = write(tmp_path / "c.yaml", "columns: {lon: a, lat: b}\nscene: {id: id}\n")
    assert load_config(f) == {"columns": {"lon": "a", "lat": "b"}, "scene": {"id": "id"}}
    assert load_config(None) == {}
    assert load_config({"crs": "local"}) == {"crs": "local"}
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config({"colour": 1})
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "x.yaml")


def test_config_drives_reader(tmp_path):
    f = write(tmp_path / "m.csv", MINE)
    c = read_catalog(f, config=mine_cfg())
    assert c.crs == "local" and c.lon.tolist() == [100, 300]
    assert np.allclose(c.depth, [-2.1, -2.3])
    assert c.time[0] == np.datetime64("2016-01-08T06:00")
    assert c.time[1] == np.datetime64("2016-01-09T12:00")
    assert set(c.extra) == {"id"}
    assert read_catalog(f, config=mine_cfg(), crs="EPSG:32719").crs == "EPSG:32719"
    assert np.allclose(read_catalog(f, config=mine_cfg(), z="depth").depth, [2.1, 2.3])


def test_config_errors(tmp_path):
    f = write(tmp_path / "m.csv", MINE)
    with pytest.raises(ValueError, match="z must be"):
        read_catalog(f, config=mine_cfg(z="height"))
    with pytest.raises(ValueError, match="time format"):
        read_catalog(f, config=mine_cfg(time={"columns": "date", "format": "julian"}))
    with pytest.raises(ValueError, match="no column 'when'"):
        read_catalog(f, config=mine_cfg(time={"columns": ["when"], "format": "excel"}))
    with pytest.raises(ValueError, match="one column"):
        read_catalog(f, config=mine_cfg(time={"columns": ["date", "frac"], "format": "decimal_year"}))
    with pytest.raises(ValueError, match="x, y, z"):
        read_catalog(f, columns={"w": "X", "lat": "Y"})


# time formats


def test_time_formats(tmp_path):
    f = write(tmp_path / "t.csv", "lon,lat,dy,ep\n1,2,2016.5,86400\n3,4,,0\n")
    c = read_catalog(f, time={"columns": "dy", "format": "decimal_year"})
    assert c.time[0] == np.datetime64("2016-07-02")
    assert np.isnat(c.time[1])
    c = read_catalog(f, time={"columns": "ep", "format": "epoch"}, time_unit="s")
    assert c.time.tolist() == [dt.datetime(1970, 1, 2), dt.datetime(1970, 1, 1)]
    assert np.isnat(from_days([np.nan], "1899-12-30")[0])
    assert decimal_years([2000.0])[0] == np.datetime64("2000-01-01")


def test_excel(tmp_path):
    pytest.importorskip("openpyxl")
    f = tmp_path / "m.xlsx"
    pd.DataFrame({"X": [1.0, 2.0], "Y": [3.0, 4.0], "Z": [100.0, 200.0],
                  "date": [dt.datetime(2016, 1, 8), dt.datetime(2016, 1, 9)],
                  "tod": [dt.time(5, 37, 59), dt.time(14, 27, 20)]}).to_excel(f, index=False)
    c = read_catalog(f, columns={"x": "X", "y": "Y", "z": "Z"}, z="elevation", z_unit="m",
                     time={"columns": ["date", "tod"], "format": "excel"})
    assert c.time.tolist() == [dt.datetime(2016, 1, 8, 5, 37, 59), dt.datetime(2016, 1, 9, 14, 27, 20)]
    assert np.allclose(c.depth, [-0.1, -0.2]) and c.crs == "local"


def test_excel_missing(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "openpyxl", None)
    f = write(tmp_path / "m.xlsx", "")
    with pytest.raises(MissingDependency, match="excel"):
        read_catalog(f)


# trim


def test_trim():
    c = Catalog([0, 1, 2, 3, 100], [0, 1, 2, 3, 0], [1, 1, 2, 2, 50], [1, 2, 3, 4, 9], crs="local")
    with pytest.warns(PituplotsWarning, match="dropped 1 of 5"):
        assert len(trim_catalog(c, box=[-1, 10, -1, 10])) == 4
    with pytest.warns(PituplotsWarning, match="dropped 1 of 5"):
        assert trim_catalog(c, mad=5).lon.tolist() == [0, 1, 2, 3]
    with pytest.warns(PituplotsWarning):
        assert len(trim_catalog(c, box=[-1, 200, -1, 10, 0, 1.5])) == 2
    with pytest.warns(PituplotsWarning):
        assert trim_catalog(c, mag=(2, 4)).mag.tolist() == [2, 3, 4]
    assert len(trim_catalog(c)) == 5
    with pytest.raises(ValueError, match="4 or 6"):
        trim_catalog(c, box=[0, 1])
    with pytest.raises(ValueError, match="magnitudes"):
        trim_catalog(Catalog([0], [0]), mag=(1, 2))
    with pytest.raises(ValueError, match="depths"):
        trim_catalog(Catalog([0], [0]), box=[0, 1, 0, 1, 0, 1])


def test_trim_from_config(tmp_path):
    f = write(tmp_path / "m.csv", MINE + "3,90000,200,2200,1.5,42379,0\n")
    with pytest.warns(PituplotsWarning, match="dropped 1 of 3"):
        assert len(read_catalog(f, config=mine_cfg(trim={"box": [0, 1000, 0, 1000]}))) == 2


# local coordinates


def test_local_xyz_projected():
    c = Catalog([100, 300], [200, 600], [-2.1, 0.5], crs="local")
    xyz, o = c.local_xyz()
    assert o == (200, 400)
    assert np.allclose(xyz, [[-100, -200, 2100], [100, 200, -500]])
    assert np.allclose(c.local_xyz(origin=(0, 0))[0][:, 0], [100, 300])
    assert np.allclose(Catalog([1], [2], crs="local").local_xyz()[0], 0)
    assert not c.geographic and Catalog([0], [0]).geographic
    assert not Catalog([0], [0], crs="EPSG:32719").geographic


def test_local_xyz_geographic():
    c = Catalog([-71, -71, 179.5, -179.5], [-33, -32, 0, 0], [10, 20, 0, 0])
    xyz, o = c[:2].local_xyz()
    assert o == (-71, -32.5)
    assert np.isclose(np.ptp(xyz[:, 1]), 111e3, rtol=0.01) and np.allclose(xyz[:, 0], 0, atol=1)
    assert np.allclose(xyz[:, 2], [-10e3, -20e3])
    xyz, o = c[2:].local_xyz()
    assert o[0] == 180 and np.isclose(np.ptp(xyz[:, 0]), 111e3, rtol=0.01)
    assert np.allclose(Catalog([1, 2], [3, 4]).local_xyz()[0][:, 2], 0)


def test_crs_kept_and_maps_refuse():
    c = Catalog([0, 1], [0, 1], [1, 2], [3, 4], crs="local")
    assert c[[1]].crs == "local"
    with pytest.raises(ValueError, match="maps and sections need lon/lat"):
        plot_catalog(c)
    with pytest.raises(ValueError, match="maps and sections need lon/lat"):
        plot_section(c, None)


def test_local_xyz_projection():
    from pyproj import Transformer

    c = Catalog([-71, -70], [-33, -32], [1, 2])
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:32719", always_xy=True).transform(c.lon, c.lat)
    xyz, o = c.local_xyz(projection="EPSG:32719")
    assert np.allclose(xyz[:, 0] + o[0], x) and np.allclose(xyz[:, 1] + o[1], y)
    assert np.allclose(c.local_xyz((0, 0), "EPSG:32719")[0][:, 0], x)
    u = Catalog(x, y, [1, 2], crs="EPSG:32719")
    assert np.allclose(u.local_xyz(projection="EPSG:32719")[0], u.local_xyz()[0])
    for bad in [(Catalog([0], [0], crs="local"), "EPSG:32719"), (c, "EPSG:4326"), (c, "local")]:
        with pytest.raises(ValueError, match="cannot project"):
            bad[0].local_xyz(projection=bad[1])
