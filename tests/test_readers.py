import json
import sys

import numpy as np
import pandas as pd
import pytest

from pituplots import Grid, MissingDependency
from pituplots.io import (cells_to_outline, read_catalog, read_csep_grid, read_csep_nodes, read_geotiff,
                          read_rate_csv, read_slab2, read_vector)

# read_catalog


def write(p, text):
    p.write_text(text)
    return p


def test_catalog_headers_and_time(data):
    c = read_catalog(data / "lacq/events.csv")
    assert c.lon.size == 596 and c.time.dtype == np.dtype("datetime64[ms]")
    assert c.time[0] == np.datetime64("1969-06-01T07:04:12")
    assert "mc_window" in c.extra


def test_catalog_split_date_columns(data):
    c = read_catalog(data / "new_zealand/cat_nz.csv")
    assert c.mag[0] == pytest.approx(3.6921)
    assert c.time[0] == np.datetime64("1987-04-06T17:35:32.989")
    assert c.lon.max() > 180


def test_catalog_comcat_preset(data):
    c = read_catalog(data / "catalogs/comcat.csv", preset="csep")
    assert c.mag[0] == pytest.approx(4.73)
    assert c.time[0] == np.datetime64("2019-07-06T03:22:35.630")


def test_catalog_csep_json(data):
    c = read_catalog(data / "catalogs/italy.json", preset="csep_json")
    assert c.lat[0] == pytest.approx(44.871) and c.lon[0] == pytest.approx(10.51)
    assert c.mag[0] == 5 and c.time[0] == np.datetime64(1327478797090, "ms")


def test_catalog_empty_json(tmp_path):
    f = write(tmp_path / "a.json", json.dumps({"catalog": []}))
    assert read_catalog(f, preset="csep_json").lon.size == 0


def test_catalog_by_index_headerless(tmp_path):
    f = write(tmp_path / "a.txt", "1.0 2.0 3.0 1600000000000\n4.0 5.0 6.0 1600000001000\n")
    c = read_catalog(f, columns={"lon": 0, "lat": 1, "mag": 2}, sep=" ")
    assert c.lon.tolist() == [1, 4] and c.mag.tolist() == [3, 6]


def test_catalog_by_header_name(tmp_path):
    f = write(tmp_path / "a.csv", "long,la,mm\n1,2,3\n")
    c = read_catalog(f, columns={"lon": "long", "lat": "la", "mag": "mm"})
    assert (c.lon[0], c.lat[0], c.mag[0]) == (1, 2, 3)
    with pytest.raises(ValueError, match="no column"):
        read_catalog(f, columns={"lon": "x", "lat": "la"})
    with pytest.raises(ValueError, match="role"):
        read_catalog(f, columns={"foo": "la"})
    with pytest.raises(ValueError, match="lon/lat"):
        read_catalog(f)


@pytest.mark.parametrize("sep", [",", ";", "\t", "|"])
def test_catalog_sniffed_delimiter(tmp_path, sep):
    f = write(tmp_path / "a.txt", sep.join(["lon", "lat", "mag"]) + "\n" + sep.join("123") + "\n" + sep.join("456") + "\n")
    assert read_catalog(f).mag.tolist() == [3, 6]


def test_catalog_time_forms(tmp_path):
    f = write(tmp_path / "a.csv", "lon,lat,time\n1,2,1600000000000\n")
    assert read_catalog(f).time[0] == np.datetime64(1600000000000, "ms")
    f = write(tmp_path / "b.csv", "lon,lat,time\n1,2,1600000000\n")
    assert read_catalog(f, time_unit="s").time[0] == np.datetime64(1600000000, "s")
    f = write(tmp_path / "c.csv", "lon,lat,time\n1,2,2020-01-02T03:04:05Z\n")
    assert read_catalog(f).time[0] == np.datetime64("2020-01-02T03:04:05")
    f = write(tmp_path / "d.csv", "lon,lat,time\n1,2,2020-01-02 03:04:05+02:00\n")
    assert read_catalog(f).time[0] == np.datetime64("2020-01-02T01:04:05")
    f = write(tmp_path / "e.csv", "lon,lat,date,time\n1,2,2020-01-02,03:04:05\n")
    assert read_catalog(f).time[0] == np.datetime64("2020-01-02T03:04:05")


def test_catalog_old_years(tmp_path):
    f = write(tmp_path / "a.csv", "lon,lat,time\n1,2,1513-01-01T00:00:00\n")
    assert read_catalog(f).time[0] == np.datetime64("1513-01-01")
    f = write(tmp_path / "b.csv", "lon,lat,year,month,day\n1,2,1513,3,2\n")
    assert read_catalog(f).time[0] == np.datetime64("1513-03-02")


def test_catalog_time_list_and_named(tmp_path):
    f = write(tmp_path / "a.csv", "lon,lat,a,b,c\n1,2,2001,2,3\n")
    assert read_catalog(f, time=["a", "b", "c"]).time[0] == np.datetime64("2001-02-03")
    f = write(tmp_path / "b.csv", "lon,lat,when\n1,2,2001-02-03\n")
    assert read_catalog(f, time="when").time[0] == np.datetime64("2001-02-03")
    with pytest.raises(ValueError, match="no column"):
        read_catalog(f, time="nope")


def test_catalog_no_time(tmp_path):
    f = write(tmp_path / "a.csv", "lon,lat\n1,2\n")
    assert read_catalog(f).time is None


def test_catalog_oq_preset(tmp_path):
    f = write(tmp_path / "a.csv", "eventID,year,month,day,hour,minute,second,longitude,latitude,depth,magnitude\n"
                                  "e1,2001,2,3,4,5,6.5,10,20,30,5.5\n")
    c = read_catalog(f, preset="oq")
    assert c.time[0] == np.datetime64("2001-02-03T04:05:06.500") and c.depth[0] == 30


def test_catalog_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_catalog(tmp_path / "x.csv")
    with pytest.raises(ValueError, match="preset"):
        read_catalog(write(tmp_path / "a.csv", "lon,lat\n1,2\n"), preset="x")


# forecasts

def dat(tmp_path, rows):
    f = tmp_path / "a.dat"
    np.savetxt(f, rows)
    return f


ROWS = np.array([
    [0, .1, 0, .1, 0, 30, 5.0, 5.1, 1, 1], [0, .1, 0, .1, 0, 30, 5.1, 5.2, 2, 1],
    [.1, .2, 0, .1, 0, 30, 5.0, 5.1, 3, 1], [.1, .2, 0, .1, 0, 30, 5.1, 5.2, 4, 1],
    [0, .1, .1, .2, 0, 30, 5.0, 5.1, 9, 0], [0, .1, .1, .2, 0, 30, 5.1, 5.2, 9, 0],
    [.1, .2, .1, .2, 0, 30, 5.0, 5.1, 5, 1], [.1, .2, .1, .2, 0, 30, 5.1, 5.2, 6, 1]])


def test_csep_dat_mask_zero(tmp_path):
    g = read_csep_grid(dat(tmp_path, ROWS))
    assert g.values.shape == (2, 2) and g.mask.tolist() == [[False, False], [True, False]]
    assert g.values[0].tolist() == [3, 7] and g.values[1, 1] == 11
    assert np.isnan(g.values[1, 0]) and g.meta["magnitudes"].tolist() == [5.0, 5.1]
    assert g.x_edges == pytest.approx([0, .1, .2])


def test_csep_mag_window(tmp_path):
    g = read_csep_grid(dat(tmp_path, ROWS), mag_min=5.05)
    assert g.values[0].tolist() == [2, 4] and g.meta["mag_bins"].tolist() == [5.1]
    assert read_csep_grid(dat(tmp_path, ROWS), mag_max=5.05).values[0].tolist() == [1, 3]
    with pytest.raises(ValueError, match="magnitude"):
        read_csep_grid(dat(tmp_path, ROWS), mag_min=9)


def test_csep_dat_dateline(tmp_path):
    r = np.array([[179.9, 180.0, 0, .1, 0, 30, 5, 5.1, 1, 1], [-180.0, -179.9, 0, .1, 0, 30, 5, 5.1, 2, 1]])
    g = read_csep_grid(dat(tmp_path, r))
    assert g.values.tolist() == [[1, 2]] and g.x_edges == pytest.approx([179.9, 180, 180.1])


def test_eshm_csv_depth_bins(tmp_path):
    f = write(tmp_path / "a.csv", "lon_min, lon_max, lat_min, lat_max, depth_min, depth_max, 5.0, 5.1\n"
                                  "0,0.1,0,0.1,0,10,1,2\n0,0.1,0,0.1,10,20,3,4\n0.1,0.2,0,0.1,0,10,5,6\n")
    g = read_csep_grid(f)
    assert g.values.tolist() == [[10, 11]] and g.mask.sum() == 0


def test_csv_mask_column(tmp_path):
    f = write(tmp_path / "a.csv", "lon_min,lon_max,lat_min,lat_max,5.0,mask\n0,1,0,1,1,1\n1,2,0,1,2,0\n")
    assert read_csep_grid(f).mask.tolist() == [[False, True]]
    with pytest.raises(ValueError, match="lon_min"):
        read_csep_grid(write(tmp_path / "b.csv", "a,b\n1,2\n"))


def test_csep_hdf5(tmp_path):
    h5py = pytest.importorskip("h5py")
    f = tmp_path / "a.hdf5"
    b = np.array([[[0, 0], [0, 1], [1, 1], [1, 0]], [[1, 0], [1, 1], [2, 1], [2, 0]]], float)
    with h5py.File(f, "w") as d:
        d["rates"], d["magnitudes"], d["bboxes"], d["poly_mask"], d["dh"] = (
            np.array([[1.0, 2], [3, 4]]), np.array([5.0, 5.1]), b, np.array([1.0, 0]), np.array([1.0]))
    g = read_csep_grid(f)
    assert g.values[0, 0] == 3 and g.mask.tolist() == [[False, True]]


def test_forecast_missing_h5py(tmp_path, monkeypatch):
    f = write(tmp_path / "a.hdf5", "x")
    monkeypatch.setitem(sys.modules, "h5py", None)
    with pytest.raises(MissingDependency, match=r"pituplots\[hdf5\]"):
        read_csep_grid(f)


def test_forecast_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_csep_grid(tmp_path / "x.dat")
    with pytest.raises(ValueError, match="format"):
        read_csep_grid(write(tmp_path / "a.xml", "x"))
    with pytest.raises(ValueError, match="columns"):
        read_csep_grid(write(tmp_path / "b.dat", "1 2 3\n"))


def test_nodes(data):
    g = read_csep_nodes(data / "new_zealand/nz.testing.nodes.dat")
    assert np.isnan(g.values).all() and np.diff(g.x_edges) == pytest.approx(0.1)
    assert g.x_edges[0] == pytest.approx(172.05)


def test_rate_csv_pattern_inferred_dh(data):
    g = read_rate_csv(data / "lacq/pattern.csv")
    assert g.values.shape == (25, 25) and g.mask is None
    assert np.diff(g.x_edges) == pytest.approx(0.01)
    assert g.x_edges[0] == pytest.approx(-0.755)


def test_rate_csv_origin_anchor(data):
    assert read_rate_csv(data / "lacq/pattern.csv", anchor="origin").x_edges[0] == pytest.approx(-0.75)


def test_rate_csv_bins(tmp_path):
    f = write(tmp_path / "a.csv", "lon,lat,rate_M4.0_4.1,rate_M4.1_4.2\n0,0,1,2\n1,0,3,4\n0,1,5,6\n1,1,7,8\n")
    assert read_rate_csv(f).values.tolist() == [[3, 7], [11, 15]]
    assert read_rate_csv(f, mag_min=4.1).values.tolist() == [[2, 4], [6, 8]]
    with pytest.raises(ValueError, match="rate column"):
        read_rate_csv(f, mag_min=9)
    with pytest.raises(ValueError, match="rate_M"):
        read_rate_csv(write(tmp_path / "b.csv", "lon,lat\n0,0\n"))


# rasters

def test_geotiff_fixture(data):
    pytest.importorskip("rasterio")
    g = read_geotiff(data / "chile/slab_depth.tif")
    assert g.crs == "EPSG:4326" and g.values.ndim == 2 and (np.diff(g.y_edges) > 0).all()
    assert np.nanmax(g.values) <= 0


def test_geotiff_nodata_and_flip(tmp_path):
    rio = pytest.importorskip("rasterio")
    from rasterio.transform import from_bounds

    f = tmp_path / "a.tif"
    z = np.array([[1.0, -9999], [3, 4]], "float32")
    with rio.open(f, "w", driver="GTiff", height=2, width=2, count=1, dtype="float32", crs="EPSG:4326",
                  nodata=-9999, transform=from_bounds(0, 0, 2, 2, 2, 2)) as d:
        d.write(z, 1)
    g = read_geotiff(f)
    assert g.values[0].tolist() == [3, 4] and g.mask.tolist() == [[False, False], [False, True]]


@pytest.mark.parametrize("bands", [3, 4])
def test_geotiff_image_uint8(tmp_path, bands):
    rio = pytest.importorskip("rasterio")
    from rasterio.transform import from_bounds

    f = tmp_path / "a.tif"
    with rio.open(f, "w", driver="GTiff", height=4, width=6, count=bands, dtype="uint8", crs="EPSG:3857",
                  transform=from_bounds(0, 0, 60, 40, 6, 4)) as d:
        d.write(np.ones((bands, 4, 6), "uint8"))
    g = read_geotiff(f)
    assert g.values.shape == (4, 6, bands) and g.values.dtype == np.uint8 and g.crs == "EPSG:3857"


def test_geotiff_window_and_decimation(tmp_path):
    rio = pytest.importorskip("rasterio")
    from rasterio.transform import from_bounds

    f = tmp_path / "a.tif"
    with rio.open(f, "w", driver="GTiff", height=100, width=100, count=1, dtype="float32", crs="EPSG:4326",
                  transform=from_bounds(0, 0, 10, 10, 100, 100)) as d:
        d.write(np.ones((1, 100, 100), "float32"))
    assert read_geotiff(f, extent=[2, 4, 3, 5]).values.shape == (20, 20)
    g = read_geotiff(f, max_px=10)
    assert g.values.shape == (10, 10) and g.x_edges[[0, -1]].tolist() == [0, 10]


def test_geotiff_missing(tmp_path, monkeypatch):
    pytest.importorskip("rasterio")
    with pytest.raises(FileNotFoundError):
        read_geotiff(tmp_path / "x.tif")
    monkeypatch.setitem(sys.modules, "rasterio", None)
    with pytest.raises(MissingDependency, match="raster"):
        read_geotiff(tmp_path / "x.tif")


def slab_file(tmp_path, name):
    return write(tmp_path / name, "280.0,-30.0,-10.0\n280.5,-30.0,-20.0\n280.0,-29.5,NaN\n280.5,-29.5,-40.0\n")


def test_slab2_dep(tmp_path):
    g = read_slab2(slab_file(tmp_path, "sam_slab2_dep_02.23.18.xyz"))
    assert g.x_edges[0] == pytest.approx(-80.25)
    assert g.values[0].tolist() == [10, 20] and g.mask.tolist() == [[False, False], [True, False]]


def test_slab2_other_keeps_sign(tmp_path):
    g = read_slab2(slab_file(tmp_path, "sam_slab2_thk_02.23.18.xyz"))
    assert g.values[0, 0] == -10


def test_slab2_crossing(tmp_path):
    f = write(tmp_path / "ker_slab2_dep_x.xyz", "179.5,0,-1\n180.0,0,-2\n180.5,0,-3\n179.5,1,-1\n180.0,1,-1\n180.5,1,-1\n")
    assert np.ptp(read_slab2(f).x_edges) == pytest.approx(1.5)


def test_slab2_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_slab2(tmp_path / "x.xyz")


# vectors

def test_read_shapefile(data):
    g, a, crs = read_vector(data / "chile/sam_nazca_trench.shp")
    assert crs == "EPSG:4326" and len(g) == len(a) and g[0].geom_type.endswith("LineString")


def test_read_shapefile_projected(data):
    g, a, crs = read_vector(data / "new_zealand/nz_cfm.shp")
    assert crs == "EPSG:2193" and "SR_pref" in a and g[0].bounds[0] > 1e6


def test_read_polygons(data):
    g, _, _ = read_vector(data / "chile/chile_buffer.shp")
    assert g[0].geom_type.endswith("Polygon")


def test_read_geojson(data):
    g, a, crs = read_vector(data / "chile/sara_faults.geojson")
    assert crs == "EPSG:4326" and len(g) == len(a)


def test_read_geojson_single_feature(tmp_path):
    f = write(tmp_path / "a.json", json.dumps({"type": "Feature", "properties": {"a": 1},
                                               "geometry": {"type": "Point", "coordinates": [1, 2]}}))
    g, a, _ = read_vector(f)
    assert g[0].x == 1 and a["a"][0] == 1


def test_gpkg_needs_pyogrio(tmp_path, monkeypatch):
    f = write(tmp_path / "a.gpkg", "x")
    monkeypatch.setitem(sys.modules, "pyogrio", None)
    with pytest.raises(MissingDependency, match="vector"):
        read_vector(f)
    with pytest.raises(FileNotFoundError):
        read_vector(tmp_path / "x.shp")


def test_gpkg_roundtrip(tmp_path):
    p = pytest.importorskip("pyogrio")
    from shapely import to_wkb
    from shapely.geometry import LineString

    f = tmp_path / "a.gpkg"
    p.raw.write(f, geometry=np.array([to_wkb(LineString([(0, 0), (1, 1)]))], object), field_data=[np.array([1])],
                fields=["a"], driver="GPKG", crs="EPSG:4326", geometry_type="LineString")
    g, a, crs = read_vector(f)
    assert crs == "EPSG:4326" and a["a"][0] == 1 and g[0].length > 1


def test_outline_holes_and_parts():
    m = np.zeros((3, 3), bool)
    m[1, 1] = True
    o = cells_to_outline(Grid(np.ones((3, 3)), np.arange(4.0), np.arange(4.0), mask=m))
    assert o.geom_type == "Polygon" and len(o.interiors) == 1 and o.area == 8
    m = np.zeros((1, 3), bool)
    m[0, 1] = True
    o = cells_to_outline(Grid(np.ones((1, 3)), np.arange(4.0), [0, 1], mask=m))
    assert o.geom_type == "MultiPolygon" and len(o.geoms) == 2


def test_outline_nodes(data):
    g = read_csep_nodes(data / "new_zealand/nz.testing.nodes.dat")
    o = cells_to_outline(g)
    assert o.is_valid and o.area == pytest.approx((~g.mask).sum() * 0.01)
