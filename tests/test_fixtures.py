import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest
import shapefile

from pituplots import Grid

MAX_BYTES = 5e6


def test_manifest_matches_files(data):
    man = json.loads((data / "manifest.json").read_text())["files"]
    found = {p.relative_to(data).as_posix() for p in data.rglob("*")
             if p.is_file() and p.parent != data and "__pycache__" not in p.parts}
    assert found == set(man)
    for k, v in man.items():
        assert hashlib.sha256((data / k).read_bytes()).hexdigest() == v, k


def test_total_size(data):
    assert sum(p.stat().st_size for p in data.rglob("*") if p.is_file()) < MAX_BYTES


def test_real_lattices_snap(data):
    p = pd.read_csv(data / "lacq/pattern.csv")
    assert Grid.from_points(p.lon, p.lat, p.p).values.shape == (25, 25)

    x, y = np.loadtxt(data / "new_zealand/nz.testing.nodes.dat").T
    g = Grid.from_points(x, y, x * np.nan, dh=0.1, anchor="origin")
    assert g.mask is None or g.mask.sum() < g.mask.size

    h = pd.read_csv(data / "chile/hazard_475_chile.csv")
    g = Grid.from_points(h.lon_center, h.lat_center, h.PGA)
    assert np.diff(g.x_edges) == pytest.approx(1 / 120, abs=1e-5)


def test_nz_catalog_crosses_180(data):
    c = pd.read_csv(data / "new_zealand/cat_nz.csv")
    assert c.lon.max() > 180 > c.lon.min()


def test_capped_grid(data):
    c = pd.read_csv(data / "chile/grid_capped.csv")
    assert {"lon", "lat", "in_buffer"} <= set(c.columns) and c.in_buffer.nunique() == 2


@pytest.mark.parametrize("f, n", [
    ("chile/sam_nazca_trench.shp", 1),
    ("chile/chile_buffer.shp", 1),
    ("chile/crustal_faults.shp", 20),
    ("new_zealand/nz_testing.shp", 1),
    ("new_zealand/nz_cfm.shp", 44),
])
def test_shapefiles(data, f, n):
    with shapefile.Reader(str(data / f)) as r:
        assert len(r) == n
    assert (data / f).with_suffix(".prj").exists()


@pytest.mark.parametrize("f, tag, n", [
    ("nrml/simple.xml", "simpleFaultSource", 5),
    ("nrml/complex.xml", "complexFaultSource", 1),
])
def test_nrml(data, f, tag, n):
    root = ET.parse(data / f).getroot()
    assert len(root.findall(f".//{{http://openquake.org/xmlns/nrml/0.4}}{tag}")) == n


def test_datastore(data):
    h5py = pytest.importorskip("h5py", reason="needs the [hdf5] extra")
    with h5py.File(data / "lacq/calc_11_subset.hdf5") as f:
        n = f["sitecol/lon"].size
        assert f["hcurves-stats"].shape == (n, 1, 1, 25) == f["hcurves-rlzs"].shape
        assert f"\"site_id\": {n}" in f["hcurves-stats"].attrs["json"]
        assert json.loads(f["oqparam"][()])["hazard_imtls"]
        assert f.attrs["engine_version"] == "3.25.1"


@pytest.mark.parametrize("f, crs, bands", [
    ("lacq/hmaps_PGA_0.002105.tiff", "EPSG:4326", 1),
    ("chile/hazard_475_PGA.tif", "EPSG:4326", 1),
    ("chile/slab_depth.tif", "EPSG:4326", 1),
    ("basemaps/california_4326_rgba.tif", "EPSG:4326", 4),
])
def test_rasters(data, f, crs, bands):
    rasterio = pytest.importorskip("rasterio", reason="needs the [raster] extra")
    with rasterio.open(data / f) as r:
        assert r.crs.to_string() == crs and r.count == bands
        assert r.read(1).size > 0
