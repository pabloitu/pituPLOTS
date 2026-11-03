import io
from pathlib import Path

import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

import pituplots.basemaps as bm
from pituplots import MissingDependency, PituplotsWarning, add_basemap, get_basemap, make_map, register_basemap


def write_tif(path, bands, crs, bounds, dtype="uint8", size=(64, 64), nodata=None):
    rio = pytest.importorskip("rasterio")
    from rasterio.transform import from_bounds

    h, w = size
    with rio.open(path, "w", driver="GTiff", height=h, width=w, count=bands, dtype=dtype, crs=crs,
                  nodata=nodata, transform=from_bounds(*bounds, w, h)) as d:
        for b in range(1, bands + 1):
            d.write((np.full((h, w), 200) if dtype == "uint8" else np.linspace(0, 1, h * w).reshape(h, w)).astype(dtype), b)
    return str(path)


BOUNDS = {"EPSG:4326": (170, -45, 176, -38), "EPSG:3857": (18.9e6, -5.6e6, 19.6e6, -4.6e6),
          "EPSG:2193": (1.3e6, 5.0e6, 1.8e6, 5.8e6)}


@pytest.fixture
def reg():
    yield
    bm.REGISTRY.clear()


def test_get_basemap_tiles():
    for n in ("ESRI_terrain", "ESRI_imagery", "ESRI_relief", "ESRI_topo", "google-satellite"):
        assert isinstance(get_basemap(n), bm.Tiles)
    t = get_basemap("https://a.b/{z}/{x}/{y}.png")
    assert t.url == "https://a.b/{z}/{x}/{y}.png"
    assert get_basemap("stock_img") == "stock_img"


def test_cache_per_source():
    a = get_basemap("https://a.b/{z}/{x}/{y}.png")
    b = get_basemap("https://c.d/{z}/{x}/{y}.png")
    assert a.folder != b.folder and a.folder.parent == b.folder.parent
    assert a.folder == get_basemap("https://a.b/{z}/{x}/{y}.png").folder
    assert a.folder.parent == Path(cartopy.config["cache_dir"]) / "pituplots"


def test_bad_sources(tmp_path):
    with pytest.raises(ValueError, match="unknown basemap"):
        get_basemap("nope")
    with pytest.raises(FileNotFoundError):
        get_basemap(str(tmp_path / "x.tif"))
    with pytest.raises(FileNotFoundError):
        get_basemap(tmp_path / "x.png")
    with pytest.raises(TypeError):
        get_basemap(3)


def test_register(reg):
    register_basemap("mine", "https://a.b/{z}/{x}/{y}.png")
    assert get_basemap("mine").url == "https://a.b/{z}/{x}/{y}.png"
    with pytest.raises(ValueError, match="built-in"):
        register_basemap("ESRI_relief", "x")


def test_needs_geoaxes():
    _, ax = plt.subplots()
    with pytest.raises(TypeError):
        add_basemap(ax, "stock_img")


def test_stock_img_is_lowest():
    ax = make_map(extent=[0, 10, 0, 10], basemap="stock_img", coastline=False)
    ax.figure.canvas.draw()
    assert ax.images[0].get_zorder() == 0


def png(color):
    b = io.BytesIO()
    Image.fromarray(np.full((256, 256, 3), color, np.uint8)).save(b, "PNG")
    return b.getvalue()


class Fake:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def read(self):
        return self.data


def test_tiles_mocked_and_cached(monkeypatch):
    calls = []

    def fake(req):
        calls.append(req.full_url)
        return Fake(png(200))

    monkeypatch.setattr(bm, "urlopen", fake)
    url = "https://mock.test/{z}/{x}/{y}.png"
    ax = make_map(extent=[0, 10, 40, 50], basemap=url, coastline=False, tile_depth=5)
    ax.figure.canvas.draw()
    n = len(calls)
    assert n and len(ax.images) >= 1
    assert ax.images[0].get_zorder() == 0
    ax = make_map(extent=[0, 10, 40, 50], basemap=url, coastline=False, tile_depth=5)
    ax.figure.canvas.draw()
    assert len(calls) == n


def test_tiles_no_network_warns(capsys):
    url = "https://nowhere.test/{z}/{x}/{y}.png"
    ax = make_map(extent=[0, 10, 40, 50], basemap=url, coastline=False, tile_depth=5)
    with pytest.warns(PituplotsWarning, match="tiles"):
        ax.figure.canvas.draw()
    assert capsys.readouterr().out == ""
    assert not get_basemap(url).folder.exists()


def test_tile_depth_auto(monkeypatch):
    seen = []
    monkeypatch.setattr(bm, "urlopen", lambda r: seen.append(r.full_url) or Fake(png(100)))
    ax = make_map(extent=[-5, 5, 40, 44], coastline=False, basemap="https://m.test/{z}/{x}/{y}.png")
    assert ax.img_factories[0][1][0] == 8
    ax = make_map(extent=[-70.9, -70.4, -33.7, -33.2], coastline=False, figsize=(8, 8),
                  basemap="https://m.test/{z}/{x}/{y}.png")
    assert ax.img_factories[0][1][0] == 12


def test_missing_rasterio(monkeypatch, tmp_path):
    f = tmp_path / "a.tif"
    f.write_bytes(b"x")
    monkeypatch.setitem(__import__("sys").modules, "rasterio", None)
    with pytest.raises(MissingDependency, match=r"pituplots\[raster\]"):
        get_basemap(str(f))


# B3 base

@pytest.mark.parametrize("crs", BOUNDS)
@pytest.mark.parametrize("bands", [1, 3, 4])
def test_geotiff_crs_and_bands(tmp_path, crs, bands):
    f = write_tif(tmp_path / "a.tif", bands, crs, BOUNDS[crs])
    ax = make_map(extent=[171, 175, -43, -40], basemap=f, coastline=False)
    ax.figure.canvas.draw()
    assert len(ax.images) == 1 and ax.images[0].get_zorder() == 0
    assert ax.get_extent() == pytest.approx((171, 175, -43, -40), abs=1e-6)
    a = np.asarray(ax.figure.canvas.buffer_rgba())
    assert (a[..., :3] < 230).any()


@pytest.mark.parametrize("crs", ["EPSG:4326", "EPSG:3857", "EPSG:2193"])
def test_native_projection(tmp_path, crs):
    f = write_tif(tmp_path / "a.tif", 4, crs, BOUNDS[crs])
    ax = make_map(extent=[171, 175, -43, -40], basemap=f, projection="native", coastline=False)
    ax.figure.canvas.draw()
    if crs != "EPSG:4326":
        assert ax.projection.to_epsg() == int(crs[5:])


def test_float_rgb_scaled(tmp_path):
    f = write_tif(tmp_path / "a.tif", 3, "EPSG:4326", BOUNDS["EPSG:4326"], dtype="float32")
    ax = make_map(extent=[171, 175, -43, -40], basemap=f, coastline=False)
    z = ax.images[0].get_array()
    assert z.min() >= 0 and z.max() <= 1


def test_large_raster_is_decimated(tmp_path, monkeypatch):
    f = write_tif(tmp_path / "a.tif", 3, "EPSG:4326", (170, -45, 176, -38), size=(3000, 3000))
    ax = make_map(extent=[170.5, 175.5, -44, -39], basemap=f, coastline=False, figsize=(3, 3))
    z = ax.images[0].get_array()
    assert max(z.shape[:2]) <= 2 * 300 * 1.2


def test_only_window_is_read(tmp_path):
    f = write_tif(tmp_path / "a.tif", 3, "EPSG:4326", (170, -45, 176, -38), size=(600, 600))
    ax = make_map(extent=[171, 172, -42, -41], basemap=f, coastline=False, figsize=(8, 8))
    e = ax.images[0].get_extent()
    assert e[1] - e[0] < 1.1 and e[3] - e[2] < 1.1


def test_extent_partly_outside(tmp_path):
    f = write_tif(tmp_path / "a.tif", 3, "EPSG:4326", (170, -45, 176, -38))
    ax = make_map(extent=[174, 178, -40, -36], basemap=f, coastline=False)
    ax.figure.canvas.draw()
    assert ax.get_extent() == pytest.approx((174, 178, -40, -36), abs=1e-6)


def test_extent_fully_outside(tmp_path):
    f = write_tif(tmp_path / "a.tif", 3, "EPSG:4326", (170, -45, 176, -38))
    ax = make_map(extent=[0, 5, 40, 45], coastline=False)
    assert add_basemap(ax, f) is None


def test_approx_aspect_survives(tmp_path):
    f = write_tif(tmp_path / "a.tif", 3, "EPSG:4326", (170, -45, 176, -38))
    ax = make_map(extent=[171, 175, -43, -40], basemap=f, projection="approx", coastline=False)
    assert ax.get_aspect() != "equal" and ax.get_aspect() != 1


def test_crossing_raster(tmp_path):
    f = write_tif(tmp_path / "a.tif", 3, "EPSG:4326", (175, -45, 185, -38))
    ax = make_map(extent=[176, -176, -44, -40], basemap=f, coastline=False)
    ax.figure.canvas.draw()
    a = np.asarray(ax.figure.canvas.buffer_rgba())[..., :3]
    assert (a < 230).any()


def test_open_dataset_stays_open(tmp_path):
    rio = pytest.importorskip("rasterio")
    f = write_tif(tmp_path / "a.tif", 3, "EPSG:4326", (170, -45, 176, -38))
    with rio.open(f) as ds:
        ax = make_map(extent=[171, 175, -43, -40], coastline=False)
        add_basemap(ax, ds)
        assert not ds.closed


def test_california_fixture(data):
    pytest.importorskip("rasterio")
    ax = make_map(extent=[-122, -116, 33, 38], basemap=data / "basemaps/california_4326_rgba.tif")
    ax.figure.canvas.draw()
    assert ax.images
