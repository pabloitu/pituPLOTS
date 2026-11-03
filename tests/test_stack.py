"""Library behaviour the package relies on, to confirm the version floors."""
import warnings

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from cartopy.img_transform import warp_array
from matplotlib.collections import Collection
from matplotlib.path import Path
from packaging.version import Version
from shapely.geometry import box


def test_floors():
    assert Version(matplotlib.__version__) >= Version("3.8")
    assert Version(cartopy.__version__) >= Version("0.23")


def test_contourset_is_collection():
    _, ax = plt.subplots()
    assert isinstance(ax.contourf(np.arange(16.0).reshape(4, 4)), Collection)


def test_inset_colorbar_constrained_tight(tmp_path):
    fig, ax = plt.subplots(layout="constrained")
    m = ax.pcolormesh(np.arange(4.0).reshape(2, 2))
    fig.colorbar(m, cax=ax.inset_axes([1.02, 0, 0.03, 1]))
    fig.savefig(tmp_path / "a.png", bbox_inches="tight")


def test_clip_path_on_geoaxes():
    ax = plt.axes(projection=ccrs.PlateCarree())
    x, y = np.meshgrid(np.arange(5.0), np.arange(5.0))
    clip = Path(np.array(box(1, 1, 3, 3).exterior.coords))
    for m in (ax.pcolormesh(x, y, x[:-1, :-1], transform=ccrs.PlateCarree()),
              ax.contourf(x, y, x, transform=ccrs.PlateCarree())):
        m.set_clip_path(clip, transform=ax.transData)
    ax.figure.canvas.draw()


@pytest.mark.parametrize("code", [2193, 3857])
def test_epsg_offline(code):
    assert ccrs.epsg(code).proj4_init


def test_rgba_warp_needs_no_network():
    z = np.random.default_rng(0).integers(0, 255, (20, 20, 4), dtype=np.uint8)
    out, _ = warp_array(z, target_proj=ccrs.PlateCarree(180), source_proj=ccrs.PlateCarree(),
                        target_res=(10, 10), source_extent=(170, 190, -50, -30),
                        target_extent=(-10, 10, -50, -30))
    assert out.shape[:2] == (10, 10)


def test_set_extent_needs_continuous_longitudes():
    pc = ccrs.PlateCarree()
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    ax.set_extent([170, 190, -10, 10], pc)
    assert ax.get_extent() == pytest.approx((-10, 10, -10, 10))
    ax.set_extent([170, -170, -10, 10], pc)
    assert ax.get_extent()[:2] == pytest.approx((-180, 180), abs=1e-2)


@pytest.mark.parametrize("res", ["110m", "50m", "10m"])
@pytest.mark.parametrize("f", [cfeature.COASTLINE, cfeature.BORDERS])
def test_natural_earth_offline(f, res):
    with warnings.catch_warnings():
        warnings.simplefilter("error", cartopy.io.DownloadWarning)
        assert list(f.with_scale(res).geometries())


@pytest.mark.parametrize("res", ["110m", "50m"])
@pytest.mark.parametrize("f", [cfeature.LAND, cfeature.OCEAN])
def test_natural_earth_polygons_offline(f, res):
    with warnings.catch_warnings():
        warnings.simplefilter("error", cartopy.io.DownloadWarning)
        assert list(f.with_scale(res).geometries())


@pytest.mark.mpl_image_compare(tolerance=20, style="default", remove_text=True)
def test_baseline_pipeline():
    fig = plt.figure(figsize=(4, 3), dpi=80)
    ax = fig.add_subplot(projection=ccrs.PlateCarree())
    ax.set_extent([-12, 32, 34, 60], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="whitesmoke")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5)
    return fig
