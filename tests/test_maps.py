import warnings

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pytest
from cartopy.mpl.geoaxes import GeoAxes

from pituplots import Grid, PituplotsWarning, add_gridlines, make_map
from pituplots.maps import central_lon, clip_path, get_ax, native, set_map_extent
from shapely.geometry import box


def test_returns_geoaxes_on_new_figure():
    ax = make_map(extent=[0, 10, 40, 50])
    assert isinstance(ax, GeoAxes)
    assert ax.get_extent() == pytest.approx((0, 10, 40, 50))


def test_no_extent_and_figsize():
    ax = make_map(figsize=(3, 2))
    assert tuple(ax.figure.get_size_inches()) == (3, 2)


def test_bad_projection():
    with pytest.raises(ValueError, match="projection"):
        make_map(projection="nope")
    with pytest.raises(ValueError, match="native"):
        make_map(projection="native")
    with pytest.raises(ValueError, match="approx"):
        make_map(projection="approx")


def test_bad_extent():
    with pytest.raises(ValueError, match="lat_min"):
        make_map(extent=[0, 1, 5, 2])


# B3 lon, at the make_map level

def test_crossing_extent_gives_central_180():
    ax = make_map(extent=[170, -170, -10, 10], coastline=False)
    assert central_lon(ax.projection) == 180
    assert ax.get_extent() == pytest.approx((-10, 10, -10, 10))


def test_noncrossing_keeps_central_0():
    ax = make_map(extent=[170, 180, -10, 10], coastline=False)
    assert central_lon(ax.projection) == 0
    assert ax.get_extent() == pytest.approx((170, 180, -10, 10))


def test_crossing_on_user_projection():
    ax = make_map(extent=[170, -170, -50, -30], projection=ccrs.Mercator(central_longitude=180),
                  coastline=False)
    a, b = ax.get_extent(ccrs.PlateCarree(central_longitude=180))[:2]
    assert (a, b) == pytest.approx((-10, 10), abs=0.01)


def test_crossing_on_seamed_projection_raises():
    with pytest.raises(ValueError, match="seam"):
        make_map(extent=[170, -170, -50, -30], projection=ccrs.Mercator(), coastline=False)


def test_set_global():
    ax = make_map(set_global=True, coastline=False)
    assert ax.get_extent() == pytest.approx((-180, 180, -90, 90))
    ax = make_map(extent=[-180, 180, -90, 90], coastline=False)
    assert ax.get_extent() == pytest.approx((-180, 180, -90, 90))


def test_set_extent_on_given_ax():
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    set_map_extent(ax, [175, -175, 0, 5])
    assert ax.get_extent() == pytest.approx((-5, 5, 0, 5))
    _, a = plt.subplots()
    set_map_extent(a, [1, 2, 3, 4])
    assert a.get_xlim() == (1, 2) and a.get_ylim() == (3, 4)


# B3 proj

PROJ = [ccrs.PlateCarree(), ccrs.Mercator(), ccrs.PlateCarree(central_longitude=180),
        ccrs.LambertConformal(central_longitude=13, central_latitude=42),
        ccrs.ObliqueMercator(central_longitude=13, central_latitude=42, azimuth=30),
        ccrs.epsg(3857), "approx"]


@pytest.mark.parametrize("proj", PROJ, ids=lambda p: str(type(p).__name__ if not isinstance(p, str) else p))
def test_projections_render(proj):
    ax = make_map(extent=[6, 19, 36, 48], projection=proj, gridlines=True)
    ax.figure.canvas.draw()


def test_approx_aspect():
    ax = make_map(extent=[0, 10, 60, 70], projection="approx", coastline=False)
    assert ax.get_aspect() == pytest.approx(110.574 / (111.32 * np.cos(np.deg2rad(65))))


def test_central_lon_epsg_projected():
    assert central_lon(ccrs.epsg(2193)) == 173


def test_nz_crossing_extent_on_nztm():
    ax = make_map(extent=[170, -175, -48, -34], projection=ccrs.epsg(2193), coastline=False)
    ax.figure.canvas.draw()


def test_gridlines_inline_and_labels():
    ax = make_map(extent=[6, 19, 36, 48], projection=ccrs.LambertConformal(13, 42), gridlines=False,
                  coastline=False)
    gl = add_gridlines(ax, inline=True, fontsize=7)
    assert gl.xlabel_style["fontsize"] == 7
    ax.figure.canvas.draw()
    ax = make_map(extent=[0, 1, 0, 1], gridlines=False, coastline=False)
    add_gridlines(ax, labels=False)
    ax.figure.canvas.draw()


def test_gridlines_need_geoaxes():
    _, ax = plt.subplots()
    with pytest.raises(TypeError):
        add_gridlines(ax)


def test_natural_earth_missing_warns(monkeypatch):
    from cartopy.io import shapereader

    def fail(*a, **k):
        raise OSError("no network")

    monkeypatch.setattr(shapereader, "natural_earth", fail)
    with pytest.warns(PituplotsWarning, match="coastline"):
        ax = make_map(extent=[0, 1, 40, 41])
    ax.figure.canvas.draw()


def test_coastline_is_offline_inside_regions():
    with warnings.catch_warnings():
        warnings.simplefilter("error", PituplotsWarning)
        for e in ([6, 19, 36, 48], [-75, -68, -40, -30], [172, 178, -44, -38]):
            make_map(extent=e, borders=True).figure.canvas.draw()


def test_get_ax_rules():
    kw = {"extent": [0, 1, 0, 1], "coastline": False, "color": "r"}
    ax = get_ax(Grid(np.zeros((1, 1)), [0, 1], [0, 1]), None, kw)
    assert kw == {"color": "r"} and isinstance(ax, GeoAxes)
    a = plt.axes(projection=ccrs.PlateCarree())
    assert get_ax(None, a, {"extent": [0, 1, 0, 1]}) is a
    with pytest.raises(ValueError, match="basemap"):
        get_ax(None, a, {"basemap": "stock_img"})


def test_native_branch():
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    x, kw = native(ax, [179, 181, -179])
    assert x == pytest.approx([-1, 1, 1]) and "transform" in kw
    _, a = plt.subplots()
    assert native(a, [190])[0].tolist() == [190] and native(a, [190])[1] == {}


def test_clip_path_holes():
    p = box(0, 0, 4, 4).difference(box(1, 1, 2, 2))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([0, 4, 0, 4], ccrs.PlateCarree())
    ax.set_axis_off()
    m = ax.pcolormesh([0, 4], [0, 4], [[1.0]], cmap="Reds", vmin=0, vmax=1, transform=ccrs.PlateCarree())
    m.set_clip_path(clip_path(ax, p), ax.transData)
    ax.figure.canvas.draw()
    a = np.asarray(ax.figure.canvas.buffer_rgba())[..., 1]
    x, y = ax.transData.transform([(1.5, 1.5), (3, 3)]).astype(int)
    h = a.shape[0]
    assert a[h - x[1], x[0]] > 200 and a[h - y[1], y[0]] < 100


def test_extent_outside_an_epsg_area_of_use_is_explained():
    with pytest.raises(ValueError, match="area of use"):
        make_map(extent=[164, 183, -48.5, -33], projection=ccrs.epsg(2193), coastline=False, gridlines=False)


def test_regional_crs_is_not_clipped_to_its_area_of_use():
    from pituplots.maps import cartopy_crs

    ax = make_map(extent=[164, 183, -48.5, -33], projection=cartopy_crs("EPSG:2193"), coastline=False,
                  gridlines=False)
    x0, x1, y0, y1 = ax.get_extent(ccrs.PlateCarree(central_longitude=180))
    assert x0 < -15 and x1 > 2 and y0 < -48 and y1 > -33.5
    assert cartopy_crs("EPSG:2193") == cartopy_crs("EPSG:2193")


def test_global_crs_keeps_cartopy_bounds():
    from pituplots.maps import cartopy_crs

    assert cartopy_crs("EPSG:3857").bounds == pytest.approx(ccrs.epsg(3857).bounds, rel=1e-6)


def test_coastline_and_borders_accept_a_scale_or_a_style(data):
    def coast(ax):
        return [c for c in ax._children if hasattr(c, "_kwargs")]

    ax = make_map(extent=[-12, 32, 34, 60], coastline={"lw": 0.3, "color": "red"}, borders="50m", gridlines=False)
    c, b = coast(ax)
    assert c._kwargs["linewidth"] == 0.3 and c._kwargs["edgecolor"] == "red"
    assert b._feature.scale == "50m"
    ax = make_map(extent=[-12, 32, 34, 60], coastline="110m", gridlines=False)
    assert coast(ax)[0]._feature.scale == "110m"
