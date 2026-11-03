from functools import partial

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from cartopy.mpl.geoaxes import GeoAxes

from pituplots import Catalog, highlight, make_map, plot_catalog, size_by_value
from pituplots.catalogs import as_catalog

KW = {"coastline": False, "gridlines": False}


def cat(n=5):
    x = np.linspace(0, 4, n)
    return Catalog(x, x + 40, x * 2, 3 + x / 2, extra={"cls": np.array(list("ababa"))[:n], "ll": x * 0.1})


def test_size_range():
    s = size_by_value([1, 2, 3], size=6, max_size=100, power=1)
    assert s.tolist() == [6, 53, 100]


def test_size_power_zero():
    assert size_by_value([1, 2], power=0).tolist() == [6, 6]


def test_size_same_values_no_division():
    assert size_by_value([4, 4, 4]).tolist() == [6, 6, 6]


def test_size_zero_and_negative_magnitudes():
    s = size_by_value([-1, 0, 1], power=1, size=0, max_size=2)
    assert s.tolist() == [0, 1, 2]
    s = size_by_value([0.0, 1.0], min_val=0, max_val=1, power=1, size=0, max_size=1)
    assert s.tolist() == [0, 1]


def test_size_ticks_below_min_clip():
    s = size_by_value([1, 5], min_val=3, max_val=6, power=0.5, size=2, max_size=10)
    assert np.isfinite(s).all() and s[0] == 2


def test_size_nan_and_empty():
    assert size_by_value([np.nan, 1, 2], power=1, size=0, max_size=1).tolist() == [0, 0, 1]
    assert size_by_value([]).size == 0


def test_inputs():
    df = pd.DataFrame({"Longitude": [1.0], "LAT": [2.0], "Mw": [3.0], "z": [4.0], "o": ["k"]})
    c = as_catalog(df)
    assert (c.lon[0], c.lat[0], c.mag[0], c.depth[0]) == (1, 2, 3, 4) and c.extra["o"][0] == "k"
    c = as_catalog(([1], [2], [3]))
    assert c.mag[0] == 3 and c.depth is None
    c = as_catalog(([1], [2], [3], [4]))
    assert c.depth[0] == 3 and c.mag[0] == 4
    assert as_catalog(([1], [2])).mag is None
    with pytest.raises(TypeError):
        as_catalog("x")
    with pytest.raises(ValueError, match="column"):
        as_catalog(pd.DataFrame({"a": [1]}))


def test_new_and_given_ax():
    ax = plot_catalog(cat(), **KW)
    assert isinstance(ax, GeoAxes) and ax.collections
    a = plt.axes(projection=ccrs.PlateCarree())
    assert plot_catalog(cat(), ax=a) is a
    _, p = plt.subplots()
    assert plot_catalog(cat(), ax=p) is p
    assert p.collections[0].get_offsets()[:, 0].max() == 4


def test_map_keyword_with_ax_raises():
    a = plt.axes(projection=ccrs.PlateCarree())
    with pytest.raises(ValueError, match="basemap"):
        plot_catalog(cat(), ax=a, basemap="stock_img")


def test_input_not_modified():
    c = cat()
    lon = c.lon.copy()
    plot_catalog(c, **KW)
    assert c.lon.tolist() == lon.tolist()


def test_defaults_gold():
    ax = plot_catalog(cat(), **KW)
    c = ax.collections[-1]
    assert np.allclose(c.get_facecolor()[0][:3], plt.matplotlib.colors.to_rgb("gold"))
    assert c.get_alpha() == 0.4 and c.get_zorder() == 6


def test_kwargs_override_and_pass():
    ax = plot_catalog(cat(), color="red", alpha=1, marker="s", linewidths=2, **KW)
    assert ax.collections[-1].get_linewidth()[0] == 2


def test_bad_kwarg_raises():
    with pytest.raises(AttributeError):
        plot_catalog(cat(), colormap="x", **KW)


def test_color_by_numeric():
    ax = plot_catalog(cat(), color_by="ll", cmap="viridis", vmin=0, vmax=1, clabel="LL", **KW)
    m = ax.collections[-1]
    assert m.get_clim() == (0, 1)
    assert ax.child_axes


def test_color_by_array_and_no_colorbar():
    c = cat()
    ax = plot_catalog(c, color_by=c.mag, colorbar=False, **KW)
    assert not ax.child_axes and ax.collections[-1].get_array() is not None
    with pytest.raises(ValueError, match="shape"):
        plot_catalog(c, color_by=[1, 2], **KW)
    with pytest.raises(ValueError, match="column"):
        plot_catalog(c, color_by="nope", **KW)


def test_color_by_categorical():
    ax = plot_catalog(cat(), color_by="cls", colors={"a": "red"}, **KW)
    t = [x.get_text() for x in ax.get_legend().get_texts()]
    assert t == ["a", "b"]


def test_legends_coexist():
    ax = plot_catalog(cat(), color_by="cls", mag_ticks=[3, 4], **KW)
    assert ax.get_legend() is not None
    assert any(isinstance(a, plt.matplotlib.legend.Legend) for a in ax.artists)


def test_size_legend_once():
    ax = plot_catalog(cat(), **KW)
    assert ax.get_legend() is None
    assert sum(isinstance(a, plt.matplotlib.legend.Legend) for a in ax.artists) == 1


def test_legend_off():
    ax = plot_catalog(cat(), legend=False, **KW)
    assert not ax.artists


def test_legend_title_and_ticks():
    ax = plot_catalog(cat(), mag_ticks=[1.5, 2.5, 3.5, 4], legend_title="M", **KW)
    lg = ax.artists[0]
    assert lg.get_title().get_text() == "M" and len(lg.get_texts()) == 4
    h = lg.legend_handles if hasattr(lg, "legend_handles") else lg.legendHandles
    assert np.isfinite([x.get_markersize() for x in h]).all()
    ax = plot_catalog(cat(), mag_ticks=2, **KW)
    assert len(ax.artists[0].get_texts()) == 2


def test_shared_sizer():
    s = partial(size_by_value, min_val=3, max_val=8)
    ax = plot_catalog(cat(), sizer=s, **KW)
    plot_catalog(cat(), ax=ax, sizer=s, color="blue")
    assert ax.collections[-1].get_sizes().tolist() == ax.collections[-2].get_sizes().tolist()


def test_title():
    assert plot_catalog(cat(), title="T", **KW).get_title() == "T"


# B3 cat

def test_empty_catalog():
    ax = plot_catalog(Catalog([], []), set_global=True, **KW)
    ax.figure.canvas.draw()
    ax = plot_catalog(Catalog([], [], mag=[]), extent=[0, 1, 0, 1], **KW)
    assert not ax.artists
    ax = plot_catalog(pd.DataFrame({"lon": [], "lat": [], "mag": []}), **KW)
    ax.figure.canvas.draw()


def test_one_event():
    ax = plot_catalog(Catalog([1], [2], mag=[3]), **KW)
    ax.figure.canvas.draw()


def test_same_magnitude():
    ax = plot_catalog(Catalog([1, 2], [2, 3], mag=[3, 3]), **KW)
    assert len(set(ax.collections[-1].get_sizes())) == 1
    ax.figure.canvas.draw()


def test_zero_and_negative_magnitude():
    ax = plot_catalog(Catalog([1, 2, 3], [2, 3, 4], mag=[-1, 0, 2]), **KW)
    assert np.isfinite(ax.collections[-1].get_sizes()).all()
    ax.figure.canvas.draw()


def test_events_outside_extent_silent():
    ax = plot_catalog(Catalog([1, 50], [2, 60], mag=[3, 4]), extent=[0, 5, 0, 5], **KW)
    ax.figure.canvas.draw()
    assert ax.get_extent() == pytest.approx((0, 5, 0, 5))


def test_unsorted_input_same_picture():
    c = cat()
    i = [3, 1, 4, 0, 2]
    a = plot_catalog(c, **KW)
    b = plot_catalog(Catalog(c.lon[i], c.lat[i], mag=c.mag[i]), **KW)
    assert (a.collections[-1].get_sizes() == b.collections[-1].get_sizes()).all()
    assert (a.collections[-1].get_offsets() == b.collections[-1].get_offsets()).all()


def test_crossing_catalog_nz(data):
    d = pd.read_csv(data / "new_zealand/cat_nz.csv")
    ax = plot_catalog(d, **KW)
    assert ax.projection.proj4_params.get("lon_0", ax.projection.proj4_params.get("pm")) == 180
    x = ax.collections[-1].get_offsets()[:, 0]
    assert np.ptp(x) < 25
    ax.figure.canvas.draw()


def test_highlight():
    c = cat()
    ax = plot_catalog(c, **KW)
    assert highlight(c, ax, where=c.mag > 4.6, labels=["a"]) is ax
    assert ax.collections[-1].get_offsets().shape[0] == 1
    assert ax.texts[0].get_text() == "a"
    ax.figure.canvas.draw()
    with pytest.raises(ValueError, match="where"):
        highlight(c, ax, where=[True])


def test_highlight_plain_axes_and_sizer():
    c = cat()
    _, a = plt.subplots()
    highlight(c, a, sizer=size_by_value, labels=list("abcde"), scale=2)
    assert a.collections[-1].get_sizes().max() == 440


def test_highlight_labels_take_the_ring_colour():
    ax = make_map(extent=[0, 2, 0, 2], coastline=False, gridlines=False)
    highlight(([1.0], [1.0], [3.0]), ax, labels=["x"], edgecolor="blue")
    assert ax.texts[-1].get_color() == "blue"
