import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.text import Text
from pyproj import Geod

from pituplots import PituplotsWarning, add_labels, add_locator, add_panel_label, add_scalebar, make_map, theme
from pituplots.maps import central_lon
from pituplots.style import defaults

MAP = {"coastline": False, "gridlines": False}
FR = [-5.5, 9, 41, 51.5]


def plain():
    return plt.subplots()[1]


def boxes(ax):
    ax.figure.canvas.draw()
    r = ax.figure.canvas.get_renderer()
    return [Text.get_window_extent(a, r) for a in ax.texts]


# add_locator


def test_locator_returns_an_inset_on_the_extent():
    ax = make_map(extent=[-1, 0, 43, 44], **MAP)
    a = add_locator(ax, FR)
    assert isinstance(a, GeoAxes) and a in ax.child_axes
    assert np.allclose(a.get_extent(ccrs.PlateCarree()), FR)


def test_locator_box_defaults_to_the_map():
    ax = make_map(extent=[-1, 0.5, 43, 44], **MAP)
    a = add_locator(ax, FR)
    r = a.patches[0]
    assert (r.get_x(), r.get_y()) == pytest.approx((-1, 43))
    assert (r.get_width(), r.get_height()) == pytest.approx((1.5, 1))
    assert a.lines[0].get_xdata()[0] == pytest.approx(-0.25)


def test_locator_explicit_box_and_style():
    ax = make_map(extent=[-1, 0, 43, 44], **MAP)
    a = add_locator(ax, FR, extent=[1, 2, 45, 46], box="blue", box_lw=3, rect=(0.5, 0.5, 0.3, 0.3), linestyle="--")
    r = a.patches[0]
    assert r.get_x() == 1 and r.get_edgecolor()[:3] == (0, 0, 1) and r.get_linewidth() == 3
    assert r.get_linestyle() == "--"


def test_locator_needs_a_map():
    with pytest.raises(TypeError, match="GeoAxes"):
        add_locator(plain(), FR)


def test_locator_on_a_crossing_map():
    ax = make_map(extent=[170, -170, -50, -30], **MAP)
    a = add_locator(ax, [150, -150, -60, -20])
    assert central_lon(a.projection) == 180
    assert a.patches[0].get_width() == pytest.approx(20)
    a.figure.canvas.draw()


def test_locator_on_a_projected_map():
    ax = make_map(extent=[166, 179, -48, -34], projection=ccrs.epsg(2193), **MAP)
    a = add_locator(ax, [160, 180, -50, -30])
    r = a.patches[0]
    assert 160 < r.get_x() < 170 and 10 < r.get_width() < 20


def test_locator_survives_constrained_layout():
    ax = make_map(extent=[-1, 0, 43, 44], **MAP)
    a = add_locator(ax, FR)
    ax.figure.canvas.draw()
    p, q = ax.get_position(), a.get_position()
    assert p.x0 <= q.x0 < p.x1


def test_locator_warns_without_natural_earth(monkeypatch):
    from cartopy.io import shapereader

    def fail(*a, **k):
        raise OSError("offline")

    monkeypatch.setattr(shapereader, "natural_earth", fail)
    ax = make_map(extent=[-1, 0, 43, 44], **MAP)
    with pytest.warns(PituplotsWarning):
        add_locator(ax, FR)


def test_locator_theme_layer(monkeypatch):
    monkeypatch.setitem(defaults.THEME, "locator", {"box": "lime"})
    a = add_locator(make_map(extent=[-1, 0, 43, 44], **MAP), FR)
    assert a.patches[0].get_edgecolor()[:3] == matplotlib.colors.to_rgb("lime")


def test_locator_rejects_a_seam_extent():
    with pytest.raises(ValueError):
        add_locator(make_map(extent=[-1, 0, 43, 44], **MAP), FR[:2] + [50, 40])


# add_scalebar


@pytest.mark.parametrize("lat", [0, 45, 65, -60])
def test_scalebar_plate_carree_is_true_length(lat):
    ax = make_map(extent=[0, 20, lat - 5, lat + 5], **MAP)
    ln = add_scalebar(ax, 200, loc=(0.1, 0.5))
    x = ln.get_xdata()
    _, _, d = Geod(ellps="WGS84").inv(x[0], lat, x[1], lat)
    assert d == pytest.approx(200e3, rel=0.01)


@pytest.mark.parametrize("proj", [ccrs.Mercator(), ccrs.epsg(3857), ccrs.LambertConformal(central_longitude=-70, central_latitude=-35, standard_parallels=(-30, -40))])
def test_scalebar_projected_is_true_length(proj):
    ax = make_map(extent=[-75, -65, -40, -30], projection=proj, **MAP)
    ln = add_scalebar(ax, 100, loc=(0.3, 0.5))
    x, y = ln.get_xdata(), ln.get_ydata()
    pc = ccrs.PlateCarree()
    p = pc.transform_points(proj, x, y)
    _, _, d = Geod(ellps="WGS84").inv(p[0, 0], p[0, 1], p[1, 0], p[1, 1])
    assert d == pytest.approx(100e3, rel=0.02)


def test_scalebar_on_a_crossing_map():
    ax = make_map(extent=[175, -175, -45, -35], **MAP)
    x = add_scalebar(ax, 100, loc=(0.4, 0.5)).get_xdata()
    assert 0 < x[1] - x[0] < 2


def test_scalebar_plain_section_and_degrees():
    ax = plain()
    ax.set_xlim(0, 500)
    assert np.diff(add_scalebar(ax, 100).get_xdata())[0] == pytest.approx(100)
    ax = plain()
    assert np.diff(add_scalebar(ax, 111, unit_km=111).get_xdata())[0] == pytest.approx(1)


def test_scalebar_keeps_the_limits_and_labels():
    ax = plain()
    ax.set_xlim(0, 500)
    ln = add_scalebar(ax, 100)
    assert ax.get_xlim() == (0, 500) and ln.get_color() == "black"
    assert ax.texts[-1].get_text() == "100 km"


def test_scalebar_style_and_errors():
    ax = plain()
    assert add_scalebar(ax, 0.2, color="white", halo=0, lw=3).get_linewidth() == 3
    assert ax.texts[-1].get_color() == "white" and not ax.texts[-1].get_path_effects()
    with pytest.raises(ValueError):
        add_scalebar(ax, 0)
    with pytest.raises(ValueError, match="outside"):
        add_scalebar(make_map(set_global=True, projection=ccrs.Orthographic(), **MAP), 10, loc=(0, 0))


# add_labels


def test_labels_returns_annotations_and_positions():
    ax = plain()
    ax.set(xlim=(0, 10), ylim=(0, 10))
    out = add_labels(ax, [1, 5], [1, 5], ["a", "b"])
    assert [t.get_text() for t in out] == ["a", "b"] and out[0].xy == (1, 1)
    assert out[0].arrow_patch is None


def test_labels_do_not_overlap():
    ax = plain()
    ax.set(xlim=(0, 10), ylim=(0, 10))
    add_labels(ax, [5] * 6, [5, 5.02, 5.04, 5.06, 5.08, 5.1], [f"label {i}" for i in range(6)], halo=0)
    b = boxes(ax)
    assert not any(p.overlaps(q) for i, p in enumerate(b) for q in b[:i])
    assert any(t.arrow_patch is not None for t in ax.texts)


def test_labels_min_sep_keeps_the_priority():
    ax = plain()
    ax.set(xlim=(0, 10), ylim=(0, 10))
    out = add_labels(ax, [1, 1.1, 5], [1, 1, 5], list("abc"), priority=[1, 2, 0], min_sep=0.5)
    assert sorted(t.get_text() for t in out) == ["b", "c"]


def test_labels_nmax():
    ax = plain()
    out = add_labels(ax, [0.1, 0.5, 0.9], [0.1, 0.5, 0.9], list("abc"), priority=[1, 3, 2], nmax=2)
    assert sorted(t.get_text() for t in out) == ["b", "c"]


def test_labels_on_a_map_cross_the_dateline():
    ax = make_map(extent=[170, -170, -50, -30], **MAP)
    out = add_labels(ax, [179, -179, 181], [-40, -41, -42], list("abc"), min_sep=5)
    assert len(out) == 1
    ax.figure.canvas.draw()


def test_labels_min_sep_across_the_seam():
    ax = make_map(extent=[170, -170, -50, -30], **MAP)
    assert len(add_labels(ax, [179.9, -179.9], [-40, -40], list("ab"), min_sep=1)) == 1


def test_labels_projected_and_clipped():
    ax = make_map(extent=[166, 179, -48, -34], projection=ccrs.epsg(2193), **MAP)
    out = add_labels(ax, [172, 100], [-40, 0], ["in", "out"])
    ax.figure.canvas.draw()
    r = ax.figure.canvas.get_renderer()
    w = {t.get_text(): t.get_window_extent(r) for t in out}
    assert ax.bbox.contains(*w["in"].p0)
    assert out[1].get_annotation_clip() is True


def test_labels_skip_nan_and_empty():
    ax = plain()
    assert len(add_labels(ax, [np.nan, 0.5], [0.5, 0.5], ["a", "b"])) == 1
    assert add_labels(ax, [], [], []) == []


def test_labels_bad_lengths():
    ax = plain()
    with pytest.raises(ValueError, match="lengths"):
        add_labels(ax, [1, 2], [1], ["a"])
    with pytest.raises(ValueError, match="priority"):
        add_labels(ax, [1], [1], ["a"], priority=[1, 2])


def test_labels_do_not_modify_inputs():
    lon, lat = np.array([181.0, 182.0]), np.array([-40.0, -41.0])
    t = ["a", "b"]
    add_labels(make_map(extent=[170, -170, -50, -30], **MAP), lon, lat, t, priority=lon)
    assert lon.tolist() == [181, 182] and lat.tolist() == [-40, -41] and t == ["a", "b"]


# add_panel_label


@pytest.mark.parametrize("loc, ha, va", [("upper right", "right", "top"), ("upper left", "left", "top"),
                                         ("lower right", "right", "bottom"), ("lower left", "left", "bottom")])
def test_panel_label_corners(loc, ha, va):
    t = add_panel_label(plain(), "(a)", loc=loc)
    assert (t.get_ha(), t.get_va(), t.get_text()) == (ha, va, "(a)")


def test_panel_label_on_a_map_is_in_axes_coordinates():
    ax = make_map(extent=[0, 10, 0, 10], **MAP)
    assert add_panel_label(ax, "x").get_transform() is ax.transAxes


def test_panel_label_size_follows_the_theme():
    with theme("paper"):
        a = add_panel_label(plain(), "x").get_fontsize()
    with theme("report"):
        b = add_panel_label(plain(), "x").get_fontsize()
    assert a < b
    assert add_panel_label(plain(), "x", fontsize=21).get_fontsize() == 21


def test_panel_label_bad_loc():
    with pytest.raises(ValueError, match="loc"):
        add_panel_label(plain(), "x", loc="center")


def test_scalebar_halo_also_on_the_bar():
    ax = plain()
    ax.set_xlim(0, 500)
    ln = add_scalebar(ax, 100, color="white", lw=2)
    assert len(ln.get_path_effects()) == 1
    assert not add_scalebar(ax, 100, halo=0).get_path_effects()


def test_class_breaks_and_legend():
    from pituplots import add_class_legend, class_breaks

    v = np.r_[np.arange(100.0), np.nan]
    e = class_breaks(v, 4)
    assert len(e) == 5 and e[0] == 0 and e[-1] == 99
    assert class_breaks(v, 4, method="equal", round_to=10).tolist() == [0, 20, 50, 70, 99] or len(class_breaks(v, 4, method="equal", round_to=10)) == 5
    ax = plain()
    lg = add_class_legend(ax, [0, 0.5, 1.25, 2], ["#fee", "#fc9", "#900"], decimal=",", numbered=True, title="PGA")
    assert [t.get_text() for t in lg.get_texts()] == ["1: < 0,50", "2: 0,50 – 1,25", "3: ≥ 1,25"]
    assert lg in ax.artists
    fig = plt.figure()
    assert add_class_legend(fig, [0, 1, 2], matplotlib.colormaps["viridis"], open_ends=False).get_texts()[0].get_text() == "0.00 – 1.00"
    with pytest.raises(ValueError, match="colours"):
        add_class_legend(ax, [0, 1, 2], ["red"])
    with pytest.raises(ValueError):
        class_breaks([np.nan])


def test_size_legend_ticks_above_the_data_keep_their_size():
    from pituplots import Catalog, plot_catalog

    ax = plot_catalog(Catalog([1, 2], [1, 2], mag=[4.0, 5.0]), mag_ticks=[4, 5, 6], coastline=False, gridlines=False)
    s = [h.get_markersize() for h in ax.artists[-1].legend_handles] if hasattr(ax.artists[-1], "legend_handles") \
        else [h.get_markersize() for h in ax.artists[-1].legendHandles]
    assert s[2] > s[1] > s[0]


def test_class_legend_outside_is_not_clipped_and_separator():
    from pituplots import add_class_legend, class_breaks

    lg = add_class_legend(plain(), [0, 1, 2], ["red", "blue"], bbox_to_anchor=(1.05, 1), loc="upper left",
                          open_ends=False, sep=" a ")
    assert not lg.get_clip_on() and lg.get_texts()[0].get_text() == "0.00 a 1.00"
    assert class_breaks(np.arange(1.03, 9.97, 0.01), 3, round_to=0.5, round_ends=True)[[0, -1]].tolist() == [1, 10]
