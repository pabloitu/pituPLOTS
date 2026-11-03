import time
import warnings

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.collections import PolyCollection
from matplotlib.colors import BoundaryNorm, LogNorm
from shapely.geometry import box

from pituplots import Grid, PituplotsWarning, Sites, plot_field, plot_grid, plot_sites

KW = {"coastline": False, "gridlines": False}


def grid(nx=4, ny=3):
    return Grid(np.arange(1.0, nx * ny + 1).reshape(ny, nx), np.arange(nx + 1.0), np.arange(ny + 1.0))


def draw(ax):
    ax.figure.canvas.draw()
    return ax


@pytest.mark.parametrize("f", [plot_grid, plot_field])
def test_new_and_given_ax(f):
    ax = f(grid(), **KW)
    assert isinstance(ax, GeoAxes) and ax.get_extent() == pytest.approx((-0.2, 4.2, -0.15, 3.15))
    a = plt.axes(projection=ccrs.PlateCarree())
    assert f(grid(), ax=a) is a
    _, p = plt.subplots()
    assert f(grid(), ax=p) is p
    draw(p)


def test_tuple_inputs():
    z = np.ones((2, 3))
    draw(plot_grid((z, [0, 1, 2, 3], [0, 1, 2]), **KW))
    draw(plot_field((z, [0, 1, 2, 3], [0, 1, 2]), **KW))
    ax = plot_field((z, [0, 1, 2], [0, 1]), **KW)
    assert ax.collections
    with pytest.raises(ValueError):
        plot_grid((z, [0, 1, 2], [0, 1]), **KW)
    with pytest.raises(TypeError):
        plot_grid("x")


def test_input_not_modified():
    g = grid()
    v = g.values.copy()
    plot_grid(g, log=True, **KW)
    assert (g.values == v).all() and g.mask is None


def test_map_keyword_with_ax_raises():
    a = plt.axes(projection=ccrs.PlateCarree())
    with pytest.raises(ValueError, match="projection"):
        plot_grid(grid(), ax=a, projection=ccrs.Mercator())


def test_style_defaults_and_override():
    ax = plot_grid(grid(), colorbar=False, **KW)
    m = [c for c in ax.collections if c.get_array() is not None][-1]
    assert m.get_cmap()(1.0)[-1] == 1 and m.get_cmap()(0.0)[-1] == 0 and m.get_zorder() == 2
    ax = plot_grid(grid(), alpha=0.3, cmap="viridis", colorbar=False, **KW)
    m = [c for c in ax.collections if c.get_array() is not None][-1]
    assert m.get_alpha() == 0.3 and m.get_cmap().name == "viridis"


def test_vmin_vmax_in_log_units():
    ax = plot_grid(grid(), log=True, vmin=-1, vmax=2, **KW)
    m = [c for c in ax.collections if c.get_array() is not None][-1]
    assert m.get_clim() == (-1, 2)


def test_norm_and_colorbar():
    ax = plot_grid(grid(), norm=LogNorm(1, 12), clabel="rate", **KW)
    cax = ax.child_axes[0]
    assert cax.get_ylabel() == "rate"
    assert not plot_grid(grid(), colorbar=False, **KW).child_axes


def test_clip_keeps_data():
    g = grid()
    ax = plot_grid(g, clip=box(1, 0, 3, 3), **KW)
    m = [c for c in ax.collections if c.get_array() is not None][-1]
    assert m.get_clip_path() is not None and m.get_array().size == g.values.size
    draw(ax)
    ax = plot_field(g, kind="contourf", clip=box(1, 0, 3, 3), **KW)
    draw(ax)


def test_projected_grid_on_map():
    g = Grid(np.ones((2, 2)), [1.5e6, 1.6e6, 1.7e6], [5.4e6, 5.5e6, 5.6e6], crs="EPSG:2193")
    ax = plot_grid(g, **KW)
    assert 170 < ax.get_extent()[0] < 175
    draw(ax)


def test_image_grid_raises():
    g = Grid(np.zeros((2, 2, 3), np.uint8), [0, 1, 2], [0, 1, 2])
    with pytest.raises(ValueError, match="basemap"):
        plot_grid(g, **KW)
    with pytest.raises(ValueError, match="band"):
        plot_field(g, **KW)


def test_crossing_grid_two_conventions():
    g = Grid.from_points([179.95, -179.95], [0, 0], [1, 2], dh=0.1)
    ax = plot_grid(g, **KW)
    assert ax.get_extent()[1] - ax.get_extent()[0] == pytest.approx(0.22)
    draw(ax)


def test_crossing_grid_on_seamed_map_two_pieces():
    g = Grid(np.ones((2, 4)), [178, 179, 180, 181, 182], [0, 1, 2])
    ax = plot_grid(g, set_global=True, **KW)
    ms = [c for c in ax.collections if c.get_array() is not None]
    assert len(ms) == 2 and ms[0].get_clim() == ms[1].get_clim()


# B3 grid

@pytest.mark.parametrize("g", [
    Grid(np.ones((1, 1)), [0, 1], [0, 1]),
    Grid(np.ones((1, 5)), np.arange(6.0), [0, 1]),
    Grid(np.ones((5, 1)), [0, 1], np.arange(6.0)),
    Grid.from_points([0, 2, 5], [0, 3, 1], [1, 2, 3], dh=1),
    Grid(np.ones((3, 3)), np.arange(4) * 0.5, np.arange(4) * 2.0),
], ids=["one", "row", "column", "gaps", "dxdy"])
def test_structure(g):
    draw(plot_grid(g, **KW))


def test_holes_and_parts():
    m = np.zeros((5, 5), bool)
    m[2, 2] = True
    m[:, 3] = True
    draw(plot_grid(Grid(np.ones((5, 5)), np.arange(6.0), np.arange(6.0), mask=m), **KW))


def test_unsorted_duplicate_off_lattice():
    a = Grid.from_points([1, 0, 1, 0, 0], [1, 0, 0, 1, 0], [4, 1, 2, 3, 1], dh=1)
    assert a.values.tolist() == [[1, 2], [3, 4]]
    g = Grid.from_points(np.arange(0, 1, 0.1) + 1e-7, np.zeros(10), np.ones(10), dh=0.1)
    assert g.values.shape == (1, 10)
    draw(plot_grid(g, **KW))


def test_large_grid_budget():
    z = np.random.default_rng(0).random((1800, 3600))
    g = Grid(z, np.linspace(-180, 180, 3601), np.linspace(-90, 90, 1801))
    t = time.perf_counter()
    ax = plot_grid(g, set_global=True, colorbar=False, **KW)
    draw(ax)
    assert time.perf_counter() - t < 30


# B3 val

def test_all_zero_log():
    with pytest.warns(PituplotsWarning, match="log=True"):
        draw(plot_grid(Grid(np.zeros((2, 2)), [0, 1, 2], [0, 1, 2]), log=True, **KW))


def test_all_nan():
    with pytest.warns(PituplotsWarning, match="no plottable"):
        draw(plot_grid(Grid(np.full((2, 2), np.nan), [0, 1, 2], [0, 1, 2]), **KW))
    with pytest.warns(PituplotsWarning):
        draw(plot_field(Grid(np.full((2, 2), np.nan), [0, 1, 2], [0, 1, 2]), kind="contourf", **KW))


def test_masked_array():
    z = np.ma.masked_array(np.ones((2, 2)), [[1, 0], [0, 0]])
    draw(plot_grid((z, [0, 1, 2], [0, 1, 2]), **KW))


def test_negative_with_log_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        draw(plot_grid(Grid(np.array([[-1.0, 1], [2, 0]]), [0, 1, 2], [0, 1, 2]), log=True, **KW))


@pytest.mark.parametrize("f", [plot_grid, plot_field])
def test_constant_widened(f):
    with warnings.catch_warnings():
        warnings.simplefilter("error", PituplotsWarning)
        ax = f(Grid(np.full((2, 2), 3.0), [0, 1, 2], [0, 1, 2]), **KW)
    m = [c for c in ax.collections if c.get_array() is not None][-1]
    assert m.get_clim() == (2.5, 3.5)
    draw(ax)


def test_constant_contourf():
    draw(plot_field(Grid(np.full((3, 3), 3.0), np.arange(4.0), np.arange(4.0)), kind="contourf", **KW))


def test_single_nonzero():
    z = np.zeros((3, 3))
    z[1, 1] = 1e-5
    draw(plot_grid(Grid(z, np.arange(4.0), np.arange(4.0)), log=True, **KW))


def test_ten_orders():
    z = np.logspace(-10, 0, 16).reshape(4, 4)
    ax = plot_grid(Grid(z, np.arange(5.0), np.arange(5.0)), log=True, **KW)
    m = [c for c in ax.collections if c.get_array() is not None][-1]
    assert m.get_clim() == pytest.approx((-10, 0))


# plot_field

def test_contourf_levels_and_over():
    g = grid()
    ax = plot_field(g, kind="contourf", levels=[0, 2, 5, 8, 10], **KW)
    draw(ax)
    cs = [c for c in ax.collections if c.get_array() is not None][0]
    assert isinstance(cs.norm, BoundaryNorm)
    assert cs.get_cmap()(1000)[:3] == pytest.approx(plt.matplotlib.colors.to_rgb("#4a0d1f"))


def test_mesh_levels():
    assert isinstance(plot_field(grid(), levels=[0, 5, 20], **KW).collections[-1].norm, (BoundaryNorm, object))


def test_levels_int_contour_labels_smooth():
    ax = plot_field(grid(8, 8), kind="contourf", levels=5, contour_labels=True, smooth=1, **KW)
    draw(ax)
    assert ax.texts


def test_smooth_keeps_nan():
    g = grid(8, 8)
    g.values[3, 3] = np.nan
    draw(plot_field(g, kind="contourf", smooth=1, **KW))


def test_field_bad_kind():
    with pytest.raises(ValueError, match="kind"):
        plot_field(grid(), kind="x", **KW)


def test_contourf_crossing_seam():
    g = Grid(np.ones((4, 8)), np.arange(176, 185.0), np.arange(5.0))
    ax = plot_field(g, kind="contourf", set_global=True, **KW)
    draw(ax)


def test_field_alpha_default():
    ax = plot_field(grid(), colorbar=False, **KW)
    m = [c for c in ax.collections if c.get_array() is not None][-1]
    assert m.get_alpha() == 0.75


# plot_sites

def test_sites_inputs():
    s = Sites([0, 1, 2], [0, 1, 2], [1.0, np.nan, 3.0])
    ax = plot_sites(s, empty_color="0.9", **KW)
    assert len([c for c in ax.collections if hasattr(c, "get_offsets")]) >= 2
    draw(ax)
    draw(plot_sites(([0, 1], [0, 1]), **KW))
    draw(plot_sites(([0, 1], [0, 1], [1, 2]), **KW))
    d = pd.DataFrame({"lon": [0, 1], "lat": [0, 1], "r": [1, 2]})
    ax = plot_sites(d, color_by="r", clabel="r", **KW)
    assert ax.child_axes
    with pytest.raises(TypeError):
        plot_sites("x")
    with pytest.raises(ValueError, match="one value"):
        plot_sites(Sites([0], [0], np.zeros((1, 2))), **KW)


def cells(ax):
    return [c for c in ax.collections if isinstance(c, PolyCollection)]


def test_sites_are_cells_in_map_coordinates():
    fig = plt.figure(figsize=(4, 4), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
    ax.set_axis_off()
    ax.set_extent([0, 10, 0, 10], ccrs.PlateCarree())
    plot_sites(([5.0], [5.0], [1.0]), ax=ax, dh=(2, 4), colorbar=False, cmap="Reds", vmin=0, vmax=1)
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba()).astype(int)
    r = a[..., 1] < 100
    assert np.ptp(np.flatnonzero(r.any(axis=0))) == pytest.approx(80, abs=2)
    assert np.ptp(np.flatnonzero(r.any(axis=1))) == pytest.approx(160, abs=2)


def test_sites_adjacent_cells_leave_no_gap():
    fig = plt.figure(figsize=(4, 4), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
    ax.set_axis_off()
    ax.set_extent([0, 5, 0, 5], ccrs.PlateCarree())
    x, y = np.meshgrid(np.arange(0.05, 5, 0.1), np.arange(0.05, 5, 0.1))
    plot_sites((x.ravel(), y.ravel(), x.ravel() * 0 + 1), ax=ax, dh=0.1, colorbar=False, cmap="Reds", vmin=0, vmax=1)
    for size in ((4, 4), (6.3, 6.3)):
        fig.set_size_inches(*size)
        fig.canvas.draw()
        a = np.asarray(fig.canvas.buffer_rgba())[..., 1].astype(int)
        b = ax.get_window_extent()
        h = a.shape[0]
        inner = a[h - int(b.y1) + 3:h - int(b.y0) - 3, int(b.x0) + 3:int(b.x1) - 3]
        assert inner.max() - inner.min() < 3


def test_sites_fill_shrinks_cells():
    ax = plot_sites(([0.0, 1], [0.0, 1], [1.0, 2]), dh=1, fill=0.5, **KW)
    p = cells(ax)[-1].get_paths()[0].vertices
    assert np.ptp(p[:, 0]) == pytest.approx(0.5)


def test_sites_no_values_and_empty_cells():
    ax = plot_sites(([0.0, 1], [0.0, 1]), dh=1, **KW)
    assert len(cells(ax)) == 1
    draw(ax)
    ax = plot_sites(([0.0, 1], [0.0, 1], [np.nan, 1.0]), dh=1, empty_color="0.9", **KW)
    assert len(cells(ax)) == 2
    draw(ax)


def test_sites_cells_across_dateline():
    ax = plot_sites(([179.5, -179.5], [0, 0], [1, 2]), dh=1, extent=[175, -175, -5, 5], **KW)
    v = np.concatenate([p.vertices for p in cells(ax)[-1].get_paths()])
    assert np.ptp(v[:, 0]) == pytest.approx(2)
    draw(ax)


def test_sites_given_ax_plain():
    _, p = plt.subplots()
    assert plot_sites(([0, 1], [0, 1], [1, 2]), ax=p, log=True, dh=0.1) is p
    draw(p)


def test_sites_all_nan_warns():
    with pytest.warns(PituplotsWarning):
        draw(plot_sites(([0, 1], [0, 1], [np.nan, np.nan]), **KW))


def test_sites_dateline():
    ax = plot_sites(([179.5, -179.5], [0, 1], [1, 2]), **KW)
    x = [c for c in ax.collections if hasattr(c, "get_offsets")][-1].get_offsets()[:, 0]
    assert np.ptp(x) == pytest.approx(1)


def test_capped_grid_fixture(data):
    d = pd.read_csv(data / "chile/grid_capped.csv")
    draw(plot_sites(d, color_by="rate_M5.00_5.10", dh=0.1, log=True, gridlines=False))


def test_field_contour_labels_follow_the_grid_crs_and_smoothing():
    x, y = np.arange(1000000, 1200001, 20000.0), np.arange(5000000, 5200001, 20000.0)
    X, Y = np.meshgrid((x[:-1] + x[1:]) / 2, (y[:-1] + y[1:]) / 2)
    g = Grid(np.hypot(X - 1.1e6, Y - 5.1e6) / 1e3, x, y, crs="EPSG:2193")
    ax = plot_field(g, contour_labels=True, levels=None, smooth=1, contour_color="red", contour_fmt="%.1f",
                    coastline=False, gridlines=False)
    lines = ax.collections[-1] if hasattr(ax.collections[-1], "levels") else None
    ax.figure.canvas.draw()
    assert lines is not None and len(lines.get_paths()) and any(len(p.vertices) for p in lines.get_paths())
    assert all("." in t.get_text() and len(t.get_text().split(".")[1]) == 1 for t in ax.texts)
