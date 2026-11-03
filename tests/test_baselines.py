"""One image per family, from tests/data only. New images are reviewed by eye.

The axes are fixed and carry no gridline labels: text moves a constrained layout
by a pixel between matplotlib versions, and these images compare the drawing.
"""
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import PowerNorm

from shapely.geometry import box

from pituplots import (Grid, add_gridlines, add_labels, add_locator, add_panel_label, add_scalebar, add_slab,
                       compare_grids, panels, strip_panels,
                       make_map, make_profile, plot_lines, plot_points, plot_polygons, plot_region,
                       plot_catalog, plot_field, plot_grid, plot_section, plot_sites, theme)
from pituplots.io import read_catalog, read_csep_nodes, read_rate_csv, read_vector

image = pytest.mark.mpl_image_compare(tolerance=20, style="default", remove_text=True)
LEVELS = [-4.4, -4.2, -4, -3.8, -3.6, -3.4, -3.2, -3]
SMALL = {"figsize": (5, 4), "gridlines": False}


def fixed(ax):
    ax.figure.set_layout_engine("none")
    ax.set_position([0.08, 0.08, 0.76, 0.84])
    return ax.figure


@image
def test_baseline_map(data):
    ax = make_map(extent=[-12, 32, 34, 60], borders=True, **SMALL)
    add_gridlines(ax, labels=False)
    return fixed(ax)


@image
def test_baseline_theme():
    with theme("paper"):
        ax = make_map(extent=[-12, 32, 34, 60], borders=True, **SMALL)
        add_gridlines(ax, labels=False)
    return fixed(ax)


@image
def test_baseline_map_dateline():
    return fixed(make_map(extent=[148, -158, -55, 0], **SMALL))


@image
def test_baseline_basemap(data):
    pytest.importorskip("rasterio")
    ax = make_map(extent=[-124, -114, 32, 42], projection=ccrs.Mercator(),
                  basemap=data / "basemaps/california_4326_rgba.tif", **SMALL)
    return fixed(ax)


@image
def test_baseline_catalog(data):
    c = read_catalog(data / "lacq/events.csv")
    ax = plot_catalog(c, mag_ticks=[1.5, 2.5, 3.5, 4], **SMALL)
    return fixed(ax)


@image
def test_baseline_catalog_dateline(data):
    ax = plot_catalog(read_catalog(data / "new_zealand/cat_nz.csv"), color_by="depth", cmap="viridis",
                      legend=False, **SMALL)
    return fixed(ax)


@image
def test_baseline_grid(data):
    ax = plot_grid(read_rate_csv(data / "lacq/pattern.csv"), log=True, **SMALL)
    return fixed(ax)


@image
def test_baseline_grid_dateline():
    x = np.arange(170.5, 190, 1.0)
    y = np.arange(-30.5, -10, 1.0)
    X, Y = np.meshgrid(x, y)
    g = Grid.from_points((X.ravel() + 180) % 360 - 180, Y.ravel(), np.hypot(X - 180, Y + 20).ravel())
    return fixed(plot_grid(g, alpha_exp=0, **SMALL))


@image
def test_baseline_field(data):
    d = pd.read_csv(data / "chile/grid_capped.csv")
    g = Grid.from_points(d["lon"], d["lat"], d["rate_M5.00_5.10"], dh=0.1)
    return fixed(plot_field(g, kind="contourf", log=True, levels=LEVELS, **SMALL))


@image
def test_baseline_sites(data):
    d = pd.read_csv(data / "chile/grid_capped.csv")
    return fixed(plot_sites(d, color_by="rate_M5.00_5.10", dh=0.1, log=True, **SMALL))


@image
def test_baseline_section(data):
    c = read_catalog(data / "new_zealand/cat_nz.csv")
    p = make_profile((178.5, -38.5), azimuth=300, length=(-50, 450), width=300)
    x = np.arange(170, 182, 0.5)
    y = np.arange(-44, -32, 0.5)
    X = np.meshgrid((x[:-1] + x[1:]) / 2, (y[:-1] + y[1:]) / 2)[0]
    top = Grid(np.maximum(5, 15 + 45 * (178.5 - X)), x, y)
    ax = plot_section(c, p, ve=1, zmax=340, figsize=(6, 4.5))
    add_slab(ax, p, top, Grid(top.values * 0 + 30, x, y))
    return fixed(ax)


@image
def test_baseline_elements():
    ax = make_map(extent=[-5, 8, 42, 50], **SMALL)
    add_locator(ax, [-12, 32, 34, 60], rect=(0.02, 0.6, 0.38, 0.38))
    add_scalebar(ax, 200, loc=(0.6, 0.08), lw=3)
    add_labels(ax, [2.35, 5.37, -1.68], [48.86, 43.3, 43.48], ["a", "b", "c"], fontsize=12)
    add_panel_label(ax, "a")
    return fixed(ax)


@image
def test_baseline_vectors(data):
    r = read_vector(data / "new_zealand/nz_cfm.shp")
    ax = plot_lines(r, values="SR_pref", norm=PowerNorm(0.5), lw_by=(0.4, 3), cmap="viridis", colorbar=False, **SMALL)
    plot_region(read_csep_nodes(data / "new_zealand/nz.testing.nodes.dat"), ax=ax)
    plot_polygons(box(166.5, -40, 169, -37), ax=ax, fill="lightsteelblue", edge="navy")
    plot_points(([174.78, 172.64], [-41.29, -43.53]), ax=ax)
    return fixed(ax)


@image
def test_baseline_composite(data):
    g = read_rate_csv(data / "lacq/pattern.csv")
    h = Grid(g.values * np.linspace(0.5, 1.5, g.values.shape[1]), g.x_edges, g.y_edges)
    fig, axs = compare_grids(g, h, log=True, coastline=False, gridlines=False, figsize=(9, 3.4))
    fig.set_layout_engine("none")
    for a, x in zip(axs, (0.02, 0.32, 0.64)):
        a.set_position([x, 0.22, 0.28, 0.74])
    next(a for a in fig.axes if a not in axs).set_position([0.06, 0.08, 0.5, 0.05])
    return fig


@image
def test_baseline_stats(data):
    import matplotlib.pyplot as plt

    from pituplots.stats import plot_curves, plot_fmd

    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3.5))
    plot_fmd(read_catalog(data / "lacq/events.csv"), ax=a, mc=2.0)
    rng = np.random.default_rng(0)
    lv = np.logspace(-3, 0.5, 25)
    c = np.clip(0.05 * (lv[None] / 0.01) ** (-1.8 * rng.uniform(0.7, 1.3, (40, 1))), 0, 1)
    plot_curves(c, imls=lv, ax=b, investigation_time=50, branches=True)
    for x, ax in zip((0.08, 0.58), (a, b)):
        ax.set_position([x, 0.12, 0.38, 0.8])
        ax.get_legend().remove()
        ax.set(xlabel="", ylabel="")
        for t in list(ax.texts):
            t.remove()
    return fig


@image
def test_baseline_sources(data):
    import pandas as pd

    from pituplots import plot_sources
    from pituplots.io import read_sources

    s = pd.concat([read_sources(data / "nrml/simple.xml"), read_sources(data / "nrml/complex.xml")],
                  ignore_index=True)
    return fixed(plot_sources(s, color_by="mmax", extent=[-77, -68, -46, -18], colorbar=False, lw=3, **SMALL))
