import ast
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import pituplots
from pituplots import Catalog, Grid, add_colorbar, add_size_legend, alpha_cmap, discrete_cmap, make_map, plot_catalog
from pituplots.style import defaults

SRC = Path(pituplots.__file__).parent


def test_zorder_table():
    z = defaults.ZORDER
    assert (z["basemap"], z["grids"], z["contours"], z["gridlines"], z["vectors"], z["catalogs"],
            z["points"], z["labels"]) == (0, 2, 3, 4, 5, 6, 7, 8)


def test_layering(monkeypatch):
    assert defaults.resolve("catalog")["color"] == "gold"
    monkeypatch.setitem(defaults.THEME, "catalog", {"color": "red", "alpha": 0.9})
    assert defaults.resolve("catalog")["color"] == "red"
    assert defaults.resolve("catalog", color="blue") == {**defaults.DEFAULTS["catalog"], "color": "blue", "alpha": 0.9}


def test_layering_reaches_the_plot(monkeypatch):
    monkeypatch.setitem(defaults.THEME, "catalog", {"alpha": 0.9})
    ax = plot_catalog(Catalog([1, 2], [2, 3], mag=[3, 4]), coastline=False, gridlines=False)
    assert ax.collections[-1].get_alpha() == 0.9
    ax = plot_catalog(Catalog([1, 2], [2, 3], mag=[3, 4]), alpha=0.1, coastline=False, gridlines=False)
    assert ax.collections[-1].get_alpha() == 0.1


def test_resolve_does_not_share_state():
    a = defaults.resolve("grid")
    a["cmap"] = "x"
    assert defaults.DEFAULTS["grid"]["cmap"] == "magma"


def test_alpha_cmap():
    c = alpha_cmap("magma", 0.8)
    assert c(0.0)[-1] == 0 and c(1.0)[-1] == 1 and c(0.5)[-1] == pytest.approx(0.5 ** 0.8, abs=0.01)
    assert alpha_cmap(matplotlib.colormaps["magma"], 0)(0.0)[-1] == 1
    assert matplotlib.colormaps["magma"](0.0)[-1] == 1


def test_discrete_cmap():
    c, n = discrete_cmap("RdYlGn_r", [0, 1, 5, 10])
    assert c.N == 3 and n(0.5) == 0 and n(7) == 2
    with pytest.raises(ValueError):
        discrete_cmap("magma", [1])


def test_colorbar_loc_and_default_mappable():
    fig, ax = plt.subplots()
    m = ax.pcolormesh(np.ones((2, 2)))
    assert add_colorbar(ax).mappable is m
    assert add_colorbar(ax, loc="bottom").orientation == "horizontal"
    add_colorbar(ax, loc="inside")
    assert len(ax.child_axes) == 3
    with pytest.raises(ValueError, match="loc"):
        add_colorbar(ax, loc="x")


def test_colorbar_skips_basemap_and_needs_mappable():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="mappable"):
        add_colorbar(ax)
    m = ax.pcolormesh(np.ones((2, 2)))
    ax.imshow(np.ones((2, 2)), zorder=0)
    assert add_colorbar(ax).mappable is m


def test_colorbar_cax_and_list():
    fig, (a, b, c) = plt.subplots(1, 3)
    m = a.pcolormesh(np.ones((2, 2)))
    assert add_colorbar(a, m, cax=c).ax is c
    fig, axs = plt.subplots(1, 2)
    m = axs[0].pcolormesh(np.ones((2, 2)))
    assert add_colorbar(axs, label="x").ax.get_ylabel() == "x"


def test_colorbar_follows_axes():
    fig, ax = plt.subplots(layout="constrained")
    m = ax.pcolormesh(np.ones((2, 2)))
    cb = add_colorbar(ax, m)
    assert cb.ax in ax.child_axes


def test_size_legend():
    _, ax = plt.subplots()
    lg = add_size_legend(ax, [1, 2], lambda v: v * 10)
    assert ax.get_legend() is None and lg in ax.artists
    assert lg.get_title().get_text() == "$M_w$"
    ax.legend(handles=[plt.Line2D([], [], label="a")])
    assert lg in ax.artists and ax.get_legend() is not lg
    add_size_legend(ax, 3, lambda v: v * 10, title="x", loc="upper left")


# conventions over the whole package

BANNED_CALLS = {"show", "savefig", "tight_layout", "gcf", "gca", "use"}
BANNED_RC = {"rcParams", "rc", "rc_context", "rcdefaults", "rc_file"}


@pytest.mark.parametrize("f", sorted(SRC.rglob("*.py")), ids=lambda p: p.name)
def test_no_banned_calls(f):
    t = ast.parse(f.read_text(encoding="utf-8"))
    for n in ast.walk(t):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            assert n.func.attr not in BANNED_CALLS, f"{f}:{n.lineno}"
        if isinstance(n, ast.Attribute) and f.name != "themes.py":
            assert n.attr not in BANNED_RC, f"{f}:{n.lineno}"
        if isinstance(n, ast.ImportFrom) and f.name != "themes.py":
            assert not {a.name for a in n.names} & BANNED_RC, f"{f}:{n.lineno}"
        if isinstance(n, ast.ExceptHandler):
            assert n.type is not None, f"{f}:{n.lineno}"


def test_optional_imports_are_local():
    opt = {"rasterio", "h5py", "pyogrio", "csep", "openquake", "geopandas"}
    for f in SRC.rglob("*.py"):
        for n in ast.parse(f.read_text(encoding="utf-8")).body:
            names = [a.name for a in n.names] if isinstance(n, ast.Import) else []
            names += [n.module or ""] if isinstance(n, ast.ImportFrom) else []
            assert not {m.partition(".")[0] for m in names} & opt, f


def test_no_side_effects_on_figures(monkeypatch):
    def fail(*a, **k):
        raise AssertionError("show or savefig called")

    monkeypatch.setattr(plt, "show", fail)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", fail)
    g = Grid(np.ones((2, 2)), [0, 1, 2], [0, 1, 2])
    pituplots.plot_grid(g, coastline=False)
    pituplots.plot_field(g, coastline=False)
    plot_catalog(Catalog([1], [1], mag=[3]), coastline=False)
    make_map(coastline=False)


def test_rcparams_untouched():
    before = dict(matplotlib.rcParams)
    plot_catalog(Catalog([1], [1], mag=[3]), coastline=False)
    assert dict(matplotlib.rcParams) == before
