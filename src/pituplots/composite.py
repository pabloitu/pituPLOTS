import numpy as np
from matplotlib.colors import LogNorm

from pituplots.elements import add_colorbar
from pituplots.grids import as_grid, masked, plot_grid
from pituplots.io.containers import Grid
from pituplots.maps import MAP_KW, add_gridlines, dress_map, extent_from, map_projection, set_map_extent
from pituplots.style.defaults import resolve


def panels(nrows=1, ncols=1, figsize=None, **map_kw):
    """Grid of maps that share their map keywords.

    Gridline labels appear only on the left column and the bottom row.

    Parameters
    ----------
    nrows, ncols : int
    figsize : tuple, optional
    **map_kw
        Keywords of ``make_map``.

    Returns
    -------
    (Figure, ndarray of GeoAxes)
        The axes are flat, row by row.
    """
    import matplotlib.pyplot as plt

    if not (nrows >= 1 and ncols >= 1):
        raise ValueError(f"need at least one panel, got {nrows} x {ncols}")
    bad = set(map_kw) - set(MAP_KW)
    if bad:
        raise TypeError(f"unknown map keywords {sorted(bad)}")
    grid = map_kw.pop("gridlines", True)
    proj = map_projection(map_kw.get("extent"), map_kw.get("projection"), map_kw.get("basemap"))
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, layout="constrained", squeeze=False,
                            subplot_kw={"projection": proj})
    for (i, j), ax in np.ndenumerate(axs):
        dress_map(ax, gridlines=False, **map_kw)
        if grid:
            gl = add_gridlines(ax)
            gl.left_labels, gl.bottom_labels = j == 0, i == nrows - 1
    return fig, axs.ravel()


def strip_panels(lat_edges, lon_centers, width_km, figsize=None, **map_kw):
    """Side-by-side maps of consecutive latitude bands, for elongated countries.

    Every panel has the same width in km and the same shape, on a cos-lat
    aspect, so the scale is nearly the same across them.

    Parameters
    ----------
    lat_edges : sequence of float
        ``n + 1`` monotonic band edges.
    lon_centers : sequence of float
        ``n`` central longitudes.
    width_km : float
    figsize : tuple, optional
    **map_kw
        Keywords of ``make_map``, except ``extent`` and ``projection``.

    Returns
    -------
    (Figure, ndarray of GeoAxes)
    """
    import matplotlib.pyplot as plt

    lat, lon = np.asarray(lat_edges, float), np.asarray(lon_centers, float)
    if lat.size < 2 or lon.size != lat.size - 1:
        raise ValueError(f"{lat.size} latitude edges need {lat.size - 1} centres, got {lon.size}")
    d = np.diff(lat)
    if not ((d > 0).all() or (d < 0).all()):
        raise ValueError("lat_edges must be monotonic")
    if not width_km > 0:
        raise ValueError(f"width_km must be positive, got {width_km}")
    bad = set(map_kw) & {"extent", "projection"}
    if bad:
        raise ValueError(f"{sorted(bad)} are set by the strips")
    grid = map_kw.pop("gridlines", True)
    fig = plt.figure(figsize=figsize, layout="constrained")
    axs = []
    for k, c in enumerate(lon):
        s, n = sorted(lat[k:k + 2])
        hw = width_km / (111.32 * np.cos(np.deg2rad((s + n) / 2))) / 2
        ext = [(c - hw + 180) % 360 - 180, (c + hw + 180) % 360 - 180, s, n]
        ax = fig.add_subplot(1, lon.size, k + 1, projection=map_projection(ext, "approx", None))
        dress_map(ax, ext, "approx", gridlines=False, **map_kw)
        if grid:
            add_gridlines(ax).left_labels = k == 0
        axs.append(ax)
    return fig, np.array(axs)


def compare_grids(a, b, kind="pct", floor=None, clip_pct=98, log=False, clabel=None, dlabel=None,
                  titles=("A", "B", None), figsize=None, **kw):
    """Two grids on a shared scale, and their difference on a symmetric one.

    Parameters
    ----------
    a, b : Grid or tuple
        On the same lattice and CRS. Regridding is not done.
    kind : {"pct", "diff", "ratio"}
        ``100 (a - b) / b``, ``a - b`` or ``a / b``. The ratio has a log scale
        centred on 1.
    floor : float, optional
        Cells with ``b <= floor`` have no ``pct`` or ``ratio``. Default 0.
    clip_pct : float
        The scale ends at this percentile of the absolute difference, so a few
        cells with a small ``b`` do not set it.
    log : bool
        Plot ``a`` and ``b`` as ``log10``. The difference uses the raw values.
    clabel, dlabel : str, optional
        Labels of the shared and of the difference colorbars.
    titles : sequence of str
    figsize : tuple, optional
    **kw
        Map keywords; ``cmap``, ``diff_cmap``, ``alpha_exp``, ``vmin``,
        ``vmax``. The rest passes to ``plot_grid``.

    Returns
    -------
    (Figure, ndarray of GeoAxes)
        The three axes.
    """
    if kind not in ("pct", "diff", "ratio"):
        raise ValueError(f"kind must be 'pct', 'diff' or 'ratio', not {kind!r}")
    a, b = as_grid(a), as_grid(b)
    same = (a.values.shape == b.values.shape and a.crs == b.crs and a.x_edges.size == b.x_edges.size
            and a.y_edges.size == b.y_edges.size and np.allclose(a.x_edges, b.x_edges, rtol=0, atol=1e-9)
            and np.allclose(a.y_edges, b.y_edges, rtol=0, atol=1e-9))
    if not same:
        raise ValueError("the grids are on different lattices: regrid them first")
    if a.values.ndim != 2:
        raise ValueError("compare_grids takes one band")

    mk = {k: kw.pop(k) for k in MAP_KW if k in kw}
    st = resolve("compare", **kw)
    diff_cmap = st.pop("diff_cmap")
    vmin, vmax = st.pop("vmin", None), st.pop("vmax", None)
    za, zb = masked(a, False), masked(b, False)
    both = np.ma.getmaskarray(za) | np.ma.getmaskarray(zb)
    x, y = np.ma.array(za.data, mask=both), np.ma.array(zb.data, mask=both)
    lim0 = 0.0 if floor is None else floor
    if kind == "diff":
        d = x - y
    else:
        w = np.ma.masked_less_equal(y, lim0)
        d = 100 * (x - w) / w if kind == "pct" else x / w
        if kind == "ratio":
            d = np.ma.masked_less_equal(d, 0)
    m = np.log10(d) if kind == "ratio" else d
    lim = float(np.nanpercentile(np.abs(m.compressed()), clip_pct)) if m.count() else 1.0
    lim = lim or 1.0

    pa, pb = masked(a, log), masked(b, log)
    v = np.ma.concatenate([pa.ravel(), pb.ravel()])
    if vmin is None and v.count():
        vmin = float(v.min())
    if vmax is None and v.count():
        vmax = float(v.max())

    if "extent" not in mk and not mk.get("set_global"):
        mk["extent"] = extent_from(Grid(a.values, a.x_edges, a.y_edges, a.crs, mask=both))
    fig, axs = panels(1, 3, figsize=figsize, **mk)
    extent = mk.get("extent")
    ms = []
    for ax, g, t in zip(axs[:2], (a, b), titles):
        plot_grid(g, ax=ax, log=log, colorbar=False, vmin=vmin, vmax=vmax, title=t, **st)
        ms.append(ax.collections[-1])
    add_colorbar(list(axs[:2]), ms[0], label=clabel, location="bottom", shrink=0.8)
    g = Grid(d, a.x_edges, a.y_edges, a.crs)
    dk = {"norm": LogNorm(10 ** -lim, 10 ** lim)} if kind == "ratio" else {"vmin": -lim, "vmax": lim}
    rest = {k: v for k, v in st.items() if k not in ("cmap", "alpha_exp")}
    plot_grid(g, ax=axs[2], cmap=diff_cmap, alpha_exp=0, colorbar=True, clabel=dlabel,
              title=titles[2], **dk, **rest)
    if extent is not None:
        for ax in axs:
            set_map_extent(ax, extent)
    return fig, axs
