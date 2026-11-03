import warnings

import cartopy.crs as ccrs
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter

from pituplots.elements import add_colorbar
from pituplots.errors import PituplotsWarning
from pituplots.io.containers import Grid, Sites, find_column
from pituplots.maps import cartopy_crs, central_lon, clip_path, extent_from, get_ax, native, seam_split
from pituplots.style.cmaps import alpha_cmap, discrete_cmap
from pituplots.style.defaults import ZORDER, resolve


def as_grid(data, centers=False):
    if isinstance(data, Grid):
        return data
    if isinstance(data, tuple) and len(data) == 3:
        z, x, y = (np.ma.asarray(data[0]), np.asarray(data[1], float), np.asarray(data[2], float))
        if centers and z.ndim == 2 and (x.size, y.size) == z.shape[::-1]:
            edge = []
            for c in (x, y):
                d = np.diff(c) if c.size > 1 else None
                if d is None:
                    raise ValueError("cell centres need at least two positions on each axis")
                edge.append(np.r_[c[0] - d[0] / 2, (c[:-1] + c[1:]) / 2, c[-1] + d[-1] / 2])
            x, y = edge
        return Grid(z, x, y)
    raise TypeError(f"unsupported grid data: {type(data).__name__}")


def as_sites(data, color_by=None):
    if isinstance(data, Sites):
        return data
    if isinstance(data, pd.DataFrame):
        v = None if color_by is None else data[color_by].to_numpy(float)
        return Sites(find_column(data, "lon").to_numpy(), find_column(data, "lat").to_numpy(), v)
    if isinstance(data, tuple) and len(data) in (2, 3):
        return Sites(*data)
    raise TypeError(f"unsupported sites data: {type(data).__name__}")


def masked(g, log, v=None):
    v = np.ma.masked_invalid(np.asarray(g.values if v is None else v, float))
    if g.mask is not None:
        v = np.ma.masked_where(g.mask, v)
    if log:
        v = np.ma.log10(np.ma.masked_less_equal(v, 0))
    if v.count() == 0:
        what = "with log=True" if log else "in the field"
        warnings.warn(f"no plottable values {what}", PituplotsWarning, stacklevel=3)
    return v


def limits(z, st):
    """Pop ``vmin``/``vmax`` and return a norm. A constant field is widened."""
    vmin, vmax = st.pop("vmin", None), st.pop("vmax", None)
    if vmin is None:
        vmin = float(z.min()) if z.count() else 0.0
    if vmax is None:
        vmax = float(z.max()) if z.count() else 1.0
    if vmin == vmax:
        vmin, vmax = vmin - 0.5, vmax + 0.5
    return Normalize(vmin, vmax)


def mesh(ax, g, z, **kw):
    if g.crs != "EPSG:4326":
        tr = {"transform": cartopy_crs(g.crs)} if hasattr(ax, "projection") else {}
        return [ax.pcolormesh(g.x_edges, g.y_edges, z, **tr, **kw)]
    if not hasattr(ax, "projection"):
        return [ax.pcolormesh(g.x_edges, g.y_edges, z, **kw)]
    c = central_lon(ax.projection)
    crs = ccrs.PlateCarree(central_longitude=c)
    return [ax.pcolormesh(x, g.y_edges, zi, transform=crs, **kw) for zi, x in seam_split(z, g.x_edges, c)]


def plot_grid(data, ax=None, log=False, clip=None, colorbar=True, **kw):
    """Cell map of a gridded dataset, by default with an alpha ramp.

    Parameters
    ----------
    data : Grid or tuple
        ``(values, x_edges, y_edges)``.
    ax : Axes, optional
        A map is built if missing, with the keywords of ``make_map``.
    log : bool
        Plot ``log10(values)`` on a linear norm. ``vmin`` and ``vmax`` are in
        log units. Values <= 0 are masked.
    clip : shapely polygon, optional
        Clip path. The data does not change.
    colorbar : bool
    **kw
        ``cmap``, ``alpha_exp`` (0.8), ``alpha`` (uniform, overrides
        ``alpha_exp``), ``vmin``, ``vmax``, ``norm``, ``clabel``, ``title``,
        ``zorder``. The rest passes to ``pcolormesh``.

    Returns
    -------
    Axes
    """
    g = as_grid(data)
    if g.values.ndim != 2:
        raise ValueError("plot_grid takes one band: use add_basemap for images")
    ax = get_ax(g, ax, kw)
    st = resolve("grid", **kw)
    z = masked(g, log)
    exp, clabel, title = st.pop("alpha_exp"), st.pop("clabel", None), st.pop("title", None)
    cmap = st.pop("cmap")
    if st.get("alpha") is None:
        st.pop("alpha", None)
        cmap = alpha_cmap(cmap, exp)
    norm = st.pop("norm", None) or limits(z, st)
    st.setdefault("zorder", ZORDER["grids"])
    ms = mesh(ax, g, z, cmap=cmap, norm=norm, shading="flat", **st)
    if clip is not None:
        p = clip_path(ax, clip)
        for m in ms:
            m.set_clip_path(p, ax.transData)
    if colorbar:
        add_colorbar(ax, ms[-1], label=clabel)
    if title:
        ax.set_title(title)
    return ax


def plot_field(data, ax=None, kind="mesh", levels=None, contour_labels=False, smooth=0, clip=None,
               colorbar=True, log=False, **kw):
    """Continuous field: shaded cells or filled contours.

    Parameters
    ----------
    data : Grid or tuple
        ``(values, x_edges, y_edges)`` or ``(values, lon, lat)`` at centres.
    ax : Axes, optional
    kind : {"mesh", "contourf"}
    levels : int or sequence, optional
        Bin edges, possibly non-uniform, or how many. A sequence gives a
        discrete colormap with ``over`` above the last level.
    contour_labels : bool
        Draw labelled contour lines at ``levels``.
    smooth : float
        Gaussian sigma in cells, for the contours only (filled and labelled).
        NaN cells stay NaN.
    clip : shapely polygon, optional
    colorbar, log : bool
    **kw
        ``cmap``, ``alpha``, ``over``, ``vmin``, ``vmax``, ``clabel``,
        ``title``, ``zorder``, ``contour_color``, ``contour_fmt`` (of the
        labels, e.g. ``"%.2f"``). The rest passes to the matplotlib call.

    Returns
    -------
    Axes
    """
    if kind not in ("mesh", "contourf"):
        raise ValueError(f"kind must be 'mesh' or 'contourf', not {kind!r}")
    g = as_grid(data, centers=True)
    if g.values.ndim != 2:
        raise ValueError("plot_field takes one band")
    ax = get_ax(g, ax, kw)
    st = resolve("field", **kw)
    z = masked(g, log)
    over, clabel, title = st.pop("over"), st.pop("clabel", None), st.pop("title", None)
    ccolor, cfmt = st.pop("contour_color"), st.pop("contour_fmt")
    cmap, zo = st.pop("cmap"), st.pop("zorder", ZORDER["grids"])
    seq = levels is not None and not np.isscalar(levels)
    cb = {"label": clabel}
    if seq:
        cmap, norm = discrete_cmap(cmap, levels)
        cmap = cmap.with_extremes(over=over)
        st.pop("vmin", None), st.pop("vmax", None)
        cb.update(boundaries=levels, ticks=levels, spacing="uniform", extend="max")
    else:
        norm = limits(z, st)

    xc, yc = (g.x_edges[:-1] + g.x_edges[1:]) / 2, (g.y_edges[:-1] + g.y_edges[1:]) / 2
    tr = {"transform": ccrs.PlateCarree() if g.crs == "EPSG:4326" else cartopy_crs(g.crs)} \
        if hasattr(ax, "projection") else {}
    zc = z
    if smooth and z.count():
        f, w = z.filled(0.0), (~np.ma.getmaskarray(z)).astype(float)
        f = gaussian_filter(f, smooth) / np.maximum(gaussian_filter(w, smooth), 1e-12)
        zc = np.ma.masked_where(np.ma.getmaskarray(z), f)
    if z.count() == 0:
        ms = []
    elif kind == "mesh":
        ms = mesh(ax, g, z, cmap=cmap, norm=norm, shading="flat", zorder=zo, **st)
    else:
        lv = levels if levels is not None else np.linspace(norm.vmin, norm.vmax, 11)
        extend = "max" if seq else "neither"
        cs = ax.contourf(xc, yc, zc, levels=lv, cmap=cmap, norm=norm, extend=extend, zorder=zo, **tr, **st)
        ms = [cs]
        if seq:
            ms.append(ax.contour(xc, yc, zc, levels=[levels[-1]], colors=ccolor, linewidths=1,
                                 zorder=ZORDER["contours"], **tr))
    if contour_labels and z.count():
        lines = ax.contour(xc, yc, zc, levels=levels if levels is not None else 7, colors=ccolor,
                           linewidths=0.6, zorder=ZORDER["contours"], **tr)
        ax.clabel(lines, fontsize="small", fmt=cfmt)
        ms.append(lines)
    if clip is not None:
        p = clip_path(ax, clip)
        for m in ms:
            m.set_clip_path(p, ax.transData)
    if colorbar and ms:
        add_colorbar(ax, ms[0], **cb)
    if title:
        ax.set_title(title)
    return ax


def plot_sites(data, ax=None, marker="s", dh=None, empty_color=None, color_by=None, log=False,
               colorbar=True, **kw):
    """Sites that need not form a lattice, as cells or as markers.

    Parameters
    ----------
    data : Sites, DataFrame or tuple
        ``(lon, lat, values)`` or ``(lon, lat)``.
    ax : Axes, optional
    marker : str
    dh : float or (float, float), optional
        Cell size in degrees. Sites are then drawn as cells in map coordinates,
        so they follow any resize, and ``marker`` and ``size`` do not apply.
        ``fill`` scales the cell.
    empty_color : color, optional
        Draw sites with no value as small markers of this colour.
    color_by : str, optional
        Column of a DataFrame holding the values.
    log, colorbar : bool
    **kw
        ``cmap``, ``vmin``, ``vmax``, ``size``, ``clabel``, ``title``,
        ``zorder``. The rest passes to ``scatter``.

    Returns
    -------
    Axes
    """
    s = as_sites(data, color_by)
    if s.values is not None and np.ndim(s.values) != 1:
        raise ValueError("plot_sites takes one value per site")
    ax = get_ax(s, ax, kw)
    st = resolve("sites", **kw)
    clabel, title, fill = st.pop("clabel", None), st.pop("title", None), st.pop("fill")
    st.setdefault("zorder", ZORDER["grids"])
    area = st.pop("size", 9)
    x, tr = native(ax, s.lon)

    def draw(sel, f=1.0, **a):
        if dh is None:
            c = a.pop("c", None)
            return ax.scatter(x[sel], s.lat[sel], c=c, s=area * f ** 2, marker=marker, **tr, **a)
        dx, dy = np.broadcast_to(dh, 2).astype(float) * fill * f / 2
        px, py = x[sel], s.lat[sel]
        v = np.stack([np.c_[px + i * dx, py + j * dy] for i, j in ((-1, -1), (1, -1), (1, 1), (-1, 1))], axis=1)
        if "color" in a:
            a["facecolors"] = a.pop("color")
        c = PolyCollection(v, array=a.pop("c", None), antialiaseds=False, **a)
        if tr:
            c.set_transform(tr["transform"]._as_mpl_transform(ax))
        geo = hasattr(ax, "projection")
        ax.add_collection(c, autolim=not geo)
        if not geo:
            ax.autoscale_view()
        return c

    every = np.ones(x.shape, bool)
    if s.values is None:
        st.pop("cmap")
        st.setdefault("color", "0.6")
        draw(every, **st)
    else:
        v = np.ma.masked_invalid(np.asarray(s.values, float))
        if log:
            v = np.ma.log10(np.ma.masked_less_equal(v, 0))
        if v.count() == 0:
            warnings.warn("no plottable values in the sites", PituplotsWarning, stacklevel=2)
        norm = st.pop("norm", None) or limits(v, st)
        ok = ~np.ma.getmaskarray(v)
        if empty_color is not None and (~ok).any():
            draw(~ok, 0.5, color=empty_color, linewidths=0, zorder=st["zorder"])
        m = draw(ok, c=v.compressed(), norm=norm, **st)
        if colorbar and ok.any():
            add_colorbar(ax, m, label=clabel)
    if title:
        ax.set_title(title)
    return ax
