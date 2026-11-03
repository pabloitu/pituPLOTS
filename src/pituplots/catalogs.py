from functools import partial

import numpy as np
import pandas as pd

from pituplots.elements import add_colorbar, add_size_legend
from pituplots.io.containers import ALIASES, Catalog, find_column
from pituplots.maps import get_ax, native
from pituplots.style.defaults import ZORDER, resolve


def size_by_value(values, size=6, max_size=220, power=4, min_val=None, max_val=None):
    """Marker areas from values.

    Parameters
    ----------
    values : array_like
    size, max_size : float
        Areas at ``min_val`` and ``max_val``.
    power : float
        Growth exponent. 0 gives a constant ``size``.
    min_val, max_val : float, optional
        Scale limits, by default the range of ``values``. Values outside are
        clipped, so ticks below ``min_val`` stay valid.

    Returns
    -------
    numpy.ndarray
    """
    v = np.asarray(values, float)
    if not np.isfinite(v).any():
        return np.full(v.shape, float(size))
    lo = np.nanmin(v) if min_val is None else min_val
    hi = np.nanmax(v) if max_val is None else max_val
    if not power or hi <= lo:
        return np.full(v.shape, float(size))
    n = (np.clip(np.nan_to_num(v, nan=lo), lo, hi) - lo) / (hi - lo)
    return size + n ** power * (max_size - size)


def as_catalog(data):
    if isinstance(data, Catalog):
        return data
    if isinstance(data, pd.DataFrame):
        lon, lat = find_column(data, "lon"), find_column(data, "lat")
        kw, used = {}, {lon.name, lat.name}
        for k in ("mag", "depth"):
            if any(str(c).lower() in ALIASES[k] for c in data.columns):
                kw[k] = find_column(data, k)
                used.add(kw[k].name)
        extra = {str(c): data[c].to_numpy() for c in data.columns if c not in used}
        return Catalog(lon.to_numpy(), lat.to_numpy(), extra=extra, **kw)
    if isinstance(data, tuple) and len(data) in (2, 3, 4):
        lon, lat, *rest = data
        depth, mag = (None, None, *rest)[-2:]
        return Catalog(lon, lat, depth, mag)
    raise TypeError(f"unsupported catalog data: {type(data).__name__}")


def by_name(cat, key):
    """Column of a catalog for ``color_by``: an attribute or ``extra`` name, or an array."""
    if isinstance(key, str):
        v = cat.extra.get(key, getattr(cat, key, None))
        if v is None:
            raise ValueError(f"no column {key!r}, have {['mag', 'depth', *cat.extra]}")
        return v
    v = np.asarray(key)
    if v.shape != cat.lon.shape:
        raise ValueError(f"color_by has shape {v.shape}, expected {cat.lon.shape}")
    return v


def draw_events(ax, x, y, s, st, color_by=None, hollow=None, **kw):
    """Scatter with a uniform, numeric or categorical colour. Returns the numeric mappable."""
    edge, face = st.pop("edgecolor"), st.pop("color")
    alpha, colors = st.pop("alpha", None), st.pop("colors", None)
    cmap, norm = st.pop("cmap", None), st.pop("norm", None)
    vmin, vmax = st.pop("vmin", None), st.pop("vmax", None)
    z = st.pop("zorder", ZORDER["catalogs"])
    base = {"alpha": alpha, "zorder": z, **kw, **st}
    hollow = np.zeros(x.shape, bool) if hollow is None else hollow

    def scatter(m, **c):
        for sel, h in ((m & ~hollow, False), (m & hollow, True)):
            if not sel.any():
                continue
            if h:
                f = c.get("color", face)
                a = {"facecolors": "none", "edgecolors": f if f not in ("none", None) else edge}
            else:
                a = {**c, "edgecolors": edge} if "c" in c else {"color": c.get("color", face),
                                                                "edgecolors": edge}
            if "c" in a:
                a["c"] = a["c"][sel]
                a.update(cmap=cmap, norm=norm)
                if norm is None:
                    a.update(vmin=vmin, vmax=vmax)
            yield ax.scatter(x[sel], y[sel], s=s[sel], **a, **base)

    if color_by is None:
        for _ in scatter(np.ones(x.shape, bool)):
            pass
        return None
    v = np.asarray(color_by)
    if np.issubdtype(v.dtype, np.number):
        vmin = np.nanmin(v) if vmin is None and v.size else vmin
        vmax = np.nanmax(v) if vmax is None and v.size else vmax
        ms = list(scatter(np.ones(x.shape, bool), c=v.astype(float)))
        return ms[0] if ms else None
    base.pop("label", None)
    for i, k in enumerate(pd.unique(v)):
        c = (colors or {}).get(k, f"C{i}")
        for _ in scatter(v == k, color=c):
            pass
        ax.scatter([], [], s=30, color=c, edgecolors=edge, alpha=alpha, label=str(k), zorder=z)
    return None


def plot_catalog(data, ax=None, color_by=None, cmap=None, sizer=None, mag_ticks=None, **kw):
    """Scatter of a catalog on a map, sized by magnitude.

    Parameters
    ----------
    data : Catalog, DataFrame or tuple
        A tuple is ``(lon, lat, mag)`` or ``(lon, lat, depth, mag)``.
    ax : Axes, optional
        A map is built if missing, with the keywords of ``make_map``.
    color_by : str or array_like, optional
        Column name or array. Numeric values use ``cmap`` and get a colorbar;
        others get one colour per category, set with ``colors={name: color}``,
        and a legend.
    cmap : str or Colormap, optional
    sizer : callable, optional
        Maps magnitudes to marker areas, e.g. ``partial(size_by_value,
        min_val=3, max_val=8)``, so several catalogs share a scale.
    mag_ticks : int or sequence, optional
        Size legend entries, or how many. Default 4.
    **kw
        Style: ``color``, ``edgecolor``, ``alpha``, ``size``, ``max_size``,
        ``power``, ``min_val``, ``max_val``, ``legend``, ``legend_loc``,
        ``legend_title``, ``colorbar``, ``clabel``, ``title``, ``zorder``. The
        rest passes to ``scatter``.

    Returns
    -------
    Axes

    Notes
    -----
    Sizes follow the magnitude range, not the number of events. The defaults
    suit a few hundred events; for thousands, lower ``alpha`` and ``size``, and
    ``linewidths``, or pass a ``sizer``.
    """
    cat = as_catalog(data)
    ax = get_ax(cat, ax, kw)
    st = resolve("catalog", **kw)
    legend, loc = st.pop("legend", True), st.pop("legend_loc")
    title = st.pop("legend_title", None)
    colorbar, clabel, ax_title = st.pop("colorbar", True), st.pop("clabel", None), st.pop("title", None)
    size = {k: st.pop(k) for k in ("size", "max_size", "power")}
    limits = {k: st.pop(k, None) for k in ("min_val", "max_val")}

    mag = cat.mag
    if sizer is None and mag is not None and mag.size:
        ticks = np.atleast_1d(mag_ticks) if isinstance(mag_ticks, (list, tuple, np.ndarray)) else np.array([])
        limits = {"min_val": np.nanmin(mag) if limits["min_val"] is None else limits["min_val"],
                  "max_val": np.nanmax(np.r_[mag, ticks]) if limits["max_val"] is None else limits["max_val"]}
        sizer = partial(size_by_value, **size, **limits)
    s = np.full(cat.lon.shape, float(size["size"])) if sizer is None or mag is None else np.asarray(sizer(mag), float)
    order = np.arange(cat.lon.size) if mag is None else np.argsort(mag, kind="stable")
    x, kw_t = native(ax, cat.lon)
    v = None if color_by is None else by_name(cat, color_by)
    if cmap is not None:
        st["cmap"] = cmap
    m = draw_events(ax, x[order], cat.lat[order], s[order], st,
                    None if v is None else v[order], **kw_t)

    if legend and mag is not None and mag.size:
        ticks = mag_ticks if isinstance(mag_ticks, (list, tuple, np.ndarray)) else np.linspace(
            np.nanmin(mag), np.nanmax(mag), mag_ticks or 4)
        ticks = np.unique(ticks) if np.ptp(mag) == 0 else ticks
        add_size_legend(ax, ticks, sizer, loc=loc, **({"title": title} if title else {}))
    if m is not None and colorbar:
        add_colorbar(ax, m, label=clabel)
    if v is not None and not np.issubdtype(np.asarray(v).dtype, np.number) and v.size:
        ax.legend()
    if ax_title:
        ax.set_title(ax_title)
    return ax


def highlight(data, ax, where=None, labels=None, sizer=None, **kw):
    """Ring selected events on an existing map.

    Parameters
    ----------
    data : Catalog, DataFrame or tuple
    ax : Axes
    where : array of bool, optional
        Which events. Default all.
    labels : sequence of str, optional
        One text per selected event.
    sizer : callable, optional
        The sizer of the underlying ``plot_catalog``. The ring is ``scale``
        times the marker area.
    **kw
        ``edgecolor``, ``linewidths``, ``scale``, ``size``. The rest passes
        to ``scatter``.

    Returns
    -------
    Axes
    """
    cat = as_catalog(data)
    keep = np.ones(cat.lon.shape, bool) if where is None else np.asarray(where, bool)
    if keep.shape != cat.lon.shape:
        raise ValueError(f"where has shape {keep.shape}, expected {cat.lon.shape}")
    st = resolve("highlight", **kw)
    scale, size = st.pop("scale"), st.pop("size", resolve("catalog")["size"])
    color = st.pop("color")
    s = np.full(cat.lon.shape, float(size)) if sizer is None or cat.mag is None else np.asarray(sizer(cat.mag), float)
    x, kw_t = native(ax, cat.lon)
    z = st.pop("zorder", ZORDER["catalogs"])
    ax.scatter(x[keep], cat.lat[keep], s=s[keep] * scale, facecolors=color, zorder=z, **st, **kw_t)
    if labels is not None:
        off = resolve("points")["offset"]
        for t, a, b in zip(labels, x[keep], cat.lat[keep]):
            kw_a = {"xycoords": kw_t["transform"]._as_mpl_transform(ax)} if kw_t else {}
            ax.annotate(t, xy=(a, b), xytext=off, textcoords="offset points", color=st.get("edgecolor"),
                        zorder=ZORDER["labels"], **kw_a)
    return ax
