import warnings

import numpy as np
import pandas as pd
import shapely
from matplotlib.collections import LineCollection, PathCollection
from shapely.affinity import translate
from shapely.geometry import LinearRing, LineString, Polygon, box
from shapely.geometry.base import BaseGeometry

from pituplots.elements import add_colorbar, add_labels
from pituplots.errors import PituplotsWarning
from pituplots.grids import as_sites, limits
from pituplots.io.containers import Grid
from pituplots.io.vectors import cells_to_outline
from pituplots.maps import central_lon, get_ax, native, polygon_path
from pituplots.style.defaults import ZORDER, resolve


def as_array(a):
    a = np.asarray(a, float)
    if a.ndim != 2 or a.shape[1] < 2 or len(a) < 2:
        raise ValueError(f"expected an (n, 2) array, got shape {a.shape}")
    return a[:, :2]


def as_geoms(data, crs, make):
    """Geometries of any vector input, with the attributes and positions of the non-null ones."""
    attrs = None
    if isinstance(data, tuple) and len(data) == 3 and isinstance(data[1], pd.DataFrame):
        data, attrs, file_crs = data
        crs = crs or file_crs
    elif hasattr(data, "geometry") and hasattr(data, "crs"):
        crs = crs or (None if data.crs is None else str(data.crs))
        data, attrs = list(data.geometry), data
    if isinstance(data, (BaseGeometry, np.ndarray)):
        data = [data]
    elif not isinstance(data, (list, tuple)):
        raise TypeError(f"unsupported vector data: {type(data).__name__}")
    gs, keep = [], []
    for i, g in enumerate(data):
        if g is None:
            continue
        gs.append(g if isinstance(g, BaseGeometry) else make(as_array(g)))
        keep.append(i)
    return gs, attrs, crs, np.array(keep, int), len(data)


def to_lonlat(g, crs, wrap):
    """2D geometry in degrees. With ``wrap``, a jump over 180 deg between vertices is a dateline crossing."""
    def f(xy):
        if crs not in (None, "EPSG:4326"):
            from pyproj import Transformer

            xy = np.c_[Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(*xy.T)]
        if wrap and xy.size and np.ptp(xy[:, 0]) < 360:
            xy = np.c_[np.unwrap(xy[:, 0], period=360), xy[:, 1]]
        return xy

    return shapely.transform(g, f)


def seam_cut(g, c):
    """Pieces of a geometry in ``PlateCarree(c)`` native longitude, cut at the seam."""
    if g.is_empty:
        return []
    x0, _, x1, _ = g.bounds
    k0, k1 = np.floor((np.array([x0, x1 - 1e-9]) - c + 180) / 360).astype(int)
    if k0 == k1:
        return [translate(g, -c - 360 * k0)]
    g = g if g.is_valid else shapely.make_valid(g)
    out = []
    for k in range(k0, k1 + 1):
        lo = c - 180 + 360 * k
        p = g.intersection(box(lo, -90, lo + 360, 90))
        if not p.is_empty:
            out.append(translate(p, -c - 360 * k))
    return out


def prepare(ax, gs, crs, kw, wrap=True):
    """Axes for a vector layer, one list of pieces per geometry, and their transform."""
    if ax is not None and not hasattr(ax, "projection"):
        return get_ax(None, ax, kw), [[shapely.force_2d(g)] for g in gs], {}
    ll = [to_lonlat(g, crs, wrap) for g in gs]
    xy = shapely.get_coordinates(ll)
    ax = get_ax((xy[:, 0], xy[:, 1]), ax, kw)
    if not hasattr(ax, "projection"):
        return ax, [[g] for g in ll], {}
    c = central_lon(ax.projection)
    return ax, [seam_cut(g, c) for g in ll], native(ax, [])[1]


def add(ax, col, tr):
    if tr:
        col.set_transform(tr["transform"]._as_mpl_transform(ax))
    geo = hasattr(ax, "projection")
    ax.add_collection(col, autolim=not geo)
    if not geo:
        ax.autoscale_view()


def segments(parts):
    """Coordinate arrays of each list of pieces, and the index of the piece list they came from."""
    segs, idx = [], []
    for i, ps in enumerate(parts):
        for p in ps:
            for q in getattr(p, "geoms", [p]):
                if q.geom_type in ("Polygon", "MultiPolygon"):
                    q = q.boundary
                for r in getattr(q, "geoms", [q]):
                    if r.geom_type in ("LineString", "LinearRing") and not r.is_empty:
                        segs.append(np.asarray(r.coords)[:, :2])
                        idx.append(i)
    return segs, np.array(idx, int)


def plot_region(data, ax=None, crs=None, **kw):
    """Outline of a region, with its holes and disjoint parts.

    Parameters
    ----------
    data : Grid, shapely polygon or array
        A ``Grid`` outlines its unmasked cells. An ``(n, 2)`` array is a ring,
        closed if it is not.
    ax : Axes, optional
        A map is built if missing, with the keywords of ``make_map``.
    crs : optional
        Of the coordinates, if not degrees. A ``Grid`` has its own.
    **kw
        ``color``, ``lw``, ``title``, ``zorder``. The rest passes to
        ``LineCollection``.

    Returns
    -------
    Axes
    """
    wrap = True
    if isinstance(data, Grid):
        gs, crs, wrap = [cells_to_outline(data)], data.crs, False
    elif isinstance(data, BaseGeometry):
        gs = [data]
    elif isinstance(data, np.ndarray):
        gs = [LinearRing(as_array(data))]
    else:
        raise TypeError(f"unsupported region data: {type(data).__name__}")
    ax, parts, tr = prepare(ax, gs, crs, kw, wrap)
    st = resolve("region", **kw)
    title = st.pop("title", None)
    st.setdefault("zorder", ZORDER["vectors"])
    color, lw = st.pop("color"), st.pop("lw")
    add(ax, LineCollection(segments(parts)[0], colors=color, linewidths=lw, **st), tr)
    if title:
        ax.set_title(title)
    return ax


def plot_points(data, ax=None, labels=None, label_kw=None, **kw):
    """Named points, such as towns or stations.

    Parameters
    ----------
    data : Sites, DataFrame or tuple
        ``(lon, lat)`` or ``(lon, lat, values)``; the values are not drawn.
    ax : Axes, optional
    labels : sequence of str or True, optional
        Text per point. True uses ``Sites.names``.
    label_kw : dict, optional
        Keywords of ``add_labels``, such as ``color``, ``fontsize``, ``halo``.
    **kw
        ``color``, ``edgecolor``, ``marker``, ``size``, ``offset``, ``title``,
        ``zorder``. The rest passes to ``scatter``.

    Returns
    -------
    Axes
    """
    s = as_sites(data)
    ax = get_ax(s, ax, kw)
    st = resolve("points", **kw)
    off, title = st.pop("offset"), st.pop("title", None)
    size = st.pop("size")
    st.setdefault("zorder", ZORDER["points"])
    x, tr = native(ax, s.lon)
    ax.scatter(x, s.lat, s=size, **tr, **st)
    if labels is True:
        labels = s.names
        if labels is None:
            raise ValueError("labels=True needs Sites with names")
    if labels is not None:
        add_labels(ax, s.lon, s.lat, list(labels), **{"offset": off, **(label_kw or {})})
    if title:
        ax.set_title(title)
    return ax


def plot_lines(data, ax=None, values=None, norm=None, lw_by=None, crs=None, colorbar=True, **kw):
    """Polylines, optionally coloured and scaled by a value. The largest values go on top.

    Parameters
    ----------
    data : shapely lines, ``read_vector`` result, GeoDataFrame or list of arrays
        Null geometries are skipped. Polygons draw their boundary.
    ax : Axes, optional
        A map is built if missing, with the keywords of ``make_map``.
    values : str or array_like, optional
        Attribute name, or one value per geometry.
    norm : Normalize, optional
        E.g. ``PowerNorm(0.5)`` for a nonlinear scale.
    lw_by : (float, float) or callable, optional
        Line width range over ``norm``, or a function of ``values``. With an
        explicit ``color``, widths vary and the colour stays fixed.
    crs : optional
        Of the coordinates, if the input does not state it. Any pyproj input.
    colorbar : bool
    **kw
        ``color``, ``lw``, ``cmap``, ``vmin``, ``vmax``, ``clabel``, ``title``,
        ``zorder``. The rest passes to ``LineCollection``.

    Returns
    -------
    Axes
    """
    gs, attrs, crs, keep, n = as_geoms(data, crs, LineString)
    ax, parts, tr = prepare(ax, gs, crs, kw)
    fixed = "color" in kw
    st = resolve("lines", **kw)
    color, lw, cmap = st.pop("color"), st.pop("lw"), st.pop("cmap")
    clabel, title = st.pop("clabel", None), st.pop("title", None)
    st.setdefault("zorder", ZORDER["vectors"])
    segs, idx = segments(parts)

    if values is None:
        if lw_by is not None:
            raise ValueError("lw_by needs values")
        for k in ("vmin", "vmax"):
            st.pop(k, None)
        add(ax, LineCollection(segs, colors=color, linewidths=lw, **st), tr)
    else:
        if isinstance(values, str):
            if attrs is None or values not in attrs:
                raise ValueError(f"no attribute {values!r}")
            v = np.asarray(attrs[values], float)
        else:
            v = np.asarray(values, float)
        if v.size != n:
            raise ValueError(f"{v.size} values for {n} geometries")
        v = np.ma.masked_invalid(v[keep])
        if v.size and v.count() == 0:
            warnings.warn("no plottable values in the lines", PituplotsWarning, stacklevel=2)
        norm = norm if norm is not None else limits(v, st)
        norm.autoscale_None(v)
        if lw_by is None:
            w = np.full(v.shape, float(lw))
        elif callable(lw_by):
            w = np.asarray(lw_by(v.filled(np.nan)), float)
        else:
            w = lw_by[0] + (lw_by[1] - lw_by[0]) * np.ma.clip(norm(v), 0, 1).filled(0)
        o = np.argsort(v[idx].filled(-np.inf), kind="stable")
        c = {"colors": color} if fixed else {"array": v[idx][o], "cmap": cmap, "norm": norm}
        col = LineCollection([segs[i] for i in o], linewidths=w[idx][o], **c, **st)
        add(ax, col, tr)
        if colorbar and v.count() and not fixed:
            add_colorbar(ax, col, label=clabel)
    if title:
        ax.set_title(title)
    return ax


def plot_polygons(data, ax=None, fill=None, edge=None, crs=None, values=None, colorbar=True, **kw):
    """Filled or outlined polygons, with holes.

    Parameters
    ----------
    data : shapely polygons, ``read_vector`` result, GeoDataFrame or list of arrays
        Null geometries are skipped.
    ax : Axes, optional
        A map is built if missing, with the keywords of ``make_map``.
    fill : color, optional
        Default no fill.
    edge : color, optional
    crs : optional
        Of the coordinates, if the input does not state it.
    values : str or array_like, optional
        Fill by value: an attribute name or one value per geometry.
    colorbar : bool
        With ``values``.
    **kw
        ``lw``, ``alpha``, ``cmap``, ``norm``, ``vmin``, ``vmax``,
        ``clabel``, ``title``, ``zorder``. The rest passes to
        ``PathCollection``.

    Returns
    -------
    Axes
    """
    gs, attrs, crs, keep, n = as_geoms(data, crs, Polygon)
    bad = {g.geom_type for g in gs} - {"Polygon", "MultiPolygon"}
    if bad:
        raise ValueError(f"plot_polygons takes polygons, got {sorted(bad)}")
    ax, parts, tr = prepare(ax, gs, crs, kw)
    st = resolve("polygons", **kw)
    style_edge = st.pop("edge")
    edge = style_edge if edge is None else edge
    lw, title = st.pop("lw"), st.pop("title", None)
    st.setdefault("zorder", ZORDER["vectors"])
    idx = [i for i, ps in enumerate(parts) for p in ps if not p.is_empty]
    paths = [polygon_path(p) for ps in parts for p in ps if not p.is_empty]
    fc = {"facecolors": "none" if fill is None else fill}
    if values is not None:
        if isinstance(values, str):
            if attrs is None or values not in attrs:
                raise ValueError(f"no attribute {values!r}")
            v = np.asarray(attrs[values], float)
        else:
            v = np.asarray(values, float)
        if v.size != n:
            raise ValueError(f"{v.size} values for {n} geometries")
        v = np.ma.masked_invalid(v[keep])
        clabel, cmap = st.pop("clabel", None), st.pop("cmap", "viridis")
        norm = st.pop("norm", None) or limits(v, st)
        fc = {"array": v[idx], "cmap": cmap, "norm": norm}
    col = PathCollection(paths, edgecolors=edge, linewidths=lw, **fc, **st)
    add(ax, col, tr)
    if values is not None and colorbar and v.count():
        add_colorbar(ax, col, label=clabel)
    if title:
        ax.set_title(title)
    return ax


def plot_sources(data, ax=None, color_by=None, log=False, colorbar=True, **kw):
    """OpenQuake sources on a map: fault traces, complex-fault edges, areas and points.

    Simple faults draw their trace, complex faults their top edge and, dashed,
    their bottom edge, area sources their outline, and point sources a marker.
    All share one colour scale.

    Parameters
    ----------
    data : DataFrame
        From ``io.oq.read_sources``.
    ax : Axes, optional
        A map is built if missing, with the keywords of ``make_map``.
    color_by : str, optional
        Numeric column, e.g. ``"rate"`` or ``"mmax"``.
    log : bool
        Colour by ``log10`` of the column.
    colorbar : bool
    **kw
        ``color``, ``lw``, ``cmap``, ``size``, ``vmin``, ``vmax``, ``clabel``,
        ``title``, ``zorder``.

    Returns
    -------
    Axes
    """
    if not isinstance(data, pd.DataFrame) or not {"kind", "geometry"} <= set(data.columns):
        raise TypeError("plot_sources takes the DataFrame of read_sources")
    d = data[data["geometry"].notna()]
    line = d["kind"] != "point"
    gs = [g.boundary if g.geom_type == "Polygon" else g for g in d["geometry"][line]]
    bottom = [LineString(e[-1][:, :2]) for e in d["edges"][line] if e is not None] if "edges" in d else []
    pts = d["geometry"][~line]
    ax, parts, tr = prepare(ax, gs + bottom + list(pts), None, kw)
    st = resolve("sources", **kw)
    color, lw, cmap, size = st.pop("color"), st.pop("lw"), st.pop("cmap"), st.pop("size")
    clabel, title = st.pop("clabel", None), st.pop("title", None)
    zo = st.pop("zorder", ZORDER["vectors"])
    v = None
    if color_by is not None:
        if color_by not in d:
            raise ValueError(f"no column {color_by!r}, have {list(d.columns)}")
        v = np.asarray(d[color_by], float)
        v = np.ma.masked_invalid(np.log10(np.where(v > 0, v, np.nan)) if log else v)
    norm = None if v is None else limits(v, st)

    k = sum(line)
    vl = None if v is None else v[line.to_numpy()]
    for ps, vals, ls in ((parts[:k], vl, "-"),
                         (parts[k:k + len(bottom)], None if v is None else vl[[e is not None for e in d["edges"][line]]],
                          "--")):
        segs, idx = segments(ps)
        c = LineCollection(segs, linewidths=lw, linestyles=ls, zorder=zo, **({"colors": color} if v is None else {}),
                           **st)
        if v is not None:
            c.set(array=vals[idx], cmap=cmap, norm=norm)
        add(ax, c, tr)
    m = None
    if len(pts):
        xy = np.array([[p.x, p.y] for ps in parts[k + len(bottom):] for p in ps if not p.is_empty])
        c = {"color": color} if v is None else {"c": v[~line.to_numpy()], "cmap": cmap, "norm": norm}
        m = ax.scatter(xy[:, 0], xy[:, 1], s=size, linewidths=0, zorder=zo, **c, **tr)
    if v is not None and colorbar and v.count():
        add_colorbar(ax, m if m is not None else ax.collections[-1], label=clabel)
    if title:
        ax.set_title(title)
    return ax
