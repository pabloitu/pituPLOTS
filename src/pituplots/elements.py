import cartopy.crs as ccrs
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.font_manager import FontProperties
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from pituplots.maps import add_natural_earth, central_lon, native, set_map_extent
from pituplots.style.defaults import ZORDER, resolve

CBAR_RECT = {"right": [1.02, 0, 0.03, 1], "bottom": [0, -0.1, 1, 0.03],
             "inside": [0.93, 0.05, 0.025, 0.4]}
PANEL_LOC = {"upper right": (0.98, 0.98, "right", "top"), "upper left": (0.02, 0.98, "left", "top"),
             "lower right": (0.98, 0.02, "right", "bottom"), "lower left": (0.02, 0.02, "left", "bottom")}


def add_colorbar(ax, mappable=None, loc="right", cax=None, **kw):
    """Colorbar attached to the axes, in axes coordinates.

    Parameters
    ----------
    ax : Axes or list of Axes
        A list gets one shared bar, through ``fig.colorbar(ax=axs)``.
    mappable : ScalarMappable, optional
        Defaults to the newest scalar mappable on ``ax``, skipping basemaps.
    loc : {"right", "bottom", "inside"}
    cax : Axes, optional
        Draw the bar here instead.
    **kw
        Passed to ``fig.colorbar``.

    Returns
    -------
    Colorbar
    """
    many = isinstance(ax, (list, tuple, np.ndarray))
    axs = list(np.ravel(ax)) if many else [ax]
    if mappable is None:
        ms = [a for x in axs for a in [*x.collections, *x.images]
              if isinstance(a, ScalarMappable) and a.get_array() is not None
              and a.get_zorder() > ZORDER["basemap"]]
        if not ms:
            raise ValueError("no scalar mappable on the axes")
        mappable = ms[-1]
    fig = axs[0].figure
    if many:
        return fig.colorbar(mappable, ax=axs, **kw)
    if cax is None:
        if loc not in CBAR_RECT:
            raise ValueError(f"loc must be one of {list(CBAR_RECT)}, not {loc!r}")
        cax = ax.inset_axes(CBAR_RECT[loc])
        kw.setdefault("orientation", "horizontal" if loc == "bottom" else "vertical")
    return fig.colorbar(mappable, cax=cax, **kw)


def add_size_legend(ax, values, sizer, **kw):
    """Legend of marker sizes. It is added as an artist, so a later legend survives.

    Parameters
    ----------
    ax : Axes
    values : array_like
        Tick values.
    sizer : callable
        Maps values to marker areas, as used on the scatter.
    **kw
        ``color``, ``alpha``, ``edgecolor``, and ``Legend`` keywords.

    Returns
    -------
    Legend
    """
    st = resolve("size_legend", **kw)
    color, alpha, edge = st.pop("color"), st.pop("alpha"), st.pop("edgecolor")
    values = np.atleast_1d(np.asarray(values, float))
    areas = np.asarray(sizer(values), float)
    hs = [Line2D([0], [0], marker="o", lw=0, markersize=np.sqrt(a), markerfacecolor=color,
                 markeredgecolor=edge, markeredgewidth=0.8, alpha=alpha) for a in areas]
    lg = Legend(ax, hs, [f"{v:g}" for v in np.round(values, 1)], **st)
    lg.set_zorder(ZORDER["legend"])
    lg.set_clip_on(False)
    ax.add_artist(lg)
    return lg


def add_locator(ax, inset_extent, extent=None, rect=(0.005, 0.75, 0.24, 0.26), **kw):
    """Inset map with a box on the study area.

    Parameters
    ----------
    ax : GeoAxes
    inset_extent : list
        ``[lon_min, lon_max, lat_min, lat_max]`` of the inset.
        ``lon_min > lon_max`` crosses the dateline.
    extent : list, optional
        The box. Defaults to the current extent of ``ax``, so set that first.
    rect : sequence of float
        Inset position, in axes coordinates.
    **kw
        ``land``, ``ocean``, ``box``, ``box_lw``, ``line``, ``lw``. The rest
        passes to the box ``Rectangle``.

    Returns
    -------
    GeoAxes
    """
    st = resolve("locator", **kw)
    land, ocean, line, lw = st.pop("land"), st.pop("ocean"), st.pop("line"), st.pop("lw")
    box, box_lw = st.pop("box"), st.pop("box_lw")
    if not hasattr(ax, "projection"):
        raise TypeError("add_locator needs a GeoAxes")
    if extent is None:
        c = central_lon(ax.projection)
        x0, x1, y0, y1 = ax.get_extent(ccrs.PlateCarree(central_longitude=c))
        extent = [x0 + c, x1 + c, y0, y1]
    x0, x1, y0, y1 = map(float, extent)
    x1 += 360 * (x1 < x0)

    crs = ccrs.PlateCarree(central_longitude=180 if inset_extent[0] > inset_extent[1] else 0)
    a = ax.inset_axes(list(rect), projection=crs, zorder=ZORDER["legend"])
    set_map_extent(a, inset_extent)
    a.set_anchor("NW")
    scale = "110m" if max((inset_extent[1] - inset_extent[0]) % 360, inset_extent[3] - inset_extent[2]) > 50 else "50m"
    add_natural_earth(a, "physical", "land", scale, facecolor=land, edgecolor="none")
    add_natural_earth(a, "physical", "ocean", scale, facecolor=ocean, edgecolor="none")
    add_natural_earth(a, "cultural", "admin_0_boundary_lines_land", scale, edgecolor=line, linewidth=lw)
    add_natural_earth(a, "physical", "coastline", scale, edgecolor="black", linewidth=lw)

    (x,), tr = native(a, [x0])
    a.add_patch(Rectangle((x, y0), x1 - x0, y1 - y0, facecolor="none", edgecolor=box, linewidth=box_lw,
                          zorder=ZORDER["vectors"], **tr, **st))
    (xc,), tr = native(a, [(x0 + x1) / 2])
    a.plot(xc, (y0 + y1) / 2, marker="s", markersize=4, color=box, zorder=ZORDER["points"], **tr)
    return a


def add_scalebar(ax, km, loc=(0.05, 0.05), unit_km=1.0, **kw):
    """Horizontal scale bar of a true length. Set the extent first.

    Parameters
    ----------
    ax : Axes
    km : float
        Bar length. On a ``GeoAxes`` it is geodesic, at the start of the bar.
    loc : (float, float)
        Start of the bar, in axes coordinates.
    unit_km : float
        Kilometres per data unit of a plain axes. Ignored on a ``GeoAxes``.
    **kw
        ``color``, ``lw``, ``halo``, ``halo_color``, ``fontsize``. The rest
        passes to ``plot``.

    Returns
    -------
    Line2D
    """
    if not km > 0:
        raise ValueError(f"km must be positive, got {km}")
    st = resolve("scalebar", **kw)
    halo, halo_color, fontsize = st.pop("halo"), st.pop("halo_color"), st.pop("fontsize")
    st.setdefault("zorder", ZORDER["labels"])
    xl, yl = ax.get_xlim(), ax.get_ylim()
    x = xl[0] + loc[0] * (xl[1] - xl[0])
    y = yl[0] + loc[1] * (yl[1] - yl[0])
    if hasattr(ax, "projection"):
        from pyproj import Geod

        pc = ccrs.PlateCarree()
        lon, lat = pc.transform_point(x, y, ax.projection)
        lon2, lat2, _ = Geod(ellps="WGS84").fwd(lon, lat, 90, 1000.0)
        x2, y2 = ax.projection.transform_point(lon2, lat2, pc)
        step = np.hypot(x2 - x, y2 - y)
        if not np.isfinite(step):
            raise ValueError(f"the bar starts outside the map, at {tuple(loc)}")
        dl = km * step
    else:
        dl = km / unit_km
    ln = ax.plot([x, x + dl], [y, y], solid_capstyle="butt", transform=ax.transData, scalex=False,
                 scaley=False, **st)[0]
    ef = [pe.withStroke(linewidth=halo, foreground=halo_color)] if halo else []
    if halo:
        ln.set_path_effects([pe.withStroke(linewidth=ln.get_linewidth() + halo, foreground=halo_color)])
    ax.annotate(f"{km:g} km", (x + dl / 2, y), xytext=(0, 3), textcoords="offset points", ha="center",
                va="bottom", fontsize=fontsize, color=st["color"], zorder=st["zorder"], path_effects=ef)
    return ln


def add_labels(ax, lon, lat, text, priority=None, min_sep=None, nmax=None, **kw):
    """Text next to points, moved aside to avoid overlaps. Set the extent first.

    A label is placed in the first free position around its point, and gets a
    leader line if that is not the nearest one. Positions are computed for the
    axes as they are, so a later layout change can bring labels closer.

    Parameters
    ----------
    ax : Axes
        A map, or a plain axes such as a section.
    lon, lat : array_like
        Positions. On a plain axes, data coordinates.
    text : sequence of str
    priority : array_like, optional
        Higher values are kept first by ``min_sep`` and ``nmax``.
    min_sep : float, optional
        Drop points closer than this to a kept one, in the units of ``lon``.
    nmax : int, optional
        Keep at most this many.
    **kw
        ``color``, ``fontsize``, ``offset`` (points), ``halo``,
        ``halo_color``. The rest passes to ``annotate``.

    Returns
    -------
    list of Annotation
    """
    st = resolve("labels", **kw)
    dx, dy = st.pop("offset")
    halo, halo_color = st.pop("halo"), st.pop("halo_color")
    x, tr = native(ax, lon)
    y = np.asarray(lat, float)
    if not (len(text) == x.size == y.size):
        raise ValueError(f"lon, lat and text have lengths {x.size}, {y.size}, {len(text)}")
    p = np.zeros(x.size) if priority is None else np.asarray(priority, float)
    if p.shape != x.shape:
        raise ValueError(f"priority has shape {p.shape}, expected {x.shape}")

    keep = []
    for i in np.argsort(-p, kind="stable"):
        if not np.isfinite(x[i] + y[i]):
            continue
        if min_sep is None or all(np.hypot(x[i] - x[j], y[i] - y[j]) > min_sep for j in keep):
            keep.append(i)
        if len(keep) == nmax:
            break

    ax.apply_aspect()
    data = tr["transform"]._as_mpl_transform(ax) if tr else ax.transData
    fs = FontProperties(size=st["fontsize"]).get_size_in_points()
    h = 1.25 * fs
    placed, out = [], []
    st.setdefault("zorder", ZORDER["labels"])
    if halo:
        st["path_effects"] = [pe.withStroke(linewidth=halo, foreground=halo_color)]
    for i in sorted(keep, key=lambda i: y[i]):
        t = text[i]
        px, py = data.transform((x[i], y[i])) * 72.0 / ax.figure.dpi
        w = 0.65 * fs * len(t)
        spots = [(a, b) for k in range(3) for b in (dy + k * (h + dy), -(k + 1) * (h + dy))
                 for a in (dx, -dx - w)]
        spots = spots[:1] + spots[2:3] + spots[1:2] + spots[3:]
        for n, (ox, oy) in enumerate(spots):
            bx, by = px + ox, py + oy
            if all(bx > q[0] + q[2] or bx + w < q[0] or by > q[1] + q[3] or by + h < q[1] for q in placed):
                break
        placed.append((bx, by, w, h))
        arrow = {"arrowprops": {"arrowstyle": "-", "lw": 0.4, "color": "0.3"}} if n else {}
        out.append(ax.annotate(t, (x[i], y[i]), xycoords=data, xytext=(ox, oy), textcoords="offset points",
                               va="bottom", annotation_clip=True, **arrow, **st))
    return out


def add_panel_label(ax, text, loc="upper right", **kw):
    """Text in a corner of the axes, such as a model name or a panel letter.

    Parameters
    ----------
    ax : Axes
    text : str
    loc : {"upper right", "upper left", "lower right", "lower left"}
    **kw
        ``fontsize``. The rest passes to ``ax.text``.

    Returns
    -------
    Text
    """
    if loc not in PANEL_LOC:
        raise ValueError(f"loc must be one of {list(PANEL_LOC)}, not {loc!r}")
    x, y, ha, va = PANEL_LOC[loc]
    st = {"zorder": ZORDER["labels"], **resolve("panel_label", **kw)}
    return ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, **st)


def add_class_legend(ax, edges, colors, fmt="{:.2f}", decimal=".", numbered=False, open_ends=True, sep=" – ", **kw):
    """Legend of discrete classes, one patch per class.

    Parameters
    ----------
    ax : Axes or Figure
        A figure gets one legend for all its panels.
    edges : sequence of float
        ``n + 1`` class edges.
    colors : sequence of colors or Colormap
        ``n`` colours, or a colormap sampled at ``n`` points.
    fmt : str
        Format of each edge.
    decimal : str
        Decimal mark, e.g. ``","``.
    numbered : bool
        Prefix ``1:``, ``2:``, ...
    open_ends : bool
        Label the first class ``< b`` and the last ``≥ a``.
    sep : str
        Between the two edges of a class, e.g. ``" a "``.
    **kw
        ``Legend`` keywords, such as ``title``, ``loc`` or ``ncol``.

    Returns
    -------
    Legend
    """
    from matplotlib.patches import Patch

    e = list(edges)
    k = len(e) - 1
    if k < 1:
        raise ValueError("need at least two edges")
    if callable(colors):
        colors = [colors(i / max(k - 1, 1)) for i in range(k)]
    if len(colors) != k:
        raise ValueError(f"{len(colors)} colours for {k} classes")
    f = lambda x: fmt.format(x).replace(".", decimal)  # noqa: E731
    lab = [f"{f(a)}{sep}{f(b)}" for a, b in zip(e[:-1], e[1:])]
    if open_ends and k > 1:
        lab[0], lab[-1] = f"< {f(e[1])}", f"≥ {f(e[-2])}"
    if numbered:
        lab = [f"{i + 1}: {t}" for i, t in enumerate(lab)]
    hs = [Patch(facecolor=c, edgecolor="none", label=t) for c, t in zip(colors, lab)]
    if hasattr(ax, "transAxes"):
        lg = Legend(ax, hs, lab, **kw)
        lg.set_zorder(ZORDER["legend"])
        lg.set_clip_on(False)
        ax.add_artist(lg)
        return lg
    return ax.legend(handles=hs, **kw)
