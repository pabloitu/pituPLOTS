import numpy as np
from pyproj import CRS, Geod, Transformer
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import LineString
from shapely.ops import linemerge

from pituplots.catalogs import as_catalog, by_name, draw_events, size_by_value
from pituplots.elements import add_size_legend
from pituplots.io.containers import Grid, Profile, unwrap_lon
from pituplots.maps import native
from pituplots.style.defaults import ZORDER, resolve

GEOD = Geod(ellps="WGS84")


def frame(p):
    aeqd = CRS.from_proj4(f"+proj=aeqd +lat_0={p.lat0} +lon_0={p.lon0} +datum=WGS84 +units=km")
    a = np.deg2rad(p.azimuth)
    return Transformer.from_crs("EPSG:4326", aeqd, always_xy=True), np.sin(a), np.cos(a)


def project(lon, lat, profile):
    """Coordinates along and across a profile.

    Parameters
    ----------
    lon, lat : array_like
    profile : Profile

    Returns
    -------
    (x, y) : arrays in km
        ``x`` along the azimuth from the origin, ``y`` to the left of it.
        Exact at any latitude and across the antimeridian, through an
        azimuthal equidistant frame centred on the origin.
    """
    t, ux, uy = frame(profile)
    e, n = t.transform(np.asarray(lon, float), np.asarray(lat, float))
    return e * ux + n * uy, -e * uy + n * ux


def unproject(x, y, profile):
    t, ux, uy = frame(profile)
    x, y = np.asarray(x, float), np.asarray(y, float)
    return t.transform(x * ux - y * uy, x * uy + y * ux, direction="INVERSE")


def make_profile(p0, azimuth=None, p1=None, length=None, width=100.0):
    """Build a profile from its origin.

    Parameters
    ----------
    p0 : (lon, lat)
    azimuth : float, optional
        Degrees clockwise from north. Give this and ``length``, or ``p1``.
    p1 : (lon, lat), optional
        End point. Sets the azimuth, and the length if ``length`` is missing.
    length : float or (x0, x1), optional
        Km. A pair sets offsets from the origin, e.g. seaward of a trench.
    width : float
        Full swath width in km.

    Returns
    -------
    Profile
    """
    if (azimuth is None) == (p1 is None):
        raise ValueError("give either azimuth or p1")
    if p1 is not None:
        azimuth, _, d = GEOD.inv(p0[0], p0[1], p1[0], p1[1])
        azimuth %= 360
        length = d / 1000 if length is None else length
    if length is None:
        raise ValueError("azimuth needs length")
    x = (0.0, float(length)) if np.isscalar(length) else tuple(length)
    return Profile(float(p0[0]), float(p0[1]), float(azimuth), x, float(width))


def profiles_from_line(line, at, width, side=None, length=(0, 800), by="lat", smooth=30.0):
    """Profiles perpendicular to a line, such as a trench.

    Parameters
    ----------
    line : LineString, sequence of them, or (n, 2) array
    at : sequence of float
        Latitudes, or km along the line, by ``by``.
    width : float
        Full swath width in km.
    side : (lon, lat) or {"left", "right"}, optional
        Where positive distances point, by default to the right of the line
        as digitized. A point picks the side facing it, per profile.
    length : float or (x0, x1)
        Km.
    by : {"lat", "km"}
    smooth : float
        Half window in km for the line direction.

    Returns
    -------
    list of Profile
    """
    if by not in ("lat", "km"):
        raise ValueError(f"by must be 'lat' or 'km', not {by!r}")
    if not isinstance(line, LineString):
        if isinstance(line, np.ndarray):
            line = LineString(line)
        else:
            m = linemerge(list(line))
            line = m if m.geom_type == "LineString" else max(m.geoms, key=lambda g: g.length)
    c = np.asarray(line.coords)[:, :2]
    lon, lat = unwrap_lon(c[:, 0]), c[:, 1]
    d = GEOD.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])[2] / 1000
    s = np.r_[0, np.cumsum(d)]
    fine = np.linspace(0, s[-1], max(int(s[-1] / 2), 50))
    lo, la = np.interp(fine, s, lon), np.interp(fine, s, lat)

    out = []
    for v in np.atleast_1d(at):
        if by == "lat":
            if not la.min() <= v <= la.max():
                raise ValueError(f"latitude {v} is outside the line, {la.min():g} to {la.max():g}")
            j = int(np.abs(la - v).argmin())
        else:
            if not 0 <= v <= s[-1]:
                raise ValueError(f"{v} km is outside the line, 0 to {s[-1]:g}")
            j = int(np.abs(fine - v).argmin())
        i0, i1 = np.searchsorted(fine, [fine[j] - smooth, fine[j] + smooth])
        i1 = min(i1, fine.size - 1)
        az = GEOD.inv(lo[i0], la[i0], lo[i1], la[i1])[0]
        if side == "left":
            az -= 90
        elif side is None or side == "right":
            az += 90
        else:
            to = GEOD.inv(lo[j], la[j], side[0], side[1])[0]
            az += 90 if np.cos(np.deg2rad(to - az - 90)) >= 0 else -90
        out.append(make_profile((lo[j], la[j]), az % 360, length=length, width=width))
    return out


def plot_section(data, profile, ax=None, ve=None, color_by=None, hollow=None, sizer=None, **kw):
    """Depth section of the events inside a swath.

    Parameters
    ----------
    data : Catalog, DataFrame or tuple
        Needs depths.
    profile : Profile
    ax : Axes, optional
        A plain axes. Created if missing.
    ve : float, optional
        Vertical exaggeration, kept when the axes is resized.
    color_by : str or array_like, optional
        As in ``plot_catalog``.
    hollow : array of bool, optional
        Events drawn hollow, such as assigned depths. NaN depths are always
        hollow, at ``nan_depth`` (0).
    sizer : callable, optional
    **kw
        Style as in ``plot_catalog``; also ``zmax``, ``nan_depth``,
        ``figsize``, ``legend``, ``mag_ticks``. The rest passes to ``scatter``.

    Returns
    -------
    Axes
    """
    import matplotlib.pyplot as plt

    cat = as_catalog(data, geo=True)
    if cat.depth is None:
        raise ValueError("plot_section needs depths")
    if ax is None:
        ax = plt.figure(figsize=kw.pop("figsize", (10, 4.5)), layout="constrained").add_subplot()
    st = resolve("section", **kw)
    zmax, nan_depth, title = st.pop("zmax", None), st.pop("nan_depth", 0.0), st.pop("title", None)
    legend, ticks = st.pop("legend"), st.pop("mag_ticks", None)
    size = {k: st.pop(k) for k in ("size", "max_size", "power")}
    limits = {k: st.pop(k) for k in ("min_val", "max_val") if k in st}

    x, y = project(cat.lon, cat.lat, profile)
    x0, x1 = profile.x_range
    keep = (np.abs(y) <= profile.width / 2 + 1e-6) & (x >= x0) & (x <= x1)
    depth = cat.depth[keep]
    empty = ~np.isfinite(depth)
    h = empty if hollow is None else empty | np.asarray(hollow, bool)[keep]
    z = np.where(empty, nan_depth, depth)
    mag = None if cat.mag is None else cat.mag[keep]
    if sizer is None and mag is not None and mag.size:
        limits = {"min_val": np.nanmin(cat.mag), "max_val": np.nanmax(cat.mag), **limits}
        sizer = lambda v: size_by_value(v, **size, **limits)  # noqa: E731
    s = np.full(z.shape, float(size["size"])) if sizer is None or mag is None else np.asarray(sizer(mag), float)
    v = None if color_by is None else by_name(cat, color_by)[keep]
    o = np.arange(z.size) if mag is None else np.argsort(mag, kind="stable")
    if "cmap" not in st and v is not None and np.issubdtype(np.asarray(v).dtype, np.number):
        st["cmap"] = "viridis"
    m = draw_events(ax, x[keep][o], z[o], s[o], st, None if v is None else v[o], h[o])
    categorical = v is not None and m is None and v.size

    if legend and sizer is not None and mag.size:
        add_size_legend(ax, ticks if ticks is not None else np.linspace(np.nanmin(mag), np.nanmax(mag), 4), sizer)
    if categorical:
        ax.legend(fontsize="small", loc="lower left")
    ax.axhline(0, color="black", lw=0.6, zorder=ZORDER["gridlines"])
    ax.set_xlim(x0, x1)
    ax.set_ylim(zmax if zmax is not None else (np.max(z) * 1.05 if z.size and np.max(z) > 0 else 100),
                -5 if zmax is None else min(0, -0.02 * zmax))
    if ve is not None:
        ax.set_aspect(ve, adjustable="box", anchor="SW")
    ax.set_xlabel("distance along profile (km)")
    ax.set_ylabel("depth (km)")
    if title:
        ax.set_title(title)
    return ax


def sample(g, lon, lat):
    """Bilinear sample of a lat/lon ``Grid`` at cell centres. NaN off the grid."""
    xc, yc = (g.x_edges[:-1] + g.x_edges[1:]) / 2, (g.y_edges[:-1] + g.y_edges[1:]) / 2
    z = np.where(g.mask, np.nan, g.values) if g.mask is not None else g.values.astype(float)
    x = g.x_edges[0] + (np.asarray(lon) - g.x_edges[0]) % 360
    return RegularGridInterpolator((yc, xc), z, bounds_error=False)(np.column_stack([lat, x]))


def add_slab(ax, profile, top, thickness=None, step=5.0, swath=False, **kw):
    """Slab surfaces along a profile.

    Parameters
    ----------
    ax : Axes
        A section axes.
    profile : Profile
    top : Grid
        Depth of the slab top, positive down, as from ``read_slab2``.
    thickness : Grid, optional
        Adds the bottom surface, and fills the plate.
    step : float
        Sampling in km.
    swath : bool
        Also draw the top at the swath edges, dotted.
    **kw
        ``color``, ``lw``, ``bottom_color``, ``fill_color``, ``fill_alpha``.

    Returns
    -------
    Line2D
        The top surface.
    """
    if not isinstance(top, Grid):
        raise TypeError("top must be a Grid")
    st = resolve("slab", **kw)
    x0, x1 = profile.x_range
    s = np.arange(x0, x1 + step / 2, step)
    lon, lat = unproject(s, s * 0, profile)
    z = sample(top, lon, lat)
    zo = ZORDER["vectors"]
    if thickness is not None:
        b = z + sample(thickness, lon, lat)
        ax.fill_between(s, z, b, color=st["fill_color"], alpha=st["fill_alpha"], lw=0, zorder=1)
        ax.plot(s, b, color=st["bottom_color"], lw=st["lw"] * 0.7, zorder=zo, label="top + thickness")
    if swath:
        for t in (-profile.width / 2, profile.width / 2):
            ax.plot(s, sample(top, *unproject(s, s * 0 + t, profile)), color=st["color"], ls=":",
                    lw=st["lw"] * 0.6, zorder=zo)
    return ax.plot(s, z, color=st["color"], lw=st["lw"], zorder=zo, label="Slab2 top")[0]


def add_profile_map(ax, profiles, labels=None, **kw):
    """Profile lines and swaths on a map.

    Parameters
    ----------
    ax : Axes
    profiles : Profile or list of Profile
    labels : sequence of str, optional
    **kw
        ``color``, ``lw``, and ``plot`` keywords.

    Returns
    -------
    list of Line2D
        The centre lines.
    """
    ps = [profiles] if isinstance(profiles, Profile) else list(profiles)
    if labels is not None and len(labels) != len(ps):
        raise ValueError(f"{len(labels)} labels for {len(ps)} profiles")
    st = resolve("profile", **kw)
    out = []
    for i, p in enumerate(ps):
        s = np.linspace(*p.x_range, 30)
        for t, ls in ((p.width / 2, "--"), (-p.width / 2, "--"), (0, "-")):
            lon, lat = unproject(s, s * 0 + t, p)
            x, tr = native(ax, unwrap_lon(lon))
            if tr:
                x = unwrap_lon(x)
            ln = ax.plot(x, lat, ls=ls, zorder=ZORDER["vectors"], **tr, **st)[0]
        out.append(ln)
        if labels is not None:
            x, tr = native(ax, lon[[0]])
            kw_a = {"xycoords": tr["transform"]._as_mpl_transform(ax)} if tr else {}
            ax.annotate(labels[i], xy=(x[0], lat[0]), xytext=(-4, 0), textcoords="offset points",
                        ha="right", va="center", zorder=ZORDER["labels"], **kw_a)
    return out
