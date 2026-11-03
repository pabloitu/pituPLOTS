import warnings

import cartopy.crs as ccrs
import numpy as np
import pandas as pd
from shapely.geometry.polygon import orient

from pituplots.errors import PituplotsWarning
from pituplots.io.containers import Catalog, Grid, Sites, find_column, unwrap_lon
from pituplots.style.defaults import ZORDER, resolve

__all__ = ["add_gridlines", "central_lon", "extent_from", "make_map", "seam_split", "set_map_extent",
           "unwrap_lon"]

MAP_KW = ("extent", "projection", "basemap", "coastline", "borders", "gridlines", "tile_depth",
          "set_global", "figsize")


def extent_from(data, pad=0.05):
    """Map extent of a dataset.

    Parameters
    ----------
    data : Catalog, Sites, Grid, DataFrame, tuple or bbox
        A tuple is ``(lon, lat, ...)`` or ``(values, x, y)``. A bbox is
        ``[lon_min, lon_max, lat_min, lat_max]``, and is taken as written. A
        ``Grid`` counts its unmasked cells, and is converted to degrees when
        its ``crs`` is projected.
    pad : float
        Margin on each side, as a fraction of the span.

    Returns
    -------
    list or None
        ``[lon_min, lon_max, lat_min, lat_max]`` with longitudes in -180..180,
        so ``lon_min > lon_max`` means the extent crosses the dateline. A span
        of 360 deg gives ``[-180, 180, ...]``. ``None`` if there are no
        finite positions.
    """
    bbox = isinstance(data, (list, tuple)) and len(data) == 4 and np.ndim(data[0]) == 0
    if bbox:
        x0, x1, y0, y1 = map(float, data)
        x1 += 360 * (x1 < x0)
    else:
        if isinstance(data, Grid):
            keep = np.ones(data.values.shape[:2], bool) if data.mask is None else ~data.mask
            keep = keep if keep.any() else ~keep
            i, j = keep.any(axis=0), keep.any(axis=1)
            x = np.r_[data.x_edges[:-1][i], data.x_edges[1:][i]]
            y = np.r_[data.y_edges[:-1][j], data.y_edges[1:][j]]
            if data.crs != "EPSG:4326":
                from pyproj import Transformer

                t = np.linspace(0, 1, 50)
                xs = x.min() + t * np.ptp(x)
                ys = y.min() + t * np.ptp(y)
                x = np.r_[xs, xs, xs * 0 + xs[0], xs * 0 + xs[-1]]
                y = np.r_[ys * 0 + ys[0], ys * 0 + ys[-1], ys, ys]
                x, y = Transformer.from_crs(data.crs, "EPSG:4326", always_xy=True).transform(x, y)
        elif isinstance(data, (Catalog, Sites)):
            x, y = data.lon, data.lat
        elif isinstance(data, pd.DataFrame):
            x, y = find_column(data, "lon").to_numpy(float), find_column(data, "lat").to_numpy(float)
        elif isinstance(data, tuple):
            a = [np.asarray(v, float) for v in data]
            x, y = a[1:3] if a[0].ndim == 2 else a[:2]
        else:
            raise TypeError(f"cannot get an extent from {type(data).__name__}")
        x, y = np.asarray(x, float), np.asarray(y, float)
        x, y = unwrap_lon(x[np.isfinite(x)]), y[np.isfinite(y)]
        if x.size == 0 or y.size == 0:
            return None
        x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()

    dx, dy = pad * ((x1 - x0) or 1), pad * ((y1 - y0) or 1)
    y0, y1 = max(y0 - dy, -90), min(y1 + dy, 90)
    if x1 - x0 + 2 * dx >= 360:
        return [-180, 180, float(y0), float(y1)]
    x0 = (x0 - dx + 180) % 360 - 180
    x1 = 180 - (180 - x1 - dx) % 360
    return [float(x0), float(x1), float(y0), float(y1)]


def central_lon(proj):
    # cartopy 0.26 keeps it in pm for PlateCarree, and epsg projections have neither
    p = proj.proj4_params
    if "lon_0" in p or "pm" in p:
        return p.get("lon_0", p.get("pm"))
    op = proj.coordinate_operation
    for q in op.params if op else ():
        if q.name.startswith("Longitude of"):
            return q.value
    return 0


def seam_split(z, x, c=0):
    """Cut a lattice at the map seam, ``c + 180``.

    Returns ``(z, x)`` pieces, each with edges in ``PlateCarree(c)`` native
    longitude. A seam inside a cell repeats that cell on both sides. ``z`` may
    be masked or an image; the pieces are views.
    """
    x = np.asarray(x, float)
    k0, k1 = np.floor((x[[0, -1]] - c + 180) / 360).astype(int)
    pieces = []
    for k in range(k0, k1 + 1):
        lo = c - 180 + 360 * k
        i = np.flatnonzero((x[1:] > lo + 1e-9) & (x[:-1] < lo + 360 - 1e-9))
        if i.size:
            e = np.clip(x[i[0]:i[-1] + 2], lo, lo + 360) - lo - 180
            pieces.append((z[:, i[0]:i[-1] + 1], e))
    return pieces


def cartopy_crs(crs):
    """Cartopy CRS of an EPSG string, WKT or pyproj CRS.

    Cartopy limits a projected CRS to its EPSG area of use, so NZTM stops at
    178.6 deg E. A regional CRS gets bounds 20 deg wider in longitude and 10 deg
    in latitude instead; a global one keeps cartopy's.
    """
    from pyproj import CRS

    c = CRS.from_user_input(crs)
    if c.is_geographic:
        return ccrs.PlateCarree()
    code = c.to_epsg()
    c = CRS.from_epsg(code) if code else c
    p = ccrs.Projection(c)
    a = c.area_of_use
    w = (-180, 180, -85, 85) if a is None else (a.west - 20, a.east + 20, max(a.south - 10, -85), min(a.north + 10, 85))
    if p.bounds is not None and w[1] - w[0] >= 340:
        return p
    lon, lat = np.meshgrid(np.linspace(w[0], w[1], 60), np.linspace(w[2], w[3], 60))
    xy = p.transform_points(p.as_geodetic(), lon.ravel(), lat.ravel())[:, :2]
    xy = xy[np.isfinite(xy).all(axis=1)]
    p.bounds = (xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max())
    p.threshold = min(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])) / 100
    return p


def native(ax, lon):
    """Longitudes on the axes' native branch, and the ``transform`` keyword to draw them."""
    if not hasattr(ax, "projection"):
        return np.asarray(lon, float), {}
    c = central_lon(ax.projection)
    x = (np.asarray(lon, float) - c + 180) % 360 - 180
    return x, {"transform": ccrs.PlateCarree(central_longitude=c)}


def set_map_extent(ax, extent):
    """Apply ``[lon_min, lon_max, lat_min, lat_max]`` on a GeoAxes or a plain axes."""
    x0, x1, y0, y1 = map(float, extent)
    if not y0 < y1:
        raise ValueError(f"extent needs lat_min < lat_max, got {list(extent)}")
    if not hasattr(ax, "projection"):
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        return
    x1 += 360 * (x1 < x0)
    if x1 - x0 >= 360:
        ax.set_global()
        return
    c = central_lon(ax.projection)
    a = (x0 - c + 180) % 360 - 180
    if a + x1 - x0 > 180 + 1e-9:
        raise ValueError(f"extent {list(extent)} crosses the seam of this projection, at "
                         f"{c + 180:g}: use a central longitude inside the extent")
    try:
        ax.set_extent([a, a + x1 - x0, y0, y1], crs=ccrs.PlateCarree(central_longitude=c))
    except ValueError as e:
        raise ValueError(f"extent {list(extent)} is outside the bounds of this projection; an EPSG "
                         "projection is limited to its area of use") from e


def add_natural_earth(ax, category, name, scale, **kw):
    from cartopy.feature import NaturalEarthFeature
    from cartopy.io import shapereader

    try:
        shapereader.natural_earth(scale, category, name)
    except OSError as e:
        warnings.warn(f"no Natural Earth {name} at {scale}, skipped: {e}", PituplotsWarning,
                      stacklevel=3)
        return None
    return ax.add_feature(NaturalEarthFeature(category, name, scale), **{"facecolor": "none", **kw})


def make_map(extent=None, projection=None, basemap=None, coastline=True, borders=False,
             gridlines=True, tile_depth="auto", set_global=False, figsize=None):
    """Build a map.

    Parameters
    ----------
    extent : list, optional
        ``[lon_min, lon_max, lat_min, lat_max]`` in degrees, whatever the
        projection. ``lon_min > lon_max`` crosses the dateline.
    projection : cartopy CRS, "approx" or "native", optional
        ``None`` is PlateCarree, centred on 180 if the extent crosses the
        dateline. "approx" is PlateCarree with a cos-lat aspect. "native" is
        the CRS of the GeoTIFF ``basemap``, so nothing is resampled.
    basemap : optional
        Anything ``get_basemap`` accepts.
    coastline, borders : bool, str or dict
        A Natural Earth scale (``"10m"``, ``"50m"``, ``"110m"``) to override
        the one chosen from the extent, or a dict of ``color``, ``lw``,
        ``scale`` and other line keywords.
    gridlines : bool
    tile_depth : int or "auto"
    set_global : bool
    figsize : tuple, optional

    Returns
    -------
    GeoAxes
    """
    import matplotlib.pyplot as plt

    proj = map_projection(extent, projection, basemap)
    ax = plt.figure(figsize=figsize, layout="constrained").add_subplot(projection=proj)
    dress_map(ax, extent, projection, basemap, coastline, borders, gridlines, tile_depth, set_global)
    return ax


def map_projection(extent, projection, basemap):
    from pituplots.basemaps import basemap_crs

    crossing = extent is not None and extent[0] > extent[1]
    if projection == "native":
        if basemap is None:
            raise ValueError("projection='native' needs a GeoTIFF basemap")
        return basemap_crs(basemap)
    if projection is None or isinstance(projection, str) and projection == "approx":
        return ccrs.PlateCarree(central_longitude=180 if crossing else 0)
    if isinstance(projection, ccrs.CRS):
        return projection
    raise ValueError(f"unknown projection {projection!r}")


def dress_map(ax, extent=None, projection=None, basemap=None, coastline=True, borders=False, gridlines=True,
              tile_depth="auto", set_global=False):
    """Extent, aspect and layers of a new map axes, the second half of ``make_map``."""
    from pituplots.basemaps import add_basemap

    if set_global:
        ax.set_global()
    elif extent is not None:
        set_map_extent(ax, extent)
    if isinstance(projection, str) and projection == "approx":
        if extent is None:
            raise ValueError("projection='approx' needs an extent")
        ax.set_aspect(110.574 / (111.320 * np.cos(np.deg2rad((extent[2] + extent[3]) / 2))))

    span = 360
    if extent is not None and not set_global:
        span = max((extent[1] - extent[0]) % 360, extent[3] - extent[2])
    scale = "10m" if span <= 15 else "50m" if span <= 50 else "110m"
    if basemap is not None:
        add_basemap(ax, basemap, tile_depth=tile_depth)
    for spec, family, cat, name in ((coastline, "coast", "physical", "coastline"),
                                    (borders, "borders", "cultural", "admin_0_boundary_lines_land")):
        if not spec:
            continue
        st, sc = resolve(family), scale
        if isinstance(spec, str):
            sc = spec
        elif isinstance(spec, dict):
            spec = dict(spec)
            sc = spec.pop("scale", scale)
            st.update(spec)
        add_natural_earth(ax, cat, name, sc, edgecolor=st.pop("color"), linewidth=st.pop("lw"),
                          zorder=ZORDER["coast"], **st)
    if gridlines:
        add_gridlines(ax)


def add_gridlines(ax, labels=True, inline=False, **kw):
    """Gridlines with labels on the left and bottom.

    Parameters
    ----------
    ax : GeoAxes
    labels : bool
    inline : bool
        Put the labels on the lines, for conic and oblique maps.
    **kw
        ``fontsize`` of the labels. The rest passes to ``ax.gridlines``.

    Returns
    -------
    Gridliner
    """
    if not hasattr(ax, "projection"):
        raise TypeError("add_gridlines needs a GeoAxes")
    st = {"zorder": ZORDER["gridlines"], **resolve("gridlines", **kw)}
    fontsize = st.pop("fontsize", None)
    gl = ax.gridlines(draw_labels=labels, x_inline=inline, y_inline=inline, **st)
    gl.right_labels = gl.top_labels = False
    if fontsize is not None:
        gl.xlabel_style["fontsize"] = gl.ylabel_style["fontsize"] = fontsize
    return gl


def get_ax(data, ax, kw):
    """Axes for a ``plot_*`` call. Takes the map keywords out of ``kw``."""
    mk = {k: kw.pop(k) for k in MAP_KW if k in kw}
    if ax is None:
        if not mk.get("set_global") and mk.get("extent") is None:
            mk["extent"] = extent_from(data)
        return make_map(**mk)
    extent = mk.pop("extent", None)
    if mk:
        raise ValueError(f"{sorted(mk)} are ignored with ax given: use add_basemap or make_map")
    if extent is not None:
        set_map_extent(ax, extent)
    return ax


def polygon_path(geom):
    """Compound matplotlib path of a shapely polygon or multipolygon, holes included."""
    from matplotlib.path import Path

    ps = [orient(p) for p in getattr(geom, "geoms", [geom]) if p.geom_type == "Polygon"]
    rings = [r for p in ps for r in (p.exterior, *p.interiors)]
    return Path.make_compound_path(*[Path(np.asarray(r.coords)[:, :2], closed=True) for r in rings])


def clip_path(ax, geom):
    """Matplotlib path of a shapely polygon, in the data coordinates of ``ax``."""
    if hasattr(ax, "projection"):
        geom = ax.projection.project_geometry(geom, ccrs.PlateCarree())
    return polygon_path(geom)
