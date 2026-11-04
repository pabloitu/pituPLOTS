from pathlib import Path

import numpy as np

from pituplots.errors import need
from pituplots.io.containers import Grid
from pituplots.scene import full_bounds, to_scene
from pituplots.style.defaults import resolve

KINDS = ("dem", "surface", "shapes", "mesh")


def quads(valid):
    """VTK faces of the quads of a ``(ny, nx)`` node lattice whose four nodes are valid."""
    ny, nx = valid.shape
    i = np.arange(ny * nx).reshape(ny, nx)
    ok = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, 1:] & valid[1:, :-1]
    q = np.stack([i[:-1, :-1], i[:-1, 1:], i[1:, 1:], i[1:, :-1]], -1)[ok]
    return np.c_[np.full(len(q), 4), q].ravel()


def inside(plotter, pts, pad):
    """Points within the events' bounds, widened by ``pad`` of their span."""
    b = np.array(full_bounds(plotter), float).reshape(3, 2)
    w = (b[:2, 1] - b[:2, 0]) * pad
    return ((pts[:, 0] >= b[0, 0] - w[0]) & (pts[:, 0] <= b[0, 1] + w[0]) & (pts[:, 1] >= b[1, 0] - w[1])
            & (pts[:, 1] <= b[1, 1] + w[1]))


def register(plotter, name, actor, kind, **info):
    plotter._pituplots["layers"][name] = {"actor": actor, "kind": kind, "mesh": actor.mapper.dataset, **info}
    if plotter._pituplots.get("section"):
        from pituplots.scene import clip_layers

        clip_layers(plotter)
    return actor


def as_grid(data, max_px, extent=None):
    from pituplots.io.rasters import read_geotiff, read_slab2

    if isinstance(data, Grid):
        return data
    p = Path(data)
    return read_slab2(p) if p.suffix.lower() == ".xyz" else read_geotiff(p, extent=extent, max_px=max_px)


def add_surface(plotter, data, crs=None, z="depth", z_unit="km", name=None, color_by="z", crop=0.15,
                max_px=1500, extent=None, colorbar=False, family="surface", **kw):
    """A gridded surface in a 3D scene: a slab top, a fault, a horizon.

    Parameters
    ----------
    plotter : pyvista.Plotter
        A scene from ``plot_catalog_3d``.
    data : Grid, str or Path
        A ``Grid``, a GeoTIFF (or any raster GDAL reads), or a Slab2 ``.xyz``.
    crs : str, optional
        Default the grid's.
    z : {"depth", "elevation"}
        What the values are.
    z_unit : {"km", "m"}
    name : str, optional
        Default the file name.
    color_by : {"z"} or None
        Colour by the values, or one ``color``.
    crop : float or None
        Keep the part within the events' bounds widened by this fraction.
    max_px : int
        Rasters are decimated to this many pixels on the longest side.
    extent : list, optional
        ``[lon_min, lon_max, lat_min, lat_max]`` window of a raster to read.
    colorbar : bool
    **kw
        Style: ``cmap``, ``clim``, ``color``, ``opacity``, ``ambient``,
        ``specular``, ``nan_color``.

    Returns
    -------
    pyvista.Actor
    """
    pv = need("pyvista", "3d")
    st = resolve(family, **kw)
    g = as_grid(data, max_px, extent)
    if g.values.ndim != 2:
        raise ValueError("a surface needs one band of values")
    k = 1000 if z_unit == "km" else 1
    v = np.where(g.mask, np.nan, g.values) if g.mask is not None else g.values.astype(float)
    up = (-v if z == "depth" else v) * k
    xc, yc = (g.x_edges[:-1] + g.x_edges[1:]) / 2, (g.y_edges[:-1] + g.y_edges[1:]) / 2
    x, y = np.meshgrid(xc, yc)
    pts = to_scene(plotter, x, y, np.nan_to_num(up), crs or g.crs)
    valid = np.isfinite(v).ravel() & np.isfinite(pts).all(1)
    if crop is not None:
        valid &= inside(plotter, pts, crop)
    faces = quads(valid.reshape(v.shape))
    if not faces.size:
        raise ValueError(f"{name or 'surface'} has no cells inside the scene")
    title = f"{'Depth' if z == 'depth' else 'Elevation'} ({z_unit})"
    mesh = pv.PolyData(np.nan_to_num(pts), faces=faces)
    mesh[title] = np.nan_to_num(v).ravel()
    mesh = mesh.clean()
    look = {"opacity": st["opacity"], "ambient": st["ambient"], "specular": st["specular"], "smooth_shading": True,
            "show_scalar_bar": False, "reset_camera": False, "pickable": False}
    if color_by == "z":
        look.update(scalars=title, cmap=st["cmap"], clim=st.get("clim"), nan_color=st["nan_color"])
    else:
        look["color"] = st["color"]
    name = name or (Path(data).stem if not isinstance(data, Grid) else family)
    a = plotter.add_mesh(mesh, name=name, **look)
    if colorbar and color_by == "z":
        plotter.add_scalar_bar(title=title, mapper=a.mapper, vertical=True, color=resolve("scene")["text_color"])
    info = {}
    if z == "elevation":
        info = {"grid": (xc, yc, v * k, crs or g.crs)}
        plotter._pituplots.setdefault("dems", []).append(info["grid"])
    return register(plotter, name, a, family, **info)


def add_dem(plotter, data, crs=None, z_unit="m", name=None, **kw):
    """A digital elevation model in a 3D scene, also used to drape shapes.

    Parameters
    ----------
    plotter : pyvista.Plotter
    data : Grid, str or Path
        A GeoTIFF of elevations, positive up, with bathymetry negative.
    crs : str, optional
        Default the file's.
    z_unit : {"m", "km"}
    name : str, optional
    **kw
        As in ``add_surface``; the style family is ``dem``.

    Returns
    -------
    pyvista.Actor
    """
    return add_surface(plotter, data, crs=crs, z="elevation", z_unit=z_unit, name=name or (
        Path(data).stem if not isinstance(data, Grid) else "dem"), family="dem", **kw)


def drape(plotter, x, y, crs, lift=0.0):
    """Elevations in metres of points on the DEMs of a scene, the first that covers each; NaN off them."""
    from scipy.interpolate import RegularGridInterpolator

    from pituplots.scene import from_frame, transformer

    h = np.full(np.shape(x), np.nan)
    if crs == "local":
        fr = plotter._pituplots["frame"]
        if fr["kind"] == "local":
            return h
        (x, y), crs = from_frame(fr, x, y), fr["crs"]
    for xc, yc, v, dcrs in plotter._pituplots.get("dems", []):
        u, w = transformer(crs, dcrs).transform(x, y) if crs != dcrs else (x, y)
        f = RegularGridInterpolator((yc, xc), v, bounds_error=False, fill_value=np.nan)
        h = np.where(np.isfinite(h), h, f(np.c_[np.asarray(w, float), np.asarray(u, float)]))
    return h + lift


def lines(geom):
    """Coordinate arrays of the lines and rings of a shapely geometry."""
    if geom is None or geom.is_empty:
        return []
    if hasattr(geom, "geoms"):
        return [c for g in geom.geoms for c in lines(g)]
    if geom.geom_type == "Polygon":
        return [np.asarray(geom.exterior.coords)] + [np.asarray(r.coords) for r in geom.interiors]
    if geom.geom_type in ("LineString", "LinearRing"):
        return [np.asarray(geom.coords)]
    return []


def add_shapes(plotter, data, crs=None, z=0.0, name=None, crop=0.15, step=None, **kw):
    """Lines and polygon outlines of a vector file in a 3D scene, flat or draped.

    Parameters
    ----------
    plotter : pyvista.Plotter
    data : str, Path or list
        A shapefile or GeoJSON, or shapely geometries (with ``crs``).
    crs : str, optional
        Default the file's, else EPSG:4326.
    z : float or "dem"
        Elevation in metres, or ``"dem"`` to drape on the scene's DEM
        (sea level off it).
    name : str, optional
    crop : float or None
        As in ``add_surface``.
    step : float, optional
        Densify segments longer than this, in the data units. Default
        0.05 for geographic data, none otherwise.
    **kw
        Style: ``color``, ``line_width``, ``opacity``, ``lift`` (metres
        above the DEM).

    Returns
    -------
    pyvista.Actor
    """
    pv = need("pyvista", "3d")
    import shapely

    from pituplots.io.vectors import read_vector

    st = resolve("shapes", **kw)
    if isinstance(data, (str, Path)):
        geoms, _, fcrs = read_vector(data)
        crs = crs or fcrs or "EPSG:4326"
        name = name or Path(data).stem
    else:
        geoms, crs, name = data, crs or "EPSG:4326", name or "shapes"
    from pyproj import CRS

    step = step if step is not None else (0.05 if crs != "local" and CRS(crs).is_geographic else None)
    if step:
        geoms = [None if g is None else shapely.segmentize(g, step) for g in geoms]
    pts, cells, n = [], [], 0
    for c in (c for g in geoms for c in lines(g)):
        h = drape(plotter, c[:, 0], c[:, 1], crs, st["lift"]) if z == "dem" else np.full(len(c), float(z))
        q = to_scene(plotter, c[:, 0], c[:, 1], np.nan_to_num(h), crs)
        keep = np.isfinite(q).all(1) & (inside(plotter, q, crop) if crop is not None else True)
        for run in np.split(np.arange(len(q)), np.flatnonzero(np.diff(keep.astype(int))) + 1):
            if keep[run[0]] and len(run) > 1:
                pts.append(q[run])
                i = np.arange(n, n + len(run))
                cells.append(np.c_[np.full(len(run) - 1, 2), i[:-1], i[1:]].ravel())
                n += len(run)
    if not pts:
        raise ValueError(f"{name} has no lines inside the scene")
    mesh = pv.PolyData(np.vstack(pts), lines=np.concatenate(cells))
    a = plotter.add_mesh(mesh, color=st["color"], line_width=st["line_width"], opacity=st["opacity"], name=name,
                         reset_camera=False, pickable=False)
    return register(plotter, name, a, "shapes")


def add_mesh(plotter, data, crs="local", z_unit="m", name=None, **kw):
    """A mesh (VTK, STL, OBJ, PLY...) in a 3D scene: mine workings, faults, models.

    Parameters
    ----------
    plotter : pyvista.Plotter
    data : str, Path or pyvista.DataSet
    crs : str
        Of the mesh x and y; ``"local"`` for the scene's own grid.
    z_unit : {"m", "km"}
        Of the mesh z, positive up.
    name : str, optional
    **kw
        Style: ``color``, ``opacity``, ``scalars``, ``cmap``,
        ``show_edges``.

    Returns
    -------
    pyvista.Actor
    """
    pv = need("pyvista", "3d")
    st = resolve("mesh", **kw)
    m = pv.read(str(data)) if isinstance(data, (str, Path)) else data.copy()
    k = 1000 if z_unit == "km" else 1
    m.points = to_scene(plotter, m.points[:, 0], m.points[:, 1], m.points[:, 2] * k, crs)
    name = name or (Path(data).stem if isinstance(data, (str, Path)) else "mesh")
    look = {"opacity": st["opacity"], "show_edges": st["show_edges"], "smooth_shading": True,
            "reset_camera": False, "pickable": False, "show_scalar_bar": False}
    look.update({"scalars": st["scalars"], "cmap": st["cmap"]} if st.get("scalars") else {"color": st["color"]})
    a = plotter.add_mesh(m, name=name, **look)
    return register(plotter, name, a, "mesh")


def add_layers(plotter, layers, base=None):
    """Add the layers of a config to a scene.

    Parameters
    ----------
    plotter : pyvista.Plotter
    layers : list of dict
        Each with ``kind`` (``dem``, ``surface``, ``shapes``, ``mesh``),
        ``path``, and the keywords of that function.
    base : str or Path, optional
        Directory that relative paths are resolved against, such as the
        config's.

    Returns
    -------
    list of str
        The names of the layers added.
    """
    out = []
    for item in layers or []:
        item = dict(item)
        kind = item.pop("kind", None)
        if kind not in KINDS:
            raise ValueError(f"layer kind must be one of {KINDS}, not {kind!r}")
        path = Path(item.pop("path"))
        path = path if path.is_absolute() or base is None else Path(base) / path
        f = {"dem": add_dem, "surface": add_surface, "shapes": add_shapes, "mesh": add_mesh}[kind]
        f(plotter, path, **item)
        out.append(item.get("name", path.stem))
    return out
