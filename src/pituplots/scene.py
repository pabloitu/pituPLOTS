import warnings
from functools import lru_cache

import numpy as np

from pituplots.catalogs import as_catalog, by_name
from pituplots.errors import PituplotsWarning, need
from pituplots.io.catalogs import load_config
from pituplots.io.containers import unwrap_lon
from pituplots.style.defaults import resolve

VIEWS = ("iso", "map", "ew", "ns")
KEYS = ("id", "hover", "color_by", "size_by", "true_size", "z_exag", "unit", "view", "projection", "color", "cmap", "clim",
        "colors", "palette", "size", "base", "stress_drop", "opacity")
HOVER = ["lon", "lat", "mag", "depth", "time", "class"]
TITLES = {"depth": "Depth (km)", "mag": "Magnitude", "time": "Year", "z": "Up ({})"}
ARGS = {"color_by": "depth", "size_by": "mag", "true_size": False, "z_exag": 1.0, "unit": "auto", "view": "iso",
        "projection": None}


def radius(mag, stress_drop=3e6):
    """Source radius in metres of a circular crack, ``(7 M0 / 16 dsigma)^(1/3)``.

    Parameters
    ----------
    mag : array_like
        Moment magnitude.
    stress_drop : float
        In Pa.

    Returns
    -------
    numpy.ndarray
    """
    m0 = 10 ** (1.5 * np.asarray(mag, float) + 9.1)
    return np.cbrt(7 * m0 / (16 * stress_drop))


@lru_cache(maxsize=32)
def transformer(src, dst):
    from pyproj import Transformer

    return Transformer.from_crs(src, dst, always_xy=True)


def make_frame(cat, projection=None):
    """The frame of a new scene, from its first catalog or a projection.

    Parameters
    ----------
    cat : Catalog
    projection : str or dict, optional
        A projected CRS, or a local grid defined from one: ``{"crs": ...,
        "a": ..., "b": ..., "tx": ..., "ty": ...}`` with
        ``x = a E + b N + tx`` and ``y = -b E + a N + ty`` (a 2D similarity,
        as mine grids are usually defined from UTM).

    Returns
    -------
    dict
        ``kind`` (``"crs"``, ``"grid"`` or ``"local"``), ``crs`` and ``grid``.
    """
    from pyproj import CRS

    if isinstance(projection, dict):
        bad = {"crs", "a", "b", "tx", "ty"} - set(projection)
        if bad:
            raise ValueError(f"a grid projection needs {sorted(bad)}")
        return {"kind": "grid", "crs": projection["crs"],
                "grid": tuple(float(projection[k]) for k in ("a", "b", "tx", "ty"))}
    if projection is not None:
        if projection == "local" or CRS(projection).is_geographic:
            raise ValueError(f"projection must be a projected CRS or a grid, not {projection!r}")
        return {"kind": "crs", "crs": projection}
    if cat.crs == "local":
        return {"kind": "local"}
    if not cat.geographic:
        return {"kind": "crs", "crs": cat.crs}
    lon, ok = unwrap_lon(cat.lon), np.isfinite(cat.lon) & np.isfinite(cat.lat)
    c = ((lon[ok].min() + lon[ok].max()) / 2, (cat.lat[ok].min() + cat.lat[ok].max()) / 2) if ok.any() else (0, 0)
    return {"kind": "crs", "crs": CRS(proj="aeqd", lon_0=c[0], lat_0=c[1], ellps="WGS84").to_wkt()}


def frame_xy(frame, x, y, crs):
    """Coordinates in ``crs`` to the frame of a scene, before the origin shift."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if crs == "local":
        if frame["kind"] == "crs":
            raise ValueError("a local layer needs a scene in a local grid")
        return x, y
    if frame["kind"] == "local":
        raise ValueError(f"the scene is a local grid with no reference CRS, so {crs!r} layers cannot be placed: "
                         "give it projection={'crs': ..., 'a': ..., 'b': ..., 'tx': ..., 'ty': ...}")
    if crs != frame["crs"]:
        x, y = transformer(crs, frame["crs"]).transform(x, y)
    x, y = np.asarray(x, float), np.asarray(y, float)
    if frame["kind"] == "grid":
        a, b, tx, ty = frame["grid"]
        x, y = a * x + b * y + tx, -b * x + a * y + ty
    return x, y


def from_frame(frame, x, y):
    """Frame coordinates back to the frame CRS: undoes the grid similarity."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if frame["kind"] != "grid":
        return x, y
    a, b, tx, ty = frame["grid"]
    d = a * a + b * b
    u, w = x - tx, y - ty
    return (a * u - b * w) / d, (b * u + a * w) / d


def to_scene(plotter, x, y, z, crs):
    """Scene coordinates of points.

    Parameters
    ----------
    plotter : pyvista.Plotter
        A scene from ``plot_catalog_3d``.
    x, y : array_like
        In ``crs``; ``"local"`` for the scene's own grid.
    z : array_like
        Up, in metres.
    crs : str

    Returns
    -------
    numpy.ndarray
        ``(n, 3)``, in the scene unit, with the exaggeration.
    """
    st = plotter._pituplots
    fx, fy = frame_xy(st["frame"], x, y, crs)
    k = 1000 if st["unit"] == "km" else 1
    z = np.broadcast_to(np.asarray(z, float), fx.shape)
    return np.c_[(fx.ravel() - st["origin"][0]) / k, (fy.ravel() - st["origin"][1]) / k,
                 z.ravel() / k * st["z_exag"]]


def plot_catalog_3d(data, plotter=None, color_by="depth", size_by="mag", true_size=False, z_exag=1.0,
                    unit="auto", view="iso", projection=None, axes=True, colorbar=True, pick=True, config=None,
                    off_screen=None, **kw):
    """Events of a catalog in 3D, as shaded spheres in a PyVista scene.

    Parameters
    ----------
    data : Catalog or DataFrame
        In any CRS (see ``Catalog.local_xyz``). Events without coordinates
        are left out, with a warning.
    plotter : pyvista.Plotter, optional
        A scene from an earlier call, whose origin, unit and exaggeration
        are reused so the layers line up. New if missing.
    color_by : str or array, optional
        ``"depth"``, ``"z"`` (up, in the scene unit), ``"mag"``, ``"time"``,
        an ``extra`` column, or None for one ``color``. Text columns are categories, coloured by ``colors``
        (``{category: colour}``) and then ``palette``, with a legend.
    size_by : str or array, optional
        Magnitude-like column. None, or ``"mag"`` without magnitudes, gives
        one size.
    true_size : bool
        Spheres of the radius of a circular crack with ``stress_drop`` (Pa),
        in metres, that grow and shrink with the zoom. Otherwise spheres
        keep their size on screen: ``classes`` magnitude classes from
        ``size[0]`` to ``size[1]`` pixels, growing exponentially, which reads
        well at any zoom and filters fast.
    z_exag : float
        Vertical exaggeration. The axis labels keep true values.
    unit : {"auto", "m", "km"}
        Scene unit; ``"auto"`` is km for spans over 50 km.
    view : {"iso", "map", "ew", "ns"}
        Camera of a new scene: oblique, from above, or facing the E–W or
        N–S plane; the last three without perspective.
    projection : str or dict, optional
        Projected CRS of a new scene, such as ``"EPSG:32719"``, or a local
        grid defined from one (see ``make_frame``), so layers in any CRS
        line up with a catalog in mine coordinates. Default an azimuthal
        equidistant projection at the centre of the first catalog, or the
        catalog CRS if it is projected or local already.
    axes, colorbar : bool
    pick : bool
        Hover labels, a click panel and flags (see ``add_picking``), with
        the ``id`` and ``hover`` of the config.
    config : str, Path or dict, optional
        A catalog config; its ``scene`` section sets the keywords left at
        their defaults.
    off_screen : bool, optional
        For a new plotter. Default PyVista's.
    **kw
        Style: ``color``, ``cmap``, ``clim``, ``colors``, ``palette``,
        ``nan_color``, ``size`` (pixels), ``classes``, ``point_size`` (one
        size, without magnitudes), ``stress_drop``, ``opacity``,
        ``ambient``, ``diffuse``, ``specular``, ``background``,
        ``background_top``, ``text_color``, ``font_size``, ``window_size``,
        and for true sizes ``max_glyphs`` (above it, point sprites instead of
        spheres) and ``resolution``.

    Returns
    -------
    pyvista.Plotter
        ``plotter._pituplots`` holds the origin, CRS, unit, exaggeration and
        the catalogs drawn, for later layers and picking.
    """
    pv = need("pyvista", "3d")
    import matplotlib as mpl

    sc = dict(load_config(config).get("scene") or {})
    bad = set(sc) - set(KEYS)
    if bad:
        raise ValueError(f"unknown scene keys {sorted(bad)}, expected {list(KEYS)}")
    ids, hover = sc.pop("id", None), sc.pop("hover", None)
    given = {"color_by": color_by, "size_by": size_by, "true_size": true_size, "z_exag": z_exag, "unit": unit,
             "view": view, "projection": projection}
    for k, d in ARGS.items():
        c = sc.pop(k, d)
        if isinstance(given[k], (str, bool, int, float, type(None))) and given[k] == d:
            given[k] = c
    st = resolve("scene", **{**sc, **kw})
    color_by, size_by, true_size, z_exag, unit, view, projection = given.values()
    if view not in VIEWS:
        raise ValueError(f"view must be one of {VIEWS}, not {view!r}")
    if unit not in ("auto", "m", "km"):
        raise ValueError(f"unit must be auto, m or km, not {unit!r}")

    cat = as_catalog(data)
    state = getattr(plotter, "_pituplots", None)
    new = state is None
    z = np.zeros(len(cat)) if cat.depth is None else -1000 * cat.depth
    if new:
        frame = make_frame(cat, projection)
        fx, fy = frame_xy(frame, cat.lon, cat.lat, cat.crs)
        f = np.isfinite(fx) & np.isfinite(fy) & np.isfinite(z)
        origin = ((fx[f].min() + fx[f].max()) / 2, (fy[f].min() + fy[f].max()) / 2) if f.any() else (0.0, 0.0)
        span = max(np.ptp(fx[f]), np.ptp(fy[f]), np.ptp(z[f])) if f.any() else 0
        state = {"frame": frame, "origin": (float(origin[0]), float(origin[1])), "crs": cat.crs, "layers": {},
                 "z_exag": float(z_exag), "unit": ("km" if span > 50e3 else "m") if unit == "auto" else unit,
                 "catalogs": []}
        plotter = pv.Plotter(off_screen=off_screen, window_size=st["window_size"]) if plotter is None else plotter
        plotter._pituplots = state
    xyz = to_scene(plotter, cat.lon, cat.lat, z, cat.crs)
    k = 1000 if state["unit"] == "km" else 1
    ok = np.isfinite(xyz).all(1)
    if not ok.all():
        warnings.warn(f"{(~ok).sum()} of {len(cat)} events without coordinates left out", PituplotsWarning,
                      stacklevel=2)
    idx = np.flatnonzero(ok)
    pts = xyz[ok]
    pc = pv.PolyData(pts)
    pc["index"] = idx

    # size
    m = None if size_by is None or (size_by == "mag" and cat.mag is None) else (
        np.asarray(by_name(cat, size_by), float)[ok])
    if true_size and m is None:
        raise ValueError("true_size needs size_by")
    px, cls = None, np.zeros(len(pts), int)
    if true_size:
        pc["radius"] = np.nan_to_num(radius(m, st["stress_drop"]) / k, nan=0.0)
    else:
        lo, hi = st["size"]
        f = np.isfinite(m) if m is not None else np.zeros(len(pts), bool)
        n = st["classes"]
        sizes = [float(st["point_size"])]
        if f.any() and np.ptp(m[f]) > 0:
            t = (np.where(f, m, np.nanmin(m)) - np.nanmin(m)) / np.ptp(m[f])
            cls = np.minimum((t * n).astype(int), n - 1)
            sizes = [lo * (hi / lo) ** (i / (n - 1)) for i in range(n)]
        px = np.asarray(sizes)[cls]
        pc["radius"] = np.zeros(len(pts))

    # colour
    p = plotter
    u = state["unit"]
    sbar = {"color": st["text_color"], "vertical": True, "title_font_size": st["font_size"] + 2,
            "label_font_size": st["font_size"]}
    look = {"opacity": st["opacity"], "show_scalar_bar": False, "ambient": st["ambient"], "diffuse": st["diffuse"],
            "specular": st["specular"], "reset_camera": new}
    legend, scalars, title = None, None, None
    if color_by is not None:
        v = pts[:, 2] / state["z_exag"] if isinstance(color_by, str) and color_by == "z" else (
            np.asarray(by_name(cat, color_by))[ok])
        title = color_by if isinstance(color_by, str) else "value"
        title = TITLES[title].format(u) if title in TITLES else title
        if np.issubdtype(v.dtype, np.datetime64):
            v = 1970 + v.astype("datetime64[ms]").astype("int64") / 86400e3 / 365.2425
        if v.dtype.kind in "OUSb":
            v = v.astype(str)
            given = dict(st.get("colors") or {})
            cats = [c for c in given if c in set(v)] + sorted(set(v) - set(given))
            cyc = mpl.colormaps["tab20" if st["palette"] == "tab10" and len(cats) > 10 else st["palette"]]
            rest = [c for c in cats if c not in given]
            col = {**{c: mpl.colors.to_rgb(given[c]) for c in cats if c in given},
                   **{c: cyc(i % cyc.N)[:3] for i, c in enumerate(rest)}}
            pc["rgb"] = np.array([col[c] for c in v]).reshape(-1, 3)
            look.update(scalars="rgb", rgb=True)
            legend = [[c, col[c]] for c in cats]
        else:
            v = v.astype(float)
            pc["value"] = v
            fin = v[np.isfinite(v)]
            clim = st.get("clim") or ((fin.min(), fin.max()) if fin.size else (0, 1))
            look.update(scalars="value", cmap=st["cmap"], clim=clim, nan_color=st["nan_color"])
            scalars = v
    else:
        look["color"] = st["color"]
    parts = []
    if not true_size:
        for c in np.unique(cls):
            sel = np.flatnonzero(cls == c)
            q = pv.PolyData(pts[sel])
            for name in pc.point_data.keys():
                q[name] = pc[name][sel]
            a = p.add_mesh(q, style="points", point_size=float(px[sel[0]]), render_points_as_spheres=True, **look)
            parts.append({"actor": a, "mesh": a.mapper.dataset, "cells": None, "sel": sel})
    elif len(pts) <= st["max_glyphs"]:
        sphere = pv.Sphere(radius=1, theta_resolution=st["resolution"][0], phi_resolution=st["resolution"][1])
        a = p.add_mesh(pc.glyph(geom=sphere, scale="radius", orient=False, factor=1.0), smooth_shading=True, **look)
        parts.append({"actor": a, "mesh": a.mapper.dataset, "cells": sphere.n_cells, "sel": np.arange(len(pts))})
    else:
        a = p.add_mesh(pc, style="points_gaussian", render_points_as_spheres=True, emissive=False, **look)
        a.mapper.scale_array = "radius"
        parts.append({"actor": a, "mesh": a.mapper.dataset, "cells": None, "sel": np.arange(len(pts))})
    for part in parts:
        part.update(scalars=look.get("scalars"), opacity=st["opacity"], mode=part["actor"].mapper.GetColorMode())
        if "scalars" in look:
            part["mesh"].set_active_scalars(look["scalars"])
            part["actor"].mapper.SetScalarModeToUsePointFieldData()
            part["actor"].mapper.SelectColorArray(look["scalars"])
    if "rgb" in pc.array_names:
        rgba = np.c_[np.asarray(pc["rgb"]) * 255, np.full(len(pts), 255)]
    elif "value" in pc.array_names:
        from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

        lut = parts[0]["actor"].mapper.GetLookupTable()
        rgba = vtk_to_numpy(lut.MapScalars(numpy_to_vtk(np.asarray(pc["value"], float)), 0, -1)).reshape(-1, 4)
    else:
        rgba = np.tile([*np.array(mpl.colors.to_rgb(st["color"])) * 255, 255], (len(pts), 1))
    if scalars is not None and colorbar:
        p.add_scalar_bar(title=title, mapper=parts[0]["actor"].mapper, fmt="%.1f" if color_by == "time" else "%.4g",
                         **sbar)
        p.scalar_bars[title].SetVerticalTitleSeparation(int(st["font_size"]))
    if legend and colorbar:
        h = min(0.035 * len(legend), 0.5)
        box = p.add_legend(legend, face="circle", bcolor=None, border=False, size=(0.15, h))
        box.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        box.SetPosition(0.01, 0.99 - h)
        box.SetPosition2(0.15, h)
    state["catalogs"].append({"catalog": cat, "index": idx, "parts": parts, "points": pts, "radius": pc["radius"],
                              "px": px, "shown": np.ones(len(idx), bool), "rgba": np.asarray(rgba, np.uint8),
                              "bar": title if scalars is not None and colorbar else None})

    # scene
    p.set_background(st["background"], top=st["background_top"])
    if axes:
        b = np.array(p.bounds, float)
        b[4:] /= state["z_exag"]
        b[:4] += np.repeat(state["origin"], 2) / k
        zt = f"Up ({u})" + (f", x{state['z_exag']:g}" if state["z_exag"] != 1 else "")
        geo = state["frame"]["kind"] != "local"
        p.show_grid(xtitle=" " if geo else f"East ({u})", ytitle=" " if geo else f"North ({u})", ztitle=zt,
                    color=st["text_color"], font_size=st["font_size"], axes_ranges=list(b), fmt="%.0f",
                    bounds=p.bounds, show_xlabels=not geo, show_ylabels=not geo, n_zlabels=4)
        if geo:
            add_graticule(p)
    if new:
        {"iso": p.view_isometric, "map": p.view_xy, "ew": p.view_xz, "ns": p.view_yz}[view]()
        if view != "iso":
            p.enable_parallel_projection()
    if pick and "picker" not in state:
        state["picker"] = Picker(p, hover, ids)
    elif pick:
        state["picker"].hover, state["picker"].id = hover or state["picker"].hover, ids or state["picker"].id
    return p


def to_lonlat(plotter, x, y):
    """Longitudes and latitudes of scene points; None for a local scene with no reference CRS."""
    st = plotter._pituplots
    fr = st["frame"]
    if fr["kind"] == "local":
        return None
    k = 1000 if st["unit"] == "km" else 1
    fx, fy = from_frame(fr, np.asarray(x, float) * k + st["origin"][0], np.asarray(y, float) * k + st["origin"][1])
    lon, lat = transformer(fr["crs"], "EPSG:4326").transform(fx, fy)
    return np.asarray(lon), np.asarray(lat)


def degrees(v, step, hemi):
    """``"70.5°W"``-style label, with the decimals of ``step``."""
    d = max(0, -int(np.floor(np.log10(step) + 1e-9)))
    return f"{abs(v):.{d}f}°{hemi[0] if v < 0 else hemi[1]}"


def add_graticule(plotter, step=None, lines=6):
    """Meridians and parallels on the floor of the events' box, labelled at its south and west edges.

    Parameters
    ----------
    plotter : pyvista.Plotter
        A georeferenced scene from ``plot_catalog_3d``.
    step : float, optional
        In degrees. Default a round number giving about ``lines`` lines.
    lines : int

    Returns
    -------
    float or None
        The step, or None for a scene with no reference CRS.
    """
    pv = need("pyvista", "3d")
    st = plotter._pituplots
    sty = resolve("scene")
    b = np.array(full_bounds(plotter), float)
    t = np.linspace(0, 1, 50)
    ex = np.r_[b[0] + t * (b[1] - b[0]), np.full(50, b[1]), b[0] + t * (b[1] - b[0]), np.full(50, b[0])]
    ey = np.r_[np.full(50, b[2]), b[2] + t * (b[3] - b[2]), np.full(50, b[3]), b[2] + t * (b[3] - b[2])]
    ll = to_lonlat(plotter, ex, ey)
    if ll is None:
        return None
    lon, lat = ll
    step = step or next(a * 10.0 ** e for e in range(-4, 3) for a in (1, 2, 5)
                        if max(np.ptp(lon), np.ptp(lat)) / (a * 10.0 ** e) <= lines)
    z = b[4] / st["z_exag"] * (1000 if st["unit"] == "km" else 1)
    pts, cells, text, at, n = [], [], [], [], 0
    grid = [(np.full(200, v), np.linspace(lat.min(), lat.max(), 200), degrees(v, step, "WE"))
            for v in np.arange(np.ceil(lon.min() / step) * step, lon.max(), step)]
    grid += [(np.linspace(lon.min(), lon.max(), 200), np.full(200, v), degrees(v, step, "SN"))
             for v in np.arange(np.ceil(lat.min() / step) * step, lat.max(), step)]
    for lo, la, label in grid:
        q = to_scene(plotter, lo, la, z, "EPSG:4326")
        i = np.flatnonzero((q[:, 0] >= b[0]) & (q[:, 0] <= b[1]) & (q[:, 1] >= b[2]) & (q[:, 1] <= b[3]))
        if len(i) < 2:
            continue
        q = q[i[0]:i[-1] + 1]
        pts.append(q)
        cells.append(np.r_[len(q), np.arange(n, n + len(q))])
        n += len(q)
        text.append(label)
        at.append(q[0])
    if not pts:
        return step
    plotter.add_mesh(pv.PolyData(np.vstack(pts), lines=np.concatenate(cells)), color=sty["text_color"],
                     opacity=0.35, line_width=1, name="graticule", pickable=False, reset_camera=False)
    plotter.add_point_labels(np.array(at), text, font_size=sty["font_size"], text_color=sty["text_color"],
                             shape=None, show_points=False, always_visible=True, bold=False, name="graticule_labels",
                             pickable=False, reset_camera=False)
    return step


class Picker:
    """Hover labels, a click panel and flags for the events of a scene.

    Moving the cursor over an event shows its ``hover`` fields; a click
    (press and release without dragging) shows all its fields in a panel;
    space flags or unflags the event under the cursor; ``x`` writes the
    flags to ``path``. Made by ``plot_catalog_3d`` with ``pick=True``.

    Parameters
    ----------
    plotter : pyvista.Plotter
        A scene from ``plot_catalog_3d``.
    hover : list of str, optional
        Fields in the label after lon, lat, mag, depth, time and class (when
        there). Lon and lat are computed for projected and grid catalogs.
    id : str, optional
        Column that names an event in labels and in the flags file.
    path : str or Path
    tol : float
        Pick distance in pixels, for events smaller than that.
    events : bool
        Listen to the window's mouse and keys, and show the click panel. Off
        when another interface, such as the app, calls the methods. With a
        section, Ctrl + wheel and ``[`` ``]`` step it.
    """

    def __init__(self, plotter, hover=None, id=None, path="flags.csv", tol=8, events=True):
        import matplotlib as mpl

        self.p, self.id, self.path, self.tol, self.events = plotter, id, path, tol, events
        self.hover = HOVER + [k for k in hover or [] if k not in HOVER]
        self.flags, self.selected, self.down, self.last = [], None, None, None
        st = resolve("picking")
        self.st = st
        self.label = plotter.add_text("", position=(0, 0), font_size=st["font_size"], color=st["text_color"])
        self.panel = plotter.add_text("", position=(0, 0), font_size=st["font_size"], color=st["text_color"])
        for t in (self.label, self.panel):
            tp = t.GetTextProperty()
            tp.SetBackgroundColor(*mpl.colors.to_rgb(st["box_color"]))
            tp.SetBackgroundOpacity(st["box_opacity"])
            tp.SetFontFamilyToCourier()
        self.panel.GetTextProperty().SetVerticalJustificationToTop()
        self.panel.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self.panel.SetPosition(0.01, 0.98)
        if events and plotter.iren is not None:
            plotter.iren.add_observer("MouseMoveEvent", self.on_move)
            plotter.iren.add_observer("LeftButtonPressEvent", self.on_press)
            plotter.iren.add_observer("LeftButtonReleaseEvent", self.on_release)
            plotter.add_key_event("space", lambda: self.toggle(self.last))
            plotter.add_key_event("x", lambda: self.export())
            step = lambda k: plotter._pituplots.get("section") and (step_section(plotter, k), self.render())  # noqa: E731
            plotter.add_key_event("bracketright", lambda: step(1))
            plotter.add_key_event("bracketleft", lambda: step(-1))
            on_wheel(plotter, step)
            plotter._pituplots["inset_on"] = True

    @property
    def catalogs(self):
        return self.p._pituplots["catalogs"]

    def at(self, x, y):
        """``(catalog number, row)`` of the front event under display pixel ``(x, y)``, or None."""
        pv = need("pyvista", "3d")
        cam = self.p.camera
        w, h = self.p.window_size
        m = pv.array_from_vtkmatrix(cam.GetCompositeProjectionTransformMatrix(w / h, -1, 1))
        best = None
        for i, e in enumerate(self.catalogs):
            q = np.c_[e["points"], np.ones(len(e["points"]))] @ m.T
            ndc = q[:, :3] / q[:, 3:]
            if cam.GetParallelProjection():
                ppu = h / (2 * cam.GetParallelScale())
            else:
                dist = (e["points"] - cam.GetPosition()) @ np.asarray(cam.GetDirectionOfProjection())
                ppu = h / (2 * np.maximum(dist, 1e-12) * np.tan(np.radians(cam.GetViewAngle()) / 2))
            d = np.hypot((ndc[:, 0] + 1) / 2 * w - x, (ndc[:, 1] + 1) / 2 * h - y)
            r = e["px"] / 2 if e["px"] is not None else e["radius"] * ppu
            hit = np.flatnonzero((d <= np.maximum(r, self.tol)) & (q[:, 3] > 0) & (np.abs(ndc[:, 2]) <= 1) & e["shown"])
            if hit.size:
                j = hit[np.argmin(ndc[hit, 2])]
                if best is None or ndc[j, 2] < best[0]:
                    best = (ndc[j, 2], i, int(e["index"][j]))
        return None if best is None else best[1:]

    def fields(self, hit, keys=None):
        """``{name: text}`` of an event, for ``keys`` or all its fields; empty values left out."""
        c = self.catalogs[hit[0]]["catalog"]
        r = hit[1]
        geo = c.geographic
        core = {"lon" if geo else "x": c.lon, "lat" if geo else "y": c.lat, "depth": c.depth, "mag": c.mag,
                "time": c.time}
        vals = {k: v[r] for k, v in {**{k: v for k, v in core.items() if v is not None}, **c.extra}.items()}
        fr = self.p._pituplots["frame"]
        if not geo and (c.crs != "local" or fr["kind"] == "grid"):
            x, y = from_frame(fr, [c.lon[r]], [c.lat[r]]) if c.crs == "local" else ([c.lon[r]], [c.lat[r]])
            lon, lat = transformer(fr["crs"] if c.crs == "local" else c.crs, "EPSG:4326").transform(x, y)
            vals = {"lon": float(np.ravel(lon)[0]), "lat": float(np.ravel(lat)[0]), **vals}
        out = {}
        for k in keys or vals:
            if k not in vals or k == self.id:
                continue
            v = vals[k]
            if isinstance(v, (float, np.floating)):
                if np.isnan(v):
                    continue
                v = f"{v:.6g}"
            elif isinstance(v, np.datetime64):
                if np.isnat(v):
                    continue
                v = str(v.astype("datetime64[s]"))
            v = str(v)
            if v not in ("", "nan", "None"):
                out[str(k)] = v[:60]
        return out

    def title(self, hit):
        c = self.catalogs[hit[0]]["catalog"]
        return f"{self.id} {c.extra[self.id][hit[1]]}" if self.id in c.extra else f"row {hit[1]}"

    def text(self, hit, keys=None):
        f = self.fields(hit, keys)
        n = max([len(k) for k in f] or [0])
        flag = "  [flagged]" if hit in self.flags else ""
        return "\n".join([self.title(hit) + flag] + [f"{k:<{n}}  {v}" for k, v in f.items()])

    def show_label(self, x, y):
        hit = self.at(x, y)
        self.last = hit
        self.label.SetInput("" if hit is None else self.text(hit, self.hover))
        self.label.SetPosition(x + 14, y + 14)

    def select(self, hit):
        self.selected = hit
        if self.events:
            self.panel.SetInput("" if hit is None else self.text(hit))
        self.markers()

    def toggle(self, hit):
        """Flag an event ``(catalog number, row)``, or unflag it."""
        if hit is None:
            return
        self.flags.remove(hit) if hit in self.flags else self.flags.append(hit)
        if hit == self.selected and self.events:
            self.panel.SetInput(self.text(hit))
        self.markers()
        self.render()

    def markers(self):
        pv = need("pyvista", "3d")
        for name, hits, color in (("flags", self.flags, self.st["flag_color"]),
                                  ("selected", [self.selected] if self.selected else [], self.st["select_color"])):
            if not hits:
                self.p.remove_actor(name)
                continue
            pts, px = [], [0.0]
            for i, r in hits:
                e = self.catalogs[i]
                j = np.searchsorted(e["index"], r)
                pts.append(e["points"][j])
                px.append(e["px"][j] if e["px"] is not None else 0.0)
            self.p.add_mesh(pv.PolyData(np.array(pts, float)), style="points", render_points_as_spheres=True,
                            point_size=max(px) + self.st["marker_size"], color=color,
                            opacity=self.st["marker_opacity"], name=name, pickable=False, reset_camera=False)

    def table(self):
        """The flagged events as a DataFrame: catalog number, row, id and fields."""
        import pandas as pd

        keys = ["lon", "lat", "x", "y", "depth", "mag", "time", *(self.hover or [])]
        rows = []
        for i, r in self.flags:
            c = self.catalogs[i]["catalog"]
            rows.append({"catalog": i, "row": r, **({self.id: c.extra[self.id][r]} if self.id in c.extra else {}),
                         **self.fields((i, r), keys)})
        return pd.DataFrame(rows, columns=None if rows else ["catalog", "row"])

    def export(self, path=None):
        """Write the flags to CSV and return the path."""
        path = path or self.path
        self.table().to_csv(path, index=False)
        return path

    def render(self):
        if self.p.iren is not None:
            self.p.render()

    def on_move(self, obj, event):
        self.show_label(*self.p.iren.interactor.GetEventPosition())
        self.render()

    def on_press(self, obj, event):
        self.down = self.p.iren.interactor.GetEventPosition()

    def on_release(self, obj, event):
        x, y = self.p.iren.interactor.GetEventPosition()
        if self.down is not None and abs(x - self.down[0]) + abs(y - self.down[1]) <= 3:
            self.select(self.at(x, y))
            self.render()
        self.down = None


def show_events(plotter, where=None, catalog=0, alpha=None):
    """Show only some events of a catalog in a scene; hidden events are not picked.

    With a section (``add_section``), only the events in it among these.
    With ``alpha``, each event is drawn with its own opacity (times the
    catalog's), as for a fading time window (see ``fade``).

    Parameters
    ----------
    plotter : pyvista.Plotter
        A scene from ``plot_catalog_3d``.
    where : array of bool, optional
        One value per catalog row. Default all.
    catalog : int
        Which catalog of the scene, in drawing order.
    alpha : array of float, optional
        One value per catalog row, 0 to 1.

    Returns
    -------
    int
        The number shown.
    """
    need("pyvista", "3d")
    e = plotter._pituplots["catalogs"][catalog]
    n = len(e["catalog"])
    w = np.ones(n, bool) if where is None else np.asarray(where, bool)
    if w.shape != (n,):
        raise ValueError(f"where has shape {w.shape}, expected ({n},)")
    e["where"] = w
    keep = w[e["index"]]
    sec = plotter._pituplots.get("section")
    if sec:
        keep = keep & (np.abs((e["points"][:, :2] - sec["center"]) @ sec["normal"]) <= sec["width"])
    for g in e["parts"]:
        sub = subset(g["mesh"], keep[g["sel"]], g["cells"])
        m = g["actor"].mapper
        if alpha is not None and sub.n_points:
            rows = np.asarray(sub["index"]).astype(int)
            c = e["rgba"][np.searchsorted(e["index"], rows)].copy()
            c[:, 3] = np.clip(np.asarray(alpha, float)[rows] * g["opacity"], 0, 1) * 255
            sub["rgba"] = c
            m.ScalarVisibilityOn()
            m.SetScalarModeToUsePointFieldData()
            m.SelectColorArray("rgba")
            m.SetColorModeToDirectScalars()
            g["actor"].prop.opacity = 1.0
        else:
            m.SetScalarVisibility(g["scalars"] is not None)
            if g["scalars"]:
                m.SelectColorArray(g["scalars"])
            m.SetColorMode(g["mode"])
            g["actor"].prop.opacity = g["opacity"]
        m.SetInputData(sub)
    e["shown"] = keep
    return int(keep.sum())


def subset(full, keep, cells=None, lines=False):
    """A shallow copy of a mesh with some of its points or cells.

    Parameters
    ----------
    full : pyvista.PolyData
    keep : array of bool
        Per point, or per group of ``cells`` consecutive cells of one size.
    cells : int, optional
    lines : bool
        Groups of lines instead of polygons.

    Returns
    -------
    pyvista.PolyData
    """
    pv = need("pyvista", "3d")
    if cells:
        sub = full.copy(deep=False)
        c = full.lines if lines else full.faces
        c = c.reshape(-1, c[0] + 1)[np.repeat(keep, cells)].ravel() if c.size else c
        if lines:
            sub.lines = c
        else:
            sub.faces = c
    elif not keep.any():
        return pv.PolyData()
    else:
        sub = pv.PolyData(full.points[keep])
        for a in full.point_data.keys():
            sub.point_data[a] = full.point_data[a][keep]
    pd, sd = full.GetPointData(), sub.GetPointData()
    for get, put in ((pd.GetScalars, sd.SetActiveScalars), (pd.GetNormals, sd.SetActiveNormals)):
        if get() is not None:
            put(get().GetName())
    return sub


def add_section(plotter, center=None, azimuth=90.0, width=None, view=False, plane=True, widget=False, crs=None):
    """Keep only the events near a vertical plane: a cross-section in 3D.

    Applies to every catalog of the scene, on top of ``show_events``.

    Parameters
    ----------
    plotter : pyvista.Plotter
        A scene from ``plot_catalog_3d``.
    center : (float, float), optional
        A point of the plane, in scene coordinates (east, north, in the scene
        unit), or in ``crs``. Default the centre of the scene.
    azimuth : float
        Strike of the section line, degrees clockwise from north.
    width : float, optional
        Half-width, in the scene unit. Default 5% of the horizontal diagonal.
    view : bool
        Look at the section face on (see ``view_section``).
    plane : bool
        Draw the plane.
    widget : bool
        A plane widget to drag and turn the section, for windows on screen.
    crs : str, optional
        Of ``center``, such as ``"EPSG:4326"`` for longitude and latitude.

    Returns
    -------
    dict
        ``center``, ``azimuth``, ``width``, ``normal``, and ``shown``, the
        number of events kept per catalog.
    """
    pv = need("pyvista", "3d")
    st = plotter._pituplots
    b = np.array(full_bounds(plotter), float)
    diag = float(np.hypot(b[1] - b[0], b[3] - b[2])) or 1.0
    c = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2] if center is None else center, float)
    if center is not None and crs is not None:
        c = to_scene(plotter, [c[0]], [c[1]], 0, crs)[0, :2]
    a = np.radians(azimuth)
    sec = {"center": c, "azimuth": float(azimuth) % 360, "width": float(width or 0.05 * diag),
           "normal": np.array([np.cos(a), -np.sin(a)])}
    st["section"] = sec
    sec["shown"] = [show_events(plotter, e.get("where"), i) for i, e in enumerate(st["catalogs"])]
    clip_layers(plotter)
    if st.get("inset_on"):
        inset(plotter)
    if plane:
        zc, zs = (b[4] + b[5]) / 2, (b[5] - b[4]) or 1.0
        pl = pv.Plane(center=(c[0], c[1], zc), direction=(*sec["normal"], 0), i_size=zs, j_size=diag)
        plotter.add_mesh(pl, color=resolve("scene")["section_color"], opacity=0.15, name="section", pickable=False,
                         reset_camera=False)
    if widget:
        def moved(normal, origin):
            az = np.degrees(np.arctan2(-normal[1], normal[0])) if np.hypot(*normal[:2]) > 1e-6 else azimuth
            add_section(plotter, origin[:2], az % 180, sec["width"], plane=plane)

        plotter.clear_plane_widgets()
        plotter.add_plane_widget(moved, normal=(*sec["normal"], 0), origin=(*c, (b[4] + b[5]) / 2),
                                 bounds=b, implicit=False, normal_rotation=True, outline_translation=False)
    if view:
        view_section(plotter)
    return sec


def remove_section(plotter):
    """Show all filtered events again and remove the section plane."""
    st = plotter._pituplots
    st.pop("section", None)
    plotter.remove_actor("section")
    plotter.clear_plane_widgets()
    for i, e in enumerate(st["catalogs"]):
        show_events(plotter, e.get("where"), i)
    clip_layers(plotter)
    if st.get("inset"):
        st["inset"]["renderer"].DrawOff()


def inset(plotter, viewport=(0.005, 0.005, 0.2, 0.3)):
    """A plan-view inset in a corner with the epicentres, the shape layers and the section, for orientation.

    Made on the first call; later calls move the section band. Hidden
    without a section. ``Picker(events=True)`` turns it on for sections.

    Parameters
    ----------
    plotter : pyvista.Plotter
    viewport : tuple of float
        ``(x0, y0, x1, y1)``, fractions of the window.

    Returns
    -------
    vtkRenderer
    """
    pv = need("pyvista", "3d")
    import matplotlib as mpl
    from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkRenderer, vtkTextActor

    mpl_rgb = mpl.colors.to_rgb
    st = plotter._pituplots
    sty = resolve("scene")
    b = np.array(full_bounds(plotter), float)
    diag = float(np.hypot(b[1] - b[0], b[3] - b[2])) or 1.0

    def actor(mesh, color, opacity=1.0, size=1.0):
        m = vtkPolyDataMapper()
        m.SetInputData(mesh)
        a = vtkActor()
        a.SetMapper(m)
        a.GetProperty().SetColor(*mpl_rgb(color))
        a.GetProperty().SetOpacity(opacity)
        a.GetProperty().SetPointSize(size)
        a.GetProperty().SetLineWidth(size)
        return a

    if "inset" not in st:
        ren = vtkRenderer()
        ren.SetViewport(*viewport)
        ren.SetBackground(*mpl_rgb("white"))
        ren.InteractiveOff()
        xy = np.vstack([e["points"][:, :2] for e in st["catalogs"]])
        xy = xy[np.random.default_rng(0).permutation(len(xy))[:3000]]
        ren.AddActor(actor(pv.PolyData(np.c_[xy, np.zeros(len(xy))]), "#7a8aa0", size=2))
        for v in st["layers"].values():
            if v["kind"] == "shapes" and v.get("mesh") is not None:
                q = v["mesh"].copy()
                q.points[:, 2] = 0.0
                ren.AddActor(actor(q, "#444444"))
        band = pv.PolyData()
        ren.AddActor(actor(band, sty["section_color"], 0.35))
        text = vtkTextActor()
        text.GetTextProperty().SetFontSize(sty["font_size"])
        text.GetTextProperty().SetColor(*mpl_rgb(sty["text_color"]))
        text.SetPosition(4, 4)
        ren.AddActor(text)
        cam = ren.GetActiveCamera()
        cam.ParallelProjectionOn()
        cam.SetFocalPoint((b[0] + b[1]) / 2, (b[2] + b[3]) / 2, 0)
        cam.SetPosition((b[0] + b[1]) / 2, (b[2] + b[3]) / 2, diag)
        cam.SetViewUp(0, 1, 0)
        ren.ResetCamera(b[0], b[1], b[2], b[3], -1, 1)
        plotter.ren_win.AddRenderer(ren)
        st["inset"] = {"renderer": ren, "band": band, "text": text}
    ins = st["inset"]
    sec = st.get("section")
    if not sec:
        ins["renderer"].DrawOff()
        return ins["renderer"]
    n = sec["normal"]
    t = np.array([-n[1], n[0]])
    q = [sec["center"] + a * sec["width"] * n + c * diag * t for a, c in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
    ins["band"].deep_copy(pv.PolyData(np.c_[np.array(q), np.zeros(4)], faces=[4, 0, 1, 2, 3]))
    ll = to_lonlat(plotter, [sec["center"][0]], [sec["center"][1]])
    ins["text"].SetInput((f"{degrees(ll[1][0], 0.01, 'SN')} {degrees(ll[0][0], 0.01, 'WE')}, " if ll else "")
                         + f"strike {sec['azimuth']:.0f}")
    ins["renderer"].DrawOn()
    return ins["renderer"]


def clip_layers(plotter):
    """Cut the layers of a scene to its section: the cells that reach into the slab; all without a section."""
    st = plotter._pituplots
    sec = st.get("section")
    for v in st["layers"].values():
        full = v.get("mesh")
        if full is None:
            continue
        if not sec:
            v["actor"].mapper.SetInputData(full)
            continue
        n, w = np.array([*sec["normal"], 0.0]), sec["width"]
        c, lines = None, False
        if hasattr(full, "faces"):
            lines = full.n_lines > 0 and full.n_lines == full.n_cells
            c = full.lines if lines else full.faces
            c = c.reshape(-1, c[0] + 1) if c.size and c.size % (c[0] + 1) == 0 else None
            c = c if c is not None and (c[:, 0] == c.shape[1] - 1).all() else None
        if c is None:
            o = np.array([*sec["center"], 0.0])
            v["actor"].mapper.SetInputData(full.clip(normal=n, origin=o + w * n).clip(normal=-n, origin=o - w * n))
            continue
        d = ((full.points[:, :2] - sec["center"]) @ sec["normal"])[c[:, 1:]]
        v["actor"].mapper.SetInputData(subset(full, (d.min(1) <= w) & (d.max(1) >= -w), 1, lines))


def step_section(plotter, steps=1, step=None):
    """Move the section along its normal, and the camera with it, to page through slabs.

    Parameters
    ----------
    plotter : pyvista.Plotter
        A scene with a section.
    steps : int
        Positive towards the normal: south for an E–W section, east for N–S.
    step : float, optional
        In the scene unit. Default twice the half-width, so slabs tile.

    Returns
    -------
    dict
        The new section.
    """
    sec = plotter._pituplots["section"]
    d = (step or 2 * sec["width"]) * steps * sec["normal"]
    pos, focal, up = plotter.camera_position
    plotter.camera_position = [np.add(pos, (*d, 0)), np.add(focal, (*d, 0)), up]
    vis = "section" in plotter.actors and plotter.actors["section"].GetVisibility()
    out = add_section(plotter, sec["center"] + d, sec["azimuth"], sec["width"])
    if "section" in plotter.actors:
        plotter.actors["section"].SetVisibility(vis)
    return out


def on_wheel(plotter, step):
    """Ctrl + wheel calls ``step(+1 or -1)``; the plain wheel still zooms."""
    style = plotter.iren.interactor.GetInteractorStyle()
    for event, k, zoom in (("MouseWheelForwardEvent", 1, style.OnMouseWheelForward),
                           ("MouseWheelBackwardEvent", -1, style.OnMouseWheelBackward)):
        def wheel(obj, ev, k=k, zoom=zoom):
            if plotter.iren.interactor.GetControlKey():
                step(k)
            else:
                zoom()

        style.AddObserver(event, wheel)


def view_section(plotter):
    """Look at the section face on, without perspective, framing the events in it.

    The plane is hidden in this view; ``add_section`` draws it again.
    """
    sec = plotter._pituplots["section"]
    b = np.array(full_bounds(plotter), float)
    f = (*sec["center"], (b[4] + b[5]) / 2)
    d = 2 * float(np.linalg.norm(b[1::2] - b[::2]))
    plotter.camera_position = [(f[0] + sec["normal"][0] * d, f[1] + sec["normal"][1] * d, f[2]), f, (0, 0, 1)]
    plotter.enable_parallel_projection()
    if "section" in plotter.actors:
        plotter.actors["section"].SetVisibility(False)
    pts = [e["points"][e["shown"]] for e in plotter._pituplots["catalogs"] if e["shown"].any()]
    if pts:
        q = np.vstack(pts)
        plotter.reset_camera(bounds=[v for lo, hi in zip(q.min(0), q.max(0)) for v in (lo, hi)])
    else:
        plotter.reset_camera()


def full_bounds(plotter):
    """Bounds of all events of a scene, filtered or not."""
    pts = np.vstack([e["points"] for e in plotter._pituplots["catalogs"]])
    return [v for lo, hi in zip(pts.min(0), pts.max(0)) for v in (lo, hi)]


def window(times, end, length=None, start=None):
    """Events in a time window: ``end - length < t <= end``, or ``start < t <= end``.

    Parameters
    ----------
    times : array of datetime64
    end : datetime64 or str
    length : str or timedelta, optional
        Such as ``"30D"``. Without it and ``start``, all events up to ``end``.
    start : datetime64 or str, optional

    Returns
    -------
    numpy.ndarray of bool
    """
    import pandas as pd

    t = np.asarray(times, "datetime64[ms]")
    end = np.datetime64(pd.Timestamp(end), "ms")
    lo = end - pd.Timedelta(length).to_timedelta64() if length is not None else (
        None if start is None else np.datetime64(pd.Timestamp(start), "ms"))
    return (t <= end) & (t > lo) if lo is not None else t <= end


def fade(times, end, length=None, start=None, floor=0.1):
    """Opacity by age in a time window: ``floor`` at its start, 1 at its end.

    Parameters
    ----------
    times : array of datetime64
    end : datetime64 or str
    length : str or timedelta, optional
        As in ``window``; without it, from ``start`` or the first event.
    start : datetime64 or str, optional
    floor : float

    Returns
    -------
    numpy.ndarray
        0 outside the window and for events without time.
    """
    import pandas as pd

    t = np.asarray(times, "datetime64[ms]").astype("int64").astype(float)
    t[np.isnat(np.asarray(times, "datetime64[ms]"))] = np.nan
    hi = float(np.datetime64(pd.Timestamp(end), "ms").astype("int64"))
    lo = hi - pd.Timedelta(length).total_seconds() * 1000 if length is not None else (
        float(np.datetime64(pd.Timestamp(start), "ms").astype("int64")) if start is not None else np.nanmin(t))
    a = floor + (1 - floor) * (t - lo) / max(hi - lo, 1.0)
    return np.where((t >= lo) & (t <= hi), np.clip(a, floor, 1), 0.0)


def animate(plotter, path, step, length=None, start=None, end=None, fps=8, catalog=0, where=None, clock=True,
            fading=False):
    """Write a movie of a scene through time.

    Each frame shows the events in the window ending at that frame: of
    ``length``, or all since ``start`` (cumulative) without it.

    Parameters
    ----------
    plotter : pyvista.Plotter
        An off-screen scene from ``plot_catalog_3d``.
    path : str or Path
        ``.mp4`` (needs imageio-ffmpeg) or ``.gif``.
    step : str or timedelta
        Between frames, such as ``"7D"``.
    length : str or timedelta, optional
    start, end : datetime64 or str, optional
        Default the first and last event.
    fps : int
    catalog : int
    where : array of bool, optional
        A filter applied in every frame.
    clock : bool
        The window end in the corner.
    fading : bool
        Older events in each window fade out (see ``fade``).

    Returns
    -------
    int
        The number of frames.
    """
    import pandas as pd

    need("imageio", "3d")
    cat = plotter._pituplots["catalogs"][catalog]["catalog"]
    if cat.time is None:
        raise ValueError("animate needs event times")
    t = cat.time.astype("datetime64[ms]")
    ok = ~np.isnat(t)
    start = np.datetime64(pd.Timestamp(start), "ms") if start is not None else t[ok].min()
    end = np.datetime64(pd.Timestamp(end), "ms") if end is not None else t[ok].max()
    dt = pd.Timedelta(step).to_timedelta64()
    first = start + (pd.Timedelta(length).to_timedelta64() if length is not None else dt)
    ends = np.arange(first, end + dt, dt).astype("datetime64[ms]")
    if str(path).lower().endswith(".gif"):
        plotter.open_gif(str(path), fps=fps)
    else:
        need("imageio_ffmpeg", "3d")
        plotter.open_movie(str(path), framerate=fps, quality=7)
    w = np.ones(len(cat), bool) if where is None else np.asarray(where, bool)
    st = resolve("scene")
    for e in ends:
        show_events(plotter, w & window(t, e, length, None if length is not None else start - np.timedelta64(1, "ms")),
                    catalog, fade(t, e, length, start) if fading else None)
        if clock:
            plotter.add_text(str(e.astype("datetime64[D]")), position="lower_right", font_size=st["font_size"] + 2,
                             color=st["text_color"], name="clock")
        plotter.write_frame()
    plotter.mwriter.close()
    plotter.remove_actor("clock")
    show_events(plotter, w, catalog)
    return len(ends)


def export_flags(plotter, path):
    """Write the events flagged in a scene to CSV.

    Parameters
    ----------
    plotter : pyvista.Plotter
        A scene from ``plot_catalog_3d`` with ``pick=True``.
    path : str or Path

    Returns
    -------
    str or Path
    """
    return plotter._pituplots["picker"].export(path)


def screenshot(plotter, path, scale=1, transparent=False):
    """Save the scene as an image.

    Parameters
    ----------
    plotter : pyvista.Plotter
        Off screen, or shown with ``auto_close=False``.
    path : str or Path, optional
        None returns the image without writing a file.
    scale : int
        Resolution multiplier of the window size.
    transparent : bool

    Returns
    -------
    numpy.ndarray
        The image.
    """
    plotter.render()
    return plotter.screenshot(None if path is None else str(path), transparent_background=transparent, scale=scale)
