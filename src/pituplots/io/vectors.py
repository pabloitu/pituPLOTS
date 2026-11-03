import json
from pathlib import Path

import numpy as np
import pandas as pd
import shapefile
from pyproj import CRS
from shapely import from_wkb
from shapely.geometry import box, shape
from shapely.ops import unary_union

from pituplots.errors import need


def crs_name(crs):
    code = 4326 if crs.to_authority() == ("OGC", "CRS84") else crs.to_epsg()
    return f"EPSG:{code}" if code else crs.to_wkt()


def read_vector(path, layer=None):
    """Read vector features.

    Parameters
    ----------
    path : str or Path
        Shapefile or GeoJSON. Other OGR formats, such as GPKG, need pyogrio.
    layer : str or int, optional
        Layer of a multi-layer file.

    Returns
    -------
    (geoms, attrs, crs)
        A list of shapely geometries (None for null shapes), a DataFrame with
        one row each, and the CRS as ``"EPSG:n"``, WKT or None.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    kind = path.suffix.lower()
    if kind == ".shp":
        with shapefile.Reader(str(path)) as r:
            names = [f[0] for f in r.fields[1:]]
            rows = [list(s.record) for s in r.iterShapeRecords()]
            geoms = [shape(s) if s.shapeType != 0 else None for s in r.shapes()]
        prj = path.with_suffix(".prj")
        crs = crs_name(CRS.from_wkt(prj.read_text())) if prj.is_file() else None
        return geoms, pd.DataFrame(rows, columns=names), crs
    if kind in (".geojson", ".json"):
        d = json.loads(path.read_text())
        fs = d["features"] if d.get("type") == "FeatureCollection" else [d]
        geoms = [shape(f["geometry"]) if f.get("geometry") else None for f in fs]
        c = d.get("crs", {}).get("properties", {}).get("name")
        return geoms, pd.DataFrame([f.get("properties") or {} for f in fs]), crs_name(CRS.from_user_input(c or 4326))
    raw = need("pyogrio", "vector").raw
    meta, _, wkb, fields = raw.read(path, layer=layer)
    attrs = pd.DataFrame({n: f for n, f in zip(meta["fields"], fields)})
    crs = meta.get("crs")
    return list(from_wkb(wkb)), attrs, crs_name(CRS.from_user_input(crs)) if crs else None


def cells_to_outline(grid):
    """Outline of the unmasked cells of a ``Grid``, with holes and disjoint parts.

    Parameters
    ----------
    grid : Grid

    Returns
    -------
    shapely Polygon or MultiPolygon
    """
    x, y = np.round(grid.x_edges, 6), np.round(grid.y_edges, 6)
    keep = np.ones(grid.values.shape[:2], bool) if grid.mask is None else ~grid.mask
    bars = []
    for j in np.flatnonzero(keep.any(axis=1)):
        d = np.diff(np.r_[0, keep[j].astype(int), 0])
        bars += [box(x[a], y[j], x[b], y[j + 1]) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]
    return unary_union(bars)
