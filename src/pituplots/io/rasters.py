from pathlib import Path

import numpy as np
import pandas as pd

from pituplots.errors import need
from pituplots.io.containers import Grid


def pixel_window(ds, l, b, r, t):
    """Whole pixels covering a box, inside the raster."""
    from rasterio.windows import Window, from_bounds

    w = from_bounds(l, b, r, t, ds.transform)
    c0, r0 = int(np.floor(w.col_off)), int(np.floor(w.row_off))
    c1, r1 = int(np.ceil(w.col_off + w.width)), int(np.ceil(w.row_off + w.height))
    c0, r0, c1, r1 = max(c0, 0), max(r0, 0), min(c1, ds.width), min(r1, ds.height)
    return Window(c0, r0, c1 - c0, r1 - r0)


def read_geotiff(path, extent=None, max_px=None, crs=None):
    """Read a GeoTIFF into a grid. Values are raw, with no sign change.

    Parameters
    ----------
    path : str or Path
    extent : list, optional
        ``[lon_min, lon_max, lat_min, lat_max]`` in degrees. Reads only that
        window.
    max_px : int, optional
        Decimate so the longest side is at most this.
    crs : optional
        For files without one, such as ESRI ASCII grids. Any GDAL format
        rasterio opens is read.

    Returns
    -------
    Grid
        An image if the file has 3 or more bands. Nodata is masked.
    """
    rio = need("rasterio", "raster")
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, bounds

    if not Path(path).is_file():
        raise FileNotFoundError(path)
    with rio.open(path) as ds:
        dcrs = ds.crs or crs
        if dcrs is None:
            raise ValueError(f"{Path(path).name} has no CRS: pass crs=")
        win = Window(0, 0, ds.width, ds.height)
        if extent is not None:
            win = pixel_window(ds, *transform_bounds("EPSG:4326", dcrs, extent[0], extent[2],
                                                     extent[1], extent[3]))
        k = max(1, max(win.width, win.height) / max_px) if max_px else 1
        shape = (max(1, int(win.height / k)), max(1, int(win.width / k)))
        rs = Resampling.average if k > 1 else Resampling.nearest
        n = min(ds.count, 4) if ds.count >= 3 else 1
        z = ds.read(list(range(1, n + 1)), window=win, out_shape=(n, *shape), resampling=rs, masked=True)
        l, b, r, t = bounds(win, ds.transform)
        crs = ds.crs.to_string() if ds.crs else str(crs)
    x = np.linspace(l, r, shape[1] + 1)
    y = np.linspace(b, t, shape[0] + 1)
    if n == 1:
        v = z[0][::-1]
        if np.issubdtype(v.dtype, np.integer):
            v = v.astype(float)
        return Grid(v, x, y, crs=crs)
    return Grid(z.data.transpose(1, 2, 0)[::-1], x, y, crs=crs)


def read_slab2(path):
    """Read a Slab2 ``.xyz`` file: no header, longitudes in 0..360, NaN off the slab.

    Parameters
    ----------
    path : str or Path
        A ``dep`` file gives depth in km, positive down. Others (``thk``,
        ``str``, ``dip``) keep their sign.

    Returns
    -------
    Grid
        Longitudes in -180..180 unless the slab crosses the dateline.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    d = pd.read_csv(path, header=None, names=["lon", "lat", "z"], na_values=["NaN", "nan"])
    if d.empty:
        raise ValueError(f"{path} is empty")
    z = d["z"].to_numpy(float) * (-1 if "dep" in path.stem.split("_") else 1)
    g = Grid.from_points((d["lon"].to_numpy(float) + 180) % 360 - 180, d["lat"].to_numpy(float), z)
    return Grid(g.values, g.x_edges, g.y_edges, mask=np.isnan(g.values))
