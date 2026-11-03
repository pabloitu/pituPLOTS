import numpy as np

from pituplots.errors import need
from pituplots.io.containers import Catalog, Grid


def from_csep_catalog(cat):
    """``Catalog`` of a pyCSEP catalog.

    Parameters
    ----------
    cat : csep.core.catalogs.AbstractBaseCatalog

    Returns
    -------
    Catalog
    """
    t = np.asarray(cat.get_epoch_times(), "int64").astype("datetime64[ms]")
    return Catalog(cat.get_longitudes(), cat.get_latitudes(), cat.get_depths(), cat.get_magnitudes(), t)


def from_csep_region(region):
    """``Grid`` of the cells of a pyCSEP ``CartesianGrid2D``, with NaN values, for outlines.

    Parameters
    ----------
    region : csep.core.regions.CartesianGrid2D

    Returns
    -------
    Grid
    """
    o = region.origins()
    return Grid.from_points(o[:, 0], o[:, 1], np.full(len(o), np.nan), dh=region.dh, anchor="origin")


def from_csep_forecast(forecast):
    """``Grid`` of the expected counts of a pyCSEP gridded forecast, summed over magnitudes.

    Parameters
    ----------
    forecast : csep.core.forecasts.GriddedForecast

    Returns
    -------
    Grid
        ``meta["magnitudes"]`` holds the bins.
    """
    o = forecast.region.origins()
    g = Grid.from_points(o[:, 0], o[:, 1], forecast.spatial_counts(), dh=forecast.region.dh, anchor="origin")
    g.meta = {"magnitudes": np.asarray(forecast.magnitudes), "name": forecast.name}
    return g


def to_csep_catalog(cat, name=None):
    """pyCSEP catalog of a ``Catalog``. Needs magnitude, depth and time.

    Parameters
    ----------
    cat : Catalog
    name : str, optional

    Returns
    -------
    csep.core.catalogs.CSEPCatalog
    """
    need("csep", "csep")
    from csep.core.catalogs import CSEPCatalog

    if cat.mag is None or cat.depth is None or cat.time is None:
        raise ValueError("a pyCSEP catalog needs magnitude, depth and time")
    t = cat.time.astype("datetime64[ms]").astype("int64")
    rows = [(str(i).encode(), t[i], cat.lat[i], cat.lon[i], cat.depth[i], cat.mag[i]) for i in range(cat.lon.size)]
    return CSEPCatalog(data=np.array(rows, dtype=CSEPCatalog.dtype), name=name)


def to_csep_forecast(grid, magnitudes=None, name=None):
    """pyCSEP ``GriddedForecast`` of a ``Grid`` in degrees, one magnitude bin unless given.

    Parameters
    ----------
    grid : Grid
        ``values`` is ``(ny, nx)``, or ``(ny, nx, nmag)`` with ``magnitudes``.
    magnitudes : sequence of float, optional
        Lower bin edges. Default ``grid.meta["magnitudes"]``, else ``[0]``.
    name : str, optional

    Returns
    -------
    csep.core.forecasts.GriddedForecast
    """
    need("csep", "csep")
    from csep.core.forecasts import GriddedForecast
    from csep.core.regions import CartesianGrid2D

    if grid.crs != "EPSG:4326":
        raise ValueError("pyCSEP regions are in degrees")
    dx, dy = np.diff(grid.x_edges), np.diff(grid.y_edges)
    if not (np.allclose(dx, dx[0]) and np.allclose(dy, dx[0])):
        raise ValueError("pyCSEP needs square cells of one size")
    m = np.asarray(magnitudes if magnitudes is not None else grid.meta.get("magnitudes", [0.0]), float)
    v = np.asarray(grid.values, float)
    v = v[..., None] if v.ndim == 2 else v
    if v.shape[2] != m.size:
        raise ValueError(f"{v.shape[2]} magnitude bins in values, {m.size} magnitudes")
    keep = np.ones(v.shape[:2], bool) if grid.mask is None else ~grid.mask
    iy, ix = np.nonzero(keep)
    o = np.c_[grid.x_edges[ix], grid.y_edges[iy]]
    region = CartesianGrid2D.from_origins(o, dh=float(dx[0]), magnitudes=m)
    data = np.zeros((len(o), m.size))
    data[region.get_index_of(o[:, 0] + dx[0] / 2, o[:, 1] + dx[0] / 2)] = v[iy, ix]
    return GriddedForecast(data=data, region=region, magnitudes=m, name=name)
