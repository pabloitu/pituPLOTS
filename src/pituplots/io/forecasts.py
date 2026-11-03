import re
from pathlib import Path

import numpy as np
import pandas as pd

from pituplots.errors import need
from pituplots.io.containers import Grid, find_column


def is_mag(c):
    try:
        return -1 < float(c) < 12
    except ValueError:
        return False


def boxes_to_grid(x0, x1, y0, y1, rates, mask, mags, mag_min, mag_max):
    use = np.ones(mags.size, bool)
    if mag_min is not None:
        use &= mags >= mag_min
    if mag_max is not None:
        use &= mags < mag_max
    if not use.any():
        raise ValueError(f"no magnitude bin in [{mag_min}, {mag_max}), have {mags.min()} to {mags.max()}")
    d = pd.DataFrame({"x0": x0, "y0": y0, "v": rates[:, use].sum(axis=1), "m": mask})
    d = d.groupby(["x0", "y0"], sort=False, as_index=False).agg(v=("v", "sum"), m=("m", "max"))
    v = np.where(d["m"] > 0, d["v"], np.nan)
    g = Grid.from_points(d["x0"], d["y0"], v, dh=(np.median(x1 - x0), np.median(y1 - y0)), anchor="origin")
    return Grid(g.values, g.x_edges, g.y_edges, mask=np.isnan(g.values),
                meta={"magnitudes": mags, "mag_bins": mags[use]})


def read_csep_grid(path, mag_min=None, mag_max=None):
    """Read a CSEP gridded forecast, summed over magnitude bins.

    Parameters
    ----------
    path : str or Path
        ``.dat`` (lon0, lon1, lat0, lat1, z0, z1, m0, m1, rate, mask),
        ``.csv`` (lon_min, lon_max, lat_min, lat_max, one column per
        magnitude, optional mask) or ``.hdf5`` (needs h5py).
    mag_min, mag_max : float, optional
        Bins to sum, by lower edge: ``mag_min <= m < mag_max``.

    Returns
    -------
    Grid
        Cells with CSEP mask 0 are masked. Bins are in ``meta``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    kind = path.suffix.lower()
    if kind == ".dat":
        a = np.loadtxt(path, ndmin=2)
        if a.shape[1] != 10:
            raise ValueError(f"{path} has {a.shape[1]} columns, expected 10")
        mags = np.unique(a[:, 6])
        cells, inv = np.unique(a[:, :4], axis=0, return_inverse=True)
        rates = np.zeros((cells.shape[0], mags.size))
        inv = np.ravel(inv)
        rates[inv, np.searchsorted(mags, a[:, 6])] += a[:, 8]
        mask = pd.Series(a[:, 9]).groupby(inv).max().to_numpy()
        x0, x1, y0, y1 = cells.T
    elif kind == ".csv":
        d = pd.read_csv(path, skipinitialspace=True)
        d.columns = [c.lstrip("#").strip() for c in d.columns]
        cols = [c for c in d.columns if is_mag(c)]
        missing = {"lon_min", "lon_max", "lat_min", "lat_max"} - set(d.columns)
        if missing or not cols:
            raise ValueError(f"{path} needs lon_min, lon_max, lat_min, lat_max and magnitude columns, "
                             f"found {list(d.columns)}")
        mags, rates = np.array(cols, float), d[cols].to_numpy(float)
        mask = d["mask"].to_numpy(float) if "mask" in d else np.ones(len(d))
        x0, x1, y0, y1 = (d[c].to_numpy(float) for c in ("lon_min", "lon_max", "lat_min", "lat_max"))
    elif kind in (".hdf5", ".h5"):
        with need("h5py", "hdf5").File(path, "r") as f:
            rates, mags, b, mask = f["rates"][:], f["magnitudes"][:], f["bboxes"][:], f["poly_mask"][:]
        x0, x1, y0, y1 = b[:, :, 0].min(1), b[:, :, 0].max(1), b[:, :, 1].min(1), b[:, :, 1].max(1)
    else:
        raise ValueError(f"unsupported forecast format {kind!r}")
    return boxes_to_grid(x0, x1, y0, y1, rates, mask, mags, mag_min, mag_max)


def read_csep_nodes(path, dh=None):
    """Read a ``*.nodes.dat`` file of cell origins.

    Parameters
    ----------
    path : str or Path
    dh : float, optional
        Inferred if missing.

    Returns
    -------
    Grid
        NaN values, for outlines. Cells outside the region are masked.
    """
    a = np.loadtxt(path, ndmin=2)
    return Grid.from_points(a[:, 0], a[:, 1], np.full(a.shape[0], np.nan), dh=dh, anchor="origin")


def read_rate_csv(path, mag_min=None, anchor="center", dh=None):
    """Read a rate table into a grid.

    Parameters
    ----------
    path : str or Path
        Columns ``lon``, ``lat`` and either ``p`` or ``rate_M<lo>_<hi>``.
    mag_min : float, optional
        Sum the bins from this magnitude up.
    anchor : {"center", "origin"}
        Whether points are cell centres or lower-left corners.
    dh : float or (float, float), optional
        Inferred if missing.

    Returns
    -------
    Grid
    """
    d = pd.read_csv(path)
    bins = {c: float(m.group(1)) for c in d.columns if (m := re.fullmatch(r"rate_M([0-9.]+)_[0-9.]+", c))}
    meta = {}
    if bins:
        cols = [c for c, m in bins.items() if mag_min is None or m >= mag_min - 1e-9]
        if not cols:
            raise ValueError(f"no rate column from M{mag_min}, have {list(bins)}")
        v = d[cols].to_numpy(float).sum(axis=1)
        meta["mag_bins"] = np.array([bins[c] for c in cols])
    elif "p" in d:
        v = d["p"].to_numpy(float)
    else:
        raise ValueError(f"no p or rate_M* columns in {list(d.columns)}")
    g = Grid.from_points(find_column(d, "lon").to_numpy(float), find_column(d, "lat").to_numpy(float), v,
                         dh=dh, anchor=anchor)
    g.meta.update(meta)
    return g
