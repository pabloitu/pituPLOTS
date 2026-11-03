from dataclasses import dataclass, field

import numpy as np


def _column(x, n, name, dtype=None):
    x = np.asarray(x, dtype=dtype)
    if x.ndim == 0 or x.shape[0] != n:
        raise ValueError(f"{name} has shape {x.shape}, expected {n} rows")
    return x


@dataclass(eq=False)
class Catalog:
    """Event catalog. Depth in km, positive down; time as naive UTC."""

    lon: np.ndarray
    lat: np.ndarray
    depth: np.ndarray | None = None
    mag: np.ndarray | None = None
    time: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.lon = np.atleast_1d(np.asarray(self.lon, float))
        if self.lon.ndim != 1:
            raise ValueError(f"lon has shape {self.lon.shape}, expected 1-D")
        n = self.lon.size
        self.lat = _column(self.lat, n, "lat", float)
        if self.depth is not None:
            self.depth = _column(self.depth, n, "depth", float)
        if self.mag is not None:
            self.mag = _column(self.mag, n, "mag", float)
        if self.time is not None:
            self.time = _column(self.time, n, "time", "datetime64[ms]")
        self.extra = {k: _column(self.extra[k], n, k) for k in self.extra}


@dataclass(eq=False)
class Grid:
    """Regular lattice defined by its cell edges.

    ``values`` is ``(ny, nx)``, or ``(ny, nx, 3|4)`` for images, with ``y``
    ascending. ``mask`` is ``True`` where there is no data.
    """

    values: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray
    crs: str = "EPSG:4326"
    mask: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        v = self.values
        if isinstance(v, np.ma.MaskedArray):
            m = np.ma.getmaskarray(v)
            m = m.any(axis=-1) if v.ndim == 3 else m
            v = v.data if v.ndim == 3 else v.astype(float).filled(np.nan)
            self.mask = m if self.mask is None else np.asarray(self.mask, bool) | m
        self.values = np.asarray(v)
        self.x_edges = np.asarray(self.x_edges, float)
        self.y_edges = np.asarray(self.y_edges, float)

        s = self.values.shape
        if not (len(s) == 2 or (len(s) == 3 and s[2] in (3, 4))):
            raise ValueError(f"values has shape {s}, expected (ny, nx) or (ny, nx, 3|4)")
        for name, e, n in (("x_edges", self.x_edges, s[1]), ("y_edges", self.y_edges, s[0])):
            if e.ndim != 1 or e.size != n + 1:
                raise ValueError(f"{name} has shape {e.shape}, expected ({n + 1},)")
            if not (np.diff(e) > 0).all():
                raise ValueError(f"{name} must be strictly ascending")
        if self.mask is not None:
            self.mask = np.asarray(self.mask, bool)
            if self.mask.shape != s[:2]:
                raise ValueError(f"mask has shape {self.mask.shape}, expected {s[:2]}")

    @classmethod
    def from_points(cls, lon, lat, values, dh=None, anchor="center"):
        """Build a lattice from scattered points.

        Parameters
        ----------
        lon, lat : array_like
            Point coordinates, in any order. Longitudes must already be
            continuous: a gap wider than 180 deg raises.
        values : array_like
            One value per point.
        dh : float or (float, float), optional
            Cell size, or ``(dx, dy)``. Inferred from the smallest spacing
            when missing, which needs at least two distinct positions on
            each axis.
        anchor : {"center", "origin"}
            Whether points are cell centres or lower-left corners.

        Returns
        -------
        Grid
            Cells without a point are NaN and masked. Repeated points must
            agree on their value.
        """
        if anchor not in ("center", "origin"):
            raise ValueError(f"anchor must be 'center' or 'origin', not {anchor!r}")
        lon = np.atleast_1d(np.asarray(lon, float))
        lat = _column(lat, lon.size, "lat", float)
        values = _column(values, lon.size, "values", float)
        if lon.size == 0:
            raise ValueError("no points")
        if np.ptp(lon) > 180 and np.diff(np.sort(lon)).max() > 180:
            raise ValueError("lon wraps the dateline: apply unwrap_lon first")

        dx, dy = (None, None) if dh is None else np.broadcast_to(dh, 2).astype(float)
        edges, idx = [], []
        for x, d, name in ((lon, dx, "x"), (lat, dy, "y")):
            x0 = x.min()
            tol = 1e-6 * max(1, np.abs(x).max())
            if d is None:
                s = np.diff(np.sort(x))
                s = s[s > tol]
                if s.size == 0:
                    raise ValueError(f"cannot infer the {name} spacing from one position: pass dh")
                d = np.ptp(x) / np.rint(np.ptp(x) / s.min())
            i = np.rint((x - x0) / d).astype(int)
            if np.abs(x - (x0 + i * d)).max() > 0.05 * d:
                raise ValueError(f"{name} positions are not on a lattice of spacing {d:g}")
            n = i.max() + 1
            edges.append(x0 + np.arange(n + 1) * d - (d / 2 if anchor == "center" else 0))
            idx.append(i)

        ix, iy = idx
        z = np.full((edges[1].size - 1, edges[0].size - 1), np.nan)
        z[iy, ix] = values
        if not np.array_equal(z[iy, ix], values, equal_nan=True):
            raise ValueError("repeated points with different values")
        mask = np.ones(z.shape, bool)
        mask[iy, ix] = False
        return cls(z, edges[0], edges[1], mask=mask if mask.any() else None)


@dataclass(eq=False)
class Sites:
    """Named or valued locations. ``values`` has one row per site."""

    lon: np.ndarray
    lat: np.ndarray
    values: np.ndarray | None = None
    names: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.lon = np.atleast_1d(np.asarray(self.lon, float))
        if self.lon.ndim != 1:
            raise ValueError(f"lon has shape {self.lon.shape}, expected 1-D")
        n = self.lon.size
        self.lat = _column(self.lat, n, "lat", float)
        if self.values is not None:
            self.values = _column(self.values, n, "values")
        if self.names is not None:
            self.names = _column(self.names, n, "names")


@dataclass(eq=False)
class Profile:
    """Cross-section line: origin, azimuth in degrees, along-line range and
    swath width in km."""

    lon0: float
    lat0: float
    azimuth: float
    x_range: tuple
    width: float

    def __post_init__(self):
        x0, x1 = self.x_range
        self.x_range = (float(x0), float(x1))
        if not x0 < x1:
            raise ValueError(f"x_range must be ascending and non-zero, got {self.x_range}")
        if self.width < 0:
            raise ValueError(f"width must not be negative, got {self.width}")
