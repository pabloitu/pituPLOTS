import json
import re
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, Polygon

from pituplots.errors import ConsistencyError, PituplotsWarning, need
from pituplots.io.containers import Sites


def text(v):
    return v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v)


def poe_at(curves, imls, poe):
    """IML at a PoE by log-log interpolation along the last axis, as OpenQuake does.

    A curve that never reaches ``poe`` gives 0; one above it at the last level
    gives the last level.
    """
    c = np.asarray(curves, float)
    lx = np.log(np.asarray(imls, float))[::-1]
    out = np.zeros(c.shape[:-1])
    for i in np.ndindex(out.shape):
        y = c[i][::-1]
        ok = y > 0
        if not ok.any() or poe > y[ok].max():
            continue
        out[i] = np.exp(np.interp(np.log(poe), np.log(y[ok]), lx[ok]))
    return out


def wquantile(c, w, q):
    """Weighted quantile across axis 1 of ``(sites, rlzs, levels)`` curves."""
    o = np.argsort(c, axis=1)
    s = np.take_along_axis(c, o, axis=1)
    cw = np.cumsum(w[o], axis=1) - 0.5 * w[o]
    out = np.empty((c.shape[0], c.shape[2]))
    for i in range(c.shape[0]):
        for j in range(c.shape[2]):
            out[i, j] = np.interp(q, cw[i, :, j], s[i, :, j])
    return out


class OQResults:
    """Hazard curves of an OpenQuake datastore, read lazily with h5py.

    Attributes
    ----------
    sites : Sites
    imtls : dict
        IMT name to its levels.
    investigation_time : float or None
    stats : list of str
        Names in ``hcurves-stats``, such as ``"mean"`` or ``"quantile-0.15"``.
    weights : ndarray or None
        Realization weights, when ``hcurves-rlzs`` is present.
    """

    def __init__(self, path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        h5py = need("h5py", "hdf5")
        with h5py.File(self.path, "r") as f:
            op = f["oqparam"][()]
            op = json.loads(text(op)) if isinstance(op, (bytes, str, np.bytes_)) else {}
            self.oqparam = op
            imtls = op.get("hazard_imtls") or op.get("imtls")
            if not imtls:
                raise ValueError(f"no IMLs in {self.path}: hazard_imtls is missing from oqparam")
            self.imtls = {k: np.asarray(v, float) for k, v in imtls.items()}
            self.investigation_time = op.get("investigation_time")
            sc = f["sitecol"]
            lon, lat = (np.asarray(sc["lon"]), np.asarray(sc["lat"]))
            names = None
            if "custom_site_id" in (sc.keys() if hasattr(sc, "keys") else sc.dtype.names or ()):
                names = np.array([text(v) for v in np.asarray(sc["custom_site_id"])])
            self.sites = Sites(lon, lat, names=names)
            self.keys = [k for k in ("hcurves-rlzs", "hcurves-stats") if k in f]
            if not self.keys:
                raise ValueError(f"no hazard curves in {self.path}")
            self.stats = []
            if "hcurves-stats" in f:
                self.stats = json.loads(f["hcurves-stats"].attrs["json"])["stat"]
            self.weights = None
            if "hcurves-rlzs" in f:
                w = np.asarray(f["weights"][()], float)
                self.weights = (w[:, -1] if w.ndim > 1 else w) / (w[:, -1] if w.ndim > 1 else w).sum()
        if "hcurves-rlzs" in self.keys and "mean" in self.stats:
            self.check()

    def __repr__(self):
        n = None if self.weights is None else self.weights.size
        return (f"OQResults({self.path.name}: {self.sites.lon.size} sites, {n} realizations, "
                f"stats {self.stats}, IMTs {list(self.imtls)})")

    def imt(self, imt):
        if imt is None:
            if len(self.imtls) > 1:
                raise ValueError(f"several IMTs, choose one of {list(self.imtls)}")
            imt = next(iter(self.imtls))
        if imt not in self.imtls:
            raise ValueError(f"no IMT {imt!r}, have {list(self.imtls)}")
        return imt, list(self.imtls).index(imt)

    def read(self, key, j, k=None):
        import h5py

        with h5py.File(self.path, "r") as f:
            return np.asarray(f[key][:, :, j, :] if k is None else f[key][:, k, j, :], float)

    def curves(self, imt=None):
        """PoE curves of every realization, ``(sites, rlzs, levels)``."""
        if "hcurves-rlzs" not in self.keys:
            raise ValueError(f"{self.path.name} has no hcurves-rlzs")
        return self.read("hcurves-rlzs", self.imt(imt)[1])

    def stat(self, name, imt=None):
        return self.read("hcurves-stats", self.imt(imt)[1], self.stats.index(name))

    def mean(self, imt=None):
        """Mean curve per site, ``(sites, levels)``; the stored one if present."""
        if "mean" in self.stats:
            return self.stat("mean", imt)
        return np.einsum("r,srl->sl", self.weights, self.curves(imt))

    def quantile(self, q, imt=None):
        """Weighted quantile per site, ``(sites, levels)``; the stored one if present."""
        name = f"quantile-{q}"
        if name in self.stats:
            return self.stat(name, imt)
        return wquantile(self.curves(imt), self.weights, q)

    def to_window(self, curves, T):
        """PoE in ``T`` years from PoE in the investigation time (Poisson)."""
        if self.investigation_time is None:
            raise ValueError("investigation_time is unknown: cannot convert PoEs to another window")
        return 1 - (1 - np.asarray(curves, float)) ** (T / self.investigation_time)

    def maps(self, poes, imt=None, window=None, curves=None):
        """Hazard map values per site at each PoE.

        Parameters
        ----------
        poes : float or sequence
        imt : str, optional
        window : float, optional
            Years the PoEs refer to. Default the investigation time.
        curves : array, optional
            ``(sites, levels)``; default the mean.

        Returns
        -------
        Sites
            ``values`` is ``(sites,)`` for one PoE, else ``(sites, poes)``.
        """
        imt, _ = self.imt(imt)
        c = self.mean(imt) if curves is None else np.asarray(curves, float)
        if window is not None:
            c = self.to_window(c, window)
        ps = np.atleast_1d(poes)
        v = np.stack([poe_at(c, self.imtls[imt], p) for p in ps], axis=-1)
        return Sites(self.sites.lon, self.sites.lat, v[:, 0] if np.ndim(poes) == 0 else v,
                     meta={"imt": imt, "poes": list(ps), "window": window or self.investigation_time})

    def check(self, tol=1e-3):
        """Weighted mean of the realizations against the stored mean.

        Compared in rate space when ``oqparam.use_rates`` is set. Raises
        ``ConsistencyError`` above ``tol``.
        """
        for imt in self.imtls:
            c, m = self.curves(imt), self.stat("mean", imt)
            if self.oqparam.get("use_rates"):
                c, m = -np.log1p(-np.minimum(c, 1 - 1e-15)), -np.log1p(-np.minimum(m, 1 - 1e-15))
            mean = np.einsum("r,srl->sl", self.weights, c)
            ok = m > 1e-8
            err = float(np.max(np.abs(mean[ok] / m[ok] - 1))) if ok.any() else 0.0
            if err > tol:
                raise ConsistencyError(f"{self.path.name}, {imt}: weighted mean of the realizations differs "
                                       f"from hcurves-stats by {err:.2e}; weights or curves are inconsistent")

    @property
    def branches(self):
        """Branch labels per realization, or None without openquake."""
        try:
            from openquake.commonlib import datastore
        except ImportError:
            return None
        ds = datastore.read(str(self.path))
        try:
            rl = sorted(ds["full_lt"].get_realizations(), key=lambda x: x.ordinal)
            trts = list(ds["full_lt"].gsim_lt.values)
            rows = [{"source": "_".join(x.sm_lt_path),
                     **{t: type(g).__name__ for t, g in zip(trts, x.gsim_rlz.value)}} for x in rl]
        finally:
            ds.close()
        return pd.DataFrame(rows)


def read_datastore(path):
    """Read an OpenQuake ``calc_*.hdf5`` datastore.

    Curves, weights and sites need h5py only. Branch labels need openquake and
    are None without it; they are never guessed. When both realizations and
    the mean are stored, their consistency is checked on opening.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    OQResults
    """
    return OQResults(path)


def read_csv(path):
    """Read an OpenQuake CSV export: hazard curves, maps or uniform hazard spectra.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    (Sites, dict)
        ``values`` is one row per site. ``meta`` has the header fields
        (``kind``, ``investigation_time``, ``imt``, ...), ``"type"``
        (``"curves"``, ``"map"`` or ``"uhs"``), and ``"imls"`` for curves or
        ``"columns"`` as ``(imt, poe)`` pairs for maps and spectra.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with open(path) as f:
        first = f.readline()
    meta = {}
    if first.startswith("#"):
        for k, v in re.findall(r"(\w+)\s*=\s*'?([^,'\"]+)'?", first):
            meta[k] = v.strip()
        if "investigation_time" in meta:
            meta["investigation_time"] = float(meta["investigation_time"])
    d = pd.read_csv(path, comment="#" if first.startswith("#") else None, skiprows=1 if first.startswith("#") else 0)
    cols = [c for c in d.columns if c not in ("lon", "lat", "depth", "custom_site_id", "site_id")]
    lon, lat = d["lon"].to_numpy(float), d["lat"].to_numpy(float)
    if all(c.startswith("poe-") for c in cols):
        meta.update(type="curves", imls=np.array([float(c[4:]) for c in cols]))
    elif all("~" in c for c in cols):
        meta.update(type="uhs", columns=[(c.split("~", 1)[1], float(c.split("~", 1)[0])) for c in cols])
    else:
        pairs = [re.fullmatch(r"(.+?)-([0-9.]+(?:[eE][+-]?\d+)?)", c) for c in cols]
        if not all(pairs):
            raise ValueError(f"unknown OpenQuake CSV columns in {path.name}: {cols[:5]}")
        meta.update(type="map", columns=[(m.group(1), float(m.group(2))) for m in pairs])
    return Sites(lon, lat, d[cols].to_numpy(float)), meta


NS = {"nrml": None, "gml": "{http://www.opengis.net/gml}"}
KINDS = {"simpleFaultSource": "simple", "complexFaultSource": "complex", "pointSource": "point",
         "areaSource": "area", "multiPointSource": "multipoint", "characteristicFaultSource": "characteristic"}


def poslist(el, ncol):
    p = el.find(f".//{NS['gml']}posList")
    if p is None:
        p = el.find(f".//{NS['gml']}pos")
    return np.array(p.text.split(), float).reshape(-1, ncol)


def mfd(src, ns):
    for el in src:
        tag = el.tag.split("}")[1]
        if tag == "incrementalMFD":
            r = np.array(el.find(f"{ns}occurRates").text.split(), float)
            m0, dm = float(el.get("minMag")), float(el.get("binWidth"))
            return {"type": "incremental", "min_mag": m0, "bin_width": dm, "rates": r}, float(r.sum()), \
                m0 + dm * (r.size - 1)
        if tag == "truncGutenbergRichterMFD":
            a, b = float(el.get("aValue")), float(el.get("bValue"))
            m0, m1 = float(el.get("minMag")), float(el.get("maxMag"))
            return {"type": "truncGR", "a": a, "b": b, "min_mag": m0, "max_mag": m1}, \
                10 ** (a - b * m0) - 10 ** (a - b * m1), m1
    return None, np.nan, np.nan


def read_sources(path):
    """Read an OpenQuake NRML source model (0.4 or 0.5) without openquake.

    Complex faults in the deprecated edge order (bottom edge to the left of the
    top edge's strike) have every edge flipped, intermediate ones included.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    DataFrame
        One row per source: ``id``, ``name``, ``kind`` (simple, complex,
        point, area, multipoint, characteristic), ``trt``, ``group``,
        ``geometry`` (trace, top edge, point or polygon, 2D), ``edges`` (list
        of ``(n, 3)`` arrays, complex faults), ``dip``, ``upper_depth``,
        ``lower_depth``, ``rake``, ``mfd`` (dict), ``rate`` (total annual
        rate), ``mmax``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    rows, group = [], None
    for ev, el in ET.iterparse(path, events=("start", "end")):
        tag = el.tag.split("}")[-1]
        ns = el.tag[:el.tag.index("}") + 1] if "}" in el.tag else ""
        if ev == "start":
            if tag == "sourceGroup":
                group = el.get("name") or el.get("tectonicRegion")
            continue
        if tag not in KINDS:
            continue
        r = {"id": el.get("id"), "name": el.get("name"), "kind": KINDS[tag], "trt": el.get("tectonicRegion"),
             "group": group, "edges": None, "dip": np.nan, "upper_depth": np.nan, "lower_depth": np.nan,
             "rake": np.nan}
        r["mfd"], r["rate"], r["mmax"] = mfd(el, ns)
        rk = el.find(f"{ns}rake")
        if rk is not None:
            r["rake"] = float(rk.text)
        g = el.find(f"{ns}simpleFaultGeometry")
        if g is not None:
            r["geometry"] = LineString(poslist(g, 2))
            r["dip"] = float(g.find(f"{ns}dip").text)
            r["upper_depth"] = float(g.find(f"{ns}upperSeismoDepth").text)
            r["lower_depth"] = float(g.find(f"{ns}lowerSeismoDepth").text)
        g = el.find(f"{ns}complexFaultGeometry")
        if g is not None:
            edges = [poslist(e, 3) for e in (g.find(f"{ns}faultTopEdge"), *g.findall(f"{ns}intermediateEdge"),
                                              g.find(f"{ns}faultBottomEdge"))]
            top, bot = edges[0], edges[-1]
            s = top[-1, :2] - top[0, :2]
            d = bot[:, :2].mean(axis=0) - top[:, :2].mean(axis=0)
            if s[1] * d[0] - s[0] * d[1] < 0:
                edges = [e[::-1] for e in edges]
            r["edges"] = edges
            r["geometry"] = LineString(edges[0][:, :2])
            r["upper_depth"], r["lower_depth"] = float(edges[0][:, 2].min()), float(edges[-1][:, 2].max())
        for gt in ("pointGeometry", "areaGeometry"):
            g = el.find(f"{ns}{gt}")
            if g is not None:
                xy = poslist(g, 2)
                r["geometry"] = Point(xy[0]) if gt == "pointGeometry" else Polygon(xy)
                r["upper_depth"] = float(g.find(f"{ns}upperSeismoDepth").text)
                r["lower_depth"] = float(g.find(f"{ns}lowerSeismoDepth").text)
                np_ = el.find(f"{ns}nodalPlaneDist/{ns}nodalPlane")
                if np_ is not None:
                    r["rake"], r["dip"] = float(np_.get("rake")), float(np_.get("dip"))
        g = el.find(f"{ns}multiPointGeometry")
        if g is not None:
            from shapely.geometry import MultiPoint

            r["geometry"] = MultiPoint(poslist(g, 2))
        if "geometry" not in r:
            warnings.warn(f"source {r['id']} ({tag}) has no geometry this reader knows", PituplotsWarning,
                          stacklevel=2)
            r["geometry"] = None
        rows.append(r)
        el.clear()
    cols = ["id", "name", "kind", "trt", "group", "geometry", "edges", "dip", "upper_depth", "lower_depth", "rake",
            "mfd", "rate", "mmax"]
    return pd.DataFrame(rows, columns=cols)
