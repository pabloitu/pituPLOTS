"""
E6. Forecast and event log-likelihood
=====================================

ESHM13 rates with the test events coloured by the log-likelihood of their
bin, rings on events in a zero-rate bin of a branch, and two colorbars.
Reference: ``eshm13_zero_rate_map.png``. As there, ``alpha=0.6`` overrides the
ramp, so the source's ``alpha_exp`` has no effect (scope §6.6).
"""
import os
from pathlib import Path

from functools import partial

import numpy as np
import pandas as pd

from pituplots import Catalog, Grid, add_colorbar, highlight, plot_catalog, plot_grid, size_by_value, theme
from pituplots.io import read_catalog

data = Path(os.environ["PITUPLOTS_DATA"]) / "testing_eshm20"
out = Path(os.environ.get("PITUPLOTS_OUT", Path(__file__).parent / "output"))
out.mkdir(parents=True, exist_ok=True)

W = {"asm_2013.csv": 0.5, "fsbg_2013.csv": 0.2, "seifa_2013.csv": 0.3}
t0, t1 = np.datetime64("2015-01-01"), np.datetime64("2026-06-01")
yrs = float((t1 - t0) / np.timedelta64(1, "D")) / 365.25
cells = np.loadtxt(data / "region_europe.txt", delimiter=",")
reg = set(zip(np.floor(cells[:, 0] * 10).astype(int), np.floor(cells[:, 1] * 10).astype(int)))

c = read_catalog(data / "catalog_emec_gcmt.json", preset="csep_json")
ix, iy = np.floor(c.lon * 10).astype(int), np.floor(c.lat * 10).astype(int)
sel = (c.time >= t0) & (c.time < t1) & (c.mag >= 4.9) & (c.depth >= -2) & (c.depth <= 70)
sel &= np.array([k in reg for k in zip(ix, iy)])
ev = Catalog(c.lon[sel], c.lat[sel], c.depth[sel], c.mag[sel])
key, tot, nz, rate = list(zip(ix[sel], iy[sel])), np.zeros(ev.lon.size), np.zeros(ev.lon.size, int), {}
for f, w in W.items():
    d = pd.read_csv(data / "eshm_13" / f)
    d.columns = [x.strip() for x in d.columns]
    mc = [x for x in d.columns[6:] if float(x) >= 4.9 - 1e-6]
    ij = list(zip(np.rint(d.lon_min * 10).astype(int), np.rint(d.lat_min * 10).astype(int)))
    k = dict(zip(ij, range(len(d))))
    R = d[mc].to_numpy(float) * yrs
    b = np.clip(np.floor((ev.mag - 4.9) / 0.2 + 1e-9).astype(int), 0, len(mc) - 1)
    r = np.array([R[k[q], i] if q in k else 0.0 for q, i in zip(key, b)])
    tot += w * r
    nz += r == 0
    for q, v in zip(ij, d[mc].to_numpy(float).sum(1)):
        rate[q] = rate.get(q, 0.0) + w * v

q = np.array([p for p in rate if p in reg])
v = np.array([rate[tuple(p)] for p in q])
g = Grid.from_points(q[:, 0] / 10 + 0.05, q[:, 1] / 10 + 0.05, v, dh=0.1)
ok = tot > 0
ll = np.log(tot[ok])
sizer = partial(size_by_value, size=12, max_size=200, power=1.5, min_val=4.9, max_val=7.8)
ext = [ev.lon.min() - 1.5, ev.lon.max() + 1.5, ev.lat.min() - 1.5, ev.lat.max() + 1.5]
with theme("report"):
    ax = plot_grid(g, log=True, cmap="Greys", alpha=0.6, extent=ext, basemap="ESRI_terrain", figsize=(13, 7.5),
                   vmin=np.nanquantile(np.log10(v), 0.02), vmax=np.nanquantile(np.log10(v), 0.995),
                   clabel=r"$\log_{10}\ \lambda\ [yr^{-1}],\ M\geq4.9$", title="ESHM13")
    good = Catalog(ev.lon[ok], ev.lat[ok], mag=ev.mag[ok], extra={"ll": ll})
    plot_catalog(good, ax=ax, color_by="ll", cmap="viridis", sizer=sizer, alpha=None, legend=False, colorbar=False,
                 vmin=np.quantile(ll, 0.02), vmax=np.quantile(ll, 0.98), linewidths=0.4)
    add_colorbar(ax, ax.collections[-1], loc="bottom", label=r"Event log-likelihood, $\ln\ \lambda_{bin}$")
    z = nz > 0
    highlight(ev, ax, where=z, sizer=sizer, labels=[f"{n}/{len(W)}" for n in nz[z]], linewidths=2.2)
    for s in ("png", "pdf"):
        ax.figure.savefig(out / f"e6_zero_rate.{s}", dpi=250, bbox_inches="tight")
