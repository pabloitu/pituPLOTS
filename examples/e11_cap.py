"""
E11. Cells with capped and empty sites
======================================

Background rate per cell, capped cells in cyan, fault buffers, and the sites of
``chile_01.csv`` that have no rate. Reference: ``s04_cap.png``.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from pituplots import plot_polygons, plot_sites, theme
from pituplots.io import read_vector

data = Path(os.environ["PITUPLOTS_DATA"]) / "psha_chile"
out = Path(os.environ.get("PITUPLOTS_OUT", Path(__file__).parent / "output"))
out.mkdir(parents=True, exist_ok=True)

o = data / "psha_chile6/outputs/crustal/ref"
t = pd.read_csv(o / "ssm/grid_total.csv")
c = pd.read_csv(o / "ssm/grid_capped.csv")
cols = [k for k in t.columns if k.startswith("rate_M") and float(k[6:].split("_")[0]) >= 6 - 1e-6]
t["n6"] = t[cols].sum(axis=1)
s = pd.read_csv(data / "data/grids/chile_01.csv")
empty = cKDTree(t[["lon", "lat"]]).query(s[["lon", "lat"]])[0] > 0.1
d = pd.concat([t[["lon", "lat", "n6"]], s.loc[empty, ["lon", "lat"]].assign(n6=np.nan)])
with theme("report"):
    ax = plot_sites(d, color_by="n6", log=True, dh=0.1, empty_color="0.6", cmap="magma_r", figsize=(6, 9),
                    clabel=r"$\log_{10}$ N(>=6)", extent=[-77, -64, -58, -16], title="background and capped cells (cyan)")
    k = c["in_buffer"].to_numpy(bool)
    plot_sites((c["lon"][k], c["lat"][k]), ax=ax, dh=0.1, color="cyan", alpha=0.4, colorbar=False)
    plot_polygons(read_vector(o / "faults/union.geojson"), ax=ax, lw=0.5)
    for s in ("png", "pdf"):
        ax.figure.savefig(out / f"e11_cap.{s}", dpi=200, bbox_inches="tight")
