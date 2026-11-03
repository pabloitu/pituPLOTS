"""
E15. OpenQuake source model
===========================

Crustal simple faults coloured by total rate, interface complex faults (top
edge, bottom edge dashed) and in-slab point sources, read from NRML without
OpenQuake, on a map and in a depth section. References: ``fig_sources`` and
``fig_section`` of ``psha_chile7/hazard/figures.py``.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cartopy.crs as ccrs

from pituplots import Catalog, add_profile_map, make_profile, plot_section, plot_sources, project, theme
from pituplots.io import read_sources
from pituplots.maps import dress_map

data = Path(os.environ["PITUPLOTS_DATA"]) / "psha_chile/psha_chile6/outputs"
out = Path(os.environ.get("PITUPLOTS_OUT", Path(__file__).parent / "output"))
out.mkdir(parents=True, exist_ok=True)

parts = [read_sources(data / "crustal/ref/nrml/faults__reference.xml"),
         read_sources(data / "interface/mmin_haz-5.5/nrml/sub__seg__seis__tgr.xml")]
slab = sorted((data / "intraslab/ref/nrml").glob("*.xml"))
if slab:
    parts.append(read_sources(slab[0]))
src = pd.concat(parts, ignore_index=True)
print(src.groupby("kind").size())

p = make_profile((-71.5, -31), azimuth=90, length=(-150, 600), width=60)
pts = src[src["kind"] == "point"]
with theme("report"):
    fig = plt.figure(figsize=(14, 9), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.6])
    ax = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
    dress_map(ax, extent=[-78, -64, -46, -17])
    plot_sources(src, ax=ax, color_by="rate", log=True, clabel=r"$\log_{10}$ annual rate", size=2)
    add_profile_map(ax, p, labels=["A"], color="#d62728")
    a = fig.add_subplot(gs[1])
    if len(pts):
        z = (pts["upper_depth"] + pts["lower_depth"]) / 2
        cat = Catalog([g.x for g in pts["geometry"]], [g.y for g in pts["geometry"]], z.to_numpy(),
                      np.log10(pts["rate"].to_numpy()))
        plot_section(cat, p, ax=a, ve=2, zmax=250, color_by="mag", cmap="viridis", size=3, max_size=40,
                     legend=False, title="In-slab point sources, section A")
    for e in src.loc[src["kind"] == "complex", "edges"]:
        for k in (0, -1):
            x, y = project(e[k][:, 0], e[k][:, 1], p)
            m = np.abs(y) <= p.width / 2
            a.plot(x[m], e[k][m, 2], "o", color="#4c72b0", ms=4)
    for s in ("png", "pdf"):
        fig.savefig(out / f"e15_sources.{s}", dpi=200)
