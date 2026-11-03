"""
E9. Faults by slip rate
=======================

NZ Community Fault Model on the EPSG:2193 basemap, coloured and scaled by slip
rate on a nonlinear norm, with the CSEP region and the Alpine fault labelled.
Reference: ``faults_map.jpg``.
"""
import os
from pathlib import Path

import numpy as np
from matplotlib.colors import SymLogNorm

from pituplots import add_labels, plot_lines, plot_region, theme
from pituplots.io import read_csep_nodes, read_vector

data = Path(os.environ["PITUPLOTS_DATA"])
out = Path(os.environ.get("PITUPLOTS_OUT", Path(__file__).parent / "output"))
out.mkdir(parents=True, exist_ok=True)

d = data / "nonpoisson_nz_src/data"
r = read_vector(d / "NZ_CFM_v1_0_shapefile/NZ_CFM_v1_0.shp")
region = read_csep_nodes(data / "pycsep/csep/artifacts/Regions/nz.testing.nodes.dat")
with theme("report"):
    ax = plot_lines(r, values="SR_pref", norm=SymLogNorm(0.1, linscale=0.2, vmin=0, vmax=30), lw_by=(0.4, 3.5),
                    cmap="magma", extent=[164.5, 181.5, -48.5, -33.5], projection="native", figsize=(7.5, 9),
                    basemap=d / "basemaps/new_zealand/basemap_2193.tiff", coastline=False,
                    clabel="Slip rate [mm/year]")
    plot_region(region, ax=ax, color="#3b4cc0", lw=0.6)
    names = r[1]["Name"].astype(str)
    alpine = next(g for g, n in zip(r[0], names) if "Alpine" in n)
    from pyproj import Transformer

    x, y = Transformer.from_crs(r[2], "EPSG:4326", always_xy=True).transform(*alpine.interpolate(0.5, True).xy)
    add_labels(ax, np.array(x), np.array(y), ["Alpine fault"], fontsize=14, offset=(-70, 12))
    for s in ("png", "pdf"):
        ax.figure.savefig(out / f"e9_faults.{s}", dpi=200, bbox_inches="tight")
