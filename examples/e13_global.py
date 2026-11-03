"""
E13. Global forecast
====================

GEAR1 on a global Robinson map. No reference figure.
"""
import os
from pathlib import Path

import cartopy.crs as ccrs

from pituplots import plot_grid, theme
from pituplots.io import read_csep_grid

data = Path(os.environ["PITUPLOTS_DATA"])
out = Path(os.environ.get("PITUPLOTS_OUT", Path(__file__).parent / "output"))
out.mkdir(parents=True, exist_ok=True)

g = read_csep_grid(data / "pycsep/csep/artifacts/ExampleForecasts/GriddedForecasts/GEAR1_downsampled.dat")
with theme("report"):
    ax = plot_grid(g, log=True, set_global=True, projection=ccrs.Robinson(), figsize=(11, 6), alpha_exp=0,
                   cmap="inferno", vmin=-9, clabel=r"$\log_{10}$ rate per cell")
    for s in ("png", "pdf"):
        ax.figure.savefig(out / f"e13_global.{s}", dpi=200, bbox_inches="tight")
