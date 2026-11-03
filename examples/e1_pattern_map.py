"""
E1. SSM pattern map
===================

Spatial pattern on the ESRI relief basemap, with a locator inset. Reference:
``pattern_map.jpg``. The points are cell centres, so the map is half a cell
from the reference, which read them as origins (F5).
"""
import os
from pathlib import Path

from pituplots import add_locator, plot_grid, theme
from pituplots.io import read_rate_csv

data = Path(os.environ["PITUPLOTS_DATA"])
out = Path(os.environ.get("PITUPLOTS_OUT", Path(__file__).parent / "output"))
out.mkdir(parents=True, exist_ok=True)

g = read_rate_csv(data / "lacq/outputs/lacq/pattern.csv", dh=0.01)
ext = [g.x_edges[0], g.x_edges[-1], g.y_edges[0], g.y_edges[-1]]
with theme("report"):
    ax = plot_grid(g, log=True, extent=ext, basemap="ESRI_relief", clabel=r"$\log_{10}$ probability per cell",
                   coastline=False)
    add_locator(ax, [-5.5, 9, 41, 51.5], extent=ext)
    for s in ("png", "pdf"):
        ax.figure.savefig(out / f"e1_pattern_map.{s}", dpi=200, bbox_inches="tight")
