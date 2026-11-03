"""
E2. Pattern and catalog
=======================

E1 plus the catalog with a magnitude legend. Reference:
``pattern_catalog_map.jpg``. Half a cell from it, as in E1.
"""
import os
from pathlib import Path

from pituplots import add_locator, plot_catalog, plot_grid, theme
from pituplots.io import read_catalog, read_rate_csv

data = Path(os.environ["PITUPLOTS_DATA"])
out = Path(os.environ.get("PITUPLOTS_OUT", Path(__file__).parent / "output"))
out.mkdir(parents=True, exist_ok=True)

g = read_rate_csv(data / "lacq/outputs/lacq/pattern.csv", dh=0.01)
ext = [g.x_edges[0], g.x_edges[-1], g.y_edges[0], g.y_edges[-1]]
with theme("report"):
    ax = plot_grid(g, log=True, extent=ext, basemap="ESRI_relief", clabel=r"$\log_{10}$ probability per cell",
                   coastline=False)
    plot_catalog(read_catalog(data / "lacq/outputs/lacq/events.csv"), ax=ax, mag_ticks=[1.5, 2.5, 3.5, 4],
                 legend_loc="upper right")
    add_locator(ax, [-5.5, 9, 41, 51.5], extent=ext)
    for s in ("png", "pdf"):
        ax.figure.savefig(out / f"e2_pattern_catalog.{s}", dpi=200, bbox_inches="tight")
