"""
E14. Slab2 depth
================

Slab2 South America depth with labelled contours. No reference figure.
"""
import os
from pathlib import Path

from pituplots import plot_field, theme
from pituplots.io import read_slab2

data = Path(os.environ["PITUPLOTS_DATA"])
out = Path(os.environ.get("PITUPLOTS_OUT", Path(__file__).parent / "output"))
out.mkdir(parents=True, exist_ok=True)

g = read_slab2(data / "psha_chile/data/slab_2.0/sam_slab2_dep_02.23.18.xyz")
levels = [0, 25, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700]
with theme("report"):
    ax = plot_field(g, kind="contourf", levels=levels, contour_labels=True, smooth=2, cmap="viridis_r", alpha=0.9,
                    clabel="Slab depth [km]", figsize=(6, 9), borders=True)
    for s in ("png", "pdf"):
        ax.figure.savefig(out / f"e14_slab.{s}", dpi=200, bbox_inches="tight")
