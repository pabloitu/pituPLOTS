"""
E7. Trench-perpendicular sections
=================================

Events by class on sections normal to the Nazca trench, with the Slab2 top and
thickness, labels on the largest events and a locator map. References:
``sections_map.png`` and ``sections.pdf`` of ``cat_handler_2``.
"""
import os
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

from pituplots import (add_gridlines, add_labels, add_profile_map, add_slab, plot_lines, plot_points, plot_section,
                       profiles_from_line, project, theme)
from pituplots.io import read_catalog, read_slab2, read_vector

data = Path(os.environ["PITUPLOTS_DATA"]) / "psha_chile"
out = Path(os.environ.get("PITUPLOTS_OUT", Path(__file__).parent / "output"))
out.mkdir(parents=True, exist_ok=True)

COLORS = {"slab_interface": "#2166ac", "patagonia_interface": "#4393c3", "intra_slab": "#f0505a",
          "slab_deep": "#a50f15", "deep_nest": "#6b3d1a", "forearc": "#ffb000", "intraarc": "#2e8b57",
          "backarc": "#c9a227", "patagonia_crustal": "#8c510a", "outer_rise": "#a6a6a6",
          "deep_unknown": "#4d4d4d", "unclassified": "#c8c8c8", "unresolved": "#7f7f7f"}

cat = read_catalog(data / "results/cat_handler_2/catalog.csv", sep=",")
top = read_slab2(data / "data/slab_2.0/sam_slab2_dep_02.23.18.xyz")
thk = read_slab2(data / "data/slab_2.0/sam_slab2_thk_02.23.18.xyz")
trench = read_vector(data / "data/shapefiles/sam_nazca_trench.shp")[0]

ps = profiles_from_line(trench, [-21, -31, -37], width=2 * 111.2, side=(-60, -30), length=(-150, 900))
big = cat.mag >= 5.5
unknown = np.isin(cat.extra["depth_status"].astype(str), ["fixed", "missing", "assigned"])
year = cat.time.astype("datetime64[Y]").astype(int) + 1970

with theme("slides"):
    fig = plt.figure(figsize=(12.5, 9), layout="constrained")
    gs = fig.add_gridspec(3, 2, width_ratios=[5, 1])
    for i, (p, name) in enumerate(zip(ps, "ABC")):
        a = fig.add_subplot(gs[i, 0])
        plot_section(cat, p, ax=a, ve=1, zmax=300, color_by=cat.extra["class"].astype(str), colors=COLORS,
                     hollow=unknown, min_val=5.5, max_val=8.8, size=2, max_size=300, power=3, alpha=0.7,
                     edgecolor="none", title=f"{name}: trench at {p.lat0:.1f}, {p.width:.0f} km wide")
        add_slab(a, p, top, thk)
        a.axvline(0, color="k", lw=0.8)
        if i:
            a.get_legend().remove()
        x, y = project(cat.lon, cat.lat, p)
        k = (np.abs(y) <= p.width / 2) & (x >= p.x_range[0]) & (x <= p.x_range[1]) & (cat.mag >= 7)
        add_labels(a, x[k], cat.depth[k], [f"{t} M{m:.1f}" for t, m in zip(year[k], cat.mag[k])],
                   priority=cat.mag[k], min_sep=40, nmax=6)

    m = fig.add_subplot(gs[:, 1], projection=ccrs.PlateCarree())
    m.set_extent([-84, -58, -60, -10], ccrs.PlateCarree())
    m.coastlines("50m", lw=0.5)
    plot_points((cat.lon[big], cat.lat[big]), ax=m, color="0.75", edgecolor="none", size=1, linewidths=0)
    plot_lines(trench, ax=m, color="0.2", lw=0.8)
    add_profile_map(m, ps, labels=list("ABC"), color="#d62728", lw=1)
    add_gridlines(m, fontsize=7)
    for s in ("png", "pdf"):
        fig.savefig(out / f"e7_sections.{s}", dpi=150)
