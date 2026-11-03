"""
E16. Catalog statistics
=======================

Frequency-magnitude distributions by family with Aki-Utsu fits, magnitude
against time, cumulative counts and b-value stability, on the gray grid of the
hazard curves. References: ``VEGA/examples/lofs_b3/vcat/{fmd,temporal,
cumulative}.png``.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pituplots import Catalog, theme
from pituplots.io import read_catalog
from pituplots.stats import mc_maxc, plot_b_stability, plot_cumulative, plot_fmd, plot_magnitude_time

data = Path(os.environ["PITUPLOTS_DATA"]) / "psha_chile"
out = Path(os.environ.get("PITUPLOTS_OUT", Path(__file__).parent / "output"))
out.mkdir(parents=True, exist_ok=True)

FAMILY = {"slab_interface": "interface", "patagonia_interface": "interface", "intra_slab": "in-slab",
          "slab_deep": "in-slab", "deep_nest": "in-slab", "forearc": "crustal", "intraarc": "crustal",
          "backarc": "crustal", "patagonia_crustal": "crustal"}
COLORS = {"interface": "#2166ac", "in-slab": "#a50f15", "crustal": "#e0641e", "excluded": "#7f7f7f"}

cat = read_catalog(data / "results/cat_handler_2/catalog.csv", sep=",")
modern = cat.time >= np.datetime64("1980-01-01")
fam = np.array([FAMILY.get(str(c), "excluded") for c in cat.extra["class"]])
cat.extra["family"] = fam
recent = Catalog(cat.lon[modern], cat.lat[modern], cat.depth[modern], cat.mag[modern], cat.time[modern],
                 extra={"family": fam[modern]})
mc = mc_maxc(recent.mag)
with theme("report"):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")
    plot_fmd(recent, ax=axs[0, 0], by="family", colors=COLORS, mc=mc)
    axs[0, 0].set_title("1980 onwards, by family")
    plot_magnitude_time(cat, ax=axs[0, 1], color_by="family", colors=COLORS, edgecolor="none")
    axs[0, 1].set_title("All events")
    plot_cumulative(recent, ax=axs[1, 0], mag_min=5.0)
    plot_b_stability(recent, ax=axs[1, 1], min_events=200, adopted=mc, label="1980 onwards")
    for s in ("png", "pdf"):
        fig.savefig(out / f"e16_catalog_stats.{s}", dpi=150)
