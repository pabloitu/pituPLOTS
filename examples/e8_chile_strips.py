"""
E8. Chile hazard in latitude strips
===================================

PGA classes on the gray QGIS basemap, in three equal-width strips, with towns
and scale bars. References: ``hazard_chile.jpeg`` (layout) and
``national_maps.py``, with the classes as quantiles.
"""
import os
from pathlib import Path

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from pituplots import add_labels, add_scalebar, plot_field, plot_points, strip_panels, theme
from pituplots.io import read_geotiff

data = Path(os.environ["PITUPLOTS_DATA"])
out = Path(os.environ.get("PITUPLOTS_OUT", Path(__file__).parent / "output"))
out.mkdir(parents=True, exist_ok=True)

CAT = ["#fee391", "#fec44f", "#fe9929", "#d95f0e", "#993404"]
TOWNS = {"Arica": (-70.31, -18.48), "Iquique": (-70.14, -20.22), "Antofagasta": (-70.40, -23.65),
         "Copiapó": (-70.33, -27.37), "La Serena": (-71.25, -29.90), "Valparaíso": (-71.62, -33.05),
         "Santiago": (-70.65, -33.45), "Talca": (-71.66, -35.43), "Concepción": (-73.05, -36.83),
         "Temuco": (-72.59, -38.74), "Valdivia": (-73.24, -39.81), "Puerto Montt": (-72.94, -41.47),
         "Coyhaique": (-72.07, -45.57), "Punta Arenas": (-70.91, -53.16)}

g = read_geotiff(data / "psha_renta/model/ghs_model_site/results_475/hazard_475_PGA.tif")
z = np.ma.masked_invalid(g.values)
edges = np.round(np.quantile(z.compressed(), np.linspace(0, 1, 6)), 2)
lon, lat = np.array(list(TOWNS.values())).T

with theme("paper"):
    fig, axs = strip_panels(np.linspace(-17.5, -56, 4), [-70.4, -72.4, -71.4], 850, coastline=False, borders=False,
                            basemap=data / "psha_chile/data/basemaps/basemap_chile_gray/basemap_gray.tif",
                            figsize=(16.5 / 2.54, 11.5 / 2.54))
    for ax in axs:
        plot_field(g, ax=ax, levels=edges, cmap=ListedColormap(CAT), alpha=1, colorbar=False)
        w, e, s, n = ax.get_extent()
        k = (lon > w) & (lon < e) & (lat > s) & (lat < n)
        plot_points((lon[k], lat[k]), ax=ax, color="white", size=5, linewidths=0.5)
        add_labels(ax, lon[k], lat[k], [t for t, i in zip(TOWNS, k) if i], offset=(3, 1))
        add_scalebar(ax, 200, loc=(0.06, 0.03))
    hs = [Patch(fc=c, label=f"{a:.2f} to {b:.2f}") for c, a, b in zip(CAT, edges[:-1], edges[1:])]
    fig.legend(handles=hs, loc="outside lower center", ncol=5, title="PGA (g), 475 yr return period")
    for s in ("png", "pdf"):
        fig.savefig(out / f"e8_chile_strips.{s}", dpi=300, bbox_inches="tight")
