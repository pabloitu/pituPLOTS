"""
E12. New Zealand catalog and region
===================================

The catalog crossing 180 deg and the CSEP region, on the EPSG:2193 basemap and on
a map centred on 180. Reference: ``forecasts.jpg`` (style). The catalog has
about 10,000 events, so the markers are smaller and more transparent than the
defaults, which suit a few hundred.
"""
import os
from pathlib import Path

import cartopy.crs as ccrs

from pituplots import make_map, plot_catalog, plot_region, theme
from pituplots.io import read_catalog, read_csep_nodes

data = Path(os.environ["PITUPLOTS_DATA"])
out = Path(os.environ.get("PITUPLOTS_OUT", Path(__file__).parent / "output"))
out.mkdir(parents=True, exist_ok=True)

d = data / "nonpoisson_nz_src/data"
cat = read_catalog(d / "catalogs/cat_nz.csv")
region = read_csep_nodes(data / "pycsep/csep/artifacts/Regions/nz.testing.nodes.dat")
ticks = [4, 5, 6, 7, 7.8]
dense = {"alpha": 0.25, "size": 2, "max_size": 300, "power": 3, "linewidths": 0.3}
with theme("report"):
    ax = make_map(extent=[164, 183, -48.5, -33], projection="native", figsize=(7.5, 9), coastline=False,
                  basemap=d / "basemaps/new_zealand/basemap_2193.tiff")
    plot_region(region, ax=ax)
    plot_catalog(cat, ax=ax, mag_ticks=ticks, legend_loc="upper left", color="goldenrod", **dense)
    for s in ("png", "pdf"):
        ax.figure.savefig(out / f"e12_nz_2193.{s}", dpi=200, bbox_inches="tight")

    ax = make_map(extent=[160, -170, -52, -30], projection=ccrs.PlateCarree(central_longitude=180), figsize=(8, 7))
    plot_region(region, ax=ax)
    plot_catalog(cat, ax=ax, mag_ticks=ticks, legend_loc="upper left", color="goldenrod", **dense)
    for s in ("png", "pdf"):
        ax.figure.savefig(out / f"e12_nz_180.{s}", dpi=200, bbox_inches="tight")
