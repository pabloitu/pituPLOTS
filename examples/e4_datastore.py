"""
E4. OpenQuake datastore
=======================

``read_datastore`` checks on opening that the weighted mean of the
realizations reproduces the stored mean, and raises ``ConsistencyError``
otherwise. The figure is the 475-year map from the mean curves next to the
hazard curve at the site of interest. References: ``lacq_hazard_475yr.jpg`` and
``lacq_hcurve_poi.png``.
OpenQuake datastores are read from ``OQ_DATA`` (default ``~/oqdata``).
"""
import os
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

from pituplots import Grid, plot_field, plot_points, theme
from pituplots.io import read_datastore
from pituplots.maps import dress_map
from pituplots.stats import plot_curves

oqdata = Path(os.environ.get("OQ_DATA", Path.home() / "oqdata"))
out = Path(os.environ.get("PITUPLOTS_OUT", Path(__file__).parent / "output"))
out.mkdir(parents=True, exist_ok=True)

oq = read_datastore(oqdata / "calc_11.hdf5")
print(oq)
poi = (-0.630, 43.400)
with theme("report"):
    fig = plt.figure(figsize=(14, 6), layout="constrained")
    ax = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    m = oq.maps(0.1, window=50)
    g = Grid.from_points(m.lon, m.lat, m.values, dh=0.01)
    dress_map(ax, extent=[g.x_edges[0], g.x_edges[-1], g.y_edges[0], g.y_edges[-1]], coastline=False)
    plot_field(g, ax=ax, clabel="PGA (g), 10% in 50 yr", title="Mean of calc_11")
    plot_points(([poi[0]], [poi[1]]), ax=ax, labels=["Site"], color="white")
    a = fig.add_subplot(1, 2, 2)
    plot_curves(oq, ax=a, site=poi, window=50, label="Lacq SSM, calc_11")
    i = int(np.argmin(np.hypot(oq.sites.lon - poi[0], oq.sites.lat - poi[1])))
    a.set_title(f"Site ({oq.sites.lon[i]:.2f}, {oq.sites.lat[i]:.2f})")
    for s in ("png", "pdf"):
        fig.savefig(out / f"e4_datastore.{s}", dpi=200)
