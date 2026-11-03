"""
E3. Lacq hazard maps
====================

Mean PGA at 475 and 2475 years from the OpenQuake datastore (``calc_11``),
as filled contours on non-uniform levels and as a shaded grid, with towns and
a locator. References: ``lacq_hazard_{475,2475}yr{,_contourf}.jpg``.
OpenQuake datastores are read from ``OQ_DATA`` (default ``~/oqdata``).
"""
import os
from pathlib import Path

from pituplots import Grid, add_locator, plot_field, plot_points, theme
from pituplots.io import read_datastore

oqdata = Path(os.environ.get("OQ_DATA", Path.home() / "oqdata"))
out = Path(os.environ.get("PITUPLOTS_OUT", Path(__file__).parent / "output"))
out.mkdir(parents=True, exist_ok=True)

LEVELS = [0.002, 0.003, 0.004, 0.005, 0.006, 0.010, 0.016, 0.025, 0.040, 0.060, 0.100, 0.160, 0.250]
TOWNS = {"Lacq": (-0.620, 43.417), "Pau": (-0.370, 43.300), "Orthez": (-0.773, 43.489),
         "Mourenx": (-0.607, 43.379), "Artix": (-0.552, 43.421), "Monein": (-0.573, 43.472)}

oq = read_datastore(oqdata / "calc_11.hdf5")
lon, lat = zip(*TOWNS.values())
with theme("report"):
    for poe, tr in ((0.002105, "475yr"), (0.000404, "2475yr")):
        m = oq.maps(poe)
        g = Grid.from_points(m.lon, m.lat, m.values, dh=0.01)
        for kind, tag in (("contourf", "_contourf"), ("mesh", "")):
            kw = {"levels": LEVELS} if kind == "contourf" else {"vmin": 0}
            ax = plot_field(g, kind=kind, basemap="ESRI_relief", coastline=False, clabel="PGA (g)",
                            title=f"Lacq mean hazard - PGA - {tr}", **kw)
            plot_points((lon, lat), ax=ax, labels=list(TOWNS))
            add_locator(ax, [-5.5, 9, 41, 51.5])
            for s in ("png", "pdf"):
                ax.figure.savefig(out / f"e3_lacq_hazard_{tr}{tag}.{s}", dpi=200, bbox_inches="tight")
