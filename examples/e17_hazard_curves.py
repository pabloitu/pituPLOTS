"""
E17. Hazard curves
==================

Three versions of the Chile model (psha_chile4, 5 and 6, 432 realizations each)
at two cities: weighted mean and 2.5-97.5% envelope, PoE in 50 years. Style:
the gray grid of ``psha_chile7/hazard/post.py``. Reference:
``testing_eshm20/results_full/hazard_cut_pazarcik_curves.png``.
OpenQuake datastores are read from ``OQ_DATA`` (default ``~/oqdata``).
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt

from pituplots import theme
from pituplots.io import read_datastore
from pituplots.stats import plot_curves

oqdata = Path(os.environ.get("OQ_DATA", Path.home() / "oqdata"))
out = Path(os.environ.get("PITUPLOTS_OUT", Path(__file__).parent / "output"))
out.mkdir(parents=True, exist_ok=True)

CALCS = {"psha_chile4": 152, "psha_chile5": 229, "psha_chile6": 268}
CITIES = {"Iquique": (-70.14, -20.21), "Antofagasta": (-70.40, -23.65)}
models = {k: read_datastore(oqdata / f"calc_{c}.hdf5") for k, c in CALCS.items()}
with theme("report"):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5), layout="constrained")
    for a, (city, xy) in zip(axs, CITIES.items()):
        for name, oq in models.items():
            plot_curves(oq, ax=a, site=xy, window=50, label=name, xlim=(1e-2, 3))
        a.set_title(city)
    for s in ("png", "pdf"):
        fig.savefig(out / f"e17_hazard_curves.{s}", dpi=200)
