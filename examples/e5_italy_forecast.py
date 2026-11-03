"""
E5. Italy forecasts
===================

Four forecasts on the EPSG:3857 QGIS basemap, with the region outline, the
catalog, panel labels and one shared colorbar. Reference: ``ALM_IT.png``, whose
panel is the first here. ``register_basemap`` names the GeoTIFF.
"""
import os
from pathlib import Path

import cartopy.crs as ccrs

from pituplots import add_colorbar, add_panel_label, panels, plot_catalog, plot_grid, plot_region, register_basemap, theme
from pituplots.io import read_catalog, read_csep_grid

data = Path(os.environ["PITUPLOTS_DATA"])
out = Path(os.environ.get("PITUPLOTS_OUT", Path(__file__).parent / "output"))
out.mkdir(parents=True, exist_ok=True)

fig_dir = data / "italy2010_experiment/figures/Fig2-3_forecasts"
models = {"ALM_IT": "schorlemmer-wiemer.ALM_IT", "LTST": "console-murru-falcone.LTST",
          "HRSS-m1": "werner.HRSS-m1", "MPS04": "meletti.MPS04"}

register_basemap("italy", fig_dir / "basemap_3857.tif")
cat = read_catalog(fig_dir / "catalog.json", preset="csep_json")
with theme("report"):
    fig, axs = panels(2, 2, extent=[5.9, 19.0, 36.0, 47.75], projection=ccrs.GOOGLE_MERCATOR, basemap="italy",
                      gridlines=False, figsize=(10, 11.5))
    for ax, (name, m) in zip(axs, models.items()):
        fc = read_csep_grid(data / f"italy2010_experiment/models/{m}.italy.10yr.2010-01-01.hdf5")
        plot_grid(fc, ax=ax, log=True, vmin=-7, vmax=0, alpha_exp=0.3, colorbar=False)
        plot_region(fc, ax=ax)
        plot_catalog(cat, ax=ax, color="white", edgecolor="black", legend=False, alpha=0.3, size=30, max_size=450)
        add_panel_label(ax, name)
        ax.gridlines(color="0.8", lw=1)
    add_colorbar(list(axs), [c for c in axs[0].collections if c.get_array() is not None][0], shrink=0.7,
                 label=r"$\log_{10}$ rate per cell, 10 yr")
    for s in ("png", "pdf"):
        fig.savefig(out / f"e5_italy_forecast.{s}", dpi=200, bbox_inches="tight")
