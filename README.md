# pituPLOTS

Spatial and geographic plotting routines for geoscience data: seismic catalogs,
forecasts, hazard maps, rasters, source models and cross-sections.

The package is under development. It has the data containers only; plotting
functions are not in yet.

## Install

Requires Python 3.10 or newer, matplotlib 3.8 or newer and cartopy 0.23 or newer.

```bash
pip install "pituPLOTS @ git+https://github.com/pabloitu/pituPLOTS.git"
```

Optional features are extras:

```bash
pip install "pituPLOTS[raster,hdf5] @ git+https://github.com/pabloitu/pituPLOTS.git"
```

| Extra | Adds | Enables |
|---|---|---|
| `raster` | rasterio | GeoTIFF basemaps and data |
| `hdf5` | h5py | OpenQuake datastore |
| `vector` | pyogrio | GPKG and other OGR formats |
| `csep` | pycsep | `adapters.csep` |
| `oq` | openquake.engine | OpenQuake branch labels |
| `all` | all of the above | |

The package imports as `pituplots`. Importing it does not load any optional
package and does not change the matplotlib backend or rcParams.

## Development

```bash
git clone https://github.com/pabloitu/pituPLOTS.git
```

```bash
pip install -e "./pituPLOTS[test,raster,hdf5,vector]"
```

```bash
pytest
```

The tests run offline. Tests that need an extra are skipped without it. Image
baselines are in `tests/baseline`; regenerate them only deliberately, with
`pytest --mpl-generate-path=tests/baseline`, and review them by eye.

Build the docs with `pip install -e ".[docs]"` and
`sphinx-build -b html docs docs/_build/html`.

## License

GPL-3.0. See `LICENSE`.
