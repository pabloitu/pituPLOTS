# pituPLOTS

Spatial and geographic plotting routines for geoscience data: seismic catalogs,
forecasts, hazard maps, rasters, source models and cross-sections.

It is a functional API: every `plot_*` function draws one layer and returns the
axes, so layers compose by passing `ax=`.

## Quick start

```python
from pituplots import add_locator, plot_catalog, plot_grid, theme
from pituplots.io import read_catalog, read_rate_csv

grid = read_rate_csv("pattern.csv")
cat = read_catalog("events.csv")
with theme("report"):
    ax = plot_grid(grid, log=True, basemap="ESRI_relief", clabel="log10 rate")
    plot_catalog(cat, ax=ax, mag_ticks=[1.5, 2.5, 3.5, 4])
    add_locator(ax, [-5.5, 9, 41, 51.5])
    ax.figure.savefig("map.png", dpi=300, bbox_inches="tight")
```

## What is in it

- Maps (`make_map`, any projection, extents across the dateline), basemaps
  (tiles, GeoTIFFs in any CRS), catalogs, gridded rates and hazard fields,
  sites, vectors (regions, lines by attribute, polygons, OpenQuake sources),
  cross-sections with Slab2, and multi-panel figures (`panels`,
  `strip_panels`, `compare_grids`).
- Map elements: colorbars, size legends, locator insets, scale bars, labels
  that avoid each other, panel labels.
- `pituplots.stats`: frequency-magnitude distributions with b-value fits
  (Aki-Utsu, or Weichert with completeness changing in time), model MFDs,
  magnitude against time, cumulative counts with simulation envelopes,
  b-value stability, and hazard curves with weighted envelopes.
- Classified maps: `class_breaks` and `add_class_legend`.
- Readers in `pituplots.io`: catalogs, CSEP forecasts and regions, rate
  tables, GeoTIFFs, Slab2, shapefiles and GeoJSON, OpenQuake datastores, CSV
  exports and NRML source models (without OpenQuake).
- `pituplots.adapters.csep` converts to and from pyCSEP objects.
- `pituplots.scene`: catalogs in 3D (PyVista), in any CRS including local
  mine grids, coloured by any column, sized by magnitude on screen (classes
  that keep their pixel size at any zoom) or by true source size, with hover
  labels, a click panel, flags for review, and vertical sections that clip
  every layer and page with Ctrl + wheel.
- `pituplots-3d`: the same scene in a browser app (trame), with controls for
  colour, size, opacity and view, layers, an event card, a flags table,
  screenshots and a static HTML export.

## 3D catalogs

A YAML config maps the catalog columns, so any catalog can be read:

```yaml
columns: {x: "LocX [m]", y: "LocY [m]", z: "LocZ [m]", mag: "Local Magnitude"}
crs: local              # or EPSG:xxxx; lon/lat catalogs need nothing
z: elevation            # or depth (default)
z_unit: m               # or km (default)
time: {columns: [EventDate, EventTimeInDay], format: excel}
trim: {mad: 10}         # drop events far from the bulk
scene:
  id: "#Index"
  color_by: z
  hover: [mag, time]
```

Layers in any CRS line up with the catalog. A mine grid defined from UTM by a
2D similarity (`x = a E + b N + tx`, `y = -b E + a N + ty`) is a scene
`projection`, so DEMs, shapefiles and meshes in UTM or lon/lat land in mine
coordinates:

```yaml
scene:
  projection: {crs: "EPSG:32719", a: 0.9736, b: -0.2302, tx: 1069146.66, ty: -6149426.81}
layers:
  - {kind: dem, path: topo.tif, opacity: 0.5}                       # GeoTIFF, any CRS
  - {kind: surface, path: sam_slab2_dep_02.23.18.xyz, opacity: 0.35}  # Slab2 or any raster
  - {kind: shapes, path: chile.shp, z: dem}                          # draped on the DEM
  - {kind: mesh, path: stopes.vtk, crs: local}                       # VTK, STL, OBJ, PLY
```

```python
from pituplots.io import load_config, read_catalog
from pituplots.layers import add_layers
from pituplots.scene import plot_catalog_3d

cat = read_catalog("events.xlsx", config="mine.yaml")
p = plot_catalog_3d(cat, config="mine.yaml")
add_layers(p, load_config("mine.yaml")["layers"], base=".")
p.show()
```

```bash
pituplots-3d events.xlsx --config mine.yaml
```

In the window: hover for a label, click for all fields, space to flag, `x` to
write `flags.csv`. The app has the same, with buttons, plus filters
(magnitude, a category column, flagged only, id search) and a time bar (range,
sliding or cumulative windows over an events histogram, play, movie export).
A vertical section (strike, offset and half-width, with a face-on view) keeps
only the events near a plane: `add_section(plotter, azimuth=90, width=40)` and
`view_section(plotter)` from scripts, or a plane widget to drag with
`widget=True`. From scripts, `show_events(plotter, mask)` filters a scene and
`animate(plotter, "movie.mp4", step="7D", length="30D")` writes a movie.

## Styles

Defaults are per plot family. A theme sets rcParams and overrides families, and
explicit keywords always win:

```python
from pituplots import theme, use_theme

use_theme("paper")            # or "slides", "report"
with theme("my_style.yaml"):  # your own, in YAML
    ...
```

A theme file has `rc` (matplotlib rcParams), `style` (per-family keywords) and
optionally `extends`. Packaged themes: `paper`, `slides`, `report`,
`dark`, `poster` and `minimal`.

```yaml
extends: report
rc:
  font.size: 11
style:
  darkgrid: {facecolor: "#EAEAF2"}
  catalog: {color: gold, alpha: 0.5}
```

Themes use the Ubuntu font when it is installed and DejaVu Sans otherwise.
Nothing changes rcParams except `use_theme` and `theme`.

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
| `excel` | openpyxl | `.xlsx` catalogs |
| `3d` | pyvista, vtk | `pituplots.scene` |
| `app` | the above, trame, trame-vtk, trame-vuetify | `pituplots-3d` |
| `all` | all of the above | |

PyYAML is a core dependency, for the themes.

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

The examples in `examples/` use the real project data. Set `PITUPLOTS_DATA`
to its root; figures go to `examples/output/`.

Build the docs with `pip install -e ".[docs]"` and
`sphinx-build -b html docs docs/_build/html`.

## License

GPL-3.0. See `LICENSE`.
