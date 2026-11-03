"""Cut the committed test fixtures from the real data.

Run locally, from anywhere. It needs refs/data, rasterio, h5py and the cartopy
Natural Earth cache; the test suite needs none of them. Output is deterministic,
so a rerun leaves git clean unless a source changed.

    python tests/data/make_fixtures.py [--refs refs/data]

grid_capped.csv (F10) is outside refs/data. It is read from $PITUPLOTS_DATA when
set, and the committed subset is kept otherwise.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import cartopy
import cartopy.io.shapereader as shpreader
import h5py
import numpy as np
import pandas as pd
import rasterio
import shapefile
import shapely
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds
from shapely.geometry import box, mapping

OUT = Path(__file__).parent
MAX_MB = 5

# Natural Earth windows: tests stay inside these at 50m and 10m
REGIONS = {
    "europe": (-12, 34, 32, 60),
    "south_america": (-90, -60, -55, -10),
    "california": (-130, 28, -107, 45),
    "new_zealand": (160, -50, 180, -30),
    "new_zealand_east": (-180, -50, -170, -30),
}
NE = [
    ("physical", "coastline", ("110m", "50m", "10m")),
    ("cultural", "admin_0_boundary_lines_land", ("110m", "50m", "10m")),
    ("physical", "land", ("110m", "50m")),
    ("physical", "ocean", ("110m", "50m")),
]

# Lacq window: 25 x 25 cells of the 81 x 81 lattice
LACQ = (-0.75, 43.25, -0.51, 43.49)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def copy_shp(src, dst):
    for f in Path(src).parent.glob(Path(src).stem + ".*"):
        if f.suffix.lower() in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            copy(f, dst.with_suffix(f.suffix.lower()))


def cut_shp(src, dst, idx, fields):
    r = shapefile.Reader(str(src))
    names = [f[0] for f in r.fields[1:]]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with shapefile.Writer(str(dst), shapeType=r.shapeType, encoding="utf-8") as w:
        for f in r.fields[1:]:
            if f[0] in fields:
                w.field(*f)
        for i in idx:
            rec = r.record(i)
            w.record(*[rec[names.index(f)] for f in names if f in fields])
            w.shape(r.shape(i))
    r.close()
    copy(Path(src).with_suffix(".prj"), dst.with_suffix(".prj"))


def cut_tif(src, dst, bounds=None, step=1):
    with rasterio.open(src) as r:
        w = from_bounds(*bounds, r.transform) if bounds else Window(0, 0, r.width, r.height)
        w = w.round_offsets().round_lengths()
        h, n = int(w.height // step), int(w.width // step)
        a = r.read(window=w, out_shape=(r.count, h, n), resampling=Resampling.nearest)
        p = r.profile | {"height": h, "width": n, "compress": "deflate", "predictor": 1,
                         "transform": r.window_transform(w) * r.transform.scale(step)}
        p.pop("blockxsize", None), p.pop("blockysize", None), p.pop("tiled", None)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst, "w", **p) as o:
        o.write(a)


def lacq(refs):
    d = OUT / "lacq"
    d.mkdir(exist_ok=True)
    x0, y0, x1, y1 = LACQ
    p = pd.read_csv(refs / "lacq/outputs/lacq/pattern.csv", dtype=str)
    lon, lat = p.lon.astype(float), p.lat.astype(float)
    p[lon.between(x0, x1) & lat.between(y0, y1)].to_csv(d / "pattern.csv", index=False)
    copy(refs / "lacq/outputs/lacq/events.csv", d / "events.csv")
    copy(refs / "lacq/oq/output/lacq_hmaps_PGA_0.002105.tiff", d / "hmaps_PGA_0.002105.tiff")

    # calc_11 restricted to the window; sids renumbered, as hcurves index them
    keep = ["oqparam", "weights", "gsims", "toms"]
    with h5py.File(refs / "lacq/oqdata/calc_11.hdf5") as f, \
            h5py.File(d / "calc_11_subset.hdf5", "w", track_order=True) as o:
        o.attrs.update({k: f.attrs[k] for k in ("engine_version", "date")})
        lon, lat = f["sitecol/lon"][:], f["sitecol/lat"][:]
        i = np.flatnonzero((lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1))
        for k in keep:
            f.copy(k, o)
        sc = o.create_group("sitecol")
        sc.attrs.update(dict(f["sitecol"].attrs))
        for k, v in f["sitecol"].items():
            sc[k] = np.arange(i.size, dtype=v.dtype) if k == "sids" else v[:][i]
        for k in ("hcurves-rlzs", "hcurves-stats", "hmaps-rlzs", "hmaps-stats"):
            o[k] = f[k][:][i]
            o[k].attrs["json"] = re.sub(r'"site_id": \d+', f'"site_id": {i.size}',
                                        f[k].attrs["json"])
        lt = o.create_group("full_lt")
        lt.attrs.update(dict(f["full_lt"].attrs))
        for k in ("gsim_lt", "sm_data", "source_model_lt"):
            f.copy(f"full_lt/{k}", lt)


def nz(refs):
    d = OUT / "new_zealand"
    c = pd.read_csv(refs / "nonpoisson_nz_src/data/catalogs/cat_nz.csv")
    east = c[c.lon > 180].iloc[::4]
    cut = pd.concat([c[c.lon <= 180].iloc[::40], east]).sort_index()
    d.mkdir(exist_ok=True)
    cut.to_csv(d / "cat_nz.csv", index=False)

    src = refs / "pycsep/csep/artifacts/Regions/nz.testing.nodes.dat"
    rows = [s for s in src.read_text().splitlines()
            if 172 <= float(s.split()[0]) <= 176 and -42 <= float(s.split()[1]) <= -38]
    (d / "nz.testing.nodes.dat").write_text("\n".join(rows) + "\n")

    copy_shp(refs / "nonpoisson_nz_src/data/regions/new_zealand/nz_testing.shp",
             d / "nz_testing.shp")
    copy(refs / "etas_sim/data/iceland_region.npy", OUT / "regions/iceland_region.npy")

    src = refs / "nonpoisson_nz_src/data/NZ_CFM_v1_0_shapefile/NZ_CFM_v1_0.shp"
    n = len(shapefile.Reader(str(src)))
    cut_shp(src, d / "nz_cfm.shp", range(0, n, 20), ("Fault_ID", "Name", "Dip_pref", "SR_pref"))


def chile(refs, ext):
    d = OUT / "chile"
    d.mkdir(exist_ok=True)
    g = pd.read_csv(refs / "psha_chile/data/grids/chile_01.csv", dtype=str)
    g[g.lat.astype(float).between(-34, -30)].to_csv(d / "chile_01.csv", index=False)

    src = refs / "psha_renta/model/ghs_model_site/results_475"
    h = pd.read_csv(src / "hazard_475_chile.csv", dtype=str)
    lat = h.lat_center.astype(float)
    h[lat.between(-33.2, -33)].to_csv(d / "hazard_475_chile.csv", index=False)
    cut_tif(src / "hazard_475_PGA.tif", d / "hazard_475_PGA.tif", (-72, -34, -70, -32))
    cut_tif(refs / "psha_chile/data/slab_2.0/slab_depth.tif", d / "slab_depth.tif",
            (-74, -36, -66, -28))

    s = refs / "psha_chile/data/shapefiles"
    copy_shp(s / "sam_nazca_trench.shp", d / "sam_nazca_trench.shp")
    copy_shp(s / "chile_buffer.shp", d / "chile_buffer.shp")
    copy(s / "pb2002_antarctic_trench.geojson", d / "pb2002_antarctic_trench.geojson")

    f = refs / "psha_chile/data/active_faults"
    cut_shp(f / "crustal_faults_chile_updated.shp", d / "crustal_faults.shp", range(0, 160, 8),
            ("name", "dip", "slip_rate"))
    j = json.loads((f / "geojson/sara-active-faults.geojson").read_text())
    j["features"] = j["features"][:: max(1, len(j["features"]) // 20)]
    (d / "sara_faults.geojson").write_text(json.dumps(j, ensure_ascii=False))

    if ext is not None and (ext / "psha_chile").exists():
        src = ext / "psha_chile/psha_chile6/outputs/crustal/ref/ssm/grid_capped.csv"
        c = pd.read_csv(src, dtype=str)
        c = c[c.lat.astype(float).between(-34, -30)]
        cols = ["lon", "lat", "in_buffer"] + [k for k in c if k.startswith("rate_M")][:3]
        c[cols].to_csv(d / "grid_capped.csv", index=False)


def nrml(refs):
    d = OUT / "nrml"
    d.mkdir(exist_ok=True)
    root = refs / "psha_chile/psha_chile6/outputs"
    for src, tag, n, name in (
        (root / "crustal/ref/nrml/faults__reference.xml", "simpleFaultSource", 5, "simple.xml"),
        (root / "interface/mmin_haz-5.5/nrml/sub__seg__seis__tgr.xml", "complexFaultSource", 1,
         "complex.xml"),
    ):
        s = src.read_text(encoding="utf-8")
        ends = [m.end() for m in re.finditer(rf"</{tag}>\n", s)]
        tail = s[s.rindex("    </sourceModel>"):]
        (d / name).write_text(s[:ends[n - 1]] + tail, encoding="utf-8")


def catalogs(refs):
    d = OUT / "catalogs"
    d.mkdir(exist_ok=True)
    copy(refs / "italy2010_experiment/figures/Fig2-3_forecasts/catalog.json", d / "italy.json")
    j = json.loads((refs / "testing_eshm20/catalog_emec_gcmt.json").read_text())
    j["catalog"] = j["catalog"][::10]
    (d / "emec_gcmt.json").write_text(json.dumps(j))
    c = pd.read_csv(refs / "pycsep/csep/artifacts/ObservedCatalogs/sample_comcat_catalog.csv",
                    dtype=str)
    c.iloc[::8].to_csv(d / "comcat.csv", index=False)


def basemaps(refs):
    cut_tif(refs / "pycsep/csep/artifacts/ExampleBasemaps/basemap_california.tif",
            OUT / "basemaps/california_4326_rgba.tif", step=6)


def natural_earth():
    """Clipped copy of the Natural Earth layers, in the cartopy cache layout."""
    root = OUT / "natural_earth/shapefiles/natural_earth"
    src = {}
    boxes = [box(*r) for r in REGIONS.values()]
    for cat, name, scales in NE:
        for res in scales:
            p = shpreader.natural_earth(res, cat, name)
            src[f"ne_{res}_{name}.shp"] = sha(p)
            dst = root / cat / f"ne_{res}_{name}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            geoms = list(shpreader.Reader(p).geometries())
            if res != "110m":
                dim = "Line" if "Line" in geoms[0].geom_type else "Polygon"
                geoms = [x for g in geoms for b in boxes if g.intersects(b)
                         for x in shapely.get_parts(g.intersection(b)) if dim in x.geom_type]
            with shapefile.Writer(str(dst), encoding="utf-8") as w:
                w.field("id", "N", 6, 0)
                for i, g in enumerate(geoms):
                    w.record(i)
                    w.shape(mapping(g))
            copy(Path(p).with_suffix(".prj"), dst.with_suffix(".prj"))
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", type=Path, default=OUT.parents[1] / "refs/data")
    a = ap.parse_args()
    ext = Path(os.environ["PITUPLOTS_DATA"]) if "PITUPLOTS_DATA" in os.environ else None
    if not a.refs.is_dir():
        raise SystemExit(f"{a.refs} not found: make_fixtures needs the local reference data")

    capped = OUT / "chile/grid_capped.csv"
    kept = capped.read_bytes() if capped.exists() else None
    for p in OUT.iterdir():
        if p.is_dir() and p.name != "__pycache__":
            shutil.rmtree(p)
    lacq(a.refs)
    nz(a.refs)
    chile(a.refs, ext)
    nrml(a.refs)
    catalogs(a.refs)
    basemaps(a.refs)
    if kept is not None and not capped.exists():
        capped.write_bytes(kept)
    ne = natural_earth()

    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.parent != OUT
                   and p.parent.name != "__pycache__")
    man = {"cartopy": cartopy.__version__, "natural_earth_sources": ne,
           "files": {p.relative_to(OUT).as_posix(): sha(p) for p in files}}
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")

    total = sum(p.stat().st_size for p in files)
    for d in sorted(p for p in OUT.iterdir() if p.is_dir() and p.name != "__pycache__"):
        print(f"{d.name:16s}{sum(p.stat().st_size for p in d.rglob('*') if p.is_file()) / 1e3:9.0f} kB")
    print(f"{'total':16s}{total / 1e3:9.0f} kB")
    if total > MAX_MB * 1e6:
        raise SystemExit(f"fixtures exceed {MAX_MB} MB")


if __name__ == "__main__":
    main()
