import json

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon

from pituplots import ConsistencyError
from pituplots.io.oq import poe_at, read_csv, read_sources, wquantile

from pituplots.io.oq import read_datastore

try:
    import h5py
except ImportError:
    h5py = None
hdf5 = pytest.mark.skipif(h5py is None, reason="needs the [hdf5] extra")

IMLS = np.logspace(-2, 0, 5)


def write_ds(path, rlzs=None, stats=None, weights=None, oqparam=None, stat_names=("mean",)):
    """Minimal OpenQuake-like datastore."""
    op = {"investigation_time": 50.0, "hazard_imtls": {"PGA": IMLS.tolist()}} if oqparam is None else oqparam
    with h5py.File(path, "w") as f:
        f["oqparam"] = json.dumps(op)
        n = (rlzs if rlzs is not None else stats).shape[0]
        f["sitecol/lon"] = np.linspace(0, 1, n)
        f["sitecol/lat"] = np.zeros(n)
        if rlzs is not None:
            f["hcurves-rlzs"] = rlzs
            f["weights"] = np.ones(rlzs.shape[1]) / rlzs.shape[1] if weights is None else weights
        if stats is not None:
            d = f.create_dataset("hcurves-stats", data=stats)
            d.attrs["json"] = json.dumps({"stat": list(stat_names)})
    return path


def curves(n=3, r=2):
    base = np.exp(-np.arange(5) * 1.5)[None, None, None, :] * 0.5
    return np.broadcast_to(base * (1 + np.arange(r))[None, :, None, None] / r, (n, r, 1, 5)).copy()


@hdf5
def test_fixture_datastore(data):
    oq = read_datastore(data / "lacq/calc_11_subset.hdf5")
    assert oq.sites.lon.size == 625 and list(oq.imtls) == ["PGA"] and oq.investigation_time == 1.0
    assert oq.curves().shape == (625, 1, 25) and oq.mean().shape == (625, 25)
    assert oq.stats == ["mean"] and oq.weights.tolist() == [1.0]
    assert "625 sites" in repr(oq)


@hdf5
def test_maps_reproduce_openquake(data):
    p = data / "lacq/calc_11_subset.hdf5"
    oq = read_datastore(p)
    m = oq.maps(0.002105)
    with h5py.File(p) as f:
        ref = f["hmaps-stats"][:, 0, 0, 0]
    assert np.allclose(m.values, ref, rtol=1e-5, atol=1e-7)
    assert m.meta["poes"] == [0.002105] and m.values.shape == (625,)
    assert oq.maps([0.002105, 0.000404]).values.shape == (625, 2)


@hdf5
def test_maps_in_a_window(data):
    oq = read_datastore(data / "lacq/calc_11_subset.hdf5")
    a = oq.maps(0.1, window=50).values
    b = oq.maps(1 - 0.9 ** (1 / 50)).values
    assert np.allclose(a, b, rtol=0.05) and not np.allclose(a, oq.maps(0.1).values)


@hdf5
def test_stats_only(tmp_path):
    oq = read_datastore(write_ds(tmp_path / "s.hdf5", stats=curves(r=1)))
    assert oq.weights is None and oq.mean().shape == (3, 5)
    with pytest.raises(ValueError, match="rlzs"):
        oq.curves()


@hdf5
def test_rlzs_only_and_weighted_mean_and_quantile(tmp_path):
    c = curves(r=2)
    oq = read_datastore(write_ds(tmp_path / "r.hdf5", rlzs=c, weights=np.array([3.0, 1.0])))
    assert oq.stats == [] and np.allclose(oq.weights, [0.75, 0.25])
    assert np.allclose(oq.mean(), 0.75 * c[:, 0, 0] + 0.25 * c[:, 1, 0])
    q = oq.quantile(0.5)
    assert q.shape == (3, 5) and (q >= c[:, 0, 0] - 1e-12).all() and (q <= c[:, 1, 0] + 1e-12).all()


@hdf5
def test_stored_quantile_wins(tmp_path):
    s = np.concatenate([curves(r=1), curves(r=1) * 0.5], axis=1)
    oq = read_datastore(write_ds(tmp_path / "q.hdf5", stats=s, stat_names=("mean", "quantile-0.15")))
    assert np.allclose(oq.quantile(0.15), s[:, 1, 0])


@hdf5
def test_single_realization_and_sampled_tree(tmp_path):
    c = curves(r=4)
    oq = read_datastore(write_ds(tmp_path / "s.hdf5", rlzs=c, stats=c.mean(axis=1, keepdims=True)))
    assert np.allclose(oq.weights, 0.25)


@hdf5
def test_mean_mismatch_raises(tmp_path):
    c = curves(r=2)
    with pytest.raises(ConsistencyError, match="hcurves-stats"):
        read_datastore(write_ds(tmp_path / "m.hdf5", rlzs=c, stats=c[:, :1] * 2))


@hdf5
def test_mean_check_in_rate_space(tmp_path):
    c = curves(r=2)
    rate = -np.log1p(-c)
    m = -np.expm1(-(rate.mean(axis=1, keepdims=True)))
    op = {"investigation_time": 1.0, "hazard_imtls": {"PGA": IMLS.tolist()}, "use_rates": True}
    read_datastore(write_ds(tmp_path / "u.hdf5", rlzs=c, stats=m, oqparam=op))
    op["use_rates"] = False
    with pytest.raises(ConsistencyError):
        read_datastore(write_ds(tmp_path / "v.hdf5", rlzs=c, stats=m, oqparam=op))


@hdf5
def test_missing_investigation_time(tmp_path):
    oq = read_datastore(write_ds(tmp_path / "t.hdf5", stats=curves(r=1),
                                 oqparam={"hazard_imtls": {"PGA": IMLS.tolist()}}))
    assert oq.investigation_time is None
    oq.maps(0.1)
    with pytest.raises(ValueError, match="investigation_time"):
        oq.to_window(oq.mean(), 50)


@hdf5
def test_missing_imls_raise(tmp_path):
    with pytest.raises(ValueError, match="IMLs"):
        read_datastore(write_ds(tmp_path / "i.hdf5", stats=curves(r=1), oqparam={"investigation_time": 1.0}))


@hdf5
def test_several_imts_need_a_choice(tmp_path):
    op = {"investigation_time": 1.0, "hazard_imtls": {"PGA": IMLS.tolist(), "SA(0.2)": IMLS.tolist()}}
    s = np.concatenate([curves(r=1), curves(r=1) * 0.5], axis=2)
    oq = read_datastore(write_ds(tmp_path / "m.hdf5", stats=s, oqparam=op))
    with pytest.raises(ValueError, match="several"):
        oq.mean()
    assert np.allclose(oq.mean("SA(0.2)"), s[:, 0, 1])
    with pytest.raises(ValueError, match="no IMT"):
        oq.mean("PGV")


@hdf5
def test_branches_without_openquake(data, monkeypatch):
    import builtins

    real = builtins.__import__

    def block(name, *a, **k):
        if name.startswith("openquake"):
            raise ImportError
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", block)
    assert read_datastore(data / "lacq/calc_11_subset.hdf5").branches is None


@hdf5
def test_missing_file():
    with pytest.raises(FileNotFoundError):
        read_datastore("nope.hdf5")


def test_poe_at_edges():
    lv = np.array([0.1, 0.2, 0.4])
    c = np.array([[0.5, 0.1, 0.01], [0.001, 0.0005, 0.0], [0.9, 0.8, 0.7]])
    v = poe_at(c, lv, 0.1)
    assert v[0] == pytest.approx(0.2) and v[1] == 0 and v[2] == 0.4


def test_wquantile_matches_unweighted_median():
    c = np.array([[[1.0], [2.0], [3.0]]])
    assert wquantile(c, np.ones(3) / 3, 0.5)[0, 0] == pytest.approx(2)


# OpenQuake CSV exports


def test_csv_curves_with_scientific_poes(tmp_path):
    p = tmp_path / "hazard_curve-mean-PGA_23.csv"
    p.write_text('#,,,,"generated_by=\'OpenQuake engine 3.16\', start_date=\'x\', checksum=1, kind=\'mean\', '
                 'investigation_time=\'50.0\', imt=\'PGA\'"\nlon,lat,depth,poe-0.0050000,poe-1e-05\n'
                 '1.0,2.0,0.0,0.5,0.01\n3.0,4.0,0.0,0.4,0.0\n')
    s, meta = read_csv(p)
    assert meta["type"] == "curves" and meta["kind"] == "mean" and meta["investigation_time"] == 50
    assert meta["imt"] == "PGA" and meta["imls"].tolist() == [0.005, 1e-5]
    assert s.values.shape == (2, 2) and s.lon.tolist() == [1, 3]


def test_csv_map_with_sa_names_and_scientific_poes(tmp_path):
    p = tmp_path / "hazard_map-mean_1.csv"
    p.write_text("#,,,,'kind=mean, investigation_time=50.0'\nlon,lat,PGA-0.1,SA(0.2)-0.02,SA(1.0)-1e-05\n"
                 "1,2,0.3,0.5,0.9\n")
    s, meta = read_csv(p)
    assert meta["type"] == "map" and meta["columns"] == [("PGA", 0.1), ("SA(0.2)", 0.02), ("SA(1.0)", 1e-5)]
    assert s.values.tolist() == [[0.3, 0.5, 0.9]]


def test_csv_uhs(tmp_path):
    p = tmp_path / "hazard_uhs-mean_1.csv"
    p.write_text("#,,'kind=mean, investigation_time=50.0'\nlon,lat,0.1~PGA,0.1~SA(0.2)\n1,2,0.3,0.6\n")
    s, meta = read_csv(p)
    assert meta["type"] == "uhs" and meta["columns"] == [("PGA", 0.1), ("SA(0.2)", 0.1)]


def test_csv_without_header_line_and_bad(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("lon,lat,PGA-0.1\n1,2,3\n")
    assert read_csv(p)[1]["type"] == "map"
    p.write_text("lon,lat,foo\n1,2,3\n")
    with pytest.raises(ValueError, match="columns"):
        read_csv(p)
    with pytest.raises(FileNotFoundError):
        read_csv(tmp_path / "none.csv")


# NRML


def test_nrml_fixtures(data):
    s = read_sources(data / "nrml/simple.xml")
    assert len(s) == 5 and set(s["kind"]) == {"simple"}
    r = s.iloc[0]
    assert r["id"] == "CL-038" and r["dip"] == 72 and r["lower_depth"] == 20 and r["rake"] == 90
    assert isinstance(r["geometry"], LineString) and r["mfd"]["type"] == "incremental"
    assert r["rate"] == pytest.approx(r["mfd"]["rates"].sum()) and r["mmax"] == pytest.approx(6.05 + 0.9)
    c = read_sources(data / "nrml/complex.xml")
    assert len(c) == 1 and len(c["edges"][0]) == 10 and c["upper_depth"][0] == 10


NRML = """<?xml version='1.0' encoding='utf-8'?>
<nrml xmlns="http://openquake.org/xmlns/nrml/0.5" xmlns:gml="http://www.opengis.net/gml">
<sourceModel name="m">
 <sourceGroup name="g1" tectonicRegion="Active Shallow Crust">
  <simpleFaultSource id="f1" name="F" tectonicRegion="Active Shallow Crust">
   <simpleFaultGeometry><gml:LineString><gml:posList>0 0 1 1</gml:posList></gml:LineString>
    <dip>60</dip><upperSeismoDepth>0</upperSeismoDepth><lowerSeismoDepth>15</lowerSeismoDepth></simpleFaultGeometry>
   <truncGutenbergRichterMFD aValue="3" bValue="1" minMag="5" maxMag="7"/><rake>-90</rake>
  </simpleFaultSource>
  <pointSource id="p1" name="P" tectonicRegion="Active Shallow Crust">
   <pointGeometry><gml:Point><gml:pos>2 3</gml:pos></gml:Point>
    <upperSeismoDepth>0</upperSeismoDepth><lowerSeismoDepth>20</lowerSeismoDepth></pointGeometry>
   <incrementalMFD minMag="5" binWidth="0.1"><occurRates>0.1 0.05</occurRates></incrementalMFD>
   <nodalPlaneDist><nodalPlane probability="1" strike="0" dip="45" rake="30"/></nodalPlaneDist>
  </pointSource>
 </sourceGroup>
 <sourceGroup name="g2" tectonicRegion="Subduction Interface">
  <complexFaultSource id="c1" name="C" tectonicRegion="Subduction Interface">
   <complexFaultGeometry>
    <faultTopEdge><gml:LineString><gml:posList>{top}</gml:posList></gml:LineString></faultTopEdge>
    <intermediateEdge><gml:LineString><gml:posList>{mid}</gml:posList></gml:LineString></intermediateEdge>
    <intermediateEdge><gml:LineString><gml:posList>{mid2}</gml:posList></gml:LineString></intermediateEdge>
    <faultBottomEdge><gml:LineString><gml:posList>{bot}</gml:posList></gml:LineString></faultBottomEdge>
   </complexFaultGeometry>
   <incrementalMFD minMag="7" binWidth="0.1"><occurRates>0.01</occurRates></incrementalMFD><rake>90</rake>
  </complexFaultSource>
  <areaSource id="a1" name="A" tectonicRegion="Subduction Interface">
   <areaGeometry><gml:Polygon><gml:exterior><gml:LinearRing><gml:posList>0 0 1 0 1 1 0 1</gml:posList>
   </gml:LinearRing></gml:exterior></gml:Polygon><upperSeismoDepth>0</upperSeismoDepth>
   <lowerSeismoDepth>10</lowerSeismoDepth></areaGeometry>
   <incrementalMFD minMag="5" binWidth="0.1"><occurRates>0.2</occurRates></incrementalMFD>
  </areaSource>
 </sourceGroup>
</sourceModel></nrml>"""


def edges(flip):
    top, mid, mid2, bot = ("0 0 10 0 1 10", "0.1 0 20 0.1 1 20", "0.2 0 30 0.2 1 30", "0.3 0 40 0.3 1 40")
    if flip:
        top, mid, mid2, bot = ("0 1 10 0 0 10", "0.1 1 20 0.1 0 20", "0.2 1 30 0.2 0 30", "0.3 1 40 0.3 0 40")
    return dict(top=top, mid=mid, mid2=mid2, bot=bot)


def test_nrml_mixed_kinds_and_groups(tmp_path):
    p = tmp_path / "m.xml"
    p.write_text(NRML.format(**edges(False)))
    s = read_sources(p).set_index("id")
    assert list(s["kind"]) == ["simple", "point", "complex", "area"]
    assert list(s["group"]) == ["g1", "g1", "g2", "g2"]
    assert s.loc["f1", "rate"] == pytest.approx(10 ** (3 - 5) - 10 ** (3 - 7)) and s.loc["f1", "mmax"] == 7
    assert isinstance(s.loc["p1", "geometry"], Point) and s.loc["p1", "rake"] == 30 and s.loc["p1", "dip"] == 45
    assert isinstance(s.loc["a1", "geometry"], Polygon)
    assert s.loc["c1", "upper_depth"] == 10 and s.loc["c1", "lower_depth"] == 40


def test_nrml_deprecated_edge_order_flips_every_edge(tmp_path):
    p = tmp_path / "m.xml"
    p.write_text(NRML.format(**edges(True)))
    e = read_sources(p).set_index("id").loc["c1", "edges"]
    assert [x[0, 1] for x in e] == [0, 0, 0, 0]
    p.write_text(NRML.format(**edges(False)))
    e = read_sources(p).set_index("id").loc["c1", "edges"]
    assert [x[0, 1] for x in e] == [0, 0, 0, 0]


def test_nrml_empty_model(tmp_path):
    p = tmp_path / "e.xml"
    p.write_text('<nrml xmlns="http://openquake.org/xmlns/nrml/0.4"><sourceModel name="x"/></nrml>')
    s = read_sources(p)
    assert s.empty and "geometry" in s.columns
    with pytest.raises(FileNotFoundError):
        read_sources(tmp_path / "none.xml")
