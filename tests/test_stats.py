import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pituplots import Catalog, darkgrid, theme
from pituplots.stats import b_value, mc_maxc, plot_b_stability, plot_cumulative, plot_curves, plot_fmd, \
    plot_magnitude_time
from pituplots.style import defaults

RNG = np.random.default_rng(1)


def gr(n=4000, b=1.0, mc=2.0, dm=0.1):
    m = mc - dm / 2 + RNG.exponential(np.log10(np.e) / b, n)
    return np.round(m / dm) * dm


def catalog(n=300):
    t = np.datetime64("2000-01-01") + np.sort(RNG.integers(0, 3650, n)).astype("timedelta64[D]")
    return Catalog(RNG.uniform(0, 1, n), RNG.uniform(0, 1, n), RNG.uniform(0, 30, n), gr(n), t.astype("datetime64[ms]"),
                   extra={"kind": np.where(RNG.random(n) < 0.5, "a", "b")})


# estimators


def test_b_value_recovers_a_synthetic_b():
    for b0 in (0.8, 1.0, 1.3):
        b, s, a, n = b_value(gr(20000, b0), 2.0)
        assert b == pytest.approx(b0, abs=4 * s) and n == 20000 and a == pytest.approx(np.log10(n))


def test_b_value_continuous_and_errors():
    m = 2 + RNG.exponential(np.log10(np.e), 20000)
    assert b_value(m, 2.0, dm=0)[0] == pytest.approx(1, abs=0.05)
    with pytest.raises(ValueError, match="two events"):
        b_value([2.0], 2.0)
    with pytest.raises(ValueError, match="undefined"):
        b_value([1.95, 1.95], 2.0)


def test_mc_maxc_mode_plus_correction():
    m = np.r_[np.full(10, 1.5), np.full(30, 1.8), np.full(5, 2.4)]
    assert mc_maxc(m) == pytest.approx(2.0)
    assert mc_maxc(m, correction=0) == pytest.approx(1.8)
    with pytest.raises(ValueError):
        mc_maxc([np.nan])


def test_mc_maxc_unrounded_input():
    assert mc_maxc(np.r_[np.full(30, 1.81), np.full(5, 2.43)], correction=0) == pytest.approx(1.8)


# plot_fmd


def test_fmd_inputs_and_return():
    c = catalog()
    for d in (c, c.mag, list(c.mag), pd.DataFrame({"lon": c.lon, "lat": c.lat, "mag": c.mag}),
              (c.lon, c.lat, c.mag)):
        ax = plot_fmd(d)
        assert ax.get_yscale() == "log"
    a = plt.subplots()[1]
    assert plot_fmd(c, ax=a) is a


def test_fmd_counts_and_fit_legend():
    m = np.array([2.0, 2.0, 2.1, 2.3])
    ax = plot_fmd(m, mc=2.0)
    cum = [l for l in ax.lines if l.get_gid() == "fmd"][0]
    assert list(cum.get_ydata()) == [4, 2, 1, 1]
    assert any(t.get_text().startswith("b = ") for t in ax.get_legend().get_texts())


def test_fmd_per_year_by_group_and_no_mc():
    c = catalog()
    ax = plot_fmd(c, by="kind", years=10, mc=None)
    lines = [l for l in ax.lines if l.get_gid() == "fmd"]
    assert len(lines) == 2 and lines[0].get_color() != lines[1].get_color()
    assert "per year" in ax.get_ylabel() and max(lines[0].get_ydata()) < 30
    with pytest.raises(ValueError, match="by has shape"):
        plot_fmd(c, by=np.ones(3))


def test_fmd_end_members():
    plot_fmd(np.array([3.0]))
    plot_fmd(np.full(20, 3.0))
    plot_fmd(np.array([np.nan, 2.0, 2.5]), mc=2.0)
    plot_fmd(np.array([-1.0, -0.5, 0.0, 0.2]))
    plot_fmd(np.array([]), mc=None)


def test_fmd_two_calls_compare():
    ax = plot_fmd(gr(500), label="a")
    plot_fmd(gr(500, 1.3), ax=ax, label="b")
    lines = [l for l in ax.lines if l.get_gid() == "fmd"]
    assert len(lines) == 2 and lines[0].get_color() != lines[1].get_color()


# plot_magnitude_time


def test_magnitude_time_categories_and_numeric():
    c = catalog()
    ax = plot_magnitude_time(c, color_by="kind")
    assert sorted(t.get_text() for t in ax.get_legend().get_texts()) == ["a", "b"]
    ax = plot_magnitude_time(c, color_by="depth", cmap="viridis", clabel="depth")
    assert len(ax.figure.axes) == 2
    ax = plot_magnitude_time(c)
    assert ax.get_facecolor()[:3] == pytest.approx(matplotlib.colors.to_rgb("0.9"))


def test_magnitude_time_largest_on_top_and_needs_time():
    c = catalog()
    ax = plot_magnitude_time(c)
    y = ax.collections[0].get_offsets()[:, 1]
    assert (np.diff(y) >= 0).all()
    with pytest.raises(ValueError, match="times"):
        plot_magnitude_time(Catalog([1], [1], mag=[2]))


def test_magnitude_time_one_event_and_same_magnitude():
    t = np.array(["2020-01-01"], "datetime64[ms]")
    plot_magnitude_time(Catalog([1], [1], mag=[3.0], time=t))
    t2 = np.array(["2020-01-01", "2021-01-01"], "datetime64[ms]")
    plot_magnitude_time(Catalog([1, 2], [1, 2], mag=[3.0, 3.0], time=t2))


# plot_cumulative


def test_cumulative_steps():
    c = catalog(50)
    ax = plot_cumulative(c)
    y = ax.lines[-1].get_ydata()
    assert y[0] == 0 and y[-1] == 50
    ax = plot_cumulative(c, mag_min=2.5)
    assert ax.lines[-1].get_ydata()[-1] == (c.mag >= 2.5).sum() and "M ≥ 2.5" in ax.get_ylabel()


def test_cumulative_simulations_keep_empty_catalogs():
    c = catalog(50)
    t = np.array([], "datetime64[ms]")
    empty = [Catalog([], [], mag=[], time=t) for _ in range(9)]
    ax = plot_cumulative(c, simulations=empty + [c], start="2000-01-01", end="2010-01-01")
    median = [l for l in ax.lines if l.get_label() == "median"][0]
    assert median.get_ydata().max() == 0
    assert len(ax.collections) == 2


def test_cumulative_period_and_errors():
    c = catalog(50)
    ax = plot_cumulative(c, start="2003-01-01", end="2004-01-01")
    assert ax.lines[-1].get_ydata()[-1] == ((c.time >= np.datetime64("2003-01-01")) &
                                           (c.time <= np.datetime64("2004-01-01"))).sum()
    with pytest.raises(ValueError, match="period"):
        plot_cumulative(Catalog([], [], mag=[], time=np.array([], "datetime64[ms]")))
    with pytest.raises(ValueError, match="five"):
        plot_cumulative(c, percentiles=(5, 95))
    with pytest.raises(ValueError, match="times"):
        plot_cumulative(Catalog([1], [1], mag=[2]))


# plot_b_stability


def test_b_stability():
    ax = plot_b_stability(gr(5000), min_events=100, adopted=2.0, label="x")
    x = ax.lines[0].get_xdata()
    assert x[0] == pytest.approx(2.0) and len(ax.collections) == 1
    assert np.abs(ax.lines[0].get_ydata()[:5] - 1).max() < 0.1
    plot_b_stability(np.array([2.0, 2.1]))


# plot_curves

IMLS = np.logspace(-2, 0.5, 20)


def rlz(n=50):
    s = RNG.uniform(0.5, 1.5, n)[:, None]
    return np.clip(0.05 * (IMLS[None] / 0.02) ** (-2.0 * s), 0, 1)


def test_curves_mean_envelope_branches():
    c, w = rlz(), RNG.random(50)
    ax = plot_curves(c, imls=IMLS, weights=w, investigation_time=50, branches=True, label="model")
    mean = [l for l in ax.lines if l.get_gid() == "curve"][0]
    assert np.allclose(mean.get_ydata(), (w / w.sum()) @ c)
    assert len(ax.collections) == 1 and len([l for l in ax.lines if l.get_lw() == 0.4]) == 50
    assert ax.get_xscale() == ax.get_yscale() == "log"
    assert "50 yr" in ax.get_ylabel()


def test_curves_poe_lines_and_window():
    c = rlz()
    ax = plot_curves(c, imls=IMLS, investigation_time=1, window=50)
    assert sorted(t.get_text() for t in ax.texts) == ["10% in 50 yr", "2% in 50 yr"]
    mean = [l for l in ax.lines if l.get_gid() == "curve"][0]
    assert np.allclose(mean.get_ydata(), (1 - (1 - c) ** 50).mean(axis=0))
    with pytest.raises(ValueError, match="investigation time"):
        plot_curves(c, imls=IMLS, window=50)


def test_curves_compare_in_new_colours():
    ax = plot_curves(rlz(), imls=IMLS, investigation_time=50, label="a")
    plot_curves(rlz(), imls=IMLS, investigation_time=50, label="b", ax=ax)
    m = [l for l in ax.lines if l.get_gid() == "curve"]
    assert m[0].get_color() != m[1].get_color() and len(ax.texts) == 2


def test_curves_zeros_and_single_curve():
    c = rlz(1)[0]
    c[-3:] = 0
    ax = plot_curves(c, imls=IMLS)
    y = [l for l in ax.lines if l.get_gid() == "curve"][0].get_ydata()
    assert np.isnan(y[-3:]).all() and not ax.texts and len(ax.collections) == 0


def test_curves_errors():
    with pytest.raises(ValueError, match="imls"):
        plot_curves(rlz())
    with pytest.raises(ValueError, match="levels"):
        plot_curves(rlz(), imls=IMLS[:5])
    with pytest.raises(ValueError, match="weights"):
        plot_curves(rlz(), imls=IMLS, weights=[1, 2])


def test_curves_from_a_datastore(data):
    pytest.importorskip("h5py")
    from pituplots.io import read_datastore

    oq = read_datastore(data / "lacq/calc_11_subset.hdf5")
    ax = plot_curves(oq, site=(-0.62, 43.42), window=50)
    assert ax.get_xlabel() == "PGA [g]" and len(ax.collections) == 0
    with pytest.raises(ValueError, match="choose one"):
        plot_curves(oq)


# style


def test_darkgrid_and_layering_through_a_theme():
    ax = darkgrid(plt.subplots()[1])
    assert not any(s.get_visible() for s in ax.spines.values())
    assert plot_curves(rlz(), imls=IMLS).lines[-1].get_lw() == 2
    with theme("paper"):
        assert plot_curves(rlz(), imls=IMLS).lines[-1].get_lw() == 1.2
        assert plot_curves(rlz(), imls=IMLS, lw=3).lines[-1].get_lw() == 3
        ax = plot_fmd(gr(100), facecolor="white")
        assert ax.get_facecolor()[:3] == (1, 1, 1)


def test_darkgrid_family_from_a_yaml_theme(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("extends: report\nstyle:\n  darkgrid: {facecolor: '#EAEAF2'}\n")
    with theme(p):
        ax = plot_fmd(gr(100))
    assert matplotlib.colors.to_hex(ax.get_facecolor()) == "#eaeaf2"
    assert defaults.THEME == {}


def test_stats_keep_rcparams():
    before = dict(matplotlib.rcParams)
    plot_fmd(gr(100))
    plot_curves(rlz(), imls=IMLS, investigation_time=50)
    plot_magnitude_time(catalog())
    assert dict(matplotlib.rcParams) == before


def test_fmd_colour_dict_by_group():
    c = catalog()
    ax = plot_fmd(c, by="kind", colors={"a": "red", "b": "blue"})
    lines = {l.get_label(): l.get_color() for l in ax.lines if l.get_gid() == "fmd"}
    assert lines == {"a": "red", "b": "blue"}


def test_curves_poe_labels_dict():
    ax = plot_curves(rlz(), imls=IMLS, investigation_time=1, poes={0.002105: "475 yr"})
    assert [t.get_text() for t in ax.texts] == ["475 yr"]


def test_curves_second_call_keeps_the_imt_label():
    ax = plot_curves(rlz(), imls=IMLS, imt="PGA")
    plot_curves(rlz(), imls=IMLS, ax=ax)
    assert ax.get_xlabel() == "PGA [g]"


# additions from the project ports


def years_since(n, y0=1900, y1=2020):
    return np.datetime64(f"{y0}-01-01") + (RNG.uniform(0, (y1 - y0) * 365.25, n)).astype("timedelta64[D]")


def test_weichert_equals_aki_with_one_level():
    from pituplots.stats import weichert

    m = gr(5000)
    t = years_since(m.size)
    b1 = b_value(m, 2.0)[0]
    b2, s, rate, n = weichert(m, t, [(2.0, 1900)], t_end=2020)
    assert b2 == pytest.approx(b1, abs=0.02) and rate == pytest.approx(n / 120, rel=0.01)


def test_weichert_recovers_b_with_time_varying_completeness():
    from pituplots.stats import weichert

    m = gr(40000, b=1.0, mc=4.0)
    t = years_since(m.size)
    yr = 1970 + (t.astype("datetime64[ms]").astype("int64") / 1000 / 86400 / 365.2425)
    keep = (m >= 5.95) | ((m >= 4.95) & (yr >= 1960)) | (yr >= 2000)
    b, s, rate, n = weichert(m[keep], t[keep], [(4.0, 2000), (5.0, 1960), (6.0, 1900)], t_end=2020)
    assert b == pytest.approx(1.0, abs=3 * s) and rate == pytest.approx(m.size / 120, rel=0.05)
    with pytest.raises(ValueError, match="two"):
        weichert([4.0], t[:1], [(4.0, 1900)])


def test_fmd_completeness_rates_and_fit():
    m = gr(3000, mc=4.0)
    c = Catalog(m * 0, m * 0, mag=m, time=years_since(m.size))
    ax = plot_fmd(c, completeness=[(4.0, 1900)], t_end=2020, incremental=False)
    cum = [l for l in ax.lines if l.get_gid() == "fmd"][0]
    assert cum.get_ydata()[0] == pytest.approx(3000 / 120, rel=0.01) and "per year" in ax.get_ylabel()
    assert not [l for l in ax.lines if l.get_marker() == "o"]
    with pytest.raises(ValueError, match="times"):
        plot_fmd(m, completeness=[(4.0, 1900)])


def test_gr_model_forms_and_plot_mfd():
    from pituplots.stats import gr_model, plot_mfd

    for form in ("tgr", "tapered", "gr"):
        mm, r = gr_model(0.5, 1.0, 5.0, 8.0, form=form)
        assert mm[0] == 5.0 and r.sum() == pytest.approx(0.5, rel=0.02 if form != "tgr" else 1e-9)
    mm, r = gr_model(0.5, 1.0, 5.0, 8.0)
    assert mm[-1] == pytest.approx(7.9)
    ax = plot_mfd(mm, r, label="model", ls="--")
    ax2 = plot_mfd(mm, r * 2, ax=ax, label="b")
    l = [x for x in ax.lines if x.get_gid() == "mfd"]
    assert l[0].get_ydata()[0] == pytest.approx(0.5) and l[0].get_linestyle() == "--"
    assert l[0].get_color() != l[1].get_color() and ax2 is ax
    with pytest.raises(ValueError, match="form"):
        gr_model(1, 1, 5, 8, form="x")
    with pytest.raises(ValueError):
        plot_mfd([1, 2], [1])


def test_magnitude_time_numeric_years_and_label():
    c = Catalog([1, 2, 3], [1, 2, 3], mag=[3, 4, 5])
    ax = plot_magnitude_time(c, time=[10.5, 20.0, 30.2], label="sim")
    assert ax.collections[0].get_offsets()[:, 0].tolist() == [10.5, 20.0, 30.2]
    assert ax.get_legend().get_texts()[0].get_text() == "sim"
    with pytest.raises(ValueError, match="shape"):
        plot_magnitude_time(c, time=[1, 2])


def test_cumulative_numeric_normalized_poisson():
    c = Catalog([1, 2, 3, 4], [1, 2, 3, 4], mag=[3, 4, 5, 6])
    ax = plot_cumulative(c, time=[1.0, 2.0, 3.0, 10.0], normalized=True, poisson=True)
    y = ax.lines[0].get_ydata()
    assert y[-1] == 1 and "Fraction" in ax.get_ylabel()
    assert [l.get_label() for l in ax.lines][-1] == "uniform rate"


def test_b_stability_from_a_table():
    ax = plot_b_stability(table=([4.0, 4.5, 5.0], [1.0, 1.1, 0.9], [0.05, 0.08, 0.1]))
    assert ax.lines[0].get_ydata().tolist() == [1.0, 1.1, 0.9]


def test_curves_linestyle_marks_and_band_first():
    c = rlz()
    ax = plot_curves(c, imls=IMLS, investigation_time=50, mean=False)
    plot_curves(c, imls=IMLS, investigation_time=50, ax=ax, ls="--", mark_iml=True, ylim=(1e-3, 1))
    assert len([l for l in ax.lines if l.get_gid() == "poe"]) == 2
    mean = [l for l in ax.lines if l.get_gid() == "curve"][0]
    assert mean.get_linestyle() == "--" and any(t.get_text().endswith(" g") for t in ax.texts)
    assert ax.get_ylim() == pytest.approx((1e-3, 1))
    ax2 = plot_curves(c, imls=IMLS, ylim=(1e-2, 1))
    plot_curves(c, imls=IMLS, ax=ax2)
    assert ax2.get_ylim() == pytest.approx((1e-2, 1))


def test_magnitude_time_period_and_mfd_legend_switch():
    from pituplots.stats import plot_mfd

    c = catalog()
    ax = plot_magnitude_time(c, start="2003-01-01", end="2004-01-01")
    assert matplotlib.dates.num2date(ax.get_xlim()[0]).year == 2003
    assert plot_mfd([5.0, 5.1], [1.0, 0.5], label="x", legend=False).get_legend() is None


def test_b_stability_without_sigma_and_mc_line_style():
    ax = plot_b_stability(table=([4.0, 4.5], [1.0, 1.1]))
    assert len(ax.collections) == 0
    ax = plot_fmd(gr(300), mc=2.0, mc_color="red", mc_ls="--")
    v = [l for l in ax.lines if l.get_xdata()[0] == l.get_xdata()[-1]][0]
    assert v.get_color() == "red" and v.get_linestyle() == "--"
