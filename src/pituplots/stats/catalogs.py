from functools import partial

import numpy as np
import pandas as pd

from pituplots.catalogs import as_catalog, by_name, draw_events, size_by_value
from pituplots.style.axes import darkgrid
from pituplots.style.defaults import resolve


def new_axes(ax, figsize):
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt

    return plt.figure(figsize=figsize, layout="constrained").add_subplot()


def decimal_years(t):
    """Decimal years of datetime64 times; numeric times as they are."""
    t = np.asarray(t)
    if np.issubdtype(t.dtype, np.datetime64):
        return 1970 + (t.astype("datetime64[ms]").astype("int64") / 1000 / 86400 / 365.2425)
    return t.astype(float)


def date_axis(ax, t):
    if np.issubdtype(np.asarray(t).dtype, np.datetime64):
        import matplotlib.dates as mdates

        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))


def magnitudes(data):
    if isinstance(data, (np.ndarray, list)) or (isinstance(data, pd.Series)):
        return np.asarray(data, float)
    c = as_catalog(data)
    if c.mag is None:
        raise ValueError("the catalog has no magnitudes")
    return c.mag


def mc_maxc(mag, dm=0.1, correction=0.2):
    """Completeness magnitude by maximum curvature, plus a correction.

    Parameters
    ----------
    mag : array_like
        Magnitudes, binned or not; they are rounded to ``dm``.
    dm : float
    correction : float
        Added to the mode, 0.2 by convention.

    Returns
    -------
    float
    """
    m = np.asarray(mag, float)
    m = m[np.isfinite(m)]
    if m.size == 0:
        raise ValueError("no magnitudes")
    c = np.round(m / dm) * dm
    v, n = np.unique(np.round(c, 6), return_counts=True)
    return float(v[np.argmax(n)] + correction)


def b_value(mag, mc, dm=0.1):
    """Gutenberg-Richter b-value by maximum likelihood (Aki-Utsu).

    Parameters
    ----------
    mag : array_like
    mc : float
        Events from ``mc - dm/2`` up are used.
    dm : float
        Bin width; 0 for continuous magnitudes.

    Returns
    -------
    (b, sigma, a, n)
        ``sigma`` is Shi and Bolt (1982); ``a`` is ``log10 N(>=mc)``.
    """
    m = np.asarray(mag, float)
    m = m[np.isfinite(m) & (m >= mc - dm / 2 - 1e-9)]
    if m.size < 2:
        raise ValueError(f"need at least two events above mc = {mc}, have {m.size}")
    d = m.mean() - (mc - dm / 2)
    if d <= 0:
        raise ValueError("all events are at mc: the b-value is undefined")
    b = np.log10(np.e) / d
    s = 2.3 * b ** 2 * np.sqrt(((m - m.mean()) ** 2).sum() / (m.size * (m.size - 1)))
    return float(b), float(s), float(np.log10(m.size)), int(m.size)


def weichert(mag, time, completeness, t_end=None, dm=0.1):
    """b-value and annual rate with completeness that changes in time (Weichert, 1980).

    Parameters
    ----------
    mag, time : array_like
        Magnitudes, and times as datetime64 or decimal years.
    completeness : sequence of (float, float)
        ``(mc, since)``: events of ``mc`` and above are complete from year
        ``since``.
    t_end : float, optional
        End of the catalog in decimal years; default the last event.
    dm : float

    Returns
    -------
    (b, sigma, rate, n)
        ``rate`` is the annual number of events from the lowest ``mc``.
    """
    m, yr = np.asarray(mag, float), decimal_years(time)
    te = yr.max() if t_end is None else float(t_end)
    lv = sorted((float(a), float(b)) for a, b in completeness)
    c = np.round(np.round(m / dm) * dm, 6)
    bins = np.round(np.arange(lv[0][0], np.nanmax(c) + dm / 2, dm), 6)
    since = np.array([min(b for a, b in lv if a <= x + 1e-9) for x in bins])
    T = te - since
    n = np.array([((c == x) & (yr >= y)).sum() for x, y in zip(bins, since)], float)
    k = T > 0
    bins, T, n = bins[k], T[k], n[k]
    N = n.sum()
    if N < 2:
        raise ValueError(f"need at least two complete events, have {int(N)}")
    mbar, beta = (n * bins).sum() / N, 2.3
    for _ in range(100):
        w = T * np.exp(-beta * bins)
        s0, s1, s2 = w.sum(), (w * bins).sum(), (w * bins ** 2).sum()
        var = s2 / s0 - (s1 / s0) ** 2
        step = (s1 / s0 - mbar) / -var
        beta -= step
        if abs(step) < 1e-10:
            break
    w = T * np.exp(-beta * bins)
    var = (w * bins ** 2).sum() / w.sum() - ((w * bins).sum() / w.sum()) ** 2
    rate = N * np.exp(-beta * bins).sum() / w.sum()
    return float(beta / np.log(10)), float(np.sqrt(1 / (N * var)) / np.log(10)), float(rate), int(N)


def plot_fmd(data, ax=None, dm=0.1, mc="maxc", fit=True, by=None, years=None, label=None, figsize=None,
             incremental=True, completeness=None, t_end=None, **kw):
    """Frequency-magnitude distribution: cumulative steps, incremental dots, GR fit.

    Parameters
    ----------
    data : Catalog, DataFrame, tuple or array of magnitudes
    ax : Axes, optional
    dm : float
        Bin width.
    mc : "maxc", float or None
        Completeness for the fit, marked with a dotted line. None skips both.
    fit : bool
        Draw the Aki-Utsu fit above ``mc``, with b in the legend.
    by : str or array_like, optional
        Category per event (a column of a Catalog); one distribution each.
    years : float, optional
        Divide counts by this duration, for annual rates.
    label : str, optional
    figsize : tuple, optional
    incremental : bool
        Draw the counts per bin as dots.
    completeness : sequence of (float, float), optional
        ``(mc, since)`` levels: magnitudes of ``mc`` and above are complete
        from year ``since``. Counts become annual rates with each bin's own
        period, and the fit is Weichert's. Needs event times.
    t_end : float, optional
        End of the catalog in decimal years, with ``completeness``.
    **kw
        ``colors`` (a list, or a dict by group), ``lw``, ``ms``, ``mew``,
        ``alpha``, ``fit_color``, ``fit_lw``, ``mc_lw``, ``title``, and the
        ``darkgrid`` keywords.

    Returns
    -------
    Axes
    """
    st = resolve("fmd", **{k: v for k, v in kw.items() if k not in resolve("darkgrid")})
    grid = {k: v for k, v in kw.items() if k in resolve("darkgrid")}
    title = st.pop("title", None)
    m = magnitudes(data)
    yr = None
    if completeness is not None:
        cat = as_catalog(data) if not isinstance(data, (np.ndarray, list, pd.Series)) else None
        if cat is None or cat.time is None:
            raise ValueError("completeness needs event times")
        yr = decimal_years(cat.time)
        t_end = np.nanmax(yr) if t_end is None else float(t_end)
        lv = sorted((float(a), float(b)) for a, b in completeness)
        mc = lv[0][0] if isinstance(mc, str) else mc
    if by is None:
        groups = [(label, np.ones(m.shape, bool))]
    else:
        g = np.asarray(by_name(as_catalog(data), by) if isinstance(by, str) else by)
        if g.shape != m.shape:
            raise ValueError(f"by has shape {g.shape}, expected {m.shape}")
        groups = [(str(k), g == k) for k in pd.unique(g)]
    ax = new_axes(ax, figsize)
    n0 = sum(1 for a in ax.lines if a.get_gid() == "fmd")
    scale = 1 / years if years else 1.0
    one = len(groups) == 1
    for i, (lab, sel) in enumerate(groups):
        sel = sel & np.isfinite(m)
        x = m[sel]
        if x.size == 0:
            continue
        cs = st["colors"]
        col = cs.get(lab, f"C{i}") if isinstance(cs, dict) else cs[(n0 + i) % len(cs)]
        c = np.round(np.round(x / dm) * dm, 6)
        v = np.round(np.arange(c.min(), c.max() + dm / 2, dm), 6)
        if yr is None:
            inc = np.array([(c == b).sum() for b in v], float) * scale
        else:
            y = yr[sel]
            ok = np.array([any(a <= b + 1e-9 for a, _ in lv) for b in v])
            v = v[ok]
            since = np.array([min(s for a, s in lv if a <= b + 1e-9) for b in v])
            inc = np.array([((c == b) & (y >= s)).sum() / (t_end - s) for b, s in zip(v, since)])
        cum = np.cumsum(inc[::-1])[::-1]
        pre = f"{lab} " if lab else ""
        ax.step(v, cum, where="mid", color=col, lw=st["lw"], label=f"{pre}cumulative" if one else lab, gid="fmd")
        k = inc > 0
        if incremental:
            ax.plot(v[k], inc[k], "o", color=col, mec="black", mew=st["mew"], ms=st["ms"], alpha=st["alpha"],
                    label=f"{pre}incremental (Δ{dm:g})" if one else None)
        if mc is None:
            continue
        mcv = mc_maxc(x, dm) if isinstance(mc, str) and mc == "maxc" else float(mc)
        ax.axvline(mcv, color=st["mc_color"] or col, ls=st["mc_ls"], lw=st["mc_lw"])
        if not fit:
            continue
        mm = np.linspace(mcv, v.max() + dm / 2, 50)
        fc = st["fit_color"] if one else col
        if yr is None and (x >= mcv - dm / 2 - 1e-9).sum() >= 2:
            b, s, a, n = b_value(x, mcv, dm)
            r = n * scale
        elif yr is not None and (c >= mcv - 1e-9).sum() >= 2:
            b, s, r, n = weichert(x, yr[sel], [(max(a, mcv), t) for a, t in lv], t_end, dm)
        else:
            continue
        ax.plot(mm, r * 10 ** (-b * (mm - mcv)), color=fc, lw=st["fit_lw"], ls="--",
                label=f"{'' if one else lab + ': '}b = {b:.2f} ± {s:.2f}, Mc {mcv:g}")
    ax.set_yscale("log")
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("N(≥M) per year" if years or yr is not None else "N(≥M)")
    if title:
        ax.set_title(title)
    darkgrid(ax, **grid)
    if any(h.get_label() and not h.get_label().startswith("_") for h in ax.lines):
        ax.legend()
    return ax


def plot_magnitude_time(data, ax=None, color_by=None, sizer=None, time=None, start=None, end=None, figsize=None,
                        **kw):
    """Magnitude against time, sized by magnitude; largest on top.

    Parameters
    ----------
    data : Catalog or DataFrame
        Needs times.
    ax : Axes, optional
    color_by : str or array_like, optional
        As in ``plot_catalog``: numeric gets a colorbar, other values a legend.
    sizer : callable, optional
    time : array_like, optional
        Times to use instead of the catalog's, e.g. model years as numbers.
    start, end : optional
        Time axis limits, as datetime64, strings or numbers like ``time``.
    figsize : tuple, optional
    **kw
        ``color``, ``edgecolor``, ``alpha``, ``size``, ``max_size``,
        ``power``, ``colors``, ``cmap``, ``clabel``, ``title``, ``label``,
        and the ``darkgrid`` keywords.

    Returns
    -------
    Axes
    """
    cat = as_catalog(data)
    t = cat.time if time is None else np.asarray(time)
    if t is None or cat.mag is None:
        raise ValueError("plot_magnitude_time needs times and magnitudes")
    if t.shape != cat.mag.shape:
        raise ValueError(f"time has shape {t.shape}, expected {cat.mag.shape}")
    dg = resolve("darkgrid")
    grid = {k: kw.pop(k) for k in list(kw) if k in dg}
    st = resolve("magtime", **kw)
    clabel, title = st.pop("clabel", None), st.pop("title", None)
    size = {k: st.pop(k) for k in ("size", "max_size", "power")}
    if sizer is None and cat.mag.size:
        sizer = partial(size_by_value, **size, min_val=np.nanmin(cat.mag), max_val=np.nanmax(cat.mag))
    s = np.asarray(sizer(cat.mag), float) if sizer is not None else np.zeros(0)
    o = np.argsort(cat.mag, kind="stable")
    ax = new_axes(ax, figsize)
    v = None if color_by is None else np.asarray(by_name(cat, color_by))
    m = draw_events(ax, t[o], cat.mag[o], s[o], st, None if v is None else v[o])
    if start is not None or end is not None:
        cast = (lambda x: None if x is None else np.datetime64(x, "ms")) if np.issubdtype(t.dtype, np.datetime64) \
            else (lambda x: x)
        ax.set_xlim(cast(start), cast(end))
    date_axis(ax, t)
    if m is not None:
        cb = ax.figure.colorbar(m, ax=ax)
        if clabel:
            cb.set_label(clabel)
    if v is not None and not np.issubdtype(v.dtype, np.number) and v.size:
        lg = ax.legend()
        for h in lg.legend_handles if hasattr(lg, "legend_handles") else lg.legendHandles:
            h.set_alpha(1)
    if st.get("label") is not None or (v is None and any(c.get_label()[:1] not in ("_", "")
                                                        for c in ax.collections)):
        lg = ax.legend()
        for h in lg.legend_handles if hasattr(lg, "legend_handles") else lg.legendHandles:
            h.set_alpha(1)
    ax.set_ylabel("Magnitude")
    if title:
        ax.set_title(title)
    darkgrid(ax, **grid)
    return ax


def plot_cumulative(data, ax=None, mag_min=None, simulations=None, start=None, end=None,
                    percentiles=(5, 25, 50, 75, 95), label="observed", time=None, normalized=False, poisson=False,
                    figsize=None, **kw):
    """Cumulative number of events against time, with simulation percentiles.

    Parameters
    ----------
    data : Catalog or DataFrame
        Needs times.
    ax : Axes, optional
    mag_min : float, optional
        Count only events at or above it, in the data and the simulations.
    simulations : sequence of Catalog, optional
        Synthetic catalogs over the same period. Empty ones count as zero.
    start, end : datetime64 or str, optional
        Period; default the span of the data.
    percentiles : sequence of 5 float
        Outer band, inner band and median.
    label : str
    time : array_like, optional
        Times to use instead of the catalog's, e.g. model years as numbers.
    normalized : bool
        Divide by the observed total, so the curve ends at 1.
    poisson : bool
        Add the uniform-rate line from the start to the observed total.
    figsize : tuple, optional
    **kw
        ``color``, ``lw``, ``sim_color``, ``band_alpha``, ``poisson_color``,
        ``poisson_label``, ``title``, and the ``darkgrid`` keywords.

    Returns
    -------
    Axes
    """
    cat = as_catalog(data)
    if cat.time is None and time is None:
        raise ValueError("plot_cumulative needs times")
    if len(percentiles) != 5:
        raise ValueError("percentiles needs five values: outer band, inner band and median")
    dg = resolve("darkgrid")
    grid = {k: kw.pop(k) for k in list(kw) if k in dg}
    st = resolve("cumulative", **kw)
    title = st.pop("title", None)

    def times(c, t=None):
        c = as_catalog(c)
        t = (c.time if c.time is not None else np.array([], "datetime64[ms]")) if t is None else np.asarray(t)
        if mag_min is not None and c.mag is not None:
            t = t[c.mag >= mag_min]
        return np.sort(t.astype("datetime64[ms]") if np.issubdtype(t.dtype, np.datetime64) else t.astype(float))

    t = times(cat, time)
    dt = np.issubdtype(t.dtype, np.datetime64)
    cast = (lambda v: np.datetime64(v, "ms")) if dt else float
    t0 = cast(start) if start is not None else (t[0] if t.size else None)
    t1 = cast(end) if end is not None else (t[-1] if t.size else None)
    if t0 is None or t1 is None:
        raise ValueError("no events to set the period: give start and end")
    t = t[(t >= t0) & (t <= t1)]
    k = 1 / t.size if normalized and t.size else 1.0
    ax = new_axes(ax, figsize)
    if simulations is not None:
        grid_t = t0 + (t1 - t0) * np.linspace(0, 1, 300)
        n = np.array([np.searchsorted(times(s), grid_t, side="right") - np.searchsorted(times(s), t0)
                      for s in simulations], float)
        if n.size:
            q = np.percentile(n, percentiles, axis=0) * k
            a = st["band_alpha"]
            ax.fill_between(grid_t, q[0], q[4], color=st["sim_color"], alpha=a[0], lw=0,
                            label=f"simulations {percentiles[0]:g}-{percentiles[4]:g}%")
            ax.fill_between(grid_t, q[1], q[3], color=st["sim_color"], alpha=a[1], lw=0,
                            label=f"{percentiles[1]:g}-{percentiles[3]:g}%")
            ax.plot(grid_t, q[2], color=st["sim_color"], lw=1, label="median")
    ax.step(np.r_[t0, t, t1], np.r_[0, np.arange(1, t.size + 1), t.size] * k, where="post", color=st["color"],
            lw=st["lw"], label=label)
    if poisson:
        ax.plot([t0, t1], [0, t.size * k], color=st["poisson_color"] or st["color"], lw=1, ls="--", alpha=0.6,
                label=st["poisson_label"])
    what = "Fraction of events" if normalized else "Cumulative number of events"
    ax.set_ylabel(what + (f", M ≥ {mag_min:g}" if mag_min is not None else ""))
    date_axis(ax, t)
    ax.set_xlim(t0, t1)
    ax.set_ylim(bottom=0)
    if title:
        ax.set_title(title)
    darkgrid(ax, **grid)
    ax.legend(loc="upper left")
    return ax


def plot_b_stability(data=None, ax=None, mcs=None, dm=0.1, min_events=50, adopted=None, label=None, table=None,
                     figsize=None, **kw):
    """b-value against the completeness magnitude, with a ±1 sigma band.

    Parameters
    ----------
    data : Catalog, DataFrame, tuple or array of magnitudes
    ax : Axes, optional
    mcs : array_like, optional
        Trial completeness magnitudes; default from the smallest bin to where
        ``min_events`` remain, in steps of ``dm``.
    dm : float
    min_events : int
        Trials with fewer events above them are skipped.
    adopted : float, optional
        Mark this Mc.
    label : str, optional
    table : tuple of arrays, optional
        Precomputed ``(mc, b)`` or ``(mc, b, sigma)``, instead of ``data``.
    figsize : tuple, optional
    **kw
        ``colors``, ``lw``, ``title``, and the ``darkgrid`` keywords.

    Returns
    -------
    Axes
    """
    dg = resolve("darkgrid")
    grid = {k: kw.pop(k) for k in list(kw) if k in dg}
    st = resolve("fmd", **kw)
    rows = []
    if table is not None:
        t = [np.asarray(a, float) for a in table]
        rows = list(zip(*(t if len(t) == 3 else [*t, np.full(t[0].shape, np.nan)])))
    else:
        m = magnitudes(data)
        m = m[np.isfinite(m)]
        if mcs is None:
            c = np.round(np.round(m / dm) * dm, 6)
            mcs = np.round(np.arange(c.min(), c.max() + dm / 2, dm), 6) if m.size else []
        for mc in mcs:
            if (m >= mc - dm / 2 - 1e-9).sum() >= max(min_events, 2):
                b, s, _, n = b_value(m, mc, dm)
                rows.append((mc, b, s))
    ax = new_axes(ax, figsize)
    n0 = sum(1 for a in ax.lines if a.get_gid() == "bstab")
    col = st["colors"][n0 % len(st["colors"])]
    if rows:
        x, b, s = np.array(rows).T
        if np.isfinite(s).any():
            ax.fill_between(x, b - s, b + s, color=col, alpha=0.3, lw=0)
        ax.plot(x, b, "o-", color=col, lw=st["lw"], ms=st["ms"], gid="bstab", label=label)
    if adopted is not None:
        ax.axvline(adopted, color="black", ls=":", lw=1)
    ax.set_xlabel("Completeness magnitude")
    ax.set_ylabel("b-value")
    if st.get("title"):
        ax.set_title(st["title"])
    darkgrid(ax, **grid)
    if label:
        ax.legend()
    return ax


def gr_model(rate, b, mmin, mmax, dm=0.1, form="tgr"):
    """Incremental annual rates of a magnitude-frequency model.

    Parameters
    ----------
    rate : float
        Annual number of events of ``mmin`` and above.
    b : float
    mmin, mmax : float
        ``mmax`` is the upper truncation, or the corner of the tapered form.
    dm : float
    form : {"tgr", "tapered", "gr"}
        Truncated, tapered (Kagan) or unbounded Gutenberg-Richter.

    Returns
    -------
    (ndarray, ndarray)
        Lower bin edges and the annual rate in each bin.
    """
    top = mmax if form == "tgr" else mmax + 1.5
    m = np.round(np.arange(mmin, top - dm / 2 + 1e-9, dm), 6)
    e = np.r_[m, m[-1] + dm]
    if form == "tgr":
        f = (10 ** (-b * (e - mmin)) - 10 ** (-b * (mmax - mmin))) / (1 - 10 ** (-b * (mmax - mmin)))
    elif form == "tapered":
        mo = lambda x: 10 ** (1.5 * x + 9.1)  # noqa: E731
        f = (mo(e) / mo(mmin)) ** (-2 * b / 3) * np.exp((mo(mmin) - mo(e)) / mo(mmax))
    elif form == "gr":
        f = 10 ** (-b * (e - mmin))
    else:
        raise ValueError(f"form must be 'tgr', 'tapered' or 'gr', not {form!r}")
    return m, rate * (f[:-1] - f[1:])


def plot_mfd(mags, rates, ax=None, cumulative=True, label=None, color=None, legend=True, figsize=None, **kw):
    """Magnitude-frequency distribution of a model or forecast, on a log axis.

    Several calls on one axes compare models, each in the next colour, and a
    ``plot_fmd`` of the observed catalog can go on the same axes.

    Parameters
    ----------
    mags : array_like
        Lower bin edges.
    rates : array_like
        Rate or expected number in each bin.
    ax : Axes, optional
    cumulative : bool
        Plot the number at or above each magnitude.
    label, color : optional
    legend : bool
        Draw the legend when there is a label.
    figsize : tuple, optional
    **kw
        ``colors``, ``lw``, ``title``, the ``darkgrid`` keywords, and line
        keywords such as ``ls``.

    Returns
    -------
    Axes
    """
    m, r = np.asarray(mags, float), np.asarray(rates, float)
    if m.shape != r.shape:
        raise ValueError(f"{m.size} magnitudes and {r.size} rates")
    dg = resolve("darkgrid")
    grid = {k: kw.pop(k) for k in list(kw) if k in dg}
    fam = resolve("fmd")
    st = {"colors": fam["colors"], "lw": fam["lw"], **kw}
    cs, lw, title = st.pop("colors"), st.pop("lw"), st.pop("title", None)
    ax = new_axes(ax, figsize)
    n = sum(1 for a in ax.lines if a.get_gid() in ("mfd", "fmd"))
    col = color or (cs[n % len(cs)] if not isinstance(cs, dict) else cs.get(label, f"C{n}"))
    y = np.cumsum(r[::-1])[::-1] if cumulative else r
    ax.plot(m, np.where(y > 0, y, np.nan), color=col, lw=lw, label=label, gid="mfd", **st)
    ax.set_yscale("log")
    ax.set_xlabel("Magnitude")
    if not ax.get_ylabel():
        ax.set_ylabel("N(≥M)" if cumulative else "N per bin")
    if title:
        ax.set_title(title)
    darkgrid(ax, **grid)
    if label and legend:
        ax.legend()
    return ax
