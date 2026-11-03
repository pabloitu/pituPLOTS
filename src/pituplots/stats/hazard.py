import numpy as np
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

from pituplots.io.oq import OQResults, wquantile
from pituplots.stats.catalogs import new_axes
from pituplots.style.axes import darkgrid
from pituplots.style.defaults import resolve


def site_index(oq, site):
    if isinstance(site, str):
        if oq.sites.names is None or site not in oq.sites.names:
            raise ValueError(f"no site named {site!r}")
        return int(np.flatnonzero(oq.sites.names == site)[0])
    if site is None:
        if oq.sites.lon.size > 1:
            raise ValueError(f"{oq.sites.lon.size} sites: choose one with site=index or site=(lon, lat)")
        return 0
    if np.ndim(site) == 0:
        return int(site)
    lon, lat = site
    d = np.hypot((oq.sites.lon - lon + 180) % 360 - 180, oq.sites.lat - lat)
    return int(np.argmin(d))


def plot_curves(data, ax=None, imls=None, site=None, imt=None, weights=None, investigation_time=None, window=None,
                poes=(0.1, 0.02), envelope=(2.5, 97.5), branches=False, mean=True, mark_iml=False, label=None,
                color=None, figsize=None, **kw):
    """Hazard curves at a site: weighted mean, quantile envelope and branches, on log axes.

    Several calls on one axes compare models, each in the next colour.

    Parameters
    ----------
    data : OQResults or array
        A datastore, or curves ``(levels,)`` or ``(realizations, levels)`` of
        PoE in the investigation time.
    ax : Axes, optional
    imls : array, optional
        Levels, for array data.
    site : int, str or (lon, lat), optional
        Site of a datastore: index, name (``custom_site_id``) or nearest one.
    imt : str, optional
        IMT of a datastore; also the x label.
    weights : array, optional
        Realization weights; default the datastore's or equal.
    investigation_time : float, optional
        Of array data, in years.
    window : float, optional
        Plot PoE in this many years instead of the investigation time.
    poes : sequence of float or dict, optional
        Reference PoE lines, in the plotted window, labelled "10% in 50 yr",
        or ``{poe: label}``.
    envelope : (float, float) or None
        Weighted percentiles of the band.
    branches : bool
        Draw every realization.
    mean : bool
    mark_iml : bool
        Mark where the mean crosses each reference PoE and write the level.
    label, color : optional
    figsize : tuple, optional
    **kw
        ``lw``, ``envelope_alpha``, ``branch_lw``, ``branch_alpha``,
        ``poe_color``, ``poe_lw``, ``poe_ls``, ``poe_fontsize``, ``ylim``,
        ``xlim``, ``title``, and the ``darkgrid`` keywords. Other line
        keywords (``ls``, ``alpha``, ...) pass to the mean curve.

    Returns
    -------
    Axes
    """
    dg = resolve("darkgrid")
    grid = {k: kw.pop(k) for k in list(kw) if k in dg}
    st = resolve("curves", **kw)
    if isinstance(data, OQResults):
        imt, j = data.imt(imt)
        imls = data.imtls[imt]
        i = site_index(data, site)
        T = data.investigation_time
        if data.weights is not None:
            c = data.curves(imt)[i]
            w = data.weights if weights is None else np.asarray(weights, float)
        else:
            c, w = data.mean(imt)[i][None], np.ones(1)
            envelope, branches = None, False
    else:
        if imls is None:
            raise ValueError("array curves need imls")
        c = np.atleast_2d(np.asarray(data, float))
        T = investigation_time
        w = np.ones(c.shape[0]) if weights is None else np.asarray(weights, float)
    imls = np.asarray(imls, float)
    if c.shape[-1] != imls.size:
        raise ValueError(f"curves have {c.shape[-1]} levels, imls {imls.size}")
    if w.size != c.shape[0]:
        raise ValueError(f"{w.size} weights for {c.shape[0]} curves")
    w = w / w.sum()
    if window is not None:
        if T is None:
            raise ValueError("window needs the investigation time")
        c = 1 - (1 - c) ** (window / T)
    T = window or T

    fam = set(resolve("curves")) | {"xlim", "title"}
    line = {k: st.pop(k) for k in list(st) if k not in fam}
    ax = new_axes(ax, figsize)
    n = sum(1 for a in ax.lines if a.get_gid() == "curve")
    first = not any(a.get_gid() in ("curve", "poe") for a in ax.lines) and not ax.collections
    col = color or st["colors"][n % len(st["colors"])]
    pos = lambda y: np.where(y > 0, y, np.nan)  # noqa: E731
    if branches and c.shape[0] > 1:
        for y in c:
            ax.plot(imls, pos(y), color=col, lw=st["branch_lw"], alpha=st["branch_alpha"])
    if envelope is not None and c.shape[0] > 1:
        lo, hi = (wquantile(c[None], w, p / 100)[0] for p in envelope)
        ax.fill_between(imls, pos(lo), pos(hi), color=col, alpha=st["envelope_alpha"], lw=0,
                        label=f"{label + ', ' if label else ''}{envelope[0]:g}-{envelope[1]:g}%")
    m = w @ c
    if mean:
        ax.plot(imls, pos(m), color=col, lw=st["lw"], gid="curve",
                label=label or ("mean" if c.shape[0] > 1 else None), **line)
    ax.set_xscale("log")
    ax.set_yscale("log")
    if first or "xlim" in kw:
        ax.set_xlim(st.get("xlim") or (imls[0], imls[-1]))
    if first or "ylim" in kw:
        ax.set_ylim(*st["ylim"])
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    labels = (poes if isinstance(poes, dict) else {p: f"{100 * p:g}% in {T:g} yr" for p in poes}) \
        if poes and T is not None else {}
    if first:
        for p, t in labels.items():
            ax.axhline(p, color=st["poe_color"], lw=st["poe_lw"], ls=st["poe_ls"], gid="poe")
            ax.text(0.02, p * 1.15, t, transform=tr, fontsize=st["poe_fontsize"], va="bottom")
    if mark_iml:
        from pituplots.io.oq import poe_at

        for p in labels:
            v = float(poe_at(m, imls, p))
            if v > 0:
                ax.plot(v, p, "o", color=col, ms=5, zorder=5)
                ax.annotate(f"{v:.2f} g", (v, p), xytext=(4, -4), textcoords="offset points", va="top",
                            fontsize=st["poe_fontsize"], color=col)
    if imt or not ax.get_xlabel():
        ax.set_xlabel(f"{imt or 'IML'} [g]")
    ax.set_ylabel(f"Probability of exceedance in {T:g} yr" if T else "Probability of exceedance")
    if st.get("title"):
        ax.set_title(st["title"])
    darkgrid(ax, **grid)
    hs = [h for h in ax.get_legend_handles_labels()[0]]
    if hs:
        ax.legend(loc="lower left")
    return ax
