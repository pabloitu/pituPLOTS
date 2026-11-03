from pituplots.style.defaults import resolve


def darkgrid(ax, **kw):
    """Gray background, white major and minor grid and gray ticks, set on one axes.

    The look of the hazard curves in ``psha_chile7/hazard/post.py``, without
    seaborn and without changing rcParams. Override it per call or in the
    ``darkgrid`` family of a theme.

    Parameters
    ----------
    ax : Axes
    **kw
        ``facecolor``, ``grid_color``, ``grid_lw``, ``minor_grid_lw``,
        ``tick_color``, ``major_length``, ``minor_length``, ``tick_width``.

    Returns
    -------
    Axes
    """
    st = resolve("darkgrid", **kw)
    ax.set_facecolor(st["facecolor"])
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(True, which="major", color=st["grid_color"], linewidth=st["grid_lw"])
    ax.grid(True, which="minor", color=st["grid_color"], linewidth=st["minor_grid_lw"])
    for w, n in (("major", "major_length"), ("minor", "minor_length")):
        ax.tick_params(which=w, length=st[n], width=st["tick_width"], color=st["tick_color"])
    return ax
