import matplotlib
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


def alpha_cmap(cmap, alpha_exp):
    """Colormap with an alpha ramp, ``linspace(0, 1) ** alpha_exp``.

    Parameters
    ----------
    cmap : str or Colormap
    alpha_exp : float
        0 keeps the colormap opaque.

    Returns
    -------
    ListedColormap
    """
    if isinstance(cmap, str):
        cmap = matplotlib.colormaps[cmap]
    c = cmap(np.arange(cmap.N))
    if alpha_exp != 0:
        c[:, -1] = np.linspace(0, 1, cmap.N) ** alpha_exp
    return ListedColormap(c, name=f"{cmap.name}_alpha")


def discrete_cmap(name, levels):
    """Listed colormap and matching norm for non-uniform levels.

    Parameters
    ----------
    name : str or Colormap
    levels : sequence of float
        Ascending bin edges, at least two.

    Returns
    -------
    (Colormap, BoundaryNorm)
    """
    n = len(levels) - 1
    if n < 1:
        raise ValueError(f"need at least two levels, got {list(levels)}")
    cmap = matplotlib.colormaps[name] if isinstance(name, str) else name
    return cmap.resampled(n), BoundaryNorm(levels, ncolors=n, clip=False)


def class_breaks(values, n=5, method="quantile", round_to=None, round_ends=False):
    """Class edges for a classified map.

    Parameters
    ----------
    values : array_like
        NaN and masked values are ignored.
    n : int
        Number of classes.
    method : {"quantile", "equal"}
        Equal counts or equal widths.
    round_to : float, optional
        Round the inner edges to a multiple of this.
    round_ends : bool
        Also round the outer edges, outwards, so they enclose the data.

    Returns
    -------
    ndarray
        ``n + 1`` ascending edges, the data range at the ends.
    """
    v = np.ma.masked_invalid(np.ma.asarray(values, float)).compressed()
    if v.size == 0:
        raise ValueError("no values to classify")
    if method == "quantile":
        e = np.quantile(v, np.linspace(0, 1, n + 1))
    elif method == "equal":
        e = np.linspace(v.min(), v.max(), n + 1)
    else:
        raise ValueError(f"method must be 'quantile' or 'equal', not {method!r}")
    if round_to:
        e[1:-1] = np.round(e[1:-1] / round_to) * round_to
        if round_ends:
            e[0], e[-1] = np.floor(e[0] / round_to) * round_to, np.ceil(e[-1] / round_to) * round_to
    e = np.unique(e)
    if e.size < 2:
        e = np.array([v.min(), v.min() + 1])
    return e
