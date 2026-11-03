import warnings
from contextlib import contextmanager
from pathlib import Path

import matplotlib
from matplotlib import font_manager

from pituplots.errors import PituplotsWarning
from pituplots.style.defaults import THEME

HERE = Path(__file__).parent / "themes"
NAMES = ("paper", "slides", "report", "dark", "poster", "minimal")

# rcParams values from before the active theme, so the next one starts clean
SAVED = {}


def load_theme(name):
    """The ``rc`` and ``style`` of a packaged theme or a YAML file, merged over ``base``.

    A file may set ``extends: <theme>`` to start from another one. Families in
    ``style`` merge key by key.

    Parameters
    ----------
    name : str or Path
        A packaged theme name, or a ``.yaml`` path.

    Returns
    -------
    dict
        ``{"rc": {...}, "style": {family: {...}}}``
    """
    import yaml

    p = Path(name)
    if p.suffix in (".yaml", ".yml"):
        if not p.is_file():
            raise FileNotFoundError(p)
    elif name in NAMES or name == "base":
        p = HERE / f"{name}.yaml"
    else:
        raise ValueError(f"unknown theme {name!r}, have {list(NAMES)} or a .yaml path")
    d = yaml.safe_load(p.read_text()) or {}
    bad = set(d) - {"rc", "style", "extends"}
    if bad:
        raise ValueError(f"{p.name}: unknown keys {sorted(bad)}, expected rc, style, extends")
    parent = load_theme(d["extends"]) if d.get("extends") else (
        {"rc": {}, "style": {}} if name == "base" else load_theme("base"))
    style = {f: dict(v) for f, v in parent["style"].items()}
    for f, v in (d.get("style") or {}).items():
        style[f] = {**style.get(f, {}), **v}
    rc = {**parent["rc"], **(d.get("rc") or {})}
    bad = [k for k in rc if k not in matplotlib.rcParams]
    if bad:
        raise ValueError(f"{p.name}: unknown rcParams {bad}")
    return {"rc": rc, "style": style}


def use_theme(name):
    """Apply a theme until the session ends or another theme replaces it.

    Sets rcParams and the per-family style layer, so the order is package
    defaults < theme < explicit keywords. The rcParams of a previous theme are
    restored first. Ubuntu is used if installed; DejaVu Sans otherwise, with a
    warning. Emphasis comes from size, not weight.

    Parameters
    ----------
    name : str or Path
        ``"paper"``, ``"slides"``, ``"report"``, ``"dark"``, ``"poster"``,
        ``"minimal"``, or a YAML file with ``rc``,
        ``style`` and optionally ``extends``.
    """
    t = load_theme(name)
    fm = font_manager.fontManager
    if not any(f.name == "Ubuntu" for f in fm.ttflist):
        for f in font_manager.findSystemFonts():
            if Path(f).name.startswith("Ubuntu"):
                fm.addfont(f)
    font = ["Ubuntu", "DejaVu Sans"]
    if not any(f.name == "Ubuntu" for f in fm.ttflist):
        warnings.warn("Ubuntu font not found, using DejaVu Sans", PituplotsWarning, stacklevel=2)
        font = ["DejaVu Sans"]
    rc = {**t["rc"], "font.family": font}
    matplotlib.rcParams.update(SAVED)
    SAVED.clear()
    SAVED.update({k: matplotlib.rcParams[k] for k in rc})
    matplotlib.rcParams.update(rc)
    THEME.clear()
    THEME.update(t["style"])


@contextmanager
def theme(name):
    """Apply a theme inside a ``with`` block, and restore everything on exit.

    Save the figure inside the block: some settings, such as gridline labels
    and font embedding, are read when the figure is drawn.

    Parameters
    ----------
    name : str or Path
        As in ``use_theme``.
    """
    style, saved = dict(THEME), dict(SAVED)
    try:
        with matplotlib.rc_context():
            use_theme(name)
            yield
    finally:
        THEME.clear()
        THEME.update(style)
        SAVED.clear()
        SAVED.update(saved)
