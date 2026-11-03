import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import font_manager

import pituplots
from pituplots import Catalog, Grid, PituplotsWarning, add_gridlines, make_map, plot_catalog, plot_grid, theme, use_theme
from pituplots.style import defaults
from pituplots.style.themes import NAMES, load_theme

THEMES = {n: load_theme(n) for n in NAMES}
RC = load_theme("base")["rc"]

MAP = {"extent": [0, 10, 0, 10], "coastline": False, "gridlines": False}


@pytest.fixture(autouse=True)
def restore():
    from pituplots.style import themes

    t, saved = dict(defaults.THEME), dict(themes.SAVED)
    with matplotlib.rc_context():
        yield
    defaults.THEME.clear()
    defaults.THEME.update(t)
    themes.SAVED.clear()
    themes.SAVED.update(saved)


@pytest.fixture
def no_ubuntu(monkeypatch):
    fm = font_manager.fontManager
    monkeypatch.setattr(fm, "ttflist", [f for f in fm.ttflist if f.name != "Ubuntu"])
    monkeypatch.setattr(font_manager, "findSystemFonts", lambda *a, **k: [])


@pytest.fixture
def ubuntu(monkeypatch):
    fm = font_manager.fontManager
    if not any(f.name == "Ubuntu" for f in fm.ttflist):
        monkeypatch.setattr(fm, "ttflist", [*fm.ttflist, font_manager.FontEntry(name="Ubuntu")])


def state():
    return dict(matplotlib.rcParams), dict(defaults.THEME)


# rcParams


@pytest.mark.parametrize("name", NAMES)
def test_use_theme_sets_rc_and_family_layer(name):
    use_theme(name)
    rc = matplotlib.rcParams
    for k, v in THEMES[name]["rc"].items():
        assert np.all(rc[k] == (list(v) if isinstance(v, tuple) else v)), k
    assert defaults.THEME == THEMES[name]["style"]


@pytest.mark.parametrize("name", NAMES)
def test_every_rc_key_is_valid(name):
    assert all(k in matplotlib.rcParams for k in {**RC, **THEMES[name]["rc"]})


@pytest.mark.parametrize("name", NAMES)
def test_theme_restores_on_exit(name):
    before = state()
    with theme(name):
        assert state() != before
    assert state() == before


def test_theme_restores_after_error():
    before = state()
    with pytest.raises(KeyError), theme("paper"):
        raise KeyError
    assert state() == before


def test_themes_nest():
    with theme("paper"):
        inner = state()
        with theme("slides"):
            assert matplotlib.rcParams["font.size"] == 12
        assert state() == inner


def test_use_theme_inside_theme_is_restored():
    before = state()
    with theme("paper"):
        use_theme("slides")
    assert state() == before


def test_theme_keeps_a_layer_set_before():
    use_theme("slides")
    before = state()
    with theme("paper"):
        assert "coast" in defaults.THEME
    assert state() == before


@pytest.mark.parametrize("a", NAMES)
@pytest.mark.parametrize("b", NAMES)
def test_switching_leaves_nothing_behind(a, b):
    use_theme(b)
    direct = state()
    use_theme(a)
    use_theme(b)
    assert state() == direct


def test_unknown_theme_changes_nothing():
    before = state()
    with pytest.raises(ValueError, match="paper"):
        use_theme("dinosaur")
    with pytest.raises(ValueError, match="paper"), theme("dinosaur"):
        pass
    assert state() == before


def test_import_does_not_fill_the_layer():
    assert defaults.THEME == {}


def test_plot_functions_keep_a_theme_rc():
    use_theme("paper")
    before = state()
    ax = plot_grid(Grid(np.ones((2, 2)), [0, 1, 2], [0, 1, 2]), **MAP)
    plot_catalog(Catalog([1], [1], mag=[3]), ax=ax)
    ax.figure.canvas.draw()
    assert state() == before


# layering, through a real theme


@pytest.mark.parametrize("family, key, base", [("coast", "lw", 1.5), ("borders", "lw", 0.5), ("region", "lw", 1)])
def test_layering_with_paper(family, key, base):
    assert defaults.resolve(family)[key] == base
    with theme("paper"):
        t = THEMES["paper"]["style"][family][key]
        assert t != base and defaults.resolve(family)[key] == t
        assert defaults.resolve(family, **{key: 9})[key] == 9
        assert defaults.resolve(family)["color"] == defaults.DEFAULTS[family]["color"]
    assert defaults.resolve(family)[key] == base


def test_layering_reaches_gridlines():
    assert add_gridlines(make_map(**MAP)).xlabel_style == {}
    with theme("paper"):
        ax = make_map(**MAP)
        assert add_gridlines(ax).xlabel_style["fontsize"] == 5.5
        assert add_gridlines(ax, fontsize=20).xlabel_style["fontsize"] == 20
        assert add_gridlines(ax).ylabel_style["fontsize"] == 5.5


def test_gridlines_keep_their_defaults():
    gl = add_gridlines(make_map(**MAP), alpha=0.1)
    assert gl.collection_kwargs["alpha"] == 0.1
    gl = add_gridlines(make_map(**MAP))
    assert gl.collection_kwargs["alpha"] == 0.5


def test_layering_reaches_the_coast(data):
    pc = {"extent": [-12, 32, 34, 60], "gridlines": False}
    lw = lambda ax: [c._kwargs["linewidth"] for c in ax._children if hasattr(c, "_kwargs")]  # noqa: E731
    assert lw(make_map(**pc)) == [1.5]
    with theme("paper"):
        assert lw(make_map(**pc)) == [0.5]


def test_paper_markers_fit_the_legend():
    c = Catalog([1, 2], [2, 3], mag=[3, 8])
    with theme("paper"):
        ax = plot_catalog(c, coastline=False, gridlines=False)
        assert ax.collections[-1].get_sizes().max() == 90
        fs = matplotlib.rcParams["legend.fontsize"]
        pad = defaults.resolve("size_legend")["labelspacing"]
    assert np.sqrt(90) <= fs * (1 + pad)


def test_map_legend_keeps_a_frame_on_slides():
    with theme("slides"):
        ax = plot_catalog(Catalog([1, 2], [2, 3], mag=[3, 8]), coastline=False, gridlines=False)
        assert ax.artists[-1].get_frame_on()


# text


@pytest.mark.parametrize("name", NAMES)
def test_size_gives_the_emphasis(name):
    r = THEMES[name]["rc"]
    assert r["axes.titlesize"] > r["axes.labelsize"] >= r["xtick.labelsize"] == r["ytick.labelsize"]
    assert r["axes.titlesize"] > r["font.size"]
    assert THEMES[name]["style"]["gridlines"]["fontsize"] <= r["xtick.labelsize"]


@pytest.mark.parametrize("name", NAMES)
def test_no_weight_emphasis(name):
    rc = {**RC, **THEMES[name]["rc"]}
    assert {v for k, v in rc.items() if k.endswith("weight")} == {"normal"}


def test_ubuntu_when_installed(ubuntu):
    with warnings.catch_warnings():
        warnings.simplefilter("error", PituplotsWarning)
        use_theme("paper")
    assert matplotlib.rcParams["font.family"] == ["Ubuntu", "DejaVu Sans"]


def test_fallback_when_missing(no_ubuntu):
    before = state()
    with pytest.warns(PituplotsWarning, match="DejaVu"), theme("report"):
        assert matplotlib.rcParams["font.family"] == ["DejaVu Sans"]
        assert matplotlib.rcParams["font.size"] == 12
        make_map(**MAP).set_title("x")
        plt.gcf().canvas.draw()
    assert state() == before


def test_font_missing_from_a_stale_cache_is_added(no_ubuntu, monkeypatch):
    added = []
    fm = font_manager.fontManager
    monkeypatch.setattr(font_manager, "findSystemFonts", lambda *a, **k: ["/x/Ubuntu-R.ttf", "/x/Arial.ttf"])

    def add(path):
        added.append(path)
        monkeypatch.setattr(fm, "ttflist", [*fm.ttflist, font_manager.FontEntry(name="Ubuntu")])

    monkeypatch.setattr(fm, "addfont", add)
    use_theme("paper")
    assert added == ["/x/Ubuntu-R.ttf"]
    assert matplotlib.rcParams["font.family"][0] == "Ubuntu"


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("fmt", ["pdf", "svg", "png"])
def test_saves_in_every_format(name, fmt, tmp_path):
    with warnings.catch_warnings(), theme(name):
        warnings.simplefilter("ignore", PituplotsWarning)
        ax = make_map(**MAP)
        add_gridlines(ax)
        ax.set_title(r"title $M_w$")
        ax.figure.savefig(tmp_path / f"a.{fmt}")
    assert (tmp_path / f"a.{fmt}").stat().st_size > 0


def test_public_names():
    assert pituplots.use_theme is use_theme and pituplots.theme is theme
    assert {"use_theme", "theme"} <= set(pituplots.__all__)


def test_map_frame_survives_open_spines():
    with theme("slides"):
        ax = make_map(**MAP)
        assert ax.spines["geo"].get_visible()
        assert not plt.figure().add_subplot().spines["top"].get_visible()


# YAML files


def test_user_yaml_extends_and_merges_families(tmp_path):
    p = tmp_path / "mine.yaml"
    p.write_text("extends: paper\nrc:\n  font.size: 9\n  axes.grid: true\nstyle:\n  coast: {color: red}\n")
    t = load_theme(p)
    assert t["rc"]["font.size"] == 9 and t["rc"]["axes.titlesize"] == 8
    assert t["style"]["coast"] == {"lw": 0.5, "color": "red"}
    with theme(p):
        assert matplotlib.rcParams["axes.grid"] and defaults.resolve("coast")["color"] == "red"


def test_user_yaml_keys_are_undone_by_the_next_theme(tmp_path):
    p = tmp_path / "mine.yaml"
    p.write_text("rc:\n  axes.grid: true\n  lines.linewidth: 4\n")
    before = matplotlib.rcParams["lines.linewidth"]
    use_theme(p)
    assert matplotlib.rcParams["lines.linewidth"] == 4
    use_theme("report")
    assert matplotlib.rcParams["lines.linewidth"] == before and not matplotlib.rcParams["axes.grid"]


def test_user_yaml_without_extends_starts_from_base(tmp_path):
    p = tmp_path / "a.yml"
    p.write_text("rc:\n  font.size: 9\n")
    assert load_theme(p)["rc"]["pdf.fonttype"] == 42


@pytest.mark.parametrize("text, match", [("foo: 1\n", "unknown keys"), ("rc:\n  not.a.key: 1\n", "rcParams"),
                                         ("extends: dinosaur\n", "unknown theme")])
def test_user_yaml_errors(tmp_path, text, match):
    p = tmp_path / "bad.yaml"
    p.write_text(text)
    with pytest.raises(ValueError, match=match):
        load_theme(p)


def test_user_yaml_missing():
    with pytest.raises(FileNotFoundError):
        use_theme("nope.yaml")


def test_packaged_themes_ship_with_the_package():
    import pituplots.style.themes as t

    assert {p.stem for p in t.HERE.glob("*.yaml")} == {"base", *NAMES}
