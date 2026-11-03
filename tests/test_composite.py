import cartopy.crs as ccrs
from cartopy.mpl.gridliner import Gridliner
import matplotlib.pyplot as plt
import numpy as np
import pytest
from pyproj import Geod

from pituplots import Grid, compare_grids, make_map, panels, plot_catalog, plot_grid, strip_panels, theme
from pituplots.maps import central_lon

MAP = {"coastline": False, "gridlines": False}
EXT = [0, 10, 0, 10]


def grid(v=None, n=4):
    v = np.arange(1.0, n * n + 1).reshape(n, n) if v is None else v
    return Grid(np.asarray(v, float), np.arange(n + 1.0), np.arange(n + 1.0))


def gridliner(ax):
    return next(a for a in ax.get_children() if isinstance(a, Gridliner))


# make_map is unchanged by the split


def test_make_map_behaviour_kept():
    ax = make_map(extent=[170, -170, -50, -30], **MAP)
    assert central_lon(ax.projection) == 180
    with pytest.raises(ValueError, match="native"):
        make_map(projection="native")
    with pytest.raises(ValueError, match="approx"):
        make_map(projection="approx")
    with pytest.raises(ValueError, match="unknown"):
        make_map(projection="x")


# panels


def test_panels_shape_and_figure():
    fig, axs = panels(2, 3, extent=EXT, **MAP)
    assert axs.shape == (6,) and all(a.figure is fig for a in axs)
    assert all(hasattr(a, "projection") for a in axs)
    assert fig.get_layout_engine() is not None


def test_panels_single():
    fig, axs = panels(extent=EXT, **MAP)
    assert axs.shape == (1,)


def test_panels_share_extent_and_projection():
    _, axs = panels(1, 2, extent=[170, -170, -50, -30], **MAP)
    assert all(central_lon(a.projection) == 180 for a in axs)
    assert np.allclose(axs[0].get_extent(), axs[1].get_extent())


def test_panels_gridline_labels_only_outside():
    _, axs = panels(2, 2, extent=EXT, coastline=False)
    g = [gridliner(a) for a in axs]
    assert [x.left_labels for x in g] == [True, False, True, False]
    assert [x.bottom_labels for x in g] == [False, False, True, True]


def test_panels_layers_compose():
    _, axs = panels(1, 2, extent=EXT, **MAP)
    g = grid()
    for a in axs:
        assert plot_grid(g, ax=a, colorbar=False) is a


def test_panels_bad():
    with pytest.raises(ValueError):
        panels(0, 1)
    with pytest.raises(TypeError, match="unknown"):
        panels(1, 1, wrong=1)


def test_panels_with_a_theme_figsize():
    with theme("paper"):
        fig, _ = panels(extent=EXT, **MAP)
    assert fig.get_size_inches() == pytest.approx((6.5, 4))


# strip_panels


def test_strips_extents_and_equal_km():
    fig, axs = strip_panels([-17.5, -30, -43, -56], [-70, -72, -71], 850, **MAP)
    assert axs.shape == (3,)
    g = Geod(ellps="WGS84")
    for a in axs:
        x0, x1, y0, y1 = a.get_extent(ccrs.PlateCarree())
        d = g.inv(x0, (y0 + y1) / 2, x1, (y0 + y1) / 2)[2] / 1e3
        assert d == pytest.approx(850, rel=0.01)


def test_strips_same_shape():
    fig, axs = strip_panels(np.linspace(-17.5, -56, 4), [-70, -72, -71], 850, figsize=(9, 5), **MAP)
    fig.canvas.draw()
    s = [(p.width, p.height) for p in (a.get_position() for a in axs)]
    assert np.ptp([w for w, h in s]) < 0.01 and np.ptp([h for w, h in s]) < 0.01


def test_strips_ascending_edges_and_dateline():
    _, axs = strip_panels([-50, -40, -30], [179, 180], 400, **MAP)
    assert central_lon(axs[1].projection) == 180


def test_strips_gridline_labels():
    _, axs = strip_panels([-20, -30, -40], [-70, -71], 500, coastline=False)
    assert [gridliner(a).left_labels for a in axs] == [True, False]


@pytest.mark.parametrize("a", [([-20], [], 100), ([-20, -30], [1, 2], 100), ([-20, -30, -25], [1, 2], 100),
                               ([-20, -30], [1], 0)])
def test_strips_bad(a):
    with pytest.raises(ValueError):
        strip_panels(*a, **MAP)


def test_strips_keywords_they_set():
    with pytest.raises(ValueError, match="extent"):
        strip_panels([-20, -30], [1], 100, extent=EXT)


# compare_grids


def test_compare_returns_three_axes():
    fig, axs = compare_grids(grid(), grid(), coastline=False, gridlines=False)
    assert axs.shape == (3,) and all(a.figure is fig for a in axs)


def test_compare_pct_values_and_symmetry():
    a, b = grid(np.full((4, 4), 2.0)), grid(np.full((4, 4), 1.0))
    a.values[0, 0] = 0.5
    _, axs = compare_grids(a, b, kind="pct", clip_pct=100, **MAP)
    m = axs[2].collections[-1]
    assert m.get_array().max() == 100 and m.get_array().min() == -50
    assert m.norm.vmin == -100 and m.norm.vmax == 100


def test_compare_diff():
    _, axs = compare_grids(grid(np.full((4, 4), 3.0)), grid(np.full((4, 4), 1.0)), kind="diff", **MAP)
    assert axs[2].collections[-1].get_array().max() == 2


def test_compare_ratio_log_scale_centred_on_one():
    a = grid(np.full((4, 4), 10.0))
    _, axs = compare_grids(a, grid(np.ones((4, 4))), kind="ratio", **MAP)
    n = axs[2].collections[-1].norm
    assert n.vmin == pytest.approx(0.1) and n.vmax == pytest.approx(10)


def test_compare_shared_scale():
    _, axs = compare_grids(grid(), grid(np.ones((4, 4)) * 100), **MAP)
    a, b = axs[0].collections[-1].norm, axs[1].collections[-1].norm
    assert (a.vmin, a.vmax) == (b.vmin, b.vmax) == (1, 100)


def test_compare_clip_percentile():
    a = np.ones((4, 4))
    a[0, 0] = 1000
    _, axs = compare_grids(grid(a), grid(np.ones((4, 4))), clip_pct=90, **MAP)
    assert axs[2].collections[-1].norm.vmax < 1e4


def test_compare_floor_and_zero_denominator():
    b = np.ones((4, 4))
    b[0, 0] = 0
    b[0, 1] = 0.01
    _, axs = compare_grids(grid(np.ones((4, 4))), grid(b), floor=0.1, **MAP)
    z = axs[2].collections[-1].get_array()
    assert np.ma.count_masked(z) == 2 and np.isfinite(z.compressed()).all()


def test_compare_masks_and_nan_are_joined():
    a, b = np.ones((4, 4)), np.ones((4, 4))
    a[1, 1] = np.nan
    gb = Grid(b, np.arange(5.0), np.arange(5.0), mask=np.eye(4, dtype=bool)[::-1])
    _, axs = compare_grids(grid(a), gb, **MAP)
    assert np.ma.count_masked(axs[2].collections[-1].get_array()) == 5


def test_compare_identical_is_a_flat_zero():
    _, axs = compare_grids(grid(), grid(), **MAP)
    n = axs[2].collections[-1].norm
    assert n.vmin < 0 < n.vmax


def test_compare_log_panels_keep_a_raw_difference():
    a, b = grid(np.full((4, 4), 100.0)), grid(np.full((4, 4), 10.0))
    _, axs = compare_grids(a, b, log=True, kind="diff", **MAP)
    assert axs[0].collections[-1].get_array().max() == 2
    assert axs[2].collections[-1].get_array().max() == 90


@pytest.mark.parametrize("b", [
    Grid(np.ones((4, 3)), np.arange(4.0), np.arange(5.0)),
    Grid(np.ones((4, 4)), np.arange(5.0) + 0.5, np.arange(5.0)),
    Grid(np.ones((4, 4)), np.arange(5.0), np.arange(5.0) * 2),
    Grid(np.ones((4, 4)), np.arange(5.0), np.arange(5.0), crs="EPSG:2193"),
], ids=["size", "shift", "spacing", "crs"])
def test_compare_different_lattices_raise(b):
    with pytest.raises(ValueError, match="lattices"):
        compare_grids(grid(), b, **MAP)


def test_compare_float_noise_is_the_same_lattice():
    b = Grid(np.ones((4, 4)), np.arange(5.0) + 1e-12, np.arange(5.0))
    compare_grids(grid(), b, **MAP)


def test_compare_bad_kind_and_bands():
    with pytest.raises(ValueError, match="kind"):
        compare_grids(grid(), grid(), kind="x")


def test_compare_inputs_untouched():
    a, b = grid(), grid()
    x = a.values.copy()
    compare_grids(a, b, **MAP)
    assert (a.values == x).all()


def test_compare_titles_labels_and_extent():
    _, axs = compare_grids(grid(), grid(), titles=("x", "y", "z"), extent=[0, 4, 0, 4], **MAP)
    assert [a.get_title() for a in axs] == ["x", "y", "z"]
    assert np.allclose(axs[0].get_extent(ccrs.PlateCarree()), [0, 4, 0, 4])


def test_compare_layout_does_not_break_save(tmp_path):
    fig, _ = compare_grids(grid(), grid(), clabel="a", dlabel="b", **MAP)
    fig.savefig(tmp_path / "a.png")
    plt.close(fig)


def test_only_composite_returns_a_tuple():
    assert not isinstance(plot_catalog(([1], [1], [3]), **MAP), tuple)


def test_compare_shared_bar_below_the_pair():
    fig, axs = compare_grids(grid(), grid(), **MAP)
    fig.canvas.draw()
    bar = [a for a in fig.axes if a not in axs and a not in axs[2].child_axes][0]
    assert bar.get_position().y1 <= min(a.get_position().y0 for a in axs[:2]) + 1e-6
