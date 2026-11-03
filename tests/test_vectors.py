import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colors import PowerNorm
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, box

from pituplots import (Grid, PituplotsWarning, Sites, make_map, plot_lines, plot_points, plot_polygons, plot_region,
                       plot_sources,
                       theme)
from pituplots.io import read_vector
from pituplots.maps import central_lon

MAP = {"coastline": False, "gridlines": False}
SQ = box(0, 0, 1, 1)


def plain():
    return plt.subplots()[1]


def lines(ax):
    return [c for c in ax.collections if isinstance(c, LineCollection)][-1]


def polys(ax):
    return [c for c in ax.collections if isinstance(c, PathCollection)][-1]


def grid(mask=None, nx=4, ny=3, dx=1.0, dy=1.0, x0=0.0):
    x, y = x0 + dx * np.arange(nx + 1), dy * np.arange(ny + 1)
    return Grid(np.ones((ny, nx)), x, y, mask=mask)


def nseg(ax):
    return len(lines(ax).get_segments())


def span(ax):
    a = np.vstack(lines(ax).get_segments())
    return a[:, 0].min(), a[:, 0].max(), a[:, 1].min(), a[:, 1].max()


# plot_region: grid structure


def test_region_returns_ax_on_new_and_given():
    ax = plot_region(SQ, **MAP)
    assert hasattr(ax, "projection")
    a = plain()
    assert plot_region(SQ, ax=a) is a


def test_region_one_cell():
    assert span(plot_region(grid(nx=1, ny=1), **MAP)) == (0, 1, 0, 1)


def test_region_one_row_and_one_column():
    assert span(plot_region(grid(nx=5, ny=1), **MAP)) == (0, 5, 0, 1)
    assert span(plot_region(grid(nx=1, ny=5), **MAP)) == (0, 1, 0, 5)


def test_region_gaps_give_disjoint_parts():
    m = np.array([[False, True, False, True]])
    assert nseg(plot_region(grid(m, nx=4, ny=1), **MAP)) == 2


def test_region_hole():
    m = np.zeros((3, 3), bool)
    m[1, 1] = True
    assert nseg(plot_region(grid(m, nx=3, ny=3), **MAP)) == 2


def test_region_irregular_spacing():
    assert span(plot_region(grid(dx=0.1, dy=0.5), **MAP)) == pytest.approx((0, 0.4, 0, 1.5))


def test_region_all_masked_is_empty():
    ax = plot_region(grid(np.ones((3, 4), bool)), extent=[0, 1, 0, 1], **MAP)
    assert nseg(ax) == 0


def test_region_float_lattice():
    x = np.arange(4) * 0.1
    assert nseg(plot_region(Grid(np.ones((1, 3)), x, [0, 0.1]), **MAP)) == 1


def test_region_from_points_is_contiguous_over_the_dateline():
    g = Grid.from_points([179.95, -179.95], [0.05, 0.05], [1, 1], dh=0.1)
    ax = plot_region(g, **MAP)
    assert nseg(ax) == 1 and central_lon(ax.projection) == 180


def test_region_global():
    g = Grid(np.ones((2, 4)), [-180, -90, 0, 90, 180], [-90, 0, 90])
    ax = plot_region(g, set_global=True, **MAP)
    x0, x1, _, _ = span(ax)
    assert x1 - x0 == 360


def test_region_polygon_holes_and_parts():
    p = box(0, 0, 3, 3).difference(box(1, 1, 2, 2))
    assert nseg(plot_region(p, **MAP)) == 2
    assert nseg(plot_region(MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)]), **MAP)) == 2


def test_region_array_closes_the_ring(data):
    a = np.load(data / "regions/iceland_region.npy")
    assert nseg(plot_region(a, **MAP)) == 1
    s = lines(plot_region(np.array([[0, 0], [1, 0], [1, 1.0]]), **MAP)).get_segments()[0]
    assert (s[0] == s[-1]).all()


def test_region_shape_on_a_map_is_kept_as_is(data):
    g, _, crs = read_vector(data / "new_zealand/nz_testing.shp")
    ax = plot_region(g[0], crs=crs, **MAP)
    x0, x1, y0, y1 = ax.get_extent(ccrs.PlateCarree(central_longitude=central_lon(ax.projection)))
    assert x1 - x0 < 30 and y0 < -35


def test_region_plain_axes_as_given():
    ax = plot_region(SQ, ax=plain(), crs="EPSG:2193")
    assert span(ax) == (0, 1, 0, 1)


def test_region_bad_input():
    with pytest.raises(TypeError):
        plot_region("x")
    with pytest.raises(ValueError, match="n, 2"):
        plot_region(np.zeros(3))


def test_region_map_keywords_with_ax_raise():
    with pytest.raises(ValueError, match="ignored"):
        plot_region(SQ, ax=make_map(**MAP), basemap="x")


def test_region_style_layers():
    assert lines(plot_region(SQ, **MAP)).get_linewidth()[0] == 1
    with theme("paper"):
        assert lines(plot_region(SQ, **MAP)).get_linewidth()[0] == 0.5
        assert lines(plot_region(SQ, lw=3, **MAP)).get_linewidth()[0] == 3


# dateline and CRS


@pytest.mark.parametrize("line", [[(170, -40), (-170, -40)], [(170, -40), (190, -40)], [(-190, -40), (-170, -40)]])
def test_line_crossing_is_one_piece_centred_on_180(line):
    ax = plot_lines(LineString(line), **MAP)
    assert central_lon(ax.projection) == 180 and nseg(ax) == 1
    assert span(ax)[:2] == pytest.approx((-10, 10))


def test_line_crossing_on_a_zero_centred_map_is_cut():
    ax = make_map(extent=[-180, 180, -90, 90], **MAP)
    plot_lines(LineString([(170, -40), (-170, -40)]), ax=ax)
    s = sorted(lines(ax).get_segments(), key=lambda a: a[0, 0])
    assert len(s) == 2
    assert sorted(np.abs(np.vstack(s)[:, 0])) == pytest.approx([170, 170, 180, 180])


def test_polygon_crossing_on_a_zero_centred_map_is_cut():
    ax = make_map(extent=[-180, 180, -90, 90], **MAP)
    plot_polygons(Polygon([(170, -40), (-170, -40), (-170, -30), (170, -30)]), ax=ax, fill="red")
    assert len(polys(ax).get_paths()) == 2


def test_polygon_crossing_fills_the_right_side():
    ax = make_map(extent=[160, -160, -50, -20], **MAP)
    plot_polygons(Polygon([(170, -40), (-170, -40), (-170, -30), (170, -30)]), ax=ax, fill="red")
    b = polys(ax).get_paths()[0].get_extents()
    assert (b.x0, b.x1) == pytest.approx((-10, 10))


def test_geometries_on_both_sides_share_an_extent():
    ax = plot_lines([LineString([(178, -40), (179, -39)]), LineString([(-179, -40), (-178, -39)])], **MAP)
    x0, x1, _, _ = ax.get_extent(ccrs.PlateCarree(central_longitude=180))
    assert x1 - x0 < 10


def test_lines_in_epsg2193_land_in_degrees(data):
    r = read_vector(data / "new_zealand/nz_cfm.shp")
    assert r[2] == "EPSG:2193"
    ax = plot_lines(r, **MAP)
    x0, x1, y0, y1 = span(ax)
    c = central_lon(ax.projection)
    assert 160 < x0 + c and x1 + c < 185 and -50 < y0 < y1 < -33


def test_lines_on_a_2193_map(data):
    r = read_vector(data / "new_zealand/nz_cfm.shp")
    ax = plot_lines(r, projection=ccrs.epsg(2193), extent=[166, 179, -48, -34], **MAP)
    ax.figure.canvas.draw()
    p = lines(ax).get_transform().transform_path(lines(ax).get_paths()[0])
    assert ax.bbox.contains(*p.vertices[0])


def test_crs_argument_applies_to_bare_geometries():
    ax = plot_lines(LineString([(1500000, 5400000), (1600000, 5500000)]), crs="EPSG:2193", **MAP)
    assert 160 < span(ax)[0] + central_lon(ax.projection) < 185


@pytest.mark.parametrize("proj", [ccrs.Mercator(), ccrs.PlateCarree(central_longitude=180),
                                  ccrs.LambertConformal(central_longitude=175, central_latitude=-40,
                                                        standard_parallels=(-30, -50))])
def test_vectors_draw_on_any_projection(proj):
    ax = make_map(extent=[170, 179, -45, -35], projection=proj, **MAP)
    plot_lines(LineString([(171, -40), (178, -38)]), ax=ax)
    plot_polygons(box(172, -42, 175, -40), ax=ax, fill="red")
    ax.figure.canvas.draw()


# plot_lines


def test_lines_inputs():
    a = np.array([[0.0, 0], [1, 1]])
    for d in (a, [a, a * 2], LineString(a), MultiLineString([a, a * 2]), [LineString(a), None, a]):
        assert isinstance(plot_lines(d, **MAP), object)
    assert nseg(plot_lines(MultiLineString([a, a * 2]), **MAP)) == 2


def test_lines_read_vector_and_values(data):
    r = read_vector(data / "new_zealand/nz_cfm.shp")
    ax = plot_lines(r, values="SR_pref", clabel="mm/yr", **MAP)
    assert lines(ax).get_array() is not None and len(ax.child_axes) == 1
    with pytest.raises(ValueError, match="no attribute"):
        plot_lines(r, values="x", **MAP)
    with pytest.raises(ValueError, match="no attribute"):
        plot_lines(LineString([(0, 0), (1, 1)]), values="x", **MAP)


def test_lines_duck_typed_geodataframe():
    class G(pd.DataFrame):
        crs = "EPSG:4326"

    d = G({"geometry": [LineString([(0, 0), (1, 1)]), None], "v": [1.0, 2.0]})
    assert nseg(plot_lines(d, values="v", **MAP)) == 1


def test_lines_values_skip_null_geometries():
    ls = [LineString([(0, 0), (1, 1)]), None, LineString([(1, 0), (2, 1)])]
    assert sorted(lines(plot_lines(ls, values=[1, 2, 3], **MAP)).get_array()) == [1, 3]
    with pytest.raises(ValueError, match="values"):
        plot_lines(ls, values=[1, 2], **MAP)


def test_lines_nonlinear_norm_and_width():
    ls = [LineString([(0, i), (1, i)]) for i in range(3)]
    n = PowerNorm(0.5, vmin=0, vmax=4)
    c = lines(plot_lines(ls, values=[4, 0, 1], norm=n, lw_by=(1, 3), **MAP))
    assert c.norm is n
    assert list(c.get_linewidth()) == pytest.approx([1, 2, 3])
    assert list(c.get_array()) == [0, 1, 4]


def test_lines_width_callable():
    ls = [LineString([(0, i), (1, i)]) for i in range(2)]
    c = lines(plot_lines(ls, values=[1, 2], lw_by=lambda v: v * 2, **MAP))
    assert list(c.get_linewidth()) == [2, 4]


def test_lines_largest_on_top_and_multipart_values():
    a = [[0.0, 0], [1, 1]]
    c = lines(plot_lines([MultiLineString([a, [[2, 2], [3, 3]]]), LineString([(0, 1), (1, 2)])], values=[5, 1], **MAP))
    assert list(c.get_array()) == [1, 5, 5]


def test_lines_nan_values():
    ls = [LineString([(0, 0), (1, 1)]), LineString([(1, 0), (2, 1)])]
    c = lines(plot_lines(ls, values=[np.nan, 2], **MAP))
    assert np.ma.count(c.get_array()) == 1
    with pytest.warns(PituplotsWarning):
        plot_lines(ls, values=[np.nan, np.nan], **MAP)


def test_lines_constant_value():
    plot_lines([LineString([(0, 0), (1, 1)])], values=[2.0], lw_by=(1, 2), **MAP).figure.canvas.draw()


def test_lines_lw_by_needs_values():
    with pytest.raises(ValueError, match="lw_by"):
        plot_lines(LineString([(0, 0), (1, 1)]), lw_by=(1, 2), **MAP)


def test_lines_empty_and_bad():
    assert nseg(plot_lines([], extent=[0, 1, 0, 1], **MAP)) == 0
    with pytest.raises(TypeError):
        plot_lines(3)
    with pytest.raises(ValueError, match="n, 2"):
        plot_lines([np.zeros((1, 2))])


def test_lines_plain_axes_autoscale():
    ax = plot_lines(LineString([(0, 0), (10, 5)]), ax=plain())
    assert ax.get_xlim()[1] >= 10


def test_lines_inputs_untouched():
    a = np.array([[179.0, 0], [-179, 1]])
    b = a.copy()
    v = np.array([1.0])
    plot_lines([a], values=v, **MAP)
    assert (a == b).all() and v[0] == 1


# plot_polygons


def test_polygons_fill_edge_and_holes():
    p = box(0, 0, 3, 3).difference(box(1, 1, 2, 2))
    ax = plot_polygons(p, fill="red", edge="blue", lw=2, **MAP)
    c = polys(ax)
    assert tuple(c.get_facecolor()[0][:3]) == (1, 0, 0) and tuple(c.get_edgecolor()[0][:3]) == (0, 0, 1)
    assert c.get_linewidth()[0] == 2 and len(c.get_paths()) == 1
    assert len(c.get_paths()[0].to_polygons()) == 2


def test_polygons_hole_stays_empty():
    p = box(0, 0, 3, 3).difference(box(1, 1, 2, 2))
    ax = plot_polygons(p, ax=plain(), fill="red", edge="none")
    ax.set(xlim=(0, 3), ylim=(0, 3))
    ax.figure.canvas.draw()
    im = np.asarray(ax.figure.canvas.buffer_rgba())
    def at(x, y):
        px, py = ax.transData.transform((x, y))
        return im[im.shape[0] - int(py), int(px), :3]
    assert (at(1.5, 1.5) == 255).all() and at(0.5, 0.5)[1] == 0


def test_polygons_default_no_fill():
    assert polys(plot_polygons(SQ, **MAP)).get_facecolor().size == 0 or polys(plot_polygons(SQ, **MAP)).get_facecolor()[0][3] == 0


def test_polygons_read_vector_and_array(data):
    assert len(polys(plot_polygons(read_vector(data / "chile/chile_buffer.shp"), **MAP)).get_paths()) >= 1
    assert len(polys(plot_polygons([np.array([[0.0, 0], [1, 0], [1, 1]])], **MAP)).get_paths()) == 1


def test_polygons_reject_lines():
    with pytest.raises(ValueError, match="polygons"):
        plot_polygons(LineString([(0, 0), (1, 1)]), **MAP)


# plot_points


def test_points_inputs():
    for d in ((np.array([1.0, 2]), np.array([3.0, 4])), Sites([1, 2], [3, 4]),
              pd.DataFrame({"lon": [1, 2], "lat": [3, 4]}), ([1, 2], [3, 4], [5, 6])):
        ax = plot_points(d, **MAP)
        assert ax.collections[-1].get_offsets().shape == (2, 2)
    with pytest.raises(TypeError):
        plot_points(3)


def test_points_default_style_and_override():
    c = plot_points(([1, 2], [3, 4]), **MAP).collections[-1]
    assert c.get_sizes()[0] == 25 and c.get_zorder() == 7
    c = plot_points(([1, 2], [3, 4]), color="red", size=9, **MAP).collections[-1]
    assert c.get_sizes()[0] == 9


def test_points_labels():
    ax = plot_points(([1, 2], [3, 4]), labels=["a", "b"], **MAP)
    assert sorted(t.get_text() for t in ax.texts) == ["a", "b"]
    ax = plot_points(Sites([1], [2], names=["x"]), labels=True, **MAP)
    assert ax.texts[0].get_text() == "x"
    with pytest.raises(ValueError, match="names"):
        plot_points(([1], [2]), labels=True, **MAP)


def test_points_on_the_seam():
    ax = plot_points(([179, -179, 181], [0, 1, 2]), **MAP)
    assert central_lon(ax.projection) == 180
    assert np.ptp(ax.collections[-1].get_offsets()[:, 0]) == pytest.approx(2)


def test_points_empty_and_plain():
    plot_points(([], []), extent=[0, 1, 0, 1], **MAP)
    a = plain()
    assert plot_points(([1], [2]), ax=a) is a


# plot_sources


def sources(tmp_path):
    from test_oq import NRML, edges

    from pituplots.io import read_sources

    p = tmp_path / "m.xml"
    p.write_text(NRML.format(**edges(False)))
    return read_sources(p)


def test_sources_draw_every_kind(tmp_path):
    ax = plot_sources(sources(tmp_path), **MAP)
    lc = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert len(lc) == 2 and len(lc[0].get_segments()) == 3 and len(lc[1].get_segments()) == 1
    assert lc[1].get_linestyle()[0][1] is not None
    assert ax.collections[-1].get_offsets().shape == (1, 2)


def test_sources_shared_colour_scale(tmp_path):
    ax = plot_sources(sources(tmp_path), color_by="rate", log=True, clabel="rate", **MAP)
    lc = [c for c in ax.collections if isinstance(c, LineCollection)]
    pts = ax.collections[-1]
    assert lc[0].norm is pts.norm and len(ax.child_axes) == 1
    assert pts.get_array()[0] == pytest.approx(np.log10(0.15))


def test_sources_fixture_on_the_dateline_free_map(data):
    from pituplots.io import read_sources

    s = pd.concat([read_sources(data / "nrml/simple.xml"), read_sources(data / "nrml/complex.xml")],
                  ignore_index=True)
    ax = plot_sources(s, color_by="mmax", **MAP)
    ax.figure.canvas.draw()


def test_sources_bad(tmp_path):
    with pytest.raises(TypeError):
        plot_sources(pd.DataFrame({"a": [1]}))
    with pytest.raises(ValueError, match="column"):
        plot_sources(sources(tmp_path), color_by="x", **MAP)


def test_points_label_style():
    ax = plot_points(([1, 2], [3, 4]), labels=["a", "b"], label_kw={"color": "red", "halo": 0}, **MAP)
    assert all(t.get_color() == "red" and not t.get_path_effects() for t in ax.texts)


def test_lines_width_by_value_with_a_fixed_colour():
    ls = [LineString([(0, i), (1, i)]) for i in range(3)]
    c = lines(plot_lines(ls, values=[1, 2, 3], lw_by=(1, 3), color="black", **MAP))
    assert c.get_array() is None and sorted(c.get_linewidth()) == [1, 2, 3]


def test_polygons_filled_by_value():
    ps = [box(0, 0, 1, 1), None, box(2, 0, 3, 1)]
    ax = plot_polygons(ps, values=[1.0, 5.0, 3.0], cmap="viridis", clabel="v", **MAP)
    assert sorted(polys(ax).get_array()) == [1, 3] and len(ax.child_axes) == 1
    with pytest.raises(ValueError, match="values"):
        plot_polygons(ps, values=[1.0], **MAP)
