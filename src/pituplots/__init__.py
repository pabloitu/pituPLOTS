from pituplots.basemaps import add_basemap, get_basemap, register_basemap
from pituplots.catalogs import highlight, plot_catalog, size_by_value
from pituplots.composite import compare_grids, panels, strip_panels
from pituplots.elements import (add_class_legend, add_colorbar, add_labels, add_locator, add_panel_label,
                                add_scalebar, add_size_legend)
from pituplots.errors import ConsistencyError, MissingDependency, PituplotsWarning
from pituplots.grids import plot_field, plot_grid, plot_sites
from pituplots.io.containers import Catalog, Grid, Profile, Sites, unwrap_lon
from pituplots.maps import add_gridlines, extent_from, make_map
from pituplots.sections import add_profile_map, add_slab, make_profile, plot_section, profiles_from_line, project
from pituplots.style.cmaps import alpha_cmap, class_breaks, discrete_cmap
from pituplots.style.axes import darkgrid
from pituplots.style.themes import theme, use_theme
from pituplots.vectors import plot_lines, plot_points, plot_polygons, plot_region, plot_sources

__version__ = "0.1.0.dev0"

__all__ = [
    "Catalog",
    "ConsistencyError",
    "Grid",
    "MissingDependency",
    "PituplotsWarning",
    "Profile",
    "Sites",
    "add_basemap",
    "add_class_legend",
    "add_colorbar",
    "add_gridlines",
    "add_labels",
    "add_locator",
    "add_panel_label",
    "add_profile_map",
    "add_scalebar",
    "add_size_legend",
    "add_slab",
    "alpha_cmap",
    "class_breaks",
    "compare_grids",
    "darkgrid",
    "discrete_cmap",
    "extent_from",
    "get_basemap",
    "highlight",
    "make_map",
    "make_profile",
    "panels",
    "plot_catalog",
    "plot_field",
    "plot_grid",
    "plot_lines",
    "plot_points",
    "plot_polygons",
    "plot_region",
    "plot_section",
    "plot_sites",
    "plot_sources",
    "profiles_from_line",
    "project",
    "register_basemap",
    "size_by_value",
    "strip_panels",
    "theme",
    "unwrap_lon",
    "use_theme",
]
