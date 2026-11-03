API
===

Containers
----------

.. autoclass:: pituplots.Catalog
   :special-members: __getitem__
.. autoclass:: pituplots.Grid
   :members: from_points, contains
.. autoclass:: pituplots.Sites
.. autoclass:: pituplots.Profile

Maps and basemaps
-----------------

.. autofunction:: pituplots.make_map
.. autofunction:: pituplots.extent_from
.. autofunction:: pituplots.unwrap_lon
.. autofunction:: pituplots.add_gridlines
.. autofunction:: pituplots.get_basemap
.. autofunction:: pituplots.add_basemap
.. autofunction:: pituplots.register_basemap

Catalogs, grids and vectors
---------------------------

.. autofunction:: pituplots.plot_catalog
.. autofunction:: pituplots.size_by_value
.. autofunction:: pituplots.highlight
.. autofunction:: pituplots.plot_grid
.. autofunction:: pituplots.plot_field
.. autofunction:: pituplots.plot_sites
.. autofunction:: pituplots.plot_region
.. autofunction:: pituplots.plot_points
.. autofunction:: pituplots.plot_lines
.. autofunction:: pituplots.plot_polygons
.. autofunction:: pituplots.plot_sources

Sections
--------

.. autofunction:: pituplots.make_profile
.. autofunction:: pituplots.profiles_from_line
.. autofunction:: pituplots.project
.. autofunction:: pituplots.plot_section
.. autofunction:: pituplots.add_slab
.. autofunction:: pituplots.add_profile_map

Elements and multi-panel figures
--------------------------------

.. autofunction:: pituplots.add_colorbar
.. autofunction:: pituplots.add_size_legend
.. autofunction:: pituplots.add_locator
.. autofunction:: pituplots.add_scalebar
.. autofunction:: pituplots.add_labels
.. autofunction:: pituplots.add_panel_label
.. autofunction:: pituplots.add_class_legend
.. autofunction:: pituplots.panels
.. autofunction:: pituplots.strip_panels
.. autofunction:: pituplots.compare_grids

Statistics
----------

.. autofunction:: pituplots.stats.plot_fmd
.. autofunction:: pituplots.stats.b_value
.. autofunction:: pituplots.stats.mc_maxc
.. autofunction:: pituplots.stats.plot_b_stability
.. autofunction:: pituplots.stats.plot_magnitude_time
.. autofunction:: pituplots.stats.plot_cumulative
.. autofunction:: pituplots.stats.plot_curves
.. autofunction:: pituplots.stats.weichert
.. autofunction:: pituplots.stats.gr_model
.. autofunction:: pituplots.stats.plot_mfd

Style
-----

.. autofunction:: pituplots.use_theme
.. autofunction:: pituplots.theme
.. autofunction:: pituplots.style.themes.load_theme
.. autofunction:: pituplots.darkgrid
.. autofunction:: pituplots.alpha_cmap
.. autofunction:: pituplots.discrete_cmap
.. autofunction:: pituplots.class_breaks

Readers
-------

.. autofunction:: pituplots.io.read_catalog
.. autofunction:: pituplots.io.read_csep_grid
.. autofunction:: pituplots.io.read_csep_nodes
.. autofunction:: pituplots.io.read_rate_csv
.. autofunction:: pituplots.io.read_geotiff
.. autofunction:: pituplots.io.read_slab2
.. autofunction:: pituplots.io.read_vector
.. autofunction:: pituplots.io.cells_to_outline
.. autofunction:: pituplots.io.read_datastore
.. autoclass:: pituplots.io.oq.OQResults
   :members: curves, mean, quantile, maps, to_window, check, branches
.. autofunction:: pituplots.io.read_csv
.. autofunction:: pituplots.io.read_sources

pyCSEP
------

.. autofunction:: pituplots.adapters.csep.from_csep_catalog
.. autofunction:: pituplots.adapters.csep.from_csep_region
.. autofunction:: pituplots.adapters.csep.from_csep_forecast
.. autofunction:: pituplots.adapters.csep.to_csep_catalog
.. autofunction:: pituplots.adapters.csep.to_csep_forecast

Errors
------

.. autoclass:: pituplots.MissingDependency
.. autoclass:: pituplots.ConsistencyError
.. autoclass:: pituplots.PituplotsWarning
