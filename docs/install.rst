Install
=======

pituPLOTS is installed from GitHub::

    pip install "pituPLOTS @ git+https://github.com/pabloitu/pituPLOTS.git"

Optional features are extras::

    pip install "pituPLOTS[raster,hdf5] @ git+https://github.com/pabloitu/pituPLOTS.git"

========== =================== =====================================
Extra      Adds                Enables
========== =================== =====================================
``raster`` rasterio            GeoTIFF basemaps and data
``hdf5``   h5py                OpenQuake datastore
``vector`` pyogrio             GPKG and other OGR formats
``csep``   pycsep              ``adapters.csep``
``oq``     openquake.engine    OpenQuake branch labels
``all``    all of the above
========== =================== =====================================
