import hashlib
import io
import warnings
from pathlib import Path
from urllib.request import Request, urlopen

import cartopy
import cartopy.crs as ccrs
import numpy as np
from cartopy.io.img_tiles import GoogleTiles
from PIL import Image

from pituplots.errors import PituplotsWarning, need
from pituplots.io.containers import unwrap_lon
from pituplots.io.rasters import pixel_window
from pituplots.maps import cartopy_crs, central_lon
from pituplots.style.defaults import ZORDER

ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/{}/MapServer/tile/{{z}}/{{y}}/{{x}}.jpg"
TILES = {
    "ESRI_terrain": ESRI.format("World_Terrain_Base"),
    "ESRI_imagery": ESRI.format("World_Imagery"),
    "ESRI_relief": ESRI.format("World_Shaded_Relief"),
    "ESRI_topo": ESRI.format("World_Topo_Map"),
}
REGISTRY = {}


class Tiles(GoogleTiles):
    """XYZ tiles cached per source. A failed fetch warns, and is not cached."""

    def __init__(self, url=None, style="satellite"):
        super().__init__(url=url) if url else super().__init__(style=style)
        key = hashlib.sha256(str(url or style).encode()).hexdigest()[:16]
        self.folder = Path(cartopy.config["cache_dir"]) / "pituplots" / key
        self.failed = False

    def get_image(self, tile):
        f = self.folder / ("_".join(map(str, tile)) + ".npy")
        if f.exists():
            img = np.load(f, allow_pickle=False)
        else:
            try:
                req = Request(self._image_url(tile), headers={"User-Agent": self.user_agent})
                with urlopen(req) as r:
                    raw = r.read()
            except OSError:
                self.failed = True
                raise
            img = np.array(Image.open(io.BytesIO(raw)).convert(self.desired_tile_form))
            self.folder.mkdir(parents=True, exist_ok=True)
            np.save(f, img, allow_pickle=False)
        return img, self.tileextent(tile), "lower"

    def image_for_domain(self, target_domain, target_z):
        self.failed = False
        try:
            out = super().image_for_domain(target_domain, target_z)
        except ValueError:
            b = target_domain.bounds
            out = np.zeros((2, 2, 4), np.uint8), (b[0], b[2], b[1], b[3]), "lower"
        if self.failed:
            warnings.warn(f"could not fetch tiles from {self.url}, the basemap is incomplete",
                          PituplotsWarning, stacklevel=2)
        return out


def register_basemap(name, src):
    """Give a name to a URL or a GeoTIFF path.

    Parameters
    ----------
    name : str
    src : str or Path
    """
    if name in TILES or name in ("stock_img", "google-satellite"):
        raise ValueError(f"{name!r} is a built-in basemap")
    REGISTRY[name] = src


def get_basemap(basemap):
    """Resolve a basemap source.

    Parameters
    ----------
    basemap : str, Path or rasterio.DatasetReader
        A built-in or registered name, a URL with ``{z}``, ``{x}`` and ``{y}``,
        or a GeoTIFF.

    Returns
    -------
    Tiles, DatasetReader or "stock_img"
        A dataset opened here is the caller's to close.
    """
    if hasattr(basemap, "crs") and hasattr(basemap, "read"):
        return basemap
    if not isinstance(basemap, (str, Path)):
        raise TypeError(f"unsupported basemap {type(basemap).__name__}")
    src = REGISTRY.get(basemap, basemap)
    s = str(src)
    if s == "stock_img":
        return s
    if s == "google-satellite":
        return Tiles()
    if s in TILES:
        return Tiles(TILES[s])
    if all(k in s for k in ("{z}", "{x}", "{y}")):
        return Tiles(s)
    if Path(s).suffix.lower() in (".tif", ".tiff") or isinstance(src, Path):
        if not Path(s).is_file():
            raise FileNotFoundError(s)
        return need("rasterio", "raster").open(s)
    raise ValueError(f"unknown basemap {s!r}; built-in: {[*TILES, 'stock_img', 'google-satellite']}, "
                     f"registered: {list(REGISTRY)}")


def basemap_crs(basemap):
    """Cartopy CRS of a GeoTIFF basemap."""
    b = get_basemap(basemap)
    if not hasattr(b, "crs"):
        raise ValueError("projection='native' needs a GeoTIFF basemap")
    crs = cartopy_crs(b.crs.to_wkt())
    if b is not basemap:
        b.close()
    return crs


def add_raster(ax, ds, **kw):
    need("rasterio", "raster")
    from rasterio.enums import Resampling
    from rasterio.windows import bounds

    crs = cartopy_crs(ds.crs.to_wkt())
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    t = np.linspace(0, 1, 30)
    xs, ys = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
    p = crs.transform_points(ax.projection, np.r_[xs, xs, xs * 0 + x0, xs * 0 + x1],
                             np.r_[ys * 0 + y0, ys * 0 + y1, ys, ys])[:, :2]
    p = p[np.isfinite(p).all(axis=1)]
    px, py = p[:, 0], p[:, 1]
    if ds.crs.is_geographic:
        px = unwrap_lon(px)
        px = px + 360 * np.round((ds.bounds.left + ds.bounds.right - px.min() - px.max()) / 720)
    l, r = max(px.min(), ds.bounds.left), min(px.max(), ds.bounds.right)
    b, u = max(py.min(), ds.bounds.bottom), min(py.max(), ds.bounds.top)
    if l >= r or b >= u:
        return None

    win = pixel_window(ds, l, b, r, u)
    size = ax.get_window_extent().size
    k = max(1, min(win.width / (2 * size[0]), win.height / (2 * size[1])))
    shape = (max(1, int(np.ceil(win.height / k))), max(1, int(np.ceil(win.width / k))))
    rs = Resampling.average if k > 1 else Resampling.nearest
    if ds.count >= 3:
        z = ds.read(indexes=list(range(1, min(ds.count, 4) + 1)), window=win,
                    out_shape=(min(ds.count, 4), *shape), resampling=rs).transpose(1, 2, 0)
        if z.dtype != np.uint8:
            z = (z - z.min()) / (np.ptp(z) or 1)
    else:
        z = ds.read(1, window=win, out_shape=shape, resampling=rs, masked=True)
        kw.setdefault("cmap", "gray")

    if crs != ax.projection:
        kw.setdefault("regrid_shape", int(2 * size.max()))
    w = bounds(win, ds.transform)
    lims, aspect = (ax.get_xlim(), ax.get_ylim()), ax.get_aspect()
    im = ax.imshow(z, origin="upper", extent=[w[0], w[2], w[1], w[3]], transform=crs, **kw)
    ax.set_xlim(*lims[0])
    ax.set_ylim(*lims[1])
    ax.set_aspect(aspect)
    return im


def add_basemap(ax, basemap, tile_depth="auto", **kw):
    """Draw a basemap at the lowest zorder.

    Parameters
    ----------
    ax : GeoAxes
    basemap : str, Path or rasterio.DatasetReader
        See ``get_basemap``. A GeoTIFF is drawn with ``transform=`` from its
        own CRS. Only the map window is read, decimated to the axes' size.
    tile_depth : int or "auto"
    **kw
        Passed to ``imshow``, e.g. ``cmap`` for a single-band raster.

    Returns
    -------
    AxesImage, Tiles or None
        None if the raster does not reach the map.
    """
    if not hasattr(ax, "projection"):
        raise TypeError("add_basemap needs a GeoAxes")
    kw["zorder"] = ZORDER["basemap"]
    b = get_basemap(basemap)
    if isinstance(b, str):
        return ax.stock_img(**kw)
    if isinstance(b, Tiles):
        if tile_depth == "auto":
            e = ax.get_extent(ccrs.PlateCarree(central_longitude=central_lon(ax.projection)))
            span = max(e[1] - e[0], 1e-6)
            px = ax.get_window_extent().width
            tile_depth = 4 if span >= 360 else int(np.clip(np.ceil(np.log2(px * 360 / (256 * span))) + 1, 2, 16))
        ax.add_image(b, tile_depth, **kw)
        return b
    try:
        return add_raster(ax, b, **kw)
    finally:
        if b is not basemap:
            b.close()
