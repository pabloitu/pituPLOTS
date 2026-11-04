import datetime as dt
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from pituplots.errors import PituplotsWarning, need
from pituplots.io.containers import ALIASES, Catalog

PRESETS = ("csep", "oq", "csep_json")
CSEP = ["lon", "lat", "mag", "time_string", "depth", "catalog_id", "event_id"]
TIME = ("time", "datetime", "time_string", "time_iso", "origin_time", "otime")
DATE = ("date", "origin_date", "ev_date")
SPLIT = (("year", "yr", "y"), ("month", "mon", "mo"), ("day", "d"), ("hour", "hr", "h"),
         ("minute", "min", "mi"), ("second", "sec", "s"))
FORMATS = ("iso", "epoch", "excel", "decimal_year")
EXCEL = (".xlsx", ".xlsm", ".xls")
ROLES = {"x": "lon", "y": "lat", "z": "depth"}
KEYS = ("columns", "time", "crs", "z", "z_unit", "sheet", "sep", "preset", "time_unit", "trim", "scene", "layers")


def parse_time(s, unit="ms"):
    if pd.api.types.is_datetime64_any_dtype(s):
        t = pd.to_datetime(s, utc=True).dt.tz_localize(None)
        return t.to_numpy().astype("datetime64[ms]")
    if pd.api.types.is_numeric_dtype(s):
        return (s.to_numpy(float) * pd.Timedelta(1, unit) / pd.Timedelta(1, "ms")).astype("int64").astype(
            "datetime64[ms]")
    text = s.astype(str).str.strip().str.replace(r"(Z|[+-]00:?00)$", "", regex=True)
    if not text.str.contains(r"T?\d\d:\d\d.*[+-]\d\d:?\d\d$").any():
        try:
            return np.array(text.to_numpy(str), "datetime64[ms]")
        except ValueError:
            pass
    return pd.to_datetime(s, utc=True, format="mixed").dt.tz_localize(None).to_numpy().astype(
        "datetime64[ms]")


def split_time(c):
    """Assemble datetimes from year, month, ... columns, as numpy dates, so old years work."""
    y, mo, d = [pd.to_numeric(x).to_numpy(float).astype(int) for x in c[:3]]
    rest = [pd.to_numeric(x).to_numpy(float) if x is not None else np.zeros(y.size) for x in c[3:]]
    t = (y - 1970).astype("datetime64[Y]").astype("datetime64[M]") + (mo - 1).astype("timedelta64[M]")
    t = t.astype("datetime64[D]") + (d - 1).astype("timedelta64[D]")
    ms = np.rint((rest[0] * 3600 + rest[1] * 60 + rest[2]) * 1000).astype("int64")
    return t.astype("datetime64[ms]") + ms.astype("timedelta64[ms]")


def from_days(d, base):
    """Datetimes ``base + d`` days, NaN to NaT."""
    d = np.asarray(d, float)
    ok = np.isfinite(d)
    ms = np.rint(np.where(ok, d, 0) * 86400000).astype("int64")
    t = np.datetime64(base, "ms") + ms.astype("timedelta64[ms]")
    t[~ok] = np.datetime64("NaT", "ms")
    return t


def excel_days(x):
    """Excel serial days of a column of numbers, dates or times of day."""
    if pd.api.types.is_datetime64_any_dtype(x):
        return (x.to_numpy().astype("datetime64[ms]") - np.datetime64("1899-12-30", "ms")) / np.timedelta64(1, "D")
    if x.dtype == object:
        x = x.map(lambda v: (v.hour * 3600 + v.minute * 60 + v.second + v.microsecond / 1e6) / 86400
                  if isinstance(v, dt.time) else v)
    try:
        return pd.to_numeric(x).to_numpy(float)
    except ValueError:
        return (pd.to_timedelta(x.astype(str)) / pd.Timedelta(1, "D")).to_numpy(float)


def decimal_years(v):
    """Datetimes of decimal years, such as 2016.5."""
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    y = np.floor(np.where(ok, v, 1970)).astype(int)
    a = (y - 1970).astype("datetime64[Y]")
    n = ((a + np.timedelta64(1, "Y")).astype("datetime64[D]") - a.astype("datetime64[D]")).astype(float)
    t = a.astype("datetime64[ms]") + from_days((np.where(ok, v, 1970) - y) * n, "1970-01-01").astype("int64").astype(
        "timedelta64[ms]")
    t[~ok] = np.datetime64("NaT", "ms")
    return t


def load_config(config):
    """A catalog config from a YAML file or a dict.

    Parameters
    ----------
    config : str, Path, dict or None

    Returns
    -------
    dict
        Top-level keys among ``columns``, ``time``, ``crs``, ``z``,
        ``z_unit``, ``sheet``, ``sep``, ``preset``, ``time_unit``, ``trim``,
        ``scene`` and ``layers``.
    """
    if config is None:
        return {}
    if not isinstance(config, dict):
        import yaml

        p = Path(config)
        if not p.is_file():
            raise FileNotFoundError(p)
        config = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    bad = set(config) - set(KEYS)
    if bad:
        raise ValueError(f"unknown config keys {sorted(bad)}, expected {list(KEYS)}")
    return dict(config)


def trim_catalog(cat, box=None, mad=None, mag=None):
    """Drop events outside a box, far from the bulk, or outside a magnitude range.

    A warning gives the number dropped.

    Parameters
    ----------
    cat : Catalog
    box : sequence of float, optional
        ``[x0, x1, y0, y1]`` or ``[x0, x1, y0, y1, d0, d1]`` in the catalog
        coordinates, depth in km.
    mad : float, optional
        Drop events farther from the median than ``mad`` times the median
        absolute deviation, on any of x, y, z in metres. Events without
        coordinates are dropped too.
    mag : (float, float), optional

    Returns
    -------
    Catalog
    """
    keep = np.ones(len(cat), bool)
    if box is not None:
        b = np.asarray(box, float)
        if b.size not in (4, 6):
            raise ValueError(f"box has {b.size} values, expected 4 or 6")
        if b.size == 6 and cat.depth is None:
            raise ValueError("a 6-value box needs depths")
        for v, lo, hi in zip([cat.lon, cat.lat, cat.depth], b[::2], b[1::2]):
            keep &= (v >= lo) & (v <= hi)
    if mad is not None:
        xyz = cat.local_xyz()[0]
        dev = np.abs(xyz - np.nanmedian(xyz, 0))
        s = np.nanmedian(dev, 0)
        keep &= np.all(dev <= mad * np.where(s > 0, s, np.inf), 1)
    if mag is not None:
        if cat.mag is None:
            raise ValueError("a magnitude range needs magnitudes")
        keep &= (cat.mag >= mag[0]) & (cat.mag <= mag[1])
    if not keep.all():
        warnings.warn(f"trim_catalog dropped {(~keep).sum()} of {len(cat)} events", PituplotsWarning, stacklevel=2)
    return cat[keep]


def read_catalog(path, columns=None, time=None, sep=None, preset=None, time_unit=None, config=None, sheet=None,
                 crs=None, z=None, z_unit=None, trim=None):
    """Read an event catalog.

    Parameters
    ----------
    path : str or Path
        Delimited text, or CSEP JSON with ``preset="csep_json"``.
        Excel files (``.xlsx``) need openpyxl.
    columns : dict, optional
        ``{"lon": name or index, ...}``, for ``lon``, ``lat``, ``depth``,
        ``mag``, or ``x``, ``y``, ``z`` for the first three. The rest are
        found through the aliases.
    time : str, list or dict, optional
        Time column, or the year, month, day, hour, minute, second columns.
        Found by name if missing. A dict ``{"columns": ..., "format": ...}``
        sets the format: ``"iso"``, ``"epoch"``, ``"excel"`` (serial days,
        summed over the columns, such as date and day fraction) or
        ``"decimal_year"``.
    sep : str, optional
        Sniffed if missing.
    preset : {"csep", "oq", "csep_json"}, optional
    time_unit : str
        Unit of numeric times, as epochs. Default ``"ms"``.
    config : str, Path or dict, optional
        The same keywords in a YAML file (see ``load_config``); explicit
        keywords win.
    sheet : str or int, optional
        Excel sheet. Default the first.
    crs : str, optional
        Default ``"EPSG:4326"``, or ``"local"`` when ``x``/``y`` columns are
        named.
    z : {"depth", "elevation"}, optional
        What the depth column holds. Default ``"depth"``.
    z_unit : {"km", "m"}, optional
        Default ``"km"``.
    trim : dict, optional
        Keywords of ``trim_catalog``.

    Returns
    -------
    Catalog
        In file order, with the other columns in ``extra``.
    """
    cfg = load_config(config)
    columns = columns if columns is not None else cfg.get("columns")
    time = time if time is not None else cfg.get("time")
    sep = sep if sep is not None else cfg.get("sep")
    preset = preset if preset is not None else cfg.get("preset")
    time_unit = time_unit or cfg.get("time_unit", "ms")
    sheet = sheet if sheet is not None else cfg.get("sheet", 0)
    z, z_unit = z or cfg.get("z", "depth"), z_unit or cfg.get("z_unit", "km")
    trim = trim if trim is not None else cfg.get("trim")
    if z not in ("depth", "elevation") or z_unit not in ("km", "m"):
        raise ValueError(f"z must be depth or elevation and z_unit km or m, not {z!r}, {z_unit!r}")
    fmt = "iso"
    if isinstance(time, dict):
        time, fmt = time.get("columns"), time.get("format", "iso")
    if fmt not in FORMATS:
        raise ValueError(f"time format must be one of {FORMATS}, not {fmt!r}")
    if preset not in (None, *PRESETS):
        raise ValueError(f"preset must be one of {PRESETS}, not {preset!r}")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if preset == "csep_json":
        rows = json.loads(path.read_text(encoding="utf-8"))["catalog"]
        c = list(zip(*rows)) if rows else [()] * 6
        t = np.array(c[1], "int64").astype("datetime64[ms]")
        return Catalog(c[3], c[2], c[4], c[5], t, extra={"id": np.array(c[0], str)})

    if path.suffix.lower() in EXCEL:
        need("openpyxl", "excel")
        df, numeric = pd.read_excel(path, sheet_name=sheet), False
    else:
        first = pd.read_csv(path, sep=sep, engine="python", header=None, nrows=1, skipinitialspace=True)
        numeric = pd.to_numeric(first.iloc[0], errors="coerce").notna().all()
        df = pd.read_csv(path, sep=sep, engine="python", header=None if numeric else "infer",
                         skipinitialspace=True)
    if numeric and preset == "csep":
        df.columns = CSEP[:df.shape[1]]
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    low = {str(c).lower(): c for c in df.columns}

    def pick(names):
        return next((low[n] for n in names if n in low), None)

    used = {}
    xy = any(k in ("x", "y") for k in columns or {})
    for k, v in (columns or {}).items():
        k = ROLES.get(k, k)
        if k not in ALIASES:
            raise ValueError(f"unknown column role {k!r}, expected one of {list(ALIASES)} or x, y, z")
        used[k] = df.columns[v] if isinstance(v, int) else v
        if used[k] not in df.columns:
            raise ValueError(f"no column {v!r} in {list(df.columns)}")
    for k in ALIASES:
        if k not in used and pick(ALIASES[k]) is not None:
            used[k] = pick(ALIASES[k])
    if "lon" not in used or "lat" not in used:
        raise ValueError(f"no lon/lat or x/y columns in {list(df.columns)}: pass columns=")

    spent = set(used.values())
    names = [time] if isinstance(time, str) else time
    for c in names or []:
        if c not in df.columns:
            raise ValueError(f"no column {c!r} in {list(df.columns)}")
    if fmt == "excel":
        if not names:
            raise ValueError("excel times need columns")
        t, tc = from_days(sum(excel_days(df[c]) for c in names), "1899-12-30"), list(names)
    elif fmt == "decimal_year":
        if not names or len(names) != 1:
            raise ValueError("decimal_year times need one column")
        t, tc = decimal_years(pd.to_numeric(df[names[0]])), list(names)
    elif fmt == "epoch" and names:
        if len(names) != 1:
            raise ValueError("epoch times need one column")
        t, tc = parse_time(pd.to_numeric(df[names[0]]), time_unit), list(names)
    elif isinstance(time, (list, tuple)):
        parts = [df[c] for c in time] + [None] * (6 - len(time))
        t, tc = split_time(parts), list(time)
    else:
        names = [time] if time is not None else None
        tc = pick(names or TIME)
        dc = pick(DATE)
        free = [pick([n for n in a]) for a in SPLIT]
        free = [None if c in spent else c for c in free]
        if time is not None and tc is None:
            raise ValueError(f"no column {time!r} in {list(df.columns)}")
        if tc is not None and dc is not None:
            t, tc = parse_time(df[dc].astype(str) + "T" + df[tc].astype(str)), [dc, tc]
        elif tc is not None or dc is not None:
            tc = tc if tc is not None else dc
            t, tc = parse_time(df[tc], time_unit), [tc]
        elif all(c is not None for c in free[:3]):
            t, tc = split_time([None if c is None else df[c] for c in free]), [c for c in free if c is not None]
        else:
            t, tc = None, []
    spent |= set(tc)
    extra = {str(c): df[c].to_numpy() for c in df.columns if c not in spent}
    get = lambda k: df[used[k]].to_numpy(float) if k in used else None  # noqa: E731
    d = get("depth")
    if d is not None:
        d = (-d if z == "elevation" else d) / (1000 if z_unit == "m" else 1)
    crs = crs or cfg.get("crs") or ("local" if xy else "EPSG:4326")
    cat = Catalog(get("lon"), get("lat"), d, get("mag"), t, extra=extra, crs=crs)
    return trim_catalog(cat, **trim) if trim else cat
