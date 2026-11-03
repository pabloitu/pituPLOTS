import json
from pathlib import Path

import numpy as np
import pandas as pd

from pituplots.io.containers import ALIASES, Catalog

PRESETS = ("csep", "oq", "csep_json")
CSEP = ["lon", "lat", "mag", "time_string", "depth", "catalog_id", "event_id"]
TIME = ("time", "datetime", "time_string", "time_iso", "origin_time", "otime")
DATE = ("date", "origin_date", "ev_date")
SPLIT = (("year", "yr", "y"), ("month", "mon", "mo"), ("day", "d"), ("hour", "hr", "h"),
         ("minute", "min", "mi"), ("second", "sec", "s"))


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


def read_catalog(path, columns=None, time=None, sep=None, preset=None, time_unit="ms"):
    """Read an event catalog.

    Parameters
    ----------
    path : str or Path
        Delimited text, or CSEP JSON with ``preset="csep_json"``.
    columns : dict, optional
        ``{"lon": name or index, ...}``, for ``lon``, ``lat``, ``depth``,
        ``mag``. The rest are found through the aliases.
    time : str or list, optional
        Time column, or the year, month, day, hour, minute, second columns.
        Found by name if missing.
    sep : str, optional
        Sniffed if missing.
    preset : {"csep", "oq", "csep_json"}, optional
    time_unit : str
        Unit of numeric times, as epochs.

    Returns
    -------
    Catalog
        In file order, with the other columns in ``extra``.
    """
    if preset not in (None, *PRESETS):
        raise ValueError(f"preset must be one of {PRESETS}, not {preset!r}")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if preset == "csep_json":
        rows = json.loads(path.read_text())["catalog"]
        c = list(zip(*rows)) if rows else [()] * 6
        t = np.array(c[1], "int64").astype("datetime64[ms]")
        return Catalog(c[3], c[2], c[4], c[5], t, extra={"id": np.array(c[0], str)})

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
    for k, v in (columns or {}).items():
        if k not in ALIASES:
            raise ValueError(f"unknown column role {k!r}, expected one of {list(ALIASES)}")
        used[k] = df.columns[v] if isinstance(v, int) else v
        if used[k] not in df.columns:
            raise ValueError(f"no column {v!r} in {list(df.columns)}")
    for k in ALIASES:
        if k not in used and pick(ALIASES[k]) is not None:
            used[k] = pick(ALIASES[k])
    if "lon" not in used or "lat" not in used:
        raise ValueError(f"no lon/lat columns in {list(df.columns)}: pass columns=")

    spent = set(used.values())
    if isinstance(time, (list, tuple)):
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
    return Catalog(get("lon"), get("lat"), get("depth"), get("mag"), t, extra=extra)
