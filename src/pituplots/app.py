import argparse
import asyncio
from pathlib import Path

import numpy as np

from pituplots.errors import need
from pituplots.io.catalogs import load_config, read_catalog
from pituplots.layers import add_layers
from pituplots.scene import (Picker, add_section, animate, degrees, full_bounds, on_wheel, plot_catalog_3d,
                              remove_section, screenshot, show_events, to_lonlat, view_section)
from pituplots.style.defaults import resolve

CMAPS = ["viridis", "magma_r", "plasma", "cividis", "turbo", "RdYlBu_r", "Spectral_r", "Greys"]
SIZES = ["magnitude", "true size", "uniform"]
MODES = ["all", "range", "sliding", "cumulative"]
VIEWS = {"iso": "mdi-cube-outline", "map": "mdi-map-outline", "ew": "mdi-arrow-left-right", "ns": "mdi-arrow-up-down"}


def columns(cat):
    """Columns a scene can be coloured by: numbers, times, and text with up to 40 values."""
    out = ["depth", "z"] + ["mag"] * (cat.mag is not None) + ["time"] * (cat.time is not None)
    for k, v in cat.extra.items():
        v = np.asarray(v)
        if v.dtype.kind in "fiu" or len(np.unique(v.astype(str))) <= 40:
            out.append(str(k))
    return out


def build_app(cat, config=None, name="pituplots", title=None):
    """A trame app around a 3D scene of a catalog.

    The left panel sets colour, size, opacity, view, the
    config's layers (shown and opacity), filters and a section; the
    right panel shows the clicked event and the flags. Hover shows a label,
    click selects, and the flag button or space flags.

    Parameters
    ----------
    cat : Catalog
    config : str, Path or dict, optional
        Its ``scene`` section sets the first state and the style.
    name : str
        Server name; one app per name in a process.
    title : str, optional

    Returns
    -------
    trame_server.Server
        Start it with ``server.start()``. ``server.plotter`` is the scene.
    """
    need("trame", "app"), need("trame_vtk", "app"), need("trame_vuetify", "app")
    from trame.app import asynchronous, get_server
    from trame.ui.vuetify3 import SinglePageWithDrawerLayout
    from trame.widgets import html, vtk as vtkw, vuetify3 as v3

    sc = dict(load_config(config).get("scene") or {})
    hover, ids = sc.pop("hover", None), sc.pop("id", None)
    fixed = {k: sc.pop(k) for k in ("unit", "projection") if k in sc}
    first = {k: sc.pop(k) for k in ("color_by", "z_exag", "true_size", "size_by", "view") if k in sc}
    sc.pop("size", None)
    style = sc
    st = resolve("scene", **style)
    server = get_server(name, client_type="vue3")
    s, ctrl = server.state, server.controller
    opts = columns(cat)
    s.update({"options": opts, "cmaps": CMAPS, "sizes": SIZES, "hover": "", "event": [], "event_title": "",
              "flagged": False, "flags": [], "flag_headers": [], "path": "flags.csv", "shot": "scene.png", "page": "scene.html",
              "message": "", "show_message": False,
              "color_by": first.get("color_by", "depth") if first.get("color_by", "depth") in opts else opts[0],
              "cmap": st["cmap"], "opacity": st["opacity"],
              "size_mode": "true size" if first.get("true_size") else
              ("uniform" if "size_by" in first and first["size_by"] is None else "magnitude"),
              "px": list(st["size"])})

    # filter state
    n = len(cat)
    mag = cat.mag if cat.mag is not None else np.full(n, np.nan)
    lim = [float(np.floor(np.nanmin(mag) * 10) / 10), float(np.ceil(np.nanmax(mag) * 10) / 10)] if np.isfinite(
        mag).any() else [0.0, 1.0]
    texts = [k for k in opts if k in cat.extra and np.asarray(cat.extra[k]).dtype.kind not in "fiu"]
    rows = (cat.extra[ids].astype(str) if ids in cat.extra else np.arange(n).astype(str))
    days = (cat.time.astype("datetime64[ms]").astype("int64") / 86400e3 if cat.time is not None
            else np.full(n, np.nan))
    days[np.isnat(cat.time) if cat.time is not None else slice(None)] = np.nan
    tl = [float(np.floor(np.nanmin(days))), float(np.ceil(np.nanmax(days)))] if np.isfinite(days).any() else [0., 1.]
    tf = 365.2425 if tl[1] - tl[0] > 20 * 365.2425 else 1.0
    s.update({"mag_lim": lim, "mag_range": list(lim), "has_mag": cat.mag is not None, "cat_cols": texts,
              "cat_col": texts[0] if texts else None, "cat_values": [], "cat_keep": [], "search": "",
              "flagged_only": False, "shown": f"{n} of {n} events", "has_time": bool(np.isfinite(days).any()),
              "t_lim": tl, "t_range": list(tl), "t_end": tl[1], "t_len": max(1.0, round((tl[1] - tl[0]) / tf / 20)),
              "t_step": max(1.0, round((tl[1] - tl[0]) / tf / 100)), "t_unit": "yr" if tf > 1 else "d",
              "t_fade": False, "t_mode": "all", "t_label": "",
              "playing": False, "movie": "scene.mp4", "hist": ""})
    p = plot_catalog_3d(cat, color_by=s.color_by, z_exag=first.get("z_exag", 1.0), view=first.get("view", "iso"),
                        true_size=s.size_mode == "true size", size_by=None if s.size_mode == "uniform" else "mag",
                        pick=False,
                        off_screen=True, **fixed, **style)
    layers = load_config(config).get("layers") or []
    base = Path(config).parent if isinstance(config, (str, Path)) else None
    add_layers(p, layers, base)
    names = list(p._pituplots["layers"])
    s.update({"layer_names": names, **{f"lv{i}": True for i in range(len(names))},
              **{f"lo{i}": float(p._pituplots["layers"][ln]["actor"].prop.opacity) for i, ln in enumerate(names)}})
    pk = p._pituplots["picker"] = Picker(p, hover, ids, events=False)
    server.plotter, down = p, {}
    fb = np.array(full_bounds(p), float)
    diag = float(np.hypot(fb[1] - fb[0], fb[3] - fb[2])) or 1.0
    mid = np.array([(fb[0] + fb[1]) / 2, (fb[2] + fb[3]) / 2])
    s.update({"sec_on": False, "sec_az": 90.0, "sec_pos": 0.0, "sec_width": round(0.05 * diag, 3),
              "sec_lim": round(diag / 2, 3), "sec_wmax": round(diag / 4, 3), "sec_step": diag / 500,
              "unit": p._pituplots["unit"]})

    # scene
    def draw():
        cam = p.camera_position
        kw = {**style, "cmap": s.cmap, "opacity": s.opacity, "size": tuple(s.px)}
        e = p._pituplots["catalogs"].pop()
        for g in e["parts"]:
            p.remove_actor(g["actor"])
        if e["bar"] in p.scalar_bars:
            p.remove_scalar_bar(e["bar"])
        p.remove_legend()
        plot_catalog_3d(cat, plotter=p, color_by=s.color_by, pick=False, true_size=s.size_mode == "true size",
                        size_by=None if s.size_mode == "uniform" else "mag", **fixed, **kw)
        p.camera_position = cam
        p._pituplots["picker"] = pk
        pk.markers()
        cut()
        apply()

    # filters
    def date(d):
        return str(np.datetime64(int(round(d)), "D"))

    def when():
        if s.t_mode == "all" or not s.has_time:
            return np.ones(n, bool), None
        lo, hi = {"range": s.t_range, "sliding": [s.t_end - s.t_len * tf, s.t_end], "cumulative": [tl[0] - 1, s.t_end]}[
            s.t_mode]
        return (days > lo) & (days <= hi) | ((days == lo) & (s.t_mode == "range")), (lo, hi)

    def where(time=True):
        keep = np.ones(n, bool)
        if s.has_mag and list(s.mag_range) != lim:
            keep &= (mag >= s.mag_range[0] - 1e-9) & (mag <= s.mag_range[1] + 1e-9)
        if s.cat_col and len(s.cat_keep) != len(s.cat_values):
            keep &= np.isin(np.asarray(cat.extra[s.cat_col]).astype(str), s.cat_keep)
        if s.flagged_only:
            keep &= np.isin(np.arange(n), [r for i, r in pk.flags if i == 0])
        return keep & when()[0] if time else keep

    def histogram(span):
        if not s.has_time:
            return ""
        h, e = np.histogram(days[np.isfinite(days)], 120, range=tl)
        w, top = 1000, max(h.max(), 1)
        bars = "".join(f'<rect x="{i * w / 120:.1f}" y="{40 - 40 * c / top:.1f}" width="{w / 120 - 1:.1f}" '
                       f'height="{40 * c / top:.1f}" fill="#7a8aa0"/>' for i, c in enumerate(h) if c)
        x = [(v - tl[0]) / (tl[1] - tl[0] or 1) * w for v in span] if span else None
        box = (f'<rect x="{max(x[0], 0):.1f}" y="0" width="{max(min(x[1], w) - max(x[0], 0), 2):.1f}" height="40" '
               f'fill="#d62728" fill-opacity="0.18"/>') if x else ""
        return f'<svg viewBox="0 0 {w} 40" preserveAspectRatio="none" style="width:100%;height:40px">{bars}{box}</svg>'

    def layer(**_):
        for i, ln in enumerate(names):
            a = p._pituplots["layers"].get(ln, {}).get("actor")
            if a is not None:
                a.SetVisibility(bool(s[f"lv{i}"]))
                a.prop.opacity = float(s[f"lo{i}"])
                for t in p._pituplots["layers"][ln].get("types", []):
                    p.actors[f"{ln}_{t}"].SetVisibility(bool(s[f"lv{i}"]))
        ctrl.view_update()

    if names:
        s.change(*[f"l{c}{i}" for i in range(len(names)) for c in "vo"])(layer)

    # mini-map of the section, in scene x and -y so north is up
    pad = 0.05 * diag
    box = f"{fb[0] - pad:.6g} {-fb[3] - pad:.6g} {fb[1] - fb[0] + 2 * pad:.6g} {fb[3] - fb[2] + 2 * pad:.6g}"
    mw = 170
    mh = int(min(mw * (fb[3] - fb[2] + 2 * pad) / (fb[1] - fb[0] + 2 * pad), 220))
    svg = f'<svg viewBox="{box}" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">{{}}</svg>'
    line = 'fill="none" vector-effect="non-scaling-stroke" stroke-linecap="round"'

    def background():
        xy = np.vstack([e["points"][:, :2] for e in p._pituplots["catalogs"]])
        xy = xy[np.random.default_rng(0).permutation(len(xy))[:3000]]
        dots = "".join(f"M{x:.5g} {-y:.5g}h0" for x, y in xy)
        segs = ""
        for v in p._pituplots["layers"].values():
            m = v.get("mesh")
            if v["kind"] == "shapes" and m is not None and m.n_lines:
                c = m.lines.reshape(-1, 3)[:, 1:]
                c = c[:: max(1, len(c) // 4000)]
                segs += "".join(f"M{a[0]:.5g} {-a[1]:.5g}L{b[0]:.5g} {-b[1]:.5g}" for a, b in m.points[c][:, :, :2])
        return svg.format(f'<path d="{segs}" stroke="#555" stroke-width="0.8" {line}/>'
                          f'<path d="{dots}" stroke="#7a8aa0" stroke-width="2" {line}/>')

    def minimap(c, a):
        n, t = np.array([np.cos(a), -np.sin(a)]), np.array([np.sin(a), np.cos(a)])
        q = [c + sn * s.sec_width * n + st_ * diag * t for sn, st_ in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
        band = " ".join(f"{x:.5g},{-y:.5g}" for x, y in q)
        e = [c - diag * t, c + diag * t]
        s.map_sec = svg.format(f'<polygon points="{band}" fill="#d62728" fill-opacity="0.25"/>'
                               f'<path d="M{e[0][0]:.5g} {-e[0][1]:.5g}L{e[1][0]:.5g} {-e[1][1]:.5g}" '
                               f'stroke="#d62728" stroke-width="1.5" {line}/>')
        ll = to_lonlat(p, [c[0]], [c[1]])
        s.sec_label = (f"{degrees(ll[1][0], 0.01, 'SN')} {degrees(ll[0][0], 0.01, 'WE')}, " if ll else "") + \
            f"strike {s.sec_az:.0f}°, ±{s.sec_width:.3g} {s.unit}"

    def cut():
        if s.sec_on:
            a = np.radians(s.sec_az)
            c = mid + s.sec_pos * np.array([np.cos(a), -np.sin(a)])
            add_section(p, c, s.sec_az, s.sec_width)
            minimap(c, a)
        else:
            remove_section(p)

    @s.change("sec_on", "sec_az", "sec_pos", "sec_width")
    def section(**_):
        cut()
        apply()

    def step(k):
        if not s.sec_on:
            return
        a = np.radians(s.sec_az)
        n = np.array([np.cos(a), -np.sin(a), 0.0])
        new = float(np.clip(s.sec_pos + 2 * s.sec_width * k, -s.sec_lim, s.sec_lim))
        pos, focal, up = p.camera_position
        p.camera_position = [np.add(pos, (new - s.sec_pos) * n), np.add(focal, (new - s.sec_pos) * n), up]
        with s:
            s.sec_pos = new

    on_wheel(p, step)

    def face():
        if s.sec_on:
            view_section(p)
            ctrl.view_update()

    def apply(push=True, **_):
        keep = where()
        span = when()[1]
        alpha = None
        if s.t_fade and span is not None and s.t_mode in ("sliding", "cumulative"):
            lo = max(span[0], tl[0])
            alpha = np.nan_to_num(np.clip(0.1 + 0.9 * (days - lo) / max(span[1] - lo, 1e-9), 0.1, 1))
        k = show_events(p, keep, alpha=alpha)
        s.shown = f"{k} of {n} events"
        s.t_label = "" if span is None else f"{date(max(span[0], tl[0]))} to {date(span[1])}"
        s.hist = histogram(span)
        if s.sec_on:
            s.shown += " in the section"
        if push:
            ctrl.view_update()

    @s.change("mag_range", "cat_keep", "flagged_only", "t_mode", "t_range", "t_end", "t_len", "t_fade")
    def refilter(**_):
        apply()

    @s.change("cat_col")
    def categories(cat_col, **_):
        v = sorted(set(np.asarray(cat.extra[cat_col]).astype(str))) if cat_col else []
        s.cat_values, s.cat_keep = v, list(v)

    def search():
        hit = np.flatnonzero(rows == str(s.search).strip())
        if not hit.size:
            note(f"no event {s.search!r}")
            return
        e = p._pituplots["catalogs"][0]
        j = np.searchsorted(e["index"], hit[0])
        if j >= len(e["index"]) or e["index"][j] != hit[0]:
            note(f"event {s.search!r} has no coordinates")
            return
        event((0, int(hit[0])))
        p.set_focus(e["points"][j])
        ctrl.view_update()

    def play():
        s.playing = not s.playing
        if s.playing:
            if s.t_mode not in ("sliding", "cumulative"):
                s.t_mode = "sliding"
            asynchronous.create_task(run())

    async def run():
        while s.playing:
            with s:
                nxt = s.t_end + s.t_step * tf
                s.t_end = nxt if nxt <= tl[1] else tl[0] + (s.t_len if s.t_mode == "sliding" else s.t_step) * tf
            await asyncio.sleep(0.3)

    def movie():
        mode = s.t_mode if s.t_mode in ("sliding", "cumulative") else "sliding"
        k = animate(p, s.movie, step=f"{s.t_step * tf:.6g}D", length=f"{s.t_len * tf:.6g}D" if mode == "sliding" else None,
                    start=date(tl[0]), end=date(tl[1]), where=where(time=False), fading=s.t_fade)
        apply()
        note(f"{k} frames written to {Path(s.movie).resolve()}")

    def event(hit):
        pk.select(hit)
        s.event = [] if hit is None else [[k, v] for k, v in pk.fields(hit).items()]
        s.event_title = "" if hit is None else pk.title(hit)
        s.flagged = hit is not None and hit in pk.flags

    def flags():
        t = pk.table()
        s.flags = t.astype(str).to_dict("records")
        s.flag_headers = [{"title": c, "key": c} for c in t.columns if c != "catalog"]
        s.flagged = pk.selected is not None and pk.selected in pk.flags

    @s.change("color_by", "cmap", "size_mode", "px")
    def restyle(**_):
        draw()

    @s.change("opacity")
    def fade(opacity, **_):
        for e in p._pituplots["catalogs"]:
            for g in e["parts"]:
                g["actor"].prop.opacity = opacity
        ctrl.view_update()

    def view(v):
        {"iso": p.view_isometric, "map": p.view_xy, "ew": p.view_xz, "ns": p.view_yz}[v]()
        p.camera.parallel_projection = v != "iso"
        p.reset_camera()
        ctrl.view_update()

    # picking
    def move(pos):
        hit = pk.at(pos["x"], pos["y"])
        pk.last = hit
        s.hover = "" if hit is None else pk.text(hit, pk.hover)

    def press(pos):
        down["pos"] = pos

    def release(pos):
        d = down.pop("pos", None)
        if d is not None and abs(pos["x"] - d["x"]) + abs(pos["y"] - d["y"]) <= 3:
            event(pk.at(pos["x"], pos["y"]))
            ctrl.view_update()

    def toggle(hit=None):
        hit = hit or pk.selected
        if hit is not None:
            pk.toggle(hit)
            flags()
            apply() if s.flagged_only else ctrl.view_update()

    def key(k):
        if k == " ":
            toggle(pk.last)
        elif k in ("[", "]"):
            step(1 if k == "]" else -1)

    def clear():
        pk.flags.clear()
        pk.markers()
        flags()
        apply()

    def note(text):
        s.message, s.show_message = text, True

    def export():
        note(f"{len(pk.flags)} flags written to {Path(pk.export(s.path)).resolve()}")

    def shot():
        screenshot(p, s.shot, scale=2)
        note(f"screenshot written to {Path(s.shot).resolve()}")

    def page():
        from trame_vtk.tools.vtksz2html import write_html

        data = vtkw.VtkLocalView(p.ren_win, trame_server=server).export(format="zip")
        with open(s.page, "w") as f:
            write_html(data, f)
        note(f"static scene written to {Path(s.page).resolve()} ({Path(s.page).stat().st_size / 1e6:.1f} MB)")

    # layout
    with SinglePageWithDrawerLayout(server) as layout:
        layout.title.set_text(title or f"pituPLOTS 3D · {len(cat)} events")
        with layout.drawer as d:
            d.width = 290
            with v3.VContainer(classes="pa-3"):
                html.Div("Colour", classes="text-subtitle-2 mb-1")
                v3.VSelect(v_model=("color_by",), items=("options",), label="column", density="compact",
                           hide_details=True, classes="mb-2")
                v3.VSelect(v_model=("cmap",), items=("cmaps",), label="colormap", density="compact",
                           hide_details=True, classes="mb-2")
                v3.VSlider(v_model=("opacity",), min=0.05, max=1, step=0.05, label="opacity", density="compact",
                           thumb_label=True, hide_details=True, classes="mb-4")
                html.Div("Size", classes="text-subtitle-2 mb-1")
                v3.VSelect(v_model=("size_mode",), items=("sizes",), label="by", density="compact",
                           hide_details=True, classes="mb-2")
                v3.VRangeSlider(v_model=("px",), min=1, max=60, step=1, label="pixels", density="compact",
                                thumb_label=True, hide_details=True, classes="mb-4",
                                disabled=("size_mode != 'magnitude'",))
                html.Div("Scene", classes="text-subtitle-2 mb-1")
                with html.Div(classes="d-flex mb-4"):
                    for v, icon in VIEWS.items():
                        v3.VBtn(icon=icon, size="small", variant="tonal", classes="mr-1", title=v,
                                click=(view, f"['{v}']"))
                html.Div("Filters", classes="text-subtitle-2 mb-1")
                html.Div("{{ shown }}", classes="text-caption mb-1")
                v3.VRangeSlider(v_model=("mag_range",), min=("mag_lim[0]",), max=("mag_lim[1]",), step=0.1,
                                label="mag", density="compact", thumb_label=True, hide_details=True,
                                v_if=("has_mag",), classes="mb-2")
                v3.VSelect(v_model=("cat_col",), items=("cat_cols",), label="category", density="compact",
                           hide_details=True, v_if=("cat_cols.length",), classes="mb-1")
                with v3.VChipGroup(v_model=("cat_keep",), multiple=True, filter=True, column=True):
                    v3.VChip("{{ v }}", v_for="v in cat_values", key="v", value=("v",), size="x-small", label=True)
                v3.VSwitch(v_model=("flagged_only",), label="flagged only", density="compact", hide_details=True,
                           color="error")
                v3.VTextField(v_model=("search",), label="find id", density="compact", hide_details=True,
                              append_inner_icon="mdi-magnify", click_appendInner=search,
                              __events=[("keydown_enter", "keydown.enter")], keydown_enter=search, classes="mb-4")
                html.Div("Layers", classes="text-subtitle-2 mb-1", v_if=("layer_names.length",))
                for i, ln in enumerate(names):
                    with html.Div(classes="d-flex align-center", style="gap: 4px"):
                        v3.VCheckbox(v_model=(f"lv{i}",), label=ln, density="compact", hide_details=True,
                                     style="flex: 0 0 125px")
                        v3.VSlider(v_model=(f"lo{i}",), min=0.05, max=1, step=0.05, density="compact",
                                   hide_details=True, disabled=(f"!lv{i}",))
                html.Div(classes="mb-4")
                html.Div("Section", classes="text-subtitle-2 mb-1")
                v3.VSwitch(v_model=("sec_on",), label="vertical section", density="compact", hide_details=True,
                           color="primary")
                v3.VSlider(v_model=("sec_az",), min=0, max=180, step=1, label="strike", density="compact",
                           thumb_label=True, hide_details=True, disabled=("!sec_on",))
                v3.VSlider(v_model=("sec_pos",), min=("-sec_lim",), max=("sec_lim",), step=("sec_step",),
                           label="offset", density="compact", hide_details=True, disabled=("!sec_on",))
                v3.VSlider(v_model=("sec_width",), min=("sec_step",), max=("sec_wmax",), step=("sec_step",),
                           label="± width", density="compact", hide_details=True, disabled=("!sec_on",))
                with html.Div(classes="d-flex align-center mb-4", style="gap: 8px"):
                    html.Div("{{ sec_pos.toFixed(1) }} ± {{ sec_width.toFixed(1) }} {{ unit }}",
                             classes="text-caption")
                    v3.VBtn(icon="mdi-chevron-left", size="small", variant="tonal", click=(step, "[-1]"),
                            disabled=("!sec_on",), title="previous slab ([ or Ctrl + wheel)")
                    v3.VBtn(icon="mdi-chevron-right", size="small", variant="tonal", click=(step, "[1]"),
                            disabled=("!sec_on",), title="next slab (] or Ctrl + wheel)")
                    v3.VBtn("face on", size="small", variant="tonal", click=face, disabled=("!sec_on",))
                v3.VTextField(v_model=("shot",), label="screenshot", density="compact", hide_details=True,
                              append_inner_icon="mdi-camera", click_appendInner=shot, classes="mb-2")
                v3.VTextField(v_model=("page",), label="static HTML", density="compact", hide_details=True,
                              append_inner_icon="mdi-download", click_appendInner=page)

        with layout.content, html.Div(classes="d-flex", style="height: calc(100vh - 64px)"):
            with html.Div(classes="d-flex flex-column", style="flex: 1 1 0; min-width: 0"):
                with html.Div(style="position: relative; flex: 1 1 0; min-height: 0"):
                    rv = vtkw.VtkRemoteView(
                        p.ren_win, interactive_ratio=1,
                        interactor_events=("events", ["MouseMove", "LeftButtonPress", "LeftButtonRelease",
                                                      "KeyPress"]),
                        MouseMove=(move, "[$event.position]"), LeftButtonPress=(press, "[$event.position]"),
                        LeftButtonRelease=(release, "[$event.position]"), KeyPress=(key, "[$event.key]"))
                    ctrl.view_update = rv.update
                    with html.Div(v_show=("sec_on",), style=f"position:absolute; left:8px; bottom:8px; width:{mw}px; "
                                  "background:rgba(255,255,255,.88); border:1px solid #bbb; border-radius:4px; "
                                  "padding:3px; pointer-events:none"):
                        with html.Div(style=f"position:relative; height:{mh}px"):
                            html.Div(v_html=("map_bg",), style="position:absolute; inset:0")
                            html.Div(v_html=("map_sec",), style="position:absolute; inset:0")
                        html.Div("{{ sec_label }}", style="font-size:10px; color:#333")
                    html.Pre("{{ hover }}", v_show=("hover",),
                             style="position:absolute; top:8px; right:8px; margin:0; padding:4px 8px; "
                                   "background:rgba(255,255,255,.9); font-size:12px; pointer-events:none")
                with html.Div(v_if=("has_time",), classes="px-3 pt-1 pb-2", style="border-top: 1px solid #ddd"):
                    html.Div(v_html=("hist",))
                    v3.VRangeSlider(v_model=("t_range",), min=("t_lim[0]",), max=("t_lim[1]",), step=1,
                                    v_if=("t_mode == 'range'",), density="compact", hide_details=True)
                    v3.VSlider(v_model=("t_end",), min=("t_lim[0]",), max=("t_lim[1]",), step=1,
                               v_if=("t_mode == 'sliding' || t_mode == 'cumulative'",), density="compact",
                               hide_details=True)
                    with html.Div(classes="d-flex flex-wrap align-center", style="gap: 6px"):
                        for m in MODES:
                            v3.VBtn(m, size="small", click=f"t_mode = '{m}'", color="primary",
                                    variant=(f"t_mode == '{m}' ? 'flat' : 'outlined'",))
                        v3.VBtn(icon=("playing ? 'mdi-pause' : 'mdi-play'",), size="small", variant="tonal",
                                click=play)
                        v3.VCheckbox(v_model=("t_fade",), label="fade", density="compact", hide_details=True,
                                     disabled=("t_mode != 'sliding' && t_mode != 'cumulative'",),
                                     style="flex: 0 0 auto")
                        v3.VTextField(v_model_number=("t_len",), type="number", label=("`window (${t_unit})`",),
                                      density="compact", hide_details=True, style="max-width: 105px",
                                      disabled=("t_mode != 'sliding'",))
                        v3.VTextField(v_model_number=("t_step",), type="number", label=("`step (${t_unit})`",),
                                      density="compact", hide_details=True, style="max-width: 90px")
                        html.Div("{{ t_label }}", classes="text-caption")
                        v3.VTextField(v_model=("movie",), label="movie", density="compact", hide_details=True,
                                      append_inner_icon="mdi-movie-open", click_appendInner=movie,
                                      style="max-width: 180px")
            with v3.VSheet(classes="pa-3 overflow-y-auto", border="s", style="flex: 0 0 330px"):
                html.Div("{{ event_title || 'Click an event' }}", classes="text-subtitle-2 mb-1")
                with html.Table(style="font-size:12px; width:100%", v_show=("event.length",)):
                    with html.Tr(v_for="(row, i) in event", key="i"):
                        html.Td("{{ row[0] }}", style="color:#666; padding-right:8px; white-space:nowrap")
                        html.Td("{{ row[1] }}", style="word-break:break-all")
                v3.VBtn("{{ flagged ? 'Unflag' : 'Flag for review' }}", v_show=("event.length",),
                        color=("flagged ? 'grey' : 'error'",), size="small", classes="my-2", click=(toggle, "[]"))
                v3.VDivider(classes="my-2")
                html.Div("Flags ({{ flags.length }})", classes="text-subtitle-2 mb-1")
                v3.VDataTable(headers=("flag_headers",), items=("flags",), density="compact", items_per_page=10,
                              style="font-size:11px")
                v3.VTextField(v_model=("path",), label="flags file", density="compact", hide_details=True,
                              classes="mt-2")
                with html.Div(classes="d-flex mt-2"):
                    v3.VBtn("Export CSV", size="small", variant="tonal", click=export, classes="mr-2")
                    v3.VBtn("Clear", size="small", variant="text", click=clear)
            v3.VSnackbar("{{ message }}", v_model=("show_message",), timeout=4000)
    s.update({"map_bg": background(), "map_sec": "", "sec_label": ""})
    categories(s.cat_col)
    apply(False)
    ctrl.on_server_ready.add(lambda **_: ctrl.view_update())
    ctrl.draw, ctrl.select, ctrl.toggle, ctrl.apply, ctrl.search = draw, event, toggle, apply, search
    ctrl.face, ctrl.cut, ctrl.layer, ctrl.step, ctrl.move = face, section, layer, step, move
    return server


def main(argv=None):
    """Command line: ``pituplots-3d <catalog> [--config yaml] [--port 8080]``."""
    a = argparse.ArgumentParser(prog="pituplots-3d", description="Explore a catalog in 3D in the browser.")
    a.add_argument("catalog")
    a.add_argument("--config")
    a.add_argument("--port", type=int, default=8080)
    a.add_argument("--host", default="localhost", help="0.0.0.0 to share on the local network")
    a.add_argument("--no-browser", action="store_true")
    a.add_argument("--title")
    o = a.parse_args(argv)
    cat = read_catalog(o.catalog, config=o.config)
    server = build_app(cat, config=o.config, title=o.title or f"{Path(o.catalog).name} · {len(cat)} events")
    server.start(port=o.port, host=o.host, open_browser=not o.no_browser)
