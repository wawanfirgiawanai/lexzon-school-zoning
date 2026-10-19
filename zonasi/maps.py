import io
import json

import folium
import numpy as np
from branca.element import Element

from .charts import CRITICAL, MUTED, TEXT
from .distance import planar_km
from .solver import UNASSIGNED

# Categorical slots 1-7 of the validated palette. Red (slot 8) is left out so it
# never competes with the red ✕ used for students without a school.
ZONE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
# Index pairs that fail the colorblind / normal-vision separation check
# (dataviz validate_palette.js, --pairs all): orange-yellow, orange-pink,
# orange-green, aqua-pink. Neighbouring schools must not receive them either.
CONFUSABLE = {(1, 3), (1, 4), (1, 5), (2, 4)}


def school_adjacency(school_df):
    """Schools whose zones border each other: edges of the Delaunay
    triangulation of school locations (a planar graph, so a handful of colors
    always suffices). Falls back to 'every pair' for fewer than 4 schools."""
    from scipy.spatial import Delaunay, QhullError

    pts = planar_km(school_df)
    m = len(pts)
    adj = [set() for _ in range(m)]
    try:
        if m < 4:
            raise QhullError("too few points")
        for simplex in Delaunay(pts).simplices:
            for a in simplex:
                for b in simplex:
                    if a != b:
                        adj[int(a)].add(int(b))
    except (QhullError, ValueError):
        for a in range(m):
            adj[a] = set(range(m)) - {a}
    return adj


def _penalty(c, other):
    if c == other:
        return 2
    return 1 if (min(c, other), max(c, other)) in CONFUSABLE else 0


def school_colors(school_df, restarts=30):
    """One color per school; a school's students share its color. With more
    schools than distinguishable hues a color must repeat, so bordering schools
    are kept apart: never the same color, never a colorblind-confusable pair.
    Greedy coloring plus local search, best of several seeded restarts."""
    m = len(school_df)
    if m == 0:
        return []
    adj = school_adjacency(school_df)
    k = len(ZONE_COLORS)
    rng = np.random.default_rng(0)

    def cost(color):
        return sum(_penalty(color[a], color[b]) for a in range(m) for b in adj[a] if a < b)

    best, best_key = None, None
    for r in range(restarts):
        order = sorted(range(m), key=lambda s: (-len(adj[s]), s)) if r == 0 else list(rng.permutation(m))
        color = [-1] * m
        for j in order:
            pen = [sum(_penalty(c, color[nb]) for nb in adj[j] if color[nb] >= 0) for c in range(k)]
            color[j] = int(np.argmin(pen))
        improved = True
        while improved:
            improved = False
            for j in range(m):
                pen = [sum(_penalty(c, color[nb]) for nb in adj[j]) for c in range(k)]
                c_best = int(np.argmin(pen))
                if pen[c_best] < pen[color[j]]:
                    color[j] = c_best
                    improved = True
        key = (cost(color), int(np.bincount(color, minlength=k).max()))
        if best_key is None or key < best_key:
            best, best_key = color, key
    return [ZONE_COLORS[c] for c in best]


def neighbour_clashes(school_df, colors):
    """Bordering school pairs whose colors are identical or confusable."""
    idx = {c: i for i, c in enumerate(ZONE_COLORS)}
    adj = school_adjacency(school_df)
    return sorted({(a, b) for a in range(len(adj)) for b in adj[a]
                   if a < b and _penalty(idx[colors[a]], idx[colors[b]])})


def _legend(highlight):
    focus_note = " (selected school)" if highlight is not None else ""
    return f"""
<div style="position:absolute; bottom:28px; right:12px; z-index:9999; background:#fcfcfb; padding:8px 12px;
            border:1px solid rgba(11,11,11,0.10); border-radius:6px; font:12px system-ui,'Segoe UI',sans-serif; color:#52514e;">
  <div><span style="display:inline-block;width:11px;height:11px;border-radius:2px;background:#2a78d6;border:2px solid {TEXT};vertical-align:middle"></span>&nbsp; School{focus_note}</div>
  <div><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#2a78d6;vertical-align:middle;margin:0 3px"></span>&nbsp; Student, same color as their school</div>
  <div><span style="display:inline-block;width:16px;height:0;border-top:2px solid {MUTED};vertical-align:middle"></span>&nbsp; Student → assigned school</div>
  <div><span style="color:{CRITICAL};font-weight:700;font-size:15px;vertical-align:middle">✕</span>&nbsp; No school</div>
  <div style="margin-top:4px;color:#898781">Neighbouring schools never share a color.</div>
</div>
"""


def zoning_map(school_df, student_df, assignment, dist_km, radius, highlight=None):
    lat = student_df["Latitude"].to_numpy()
    lon = student_df["Longitude"].to_numpy()
    s_lat = school_df["Latitude"].to_numpy()
    s_lon = school_df["Longitude"].to_numpy()
    names = school_df["School"].to_numpy()
    students = student_df["Student"].to_numpy()
    quotas = school_df["Quota"].to_numpy()
    admitted = assignment != UNASSIGNED
    used = np.bincount(assignment[admitted], minlength=len(school_df))
    cap = np.inf if radius is None else radius
    colors = school_colors(school_df)
    faded = "#c3c2b7"

    m = folium.Map(location=[float(lat.mean()), float(lon.mean())], zoom_start=14, tiles=None, control_scale=True, zoom_snap=0.25)
    folium.TileLayer("Esri.WorldGrayCanvas", name="Gray basemap").add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", show=False).add_to(m)

    def in_focus(j):
        return highlight is None or j == highlight

    lines = folium.FeatureGroup(name="Student → school lines")
    for j in range(len(school_df)):
        members = np.nonzero(assignment == j)[0]
        if not len(members):
            continue
        segments = [[[lat[i], lon[i]], [s_lat[j], s_lon[j]]] for i in members]
        focus = in_focus(j)
        folium.PolyLine(segments, color=colors[j] if focus else faded, weight=1.2 if focus else 0.8,
                        opacity=0.55 if focus else 0.25).add_to(lines)
    lines.add_to(m)

    student_layer = folium.FeatureGroup(name="Admitted students")
    for i in np.nonzero(admitted)[0]:
        j = assignment[i]
        focus = in_focus(j)
        folium.CircleMarker(
            [lat[i], lon[i]], radius=3.5 if focus else 2, weight=0, fill=True,
            fill_color=colors[j] if focus else faded, fill_opacity=0.9 if focus else 0.5,
            tooltip=f"{students[i]} → {names[j]} ({dist_km[i, j]:.2f} km)",
        ).add_to(student_layer)
    student_layer.add_to(m)

    nearest = dist_km.min(axis=1)
    rejected = folium.FeatureGroup(name="No school (✕)")
    for i in np.nonzero(~admitted)[0]:
        reason = "no school within the radius" if nearest[i] > cap else "every school within the radius is full"
        folium.Marker(
            [lat[i], lon[i]],
            icon=folium.DivIcon(html=f'<div style="color:{CRITICAL};font-weight:700;font-size:16px;line-height:16px;'
                                     f'text-shadow:0 0 2px #fff,0 0 2px #fff">✕</div>', icon_size=(16, 16), icon_anchor=(8, 8)),
            tooltip=f"✕ {students[i]}: {reason} (nearest school {nearest[i]:.2f} km)",
        ).add_to(rejected)
    rejected.add_to(m)

    if highlight is not None and radius is not None:
        folium.Circle([s_lat[highlight], s_lon[highlight]], radius=radius * 1000, color=colors[highlight], weight=1.5,
                      fill=True, fill_opacity=0.06, tooltip=f"Zoning radius {radius:g} km").add_to(m)

    school_layer = folium.FeatureGroup(name="Schools")
    for j in range(len(school_df)):
        focus = in_focus(j)
        size = 18 if highlight == j else 14
        fill = colors[j] if focus else faded
        folium.Marker(
            [s_lat[j], s_lon[j]],
            icon=folium.DivIcon(
                html=f'<div style="width:{size}px;height:{size}px;background:{fill};border:2.5px solid {TEXT};'
                     f'border-radius:3px;box-shadow:0 0 0 2px #fff"></div>',
                icon_size=(size, size), icon_anchor=(size // 2, size // 2)),
            tooltip=f"{names[j]} — {used[j]}/{quotas[j]} seats filled",
        ).add_to(school_layer)
    school_layer.add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)
    m.get_root().html.add_child(Element(_legend(highlight)))

    if highlight is None:
        b_lat, b_lon = np.concatenate([lat, s_lat]), np.concatenate([lon, s_lon])
    else:
        members = np.nonzero(assignment == highlight)[0]
        b_lat = np.concatenate([lat[members], [s_lat[highlight]]])
        b_lon = np.concatenate([lon[members], [s_lon[highlight]]])
        if radius is not None:
            dlat = radius / 110.574
            dlon = radius / (111.320 * np.cos(np.radians(s_lat[highlight])))
            b_lat = np.concatenate([b_lat, [s_lat[highlight] - dlat, s_lat[highlight] + dlat]])
            b_lon = np.concatenate([b_lon, [s_lon[highlight] - dlon, s_lon[highlight] + dlon]])
    bounds = [[float(b_lat.min()), float(b_lon.min())], [float(b_lat.max()), float(b_lon.max())]]
    # The iframe gets its final width only after Leaflet initialises, so re-measure and refit once loaded.
    m.get_root().script.add_child(Element(f"""
window.addEventListener("load", function () {{
  var map = window["{m.get_name()}"];
  if (!map) return;
  var bounds = {json.dumps(bounds)};
  var el = map.getContainer();
  function refit() {{
    if (!el.clientWidth || !el.clientHeight) return;
    map.invalidateSize();
    map.fitBounds(bounds, {{padding: [40, 40]}});
  }}
  if (window.ResizeObserver) new ResizeObserver(refit).observe(el);
  [0, 500, 1500].forEach(function (t) {{ setTimeout(refit, t); }});
}});
"""))
    return m


def static_map_png(school_df, student_df, assignment, radius, title, basemap=True):
    """Publication figure (300 DPI PNG bytes)."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    lat = student_df["Latitude"].to_numpy()
    lon = student_df["Longitude"].to_numpy()
    s_lat = school_df["Latitude"].to_numpy()
    s_lon = school_df["Longitude"].to_numpy()
    admitted = assignment != UNASSIGNED
    idx = np.nonzero(admitted)[0]
    colors = np.array(school_colors(school_df))

    fig = Figure(figsize=(8, 8.5))
    ax = fig.add_subplot(111)
    segments = np.stack([np.column_stack([lon[idx], lat[idx]]),
                         np.column_stack([s_lon[assignment[idx]], s_lat[assignment[idx]]])], axis=1)
    ax.add_collection(LineCollection(segments, colors=colors[assignment[idx]], linewidths=0.4, alpha=0.45, zorder=1))
    ax.scatter(lon[idx], lat[idx], s=7, c=colors[assignment[idx]], linewidths=0, zorder=2)
    ax.scatter(lon[~admitted], lat[~admitted], s=46, marker="x", color=CRITICAL, linewidths=1.6, zorder=4)
    ax.scatter(s_lon, s_lat, s=70, marker="s", c=colors, edgecolors=TEXT, linewidths=1.6, zorder=3)

    all_lon = np.concatenate([lon, s_lon])
    all_lat = np.concatenate([lat, s_lat])
    pad_lon = (all_lon.max() - all_lon.min()) * 0.04 or 0.005
    pad_lat = (all_lat.max() - all_lat.min()) * 0.04 or 0.005
    ax.set_xlim(all_lon.min() - pad_lon, all_lon.max() + pad_lon)
    ax.set_ylim(all_lat.min() - pad_lat, all_lat.max() + pad_lat)
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title, fontsize=11, color=TEXT)
    ax.set_aspect("equal", adjustable="box")
    handles = [
        Line2D([0], [0], marker="s", ls="", markerfacecolor=ZONE_COLORS[0], markeredgecolor=TEXT, markersize=8,
               label="School (color = its zone)"),
        Line2D([0], [0], marker="o", ls="", markerfacecolor=ZONE_COLORS[0], markeredgewidth=0, markersize=5,
               label="Student (color of assigned school)"),
        Line2D([0], [0], marker="x", ls="", color=CRITICAL, markersize=8, markeredgewidth=1.6, label="No school"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=8)
    if basemap:
        try:
            import contextily as ctx
            ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.Esri.WorldGrayCanvas, attribution_size=6)
        except Exception:  # noqa: BLE001 - map still useful without tiles (offline)
            pass
    fig.set_layout_engine("constrained")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    return buf.getvalue()
