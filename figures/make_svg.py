"""Rebuild every manuscript figure as SVG (with a PNG fallback) from the project data.

Figure 1  framework diagram
Figure 2  unassigned students by method (results/scenarios/S*_comparison.csv + fairness/b7_comparison.csv)
Figure 3  LEXZON zoning map, S2 (dataset + solver)
Figure 4  school-highlighting views, S2 (dataset + solver)
Figure 5  radius sensitivity (results/scenarios/S*_radius.csv)
Figure 6  enrollment gain under violation budgets (results/pareto_ortools915/pareto_points.csv)
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Repository root. Set LEXZON_ROOT to run against data kept outside the repository.
ROOT = Path(os.environ.get("LEXZON_ROOT", Path(__file__).resolve().parents[1]))
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from lexzon import maps, solver  # noqa: E402
from lexzon.distance import haversine_matrix  # noqa: E402

# IJIES asks for at least 10 pt inside figures. Each figure below is drawn at the
# width at which it is placed in the manuscript, so the sizes set here are also
# the printed sizes: FULL_IN for figures spanning both columns, COL_IN for a
# figure inside one column.
FULL_IN, COL_IN = 6.78, 3.26

plt.rcParams.update({
    "font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 10,
    "axes.titlesize": 10, "axes.labelsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "axes.linewidth": 0.7, "lines.linewidth": 1.2,
    "svg.fonttype": "path",          # text as outlines: identical shape in any renderer
    "figure.dpi": 600, "savefig.dpi": 600,
})
TEAL, GREY, ORANGE, INK, RED = "#1b6b76", "#c9c9c9", "#d1702f", "#1f1f1f", "#d03b3b"
SCEN = {"S1": "S1: matched capacity", "S2": "S2: surplus capacity", "S3": "S3: insufficient capacity"}


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("  wrote", name)


# ---------------------------------------------------------------- Figure 1: framework
def figure1():
    """Four phase columns read left to right, each column top to bottom.

    Drawn at the full two-column width so that the 10 pt labels print at 10 pt;
    the earlier portrait version had to be scaled to half size in the manuscript.
    """
    fig, ax = plt.subplots(figsize=(FULL_IN, 3.45))
    ax.set_position([0, 0, 1, 1])        # the boxes are sized against the full figure width
    ax.set_xlim(0, 4); ax.set_ylim(0.55, 5.45); ax.axis("off")
    COLX = {1: 0.50, 2: 1.50, 3: 2.50, 4: 3.50}
    ROWY = {1: 4.55, 2: 3.53, 3: 2.65, 4: 1.80, 5: 0.93}
    HW = 0.42                                        # half the box width

    def box(col, row, text, fc, h=0.62, ls="-"):
        x, y = COLX[col], ROWY[row]
        ax.add_patch(FancyBboxPatch((x - HW, y - h / 2), 2 * HW, h, boxstyle="round,pad=0.015",
                                    linewidth=0.7, edgecolor=INK, facecolor=fc, linestyle=ls, zorder=3))
        ax.text(x, y, text, ha="center", va="center", fontsize=10, color=INK, zorder=4)
        return dict(x=x, y=y, t=y + h / 2, b=y - h / 2, l=x - HW, r=x + HW)

    def arrow(pts, label=None, dx=0.045):
        for p, q in zip(pts, pts[1:-1]):             # plain segments up to the last one
            ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-", linewidth=0.7,
                                         color=INK, shrinkA=0, shrinkB=0, zorder=1))
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=8,
                                     linewidth=0.7, color=INK, shrinkA=0, shrinkB=0, zorder=1))
        if label:
            p, q = pts[0], pts[1]
            if abs(p[1] - q[1]) < 1e-9:                      # horizontal stub: label above it
                ax.text((p[0] + q[0]) / 2, p[1] + 0.07, label, fontsize=10, color=INK,
                        ha="center", va="bottom", zorder=4)
            else:                                            # vertical stub: label beside it
                ax.text((p[0] + q[0]) / 2 + dx, (p[1] + q[1]) / 2, label, fontsize=10, color=INK,
                        ha="left", va="center", zorder=4)

    BLUE, GREEN, SAND, PLAIN = "#e4ecf5", "#dff0ea", "#fbeee3", "#eef1f4"
    for col, head in ((1, "Data preparation"), (2, "LEXZON"),
                      (3, "Distance-priority analysis"), (4, "Evaluation")):
        ax.text(COLX[col], 5.26, head, ha="center", va="center", fontsize=10, color=INK, weight="bold")

    a = box(1, 1, "School and applicant\ndata, 2025", BLUE)
    b = box(1, 2, "Set capacity scenario\nand eligibility radius", BLUE)
    c = box(1, 3, "Haversine distances\nand eligible pairs", BLUE)
    d = box(2, 1, "Stage 1: maximize\nenrollment", GREEN)
    e = box(2, 2, "Positive maximum\nenrollment?", "#f7f7f2", ls=(0, (2, 2)))
    f = box(2, 3, "Stage 2: minimize\nmaximum distance", GREEN)
    g = box(2, 4, "Stage 3: minimize\ntotal distance", GREEN)
    i = box(2, 5, "Return empty\nassignment", "#f3e7e1")
    h = box(3, 1, "LEXZON assignment\nand benchmark", GREEN)
    p = box(3, 2, "Deferred acceptance,\nviolation budget,\nLEXZON-F", SAND, h=0.80)
    m = box(4, 1, "Verify feasibility,\noptimality, and\nsolver status", PLAIN, h=0.80)
    n = box(4, 2, "Evaluate enrollment,\ndistance, violations,\nruntime", PLAIN, h=0.80)

    for u, v in ((a, b), (b, c), (d, e), (f, g), (h, p), (m, n)):
        arrow([(u['x'], u['b']), (v['x'], v['t'])])
    arrow([(e['x'], e['b']), (f['x'], f['t'])], "yes")
    arrow([(c['r'], c['y']), (1.00, c['y']), (1.00, d['y']), (d['l'], d['y'])])
    arrow([(e['r'], e['y']), (1.96, e['y']), (1.96, i['y']), (i['r'], i['y'])], "no", dx=0.03)
    arrow([(g['r'], g['y']), (2.04, g['y']), (2.04, h['y']), (h['l'], h['y'])])
    arrow([(p['r'], p['y']), (3.00, p['y']), (3.00, m['y']), (m['l'], m['y'])])
    save(fig, "figure1_framework")


# ---------------------------------------------------------------- Figure 2: unassigned students by method
def figure2():
    b7 = pd.read_csv(RES / "fairness" / "b7_comparison.csv").set_index("Scenario")["rejected"]
    fig, axes = plt.subplots(1, 3, figsize=(FULL_IN, 2.45), sharex=True)
    for ax, (sc, title) in zip(axes, SCEN.items()):
        df = pd.read_csv(RES / "scenarios" / f"{sc}_comparison.csv")
        df = df[df.Code != "B1"]
        rows = [("LEXZON", float(df[df.Code == "P"].rejected.iloc[0]))]
        for code in ("B2", "B3", "B4", "B5", "B6"):
            rows.append((code, float(df[df.Code == code].rejected.iloc[0])))
        rows.append(("B7", float(b7[sc])))
        labels = [r[0] for r in rows][::-1]
        vals = [r[1] for r in rows][::-1]
        ypos = np.arange(len(vals))
        colors = [TEAL if l == "LEXZON" else GREY for l in labels]
        bars = ax.barh(ypos, vals, color=colors, edgecolor=INK, linewidth=0.5, height=0.68)
        for bar, lab in zip(bars, labels):
            if lab == "B3":
                bar.set_hatch("///")
        for y, v in zip(ypos, vals):
            ax.text(v + 8, y, f"{v:.0f}" if abs(v - round(v)) < 0.05 else f"{v:.1f}", va="center", fontsize=10)
        ax.axvline(min(vals), color=INK, linestyle=(0, (3, 2)), linewidth=0.7)
        ax.set_yticks(ypos); ax.set_yticklabels(labels)
        ax.set_xlim(0, 345); ax.set_xticks([0, 100, 200, 300])
        ax.set_title(title)
        ax.set_xlabel("Unassigned students")
        ax.tick_params(length=2.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout(w_pad=1.0)
    save(fig, "figure2_unassigned")


# ---------------------------------------------------------------- Figures 3 and 4: maps
def _s2_allocation():
    sch = pd.read_csv(ROOT / "dataset" / "scenario_2_surplus" / "schools.csv")
    stu = pd.read_csv(ROOT / "dataset" / "scenario_2_surplus" / "students.csv")
    dist = haversine_matrix(stu, sch)
    alloc = solver.proposed(dist, sch["Quota"].to_numpy(int), 1.0)
    return sch, stu, dist, alloc.assignment


def _jitter(lat, lon, metres=100.0, seed=2026):
    rng = np.random.default_rng(seed)
    r = rng.uniform(0, metres, len(lat)); th = rng.uniform(0, 2 * np.pi, len(lat))
    jlat = lat + (r * np.sin(th)) / 111320.0
    jlon = lon + (r * np.cos(th)) / (111320.0 * np.cos(np.radians(lat)))
    return jlat, jlon


def _basemap(ax):
    try:
        import contextily as ctx
        ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.Esri.WorldGrayCanvas, attribution=False)
    except Exception as exc:  # noqa: BLE001 - the map is still readable without tiles
        print("  basemap skipped:", exc)


def figure3(sch, stu, dist, assign):
    lat, lon = _jitter(stu.Latitude.to_numpy(), stu.Longitude.to_numpy())
    slat, slon = sch.Latitude.to_numpy(), sch.Longitude.to_numpy()
    col = np.array(maps.school_colors(sch))
    ok = assign >= 0; idx = np.nonzero(ok)[0]
    noel = ~(dist <= 1.0).any(1); un = ~ok
    fill = np.bincount(assign[ok], minlength=len(sch)); full = fill == sch.Quota.to_numpy()

    fig, ax = plt.subplots(figsize=(COL_IN, 3.55))
    seg = np.stack([np.column_stack([lon[idx], lat[idx]]),
                    np.column_stack([slon[assign[idx]], slat[assign[idx]]])], 1)
    ax.add_collection(LineCollection(seg, colors=col[assign[idx]], linewidths=0.18, alpha=0.5, zorder=1))
    ax.scatter(lon[idx], lat[idx], s=1.6, c=col[assign[idx]], linewidths=0, zorder=2)
    ax.scatter(lon[un & ~noel], lat[un & ~noel], s=11, marker="x", color=RED, linewidths=0.6, zorder=5)
    ax.scatter(lon[noel], lat[noel], s=11, marker="+", color=RED, linewidths=0.7, zorder=5)
    ax.scatter(slon[full], slat[full], s=16, marker="s", c=col[full], edgecolors=INK, linewidths=0.5, zorder=4)
    ax.scatter(slon[~full], slat[~full], s=16, marker="D", c=col[~full], edgecolors=INK, linewidths=0.5, zorder=4)
    lab = {6: ("ES 07 / ES 17", (0.0008, 0.0062)), 21: ("ES 11 / ES 22", (0.0025, -0.0040)),
           22: ("ES 23", (0.0035, 0.0010)), 20: ("ES 21", (0.0028, 0.0022)), 11: ("ES 12", (0.0030, 0.0010))}
    for j, (txt, (dx, dy)) in lab.items():
        ax.annotate(txt, (slon[j], slat[j]), (slon[j] + dx, slat[j] + dy), fontsize=10,
                    ha="left" if dx > 0 else "right",
                    arrowprops=dict(arrowstyle="-", lw=0.3, color=INK),
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85), zorder=6)
    pad = 0.0015
    ax.set_xlim(lon.min() - pad, lon.max() + pad); ax.set_ylim(lat.min() - pad, lat.max() + pad)
    ax.set_aspect(1 / np.cos(np.radians(-3.545))); ax.ticklabel_format(useOffset=False, style="plain")
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°)")
    ax.xaxis.set_major_locator(plt.MaxNLocator(4)); ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _basemap(ax)
    save(fig, "figure3_zoning_map")


def figure4(sch, stu, dist, assign):
    lat, lon = _jitter(stu.Latitude.to_numpy(), stu.Longitude.to_numpy())
    slat, slon = sch.Latitude.to_numpy(), sch.Longitude.to_numpy()
    col = np.array(maps.school_colors(sch))
    ok = assign >= 0; noel = ~(dist <= 1.0).any(1); un = ~ok

    def panel(ax, js, title, half_km):
        c0, c1 = np.mean(slat[js]), np.mean(slon[js])
        dl = half_km / 111.32; dn = half_km / (111.32 * np.cos(np.radians(c0)))
        other = ok & ~np.isin(assign, js)
        ax.scatter(lon[other], lat[other], s=2.0, c="#9a9a9a", alpha=0.55, linewidths=0, zorder=1)
        for j in js:
            m = assign == j
            ax.add_collection(LineCollection(
                np.stack([np.column_stack([lon[m], lat[m]]),
                          np.column_stack([np.full(m.sum(), slon[j]), np.full(m.sum(), slat[j])])], 1),
                colors=col[j], linewidths=0.3, alpha=0.7, zorder=2))
            ax.scatter(lon[m], lat[m], s=4, c=col[j], linewidths=0, zorder=3)
            th = np.linspace(0, 2 * np.pi, 240)
            ax.plot(slon[j] + np.cos(th) / (111.32 * np.cos(np.radians(slat[j]))),
                    slat[j] + np.sin(th) / 111.32, color=col[j], lw=0.7, ls="--", zorder=2)
        ax.scatter(lon[un & ~noel], lat[un & ~noel], s=14, marker="x", color=RED, linewidths=0.7, zorder=5)
        ax.scatter(lon[noel], lat[noel], s=14, marker="+", color=RED, linewidths=0.8, zorder=5)
        rest = [j for j in range(len(sch)) if j not in js]
        ax.scatter(slon[rest], slat[rest], s=14, marker="s", c="#cfcfcf", edgecolors=INK, linewidths=0.5, zorder=4)
        ax.scatter(slon[js], slat[js], s=26, marker="s", c=col[js], edgecolors=INK, linewidths=0.8, zorder=6)
        ax.set_xlim(c1 - dn, c1 + dn); ax.set_ylim(c0 - dl, c0 + dl)
        ax.set_aspect(1 / np.cos(np.radians(c0)))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title)
        for s in ax.spines.values():      # IJIES: no border around a figure
            s.set_visible(False)
        x0 = c1 + dn * 0.42; y0 = c0 - dl * 0.88
        ax.plot([x0, x0 + 0.5 / (111.32 * np.cos(np.radians(c0)))], [y0, y0], color=INK, lw=1.4)
        ax.text(x0, y0 + dl * 0.05, "500 m", fontsize=10)
        _basemap(ax)

    fig, axes = plt.subplots(1, 3, figsize=(FULL_IN, 2.95))
    panel(axes[0], [6, 16], "ES 07 and ES 17", 1.25)
    panel(axes[1], [20], "ES 21", 1.15)
    panel(axes[2], [11], "ES 12", 1.15)
    fig.tight_layout(w_pad=0.5)
    save(fig, "figure4_highlight")


# ---------------------------------------------------------------- Figure 5: radius sensitivity
def figure5():
    bound = {"S1": 0, "S2": 0, "S3": 209}
    fig, axes = plt.subplots(1, 3, figsize=(FULL_IN, 2.40), sharey=True)
    for ax, (sc, title) in zip(axes, SCEN.items()):
        df = pd.read_csv(RES / "scenarios" / f"{sc}_radius.csv")
        df = df[df["Radius (km)"] != np.inf]
        series = [("P", "LEXZON = B5", TEAL, "o", "-"),
                  ("B2", "B2: greedy", ORANGE, "s", "--"),
                  ("B4", "B4: deferred acceptance", "#3c6e9f", "^", "-.")]
        for code, lab, color, mk, ls in series:
            sub = df[df.Code == code].sort_values("Radius (km)")
            ax.plot(sub["Radius (km)"], sub["rejected"], ls, color=color, marker=mk,
                    markersize=3, linewidth=1.1, label=lab)
        ax.axhline(bound[sc], color=INK, linestyle=(0, (1, 2)), linewidth=0.8,
                   label="Aggregate capacity lower bound")
        ax.set_title(f"({'abc'[list(SCEN).index(sc)]}) {title}")
        ax.set_xlabel("Eligibility radius (km)")
        ax.set_xticks([0.5, 1.5, 2.5])
        ax.set_ylim(-10, 350); ax.tick_params(length=2.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Unassigned students")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout(w_pad=0.8)
    save(fig, "figure5_radius")


# ---------------------------------------------------------------- Figure 6: violation budgets
def figure6():
    pp = pd.read_csv(RES / "pareto_ortools915" / "pareto_points.csv")
    fig, axes = plt.subplots(1, 3, figsize=(FULL_IN, 2.50), sharey=True)
    for ax, sc in zip(axes, SCEN):
        sub = pp[pp.Scenario == sc]
        da = float(sub[sub.Point == "DA"].admitted.iloc[0])
        nstar = float(sub[sub.Point == "LEXZON"].admitted.iloc[0])
        bud = sub[sub.Point == "budget"].sort_values("budget")
        xs = bud.budget.to_numpy(float); ys = bud.admitted.to_numpy(float) - da
        ub = bud.bound.to_numpy(float) - da
        ax.plot(xs, ys, "-o", color=TEAL, markersize=3, linewidth=1.1, label="Feasible enrollment gain")
        for x, y, u in zip(xs, ys, ub):
            if u > y + 0.5:
                ax.plot([x, x], [y, u], color=ORANGE, linewidth=0.9,
                        solid_capstyle="butt")
                ax.plot([x - 2.5, x + 2.5], [u, u], color=ORANGE, linewidth=0.9)
        ax.axhline(nstar - da, color=INK, linestyle=(0, (4, 2)), linewidth=0.8,
                   label="Maximum possible enrollment gain")
        f = sub[sub.Point == "min_envy_at_Nstar"]
        if len(f):
            kf = float(f.envy.iloc[0])
            ax.plot([kf], [nstar - da], marker="*", markersize=8, color=ORANGE, linestyle="none",
                    label="LEXZON-F feasible solution")
            ax.annotate(f"k = {kf:.0f}", (kf, nstar - da), textcoords="offset points",
                        xytext=(2, 6), fontsize=10)
        ax.set_title(sc)
        ax.set_xlabel("Violation budget k (students)")
        ax.set_xlim(-12, 270); ax.set_xticks([0, 100, 200]); ax.tick_params(length=2.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Additional assignments\nrelative to DA")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(w_pad=0.8)
    save(fig, "figure6_violation_budget")


if __name__ == "__main__":
    print("Figure 1"); figure1()
    print("Figure 2"); figure2()
    sch, stu, dist, assign = _s2_allocation()
    print("Figure 3"); figure3(sch, stu, dist, assign)
    print("Figure 4"); figure4(sch, stu, dist, assign)
    print("Figure 5"); figure5()
    print("Figure 6"); figure6()
    print("done")
