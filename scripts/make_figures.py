"""
Regenerate every figure and results table used in README.md, with the same
functions the Streamlit app calls.

Run from the project root:  python scripts/make_figures.py
Outputs: docs/figures/*.png (300 DPI) and docs/results/*.csv
"""
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from zonasi import data, maps, solver  # noqa: E402
from zonasi import experiments as ex  # noqa: E402
from zonasi.charts import AXIS, GRID, MUTED, SERIES, SURFACE, TEXT, TEXT_2  # noqa: E402
from zonasi.distance import haversine_matrix  # noqa: E402

FIG = ROOT / "docs" / "figures"
RES = ROOT / "docs" / "results"
RADIUS = 1.0
SEEDS = 30
RADII = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, None]
STAT_RUNS, STAT_FRACTION, STAT_SEED = 30, 0.8, 42
SIZES = [1000, 2000, 5000, 10000, 20000, 50000]
NAMES = ex.METHOD_NAMES
P = ex.PROPOSED_CODE

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"], "font.size": 10,
    "axes.edgecolor": AXIS, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save(fig, name):
    fig.savefig(FIG / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"  saved docs/figures/{name}")


def fig_comparison(table):
    df = table.iloc[::-1]
    labels = [f"{r.Code} · {r.Method}" for r in df.itertuples()]
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(y, df["rejected_full"], color=SERIES[0], height=0.62, label="All schools in radius full")
    ax.barh(y, df["rejected_no_school"], left=df["rejected_full"], color=SERIES[1], height=0.62, label="No school within radius")
    for yi, total in zip(y, df["rejected"]):
        ax.text(total + 1.5, yi, f"{total:.0f}" if float(total).is_integer() else f"{total:.1f}", va="center", color=TEXT_2)
    ax.set_yticks(y, labels)
    for tick, code in zip(ax.get_yticklabels(), df["Code"]):
        tick.set_color(TEXT if code == P else TEXT_2)
        tick.set_fontweight("bold" if code == P else "normal")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Students without a school")
    ax.set_xlim(0, df["rejected"].max() * 1.15)
    ax.set_title(f"Students left without a school, radius {RADIUS:g} km")
    ax.legend(loc="lower right")
    save(fig, "comparison.png")


def fig_radius(sweep):
    xs = {r: i for i, r in enumerate(sorted(sweep["Radius (km)"].unique()))}
    ticks = ["∞" if np.isinf(r) else f"{r:g}" for r in xs]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))

    def draw(ax, code, idx, column):
        part = sweep[sweep["Code"] == code].sort_values("Radius (km)")
        x = [xs[r] for r in part["Radius (km)"]]
        if code == P:
            # Drawn as a wide band underneath: where B5 coincides with it, both stay visible.
            ax.plot(x, part[column], "-", color=SERIES[idx], lw=7, alpha=0.35, solid_capstyle="round",
                    label=f"{code} · {NAMES[code]}")
        else:
            ax.plot(x, part[column], "-o", color=SERIES[idx], lw=2, ms=6, mec=SURFACE, mew=1.5, label=f"{code} · {NAMES[code]}")

    for idx, code in enumerate(ex.SWEEP_METHODS):
        draw(a, code, idx, "rejected")
    a.set_xticks(range(len(ticks)), ticks)
    a.set_xlabel("Zoning radius (km)")
    a.set_ylabel("Students without a school")
    a.set_title("(a) Students without a school")
    a.legend(loc="upper right")
    for idx, code in enumerate([P, "B5"]):
        draw(b, code, idx, "max_km")
    b.set_xticks(range(len(ticks)), ticks)
    b.set_xlabel("Zoning radius (km)")
    b.set_ylabel("Longest trip of an admitted student (km)")
    b.set_title("(b) Effect of the min-max stage")
    b.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "radius_sensitivity.png")


def fig_scenarios(sc):
    order = [P, "B2", "B4", "B5"]
    scen = list(dict.fromkeys(sc["Scenario"]))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 4))
    for idx, code in enumerate(order):
        part = sc[sc["Code"] == code].set_index("Scenario").reindex(scen)
        x = np.arange(len(scen)) + (idx - 1.5) * width
        bars = ax.bar(x, part["rejected"], width * 0.92, color=SERIES[idx], label=f"{code} · {NAMES[code]}")
        if code == P:
            ax.bar_label(bars, fmt="%.0f", padding=2, color=TEXT_2, fontsize=9)
    seats = sc.drop_duplicates("Scenario").set_index("Scenario").reindex(scen)["Total seats"]
    ax.set_xticks(np.arange(len(scen)), [f"{s}\n{int(q):,} seats / 1,209 students" for s, q in zip(scen, seats)])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Students without a school")
    ax.set_title(f"Capacity scenarios, radius {RADIUS:g} km")
    ax.legend(ncol=2)
    save(fig, "capacity_scenarios.png")


def fig_statistics(runs_df):
    order = [P] + ex.TEST_METHODS
    groups = [runs_df[runs_df["Code"] == c]["rejected"].to_numpy() for c in order]
    fig, ax = plt.subplots(figsize=(8, 4))
    rng = np.random.default_rng(0)
    for i, (code, vals) in enumerate(zip(order, groups)):
        color = SERIES[0] if code == P else MUTED
        ax.boxplot(vals, positions=[i], widths=0.5, showfliers=False, patch_artist=True,
                   boxprops={"facecolor": "none", "edgecolor": color, "linewidth": 1.6},
                   medianprops={"color": color, "linewidth": 2}, whiskerprops={"color": color}, capprops={"color": color})
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(vals)), vals, s=14, color=color, alpha=0.75, linewidths=0)
    ax.set_xticks(range(len(order)), [f"{c}\n" + NAMES[c].replace(" ", "\n", 1) for c in order], fontsize=8)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Students without a school")
    ax.set_title(f"{STAT_RUNS} resampled runs ({STAT_FRACTION:.0%} of students), radius {RADIUS:g} km")
    save(fig, "statistical_test.png")


def fig_ablation(abl):
    ok = abl[abl["Status"] == "Optimal"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(ok["Code"], ok["beyond_radius"], "-o", color=SERIES[1], lw=2, ms=6, mec=SURFACE, mew=1.5, label="Admitted beyond the radius")
    ax.plot(ok["Code"], ok["rejected"], "-o", color=SERIES[0], lw=2, ms=6, mec=SURFACE, mew=1.5, label="Without a school")
    ax.set_xlabel("Ablation configuration")
    ax.set_ylabel("Students")
    ax.set_title("Ablation: each component added in turn")
    ax.legend()
    save(fig, "ablation.png")


def fig_scalability(scale):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(scale["Students"], scale["Proposed total runtime (s)"], "-o", color=SERIES[0], lw=2, ms=6, mec=SURFACE, mew=1.5)
    last = scale.iloc[-1]
    ax.annotate(f"{last['Proposed total runtime (s)']:.1f} s", (last["Students"], last["Proposed total runtime (s)"]),
                textcoords="offset points", xytext=(-8, 8), ha="right", color=TEXT_2)
    ax.set_xlabel("Students (synthetic)")
    ax.set_ylabel("Runtime of the proposed method (s)")
    ax.set_title("Scalability")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
    save(fig, "scalability.png")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    school_df, student_df, _ = data.load_bundled("Main dataset (original quotas)")
    dist = haversine_matrix(student_df, school_df)
    quotas = school_df["Quota"].to_numpy(dtype=int)

    log("Allocation map")
    alloc = solver.proposed(dist, quotas, RADIUS)
    (FIG / "zoning_map.png").write_bytes(
        maps.static_map_png(school_df, student_df, alloc.assignment, RADIUS, f"Allocation by the proposed method (radius {RADIUS:g} km)"))
    log("  saved docs/figures/zoning_map.png")

    log("Method comparison")
    table, _, max_adm = ex.comparison_table(dist, school_df, student_df, RADIUS, SEEDS)
    table.round(4).to_csv(RES / "comparison.csv", index=False)
    log(f"  max-flow upper bound = {max_adm}")
    fig_comparison(table)

    log("Ablation")
    abl = ex.ablation(school_df, student_df, RADIUS)
    abl.round(4).to_csv(RES / "ablation.csv", index=False)
    fig_ablation(abl)

    log("Radius sensitivity")
    sweep = ex.radius_sweep(dist, school_df, student_df, RADII)
    sweep.round(4).to_csv(RES / "radius_sensitivity.csv", index=False)
    fig_radius(sweep)

    log("Capacity scenarios")
    sc = ex.scenario_study(RADIUS, SEEDS)
    sc.round(4).to_csv(RES / "capacity_scenarios.csv", index=False)
    fig_scenarios(sc)

    log("Statistical test")
    runs_df, summary = ex.resampling_test(school_df, student_df, RADIUS, STAT_RUNS, STAT_FRACTION, STAT_SEED)
    runs_df.round(4).to_csv(RES / "statistical_test_runs.csv", index=False)
    summary.to_csv(RES / "statistical_test_summary.csv", index=False)
    fig_statistics(runs_df)

    log("Scalability")
    scale = ex.scalability(school_df, student_df, SIZES, RADIUS, 7)
    scale.round(4).to_csv(RES / "scalability.csv", index=False)
    fig_scalability(scale)
    log("Done.")


if __name__ == "__main__":
    main()
