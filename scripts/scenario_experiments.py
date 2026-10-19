"""
Full experiment for each of the three capacity scenarios (S1 exact fit,
S2 surplus, S3 shortage): method comparison, radius sweep, resampled
Wilcoxon test and school utilisation, plus a repeated scalability timing.

Run from the project root:  python scripts/scenario_experiments.py
Outputs: docs/results/scenarios/*.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from zonasi import data, solver  # noqa: E402
from zonasi import experiments as ex  # noqa: E402
from zonasi.distance import haversine_matrix  # noqa: E402
from zonasi.metrics import school_utilization  # noqa: E402

OUT = ROOT / "docs" / "results" / "scenarios"
RADIUS = 1.0
SEEDS = 30
RADII = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, None]
STAT_RUNS, STAT_FRACTION, STAT_SEED = 30, 0.8, 42
SIZES = [1000, 2000, 5000, 10000, 20000, 50000]
REPEATS = 3


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary_rows, stat_rows = [], []
    for k, key in enumerate(data.SCENARIO_KEYS, start=1):
        code = f"S{k}"
        log(f"{code}: {key}")
        school_df, student_df, _ = data.load_bundled(key)
        quotas = school_df["Quota"].to_numpy(dtype=int)
        dist = haversine_matrix(student_df, school_df)

        table, allocations, max_adm = ex.comparison_table(dist, school_df, student_df, RADIUS, SEEDS)
        full = {}
        for c, alloc in allocations.items():
            used = np.bincount(alloc.assignment[alloc.assignment >= 0], minlength=len(quotas))
            full[c] = int((used >= quotas).sum())
        table["schools_full"] = table["Code"].map(full)
        table.insert(0, "Scenario", code)
        table.round(4).to_csv(OUT / f"{code}_comparison.csv", index=False)

        p = allocations[ex.PROPOSED_CODE]
        forced = solver.forced_full_assignment(dist, quotas)
        beyond = np.nan
        if forced.feasible:
            adm = forced.assignment >= 0
            beyond = int((dist[np.nonzero(adm)[0], forced.assignment[adm]] > RADIUS).sum())
        util = school_utilization(p.assignment, dist, school_df)
        summary_rows.append({
            "Scenario": code, "Description": key.split(" — ")[1], "Students": len(student_df),
            "Seats": int(quotas.sum()), "Quota values": ", ".join(map(str, sorted(set(quotas.tolist())))),
            "Max-flow bound N*": max_adm, "z* (km)": round(p.info["z_star"], 4),
            "Bisection steps": p.info["bisection_iterations"],
            "Certificate flow below z*": p.info["certificate"]["flow_just_below_z_star"],
            "Forced model feasible": forced.feasible, "Forced model beyond R": beyond,
            "Proposed schools full": full[ex.PROPOSED_CODE],
        })
        if True:
            util.to_csv(OUT / f"{code}_utilization.csv", index=False)

        log(f"  radius sweep")
        sweep = ex.radius_sweep(dist, school_df, student_df, RADII)
        sweep.insert(0, "Scenario", code)
        sweep.round(4).to_csv(OUT / f"{code}_radius.csv", index=False)

        log(f"  resampling test")
        runs_df, stat = ex.resampling_test(school_df, student_df, RADIUS, STAT_RUNS, STAT_FRACTION, STAT_SEED)
        runs_df.insert(0, "Scenario", code)
        runs_df.round(4).to_csv(OUT / f"{code}_stat_runs.csv", index=False)
        stat.insert(0, "Scenario", code)
        stat_rows.append(stat)

    pd.DataFrame(summary_rows).to_csv(OUT / "scenario_summary.csv", index=False)
    pd.concat(stat_rows).to_csv(OUT / "scenario_stat_summary.csv", index=False)

    log("Scalability (median of repeated runs)")
    school_df, student_df, _ = data.load_bundled("Main dataset (original quotas)")
    rows = []
    for rep in range(REPEATS):
        part = ex.scalability(school_df, student_df, SIZES, RADIUS, 7)
        part["Repeat"] = rep + 1
        rows.append(part)
        log(f"  repeat {rep + 1} done")
    allrep = pd.concat(rows)
    allrep.round(4).to_csv(OUT / "scalability_repeats.csv", index=False)
    num = [c for c in allrep.columns if c not in ("Status", "Repeat")]
    med = allrep.groupby("Students", as_index=False)[num[1:]].median()
    med.round(4).to_csv(OUT / "scalability_median.csv", index=False)
    log("Done.")


if __name__ == "__main__":
    main()
