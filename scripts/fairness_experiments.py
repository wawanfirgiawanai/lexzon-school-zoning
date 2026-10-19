"""
Distance-priority fairness and the constrained k-means baseline (B7) on the
three capacity scenarios: full data at R = 1 km plus 30 resampled instances.

Run from the project root:  python scripts/fairness_experiments.py
Outputs: results/fairness/*.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lexzon import baselines, data, fairness, solver  # noqa: E402
from lexzon.distance import haversine_matrix  # noqa: E402

OUT = ROOT / "results" / "fairness"
RADIUS, SEEDS = 1.0, 30
RUNS, FRACTION, SEED = 30, 0.8, 42


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def methods(student_df, school_df, dist, quotas, seed):
    return {
        "LEXZON": solver.proposed(dist, quotas, RADIUS),
        "B2": baselines.greedy_nearest_available(dist, quotas, RADIUS),
        "B3": baselines.greedy_random_order(dist, quotas, RADIUS, seed),
        "B4": baselines.deferred_acceptance(dist, quotas, RADIUS),
        "B5": solver.min_sum_within_radius(dist, quotas, RADIUS),
        "B6": solver.min_max_within_radius(dist, quotas, RADIUS),
        "B7": fairness.constrained_kmeans(student_df, school_df, dist, quotas, RADIUS),
    }


def row(code, alloc, dist, quotas):
    a = alloc.assignment
    return {"Code": code, "admitted": int((a >= 0).sum()), "rejected": int((a < 0).sum()),
            "runtime_s": alloc.runtime_s, **fairness.priority_violations(a, dist, quotas, RADIUS)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    full_rows, run_rows = [], []
    for k, key in enumerate(data.SCENARIO_KEYS, start=1):
        sc = f"S{k}"
        school_df, student_df, _ = data.load_bundled(key)
        quotas = school_df["Quota"].to_numpy(dtype=int)
        dist = haversine_matrix(student_df, school_df)
        n_star = solver.max_flow(dist, quotas, RADIUS)
        log(f"{sc}: full data")
        allocs = methods(student_df, school_df, dist, quotas, 0)
        for code, alloc in allocs.items():
            if code == "B3":
                reps = [row("B3", baselines.greedy_random_order(dist, quotas, RADIUS, s), dist, quotas)
                        for s in range(SEEDS)]
                r = {c: (np.mean([x[c] for x in reps]) if c != "Code" else "B3") for c in reps[0]}
            else:
                r = row(code, alloc, dist, quotas)
            full_rows.append({"Scenario": sc, "N*": n_star, **r})
        log(f"{sc}: resampling")
        for run in range(RUNS):
            rng = np.random.default_rng(SEED + run)
            schools, students = data.subsample(school_df, student_df, FRACTION, rng)
            q = schools["Quota"].to_numpy(dtype=int)
            d = haversine_matrix(students, schools)
            for code, alloc in methods(students, schools, d, q, SEED + run).items():
                run_rows.append({"Scenario": sc, "Run": run + 1, **row(code, alloc, d, q)})
    full = pd.DataFrame(full_rows)
    runs = pd.DataFrame(run_rows)
    full.round(4).to_csv(OUT / "fairness_full.csv", index=False)
    runs.round(4).to_csv(OUT / "fairness_runs.csv", index=False)

    tests = []
    for sc, g in runs.groupby("Scenario"):
        wide = g.pivot(index="Run", columns="Code", values="rejected")
        envy = g.pivot(index="Run", columns="Code", values="envy_unassigned")
        for code in ["B7"]:
            diff = wide[code] - wide["LEXZON"]
            res = wilcoxon(wide[code], wide["LEXZON"], alternative="greater") if (diff != 0).any() else None
            tests.append({"Scenario": sc, "Baseline": code,
                          "LEXZON rejected mean": wide["LEXZON"].mean(), "LEXZON rejected sd": wide["LEXZON"].std(),
                          "Baseline rejected mean": wide[code].mean(), "Baseline rejected sd": wide[code].std(),
                          "better": int((diff > 0).sum()), "tied": int((diff == 0).sum()), "worse": int((diff < 0).sum()),
                          "W": res.statistic if res else np.nan, "p": res.pvalue if res else np.nan})
        summ = g.groupby("Code")[["rejected", "envy_students", "envy_unassigned", "envy_share_pct"]].mean()
        log(f"{sc} resampled means:\n{summ.round(2)}")
        log(f"{sc} max envy_unassigned for LEXZON over runs: {envy['LEXZON'].max()}")
    pd.DataFrame(tests).to_csv(OUT / "b7_wilcoxon.csv", index=False)
    log("Done.")


if __name__ == "__main__":
    main()
