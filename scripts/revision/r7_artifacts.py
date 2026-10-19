"""R1.3: machine-readable artifacts for independent verification.

The bundle describes every experimental instance and every reported output
without releasing applicant coordinates. Three privacy modes are available:

  structure  eligibility and the rank order of schools per student only.
             Enrollment, deferred acceptance and the violation counts are
             reproducible exactly; distance objectives are not.
  rounded    as above plus distances rounded to ROUNDING km (default 0.05).
             All results reproduce approximately.
  exact      as above with full-precision distances. Every reported number
             reproduces exactly. Distances to three or more schools allow the
             home location to be reconstructed, so this mode releases the
             geometry of the demand even though identities stay pseudonymous.

Usage:  python scripts/revision/r7_artifacts.py [structure|rounded|exact]
Output: docs/artifacts/
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lexzon import baselines, solver  # noqa: E402
from lexzon.distance import haversine_matrix  # noqa: E402
from lexzon.fairness import constrained_kmeans, priority_violations  # noqa: E402

RADIUS = 1.0
ROUNDING = 0.05
SEEDS_B3 = list(range(30))
SUBSAMPLE_SEED, SUBSAMPLE_RUNS, SUBSAMPLE_FRACTION = 42, 30, 0.8
SCENARIOS = {"S1": "scenario_1_matched", "S2": "scenario_2_surplus", "S3": "scenario_3_insufficient"}
OUT = ROOT / "docs" / "artifacts"


def school_codes(n):
    return [f"ES {i + 1:02d}" for i in range(n)]


def export_scenario(scen, folder, mode):
    schools = pd.read_csv(ROOT / "dataset" / folder / "schools.csv")
    students = pd.read_csv(ROOT / "dataset" / folder / "students.csv")
    codes = school_codes(len(schools))
    dist = haversine_matrix(students, schools)
    quotas = schools["Quota"].to_numpy(dtype=int)
    n, m = dist.shape

    pd.DataFrame({"School": codes, "Capacity": quotas}).to_csv(OUT / f"capacities_{scen}.csv", index=False)

    rows, cols = np.nonzero(dist <= RADIUS + 1e-9)
    order = np.argsort(dist[rows, cols], kind="stable")
    rows, cols = rows[order], cols[order]
    rank = np.zeros(len(rows), dtype=int)
    for i in np.unique(rows):                       # rank of each eligible school for this student
        sel = rows == i
        rank[sel] = np.argsort(np.argsort(dist[i, cols[sel]], kind="stable")) + 1
    pairs = pd.DataFrame({"Student": [f"A{r + 1:04d}" for r in rows],
                          "School": [codes[c] for c in cols],
                          "Student rank of school": rank,
                          "School rank of student": 0})
    for c in np.unique(cols):                       # priority order at each school (distance, index tie-break)
        sel = cols == c
        d = dist[rows[sel], c] + rows[sel] * 1e-13
        pairs.loc[sel, "School rank of student"] = np.argsort(np.argsort(d)) + 1
    if mode == "exact":
        pairs["Distance (km)"] = np.round(dist[rows, cols], 6)
    elif mode == "rounded":
        pairs["Distance (km)"] = np.round(dist[rows, cols] / ROUNDING) * ROUNDING
    pairs.to_csv(OUT / f"eligible_pairs_{scen}.csv.gz", index=False, compression="gzip")

    lex = solver.proposed(dist, quotas, RADIUS)
    da = baselines.deferred_acceptance(dist, quotas, RADIUS)
    b7 = constrained_kmeans(students, schools, dist, quotas, RADIUS)
    out = pd.DataFrame({"Student": [f"A{i + 1:04d}" for i in range(n)]})
    for name, a in (("LEXZON", lex.assignment), ("DeferredAcceptance", da.assignment),
                    ("B2_greedy", baselines.greedy_nearest_available(dist, quotas, RADIUS).assignment),
                    ("B5_min_total", solver.min_sum_within_radius(dist, quotas, RADIUS).assignment),
                    ("B6_min_max", solver.min_max_within_radius(dist, quotas, RADIUS).assignment),
                    ("B7_constrained_kmeans", b7.assignment)):
        out[name] = [codes[j] if j >= 0 else "" for j in a]
    out.to_csv(OUT / f"assignments_{scen}.csv", index=False)

    v = priority_violations(lex.assignment, dist, quotas, RADIUS)
    return {
        "scenario": scen, "students": int(n), "schools": int(m), "seats": int(quotas.sum()),
        "eligible_pairs": int(len(rows)), "radius_km": RADIUS,
        "lexzon": {"assigned": int((lex.assignment >= 0).sum()), "status": lex.status,
                   "n_star": int(lex.info.get("n_star", 0)),
                   "z_star_km": round(float(lex.info.get("z_star", float("nan"))), 6)
                   if lex.info.get("z_star") is not None else None,
                   "stage_times_s": {k: round(x, 4) for k, x in lex.info.get("stage_times", {}).items()}},
        "deferred_acceptance_assigned": int((da.assignment >= 0).sum()),
        "lexzon_students_with_violations": int(v["envy_students"]),
        "b7_iterations": int(b7.info["iterations"]), "b7_removed_beyond_radius": int(b7.info["beyond_radius_removed"]),
    }


def subsample_membership():
    rng_rows = []
    for scen in SCENARIOS:
        for run in range(SUBSAMPLE_RUNS):
            rng = np.random.default_rng(SUBSAMPLE_SEED + run)
            idx = rng.choice(1209, size=int(round(SUBSAMPLE_FRACTION * 1209)), replace=False)
            rng_rows.append({"Scenario": scen, "Run": run + 1,
                             "Seed": SUBSAMPLE_SEED + run, "Size": len(idx),
                             "Members": " ".join(f"A{i + 1:04d}" for i in sorted(idx))})
    pd.DataFrame(rng_rows).to_csv(OUT / "subsample_membership.csv.gz", index=False, compression="gzip")


def main(mode):
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"privacy_mode": mode, "radius_km": RADIUS,
               "distance": "Haversine, Earth radius 6371 km",
               "tie_breaking": "lower student index, lower school index (1e-13 key offset)",
               "random_order_greedy_seeds": SEEDS_B3,
               "subsampling": {"runs": SUBSAMPLE_RUNS, "fraction": SUBSAMPLE_FRACTION,
                               "seed_rule": "42 + repetition index",
                               "capacity_rounding": "largest remainder, minimum one seat per school"},
               "synthetic_scalability": {"resampling": "with replacement from the recorded locations",
                                         "jitter_km": 0.15, "schools": "unchanged (28)",
                                         "quota_rule": "seats/student ratio preserved, largest remainder",
                                         "seed": 42, "repetitions": 3, "statistic": "median"},
               "constrained_kmeans_B7": {"init": "centres at school locations (planar km)",
                                         "assignment_step": "capacitated transportation LP on squared distance",
                                         "capacity": "cluster size <= school quota",
                                         "centre_update": "mean of members", "max_iter": 50,
                                         "stop": "assignment repeats", "restarts": 1,
                                         "post_processing": "members beyond R are left unassigned, not reassigned"},
               "scenarios": []}
    for scen, folder in SCENARIOS.items():
        summary["scenarios"].append(export_scenario(scen, folder, mode))
        print("exported", scen, flush=True)
    subsample_membership()
    (OUT / "instances.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("written to", OUT)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "rounded")
