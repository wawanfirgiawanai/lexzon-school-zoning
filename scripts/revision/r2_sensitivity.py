"""R2.2: sensitivity of the spatial diagnosis to the modelling assumptions.

Three assumptions are varied, one at a time, around the surplus scenario S2:

  capacity    every quota is scaled by a factor (published quotas may differ
              from the experimental configuration);
  access      a detour factor multiplies the great-circle distance, which
              approximates travel along a road network;
  priority    the ranking that defines stability is replaced by alternatives
              (random tie-breaking, a common lottery, and a school-specific
              lottery), which changes deferred acceptance and the violation
              count but not the LEXZON assignment.

Output: results/revision/sensitivity_capacity.csv
        results/revision/sensitivity_access.csv
        results/revision/sensitivity_priority.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lexzon import baselines, solver  # noqa: E402
from lexzon.data import scale_quotas  # noqa: E402
from lexzon.distance import haversine_matrix  # noqa: E402
from lexzon.fairness import priority_violations  # noqa: E402

RADIUS = 1.0
FOLDER = "scenario_2_surplus"
CAPACITY_FACTORS = (0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15)
DETOURS = (1.00, 1.10, 1.20, 1.30, 1.40)
SEEDS = (1, 2, 3, 4, 5)
OUT = ROOT / "results" / "revision"


def summarise(dist, quotas, radius, assignment):
    ok = assignment >= 0
    d = dist[np.nonzero(ok)[0], assignment[ok]] if ok.any() else np.array([0.0])
    v = priority_violations(assignment, dist, quotas, radius)
    return {"Assigned": int(ok.sum()), "Unassigned": int((~ok).sum()),
            "Max distance (km)": round(float(d.max()), 4), "Total distance (km)": round(float(d.sum()), 1),
            "Students with violations": int(v["envy_students"]),
            "Violation rate (%)": round(float(v["envy_share_pct"]), 2)}


def capacity_study(dist, schools, students):
    base = schools["Quota"].to_numpy(dtype=int)
    rows = []
    for f in CAPACITY_FACTORS:
        quotas = scale_quotas(base, int(round(base.sum() * f)))
        lex = solver.proposed(dist, quotas, RADIUS)
        da = baselines.deferred_acceptance(dist, quotas, RADIUS)
        rows.append({"Capacity factor": f, "Total seats": int(quotas.sum()),
                     **summarise(dist, quotas, RADIUS, lex.assignment),
                     "Deferred acceptance assigned": int((da.assignment >= 0).sum())})
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def access_study(dist, schools):
    quotas = schools["Quota"].to_numpy(dtype=int)
    rows = []
    for f in DETOURS:
        d = dist * f                       # travel distance approximated by a detour factor
        lex = solver.proposed(d, quotas, RADIUS)
        da = baselines.deferred_acceptance(d, quotas, RADIUS)
        rows.append({"Detour factor": f, "Eligible pairs": int((d <= RADIUS + 1e-9).sum()),
                     **summarise(d, quotas, RADIUS, lex.assignment),
                     "Deferred acceptance assigned": int((da.assignment >= 0).sum())})
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def priority_study(dist, schools):
    """Student preferences stay distance-based; the school priority rule changes.

    distance                the rule used in the manuscript
    distance + noise        imperfect distance measurement (100 m and 250 m)
    school-specific lottery priorities drawn independently at each school
    common lottery          one random order shared by all schools
    """
    quotas = schools["Quota"].to_numpy(dtype=int)
    n, m = dist.shape
    lex = solver.proposed(dist, quotas, RADIUS)
    scale = max(float(dist.max()), 1.0)
    rows = []
    variants = [("distance (manuscript)", (None,)),
                ("distance + 100 m noise", SEEDS),
                ("distance + 250 m noise", SEEDS),
                ("school-specific lottery", SEEDS),
                ("common lottery", SEEDS)]
    for name, seeds in variants:
        for seed in seeds:
            rng = np.random.default_rng(0 if seed is None else seed)
            if name.startswith("distance (") :
                priority = None
            elif "100 m" in name:
                priority = dist + rng.normal(0, 0.1, size=dist.shape)
            elif "250 m" in name:
                priority = dist + rng.normal(0, 0.25, size=dist.shape)
            elif name == "school-specific lottery":
                priority = rng.uniform(0, scale, size=dist.shape)
            else:
                priority = np.repeat(rng.permutation(n)[:, None], m, axis=1) * (scale / n)
            da = baselines.deferred_acceptance(dist, quotas, RADIUS, priority=priority)
            v_lex = priority_violations(lex.assignment, dist, quotas, RADIUS, priority=priority)
            v_da = priority_violations(da.assignment, dist, quotas, RADIUS, priority=priority)
            rows.append({"Priority rule": name, "Seed": seed,
                         "Deferred acceptance assigned": int((da.assignment >= 0).sum()),
                         "DA violations": int(v_da["envy_students"]),
                         "LEXZON assigned": int((lex.assignment >= 0).sum()),
                         "LEXZON students with violations": int(v_lex["envy_students"]),
                         "LEXZON violation rate (%)": round(float(v_lex["envy_share_pct"]), 2)})
            print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    schools = pd.read_csv(ROOT / "dataset" / FOLDER / "schools.csv")
    students = pd.read_csv(ROOT / "dataset" / FOLDER / "students.csv")
    dist = haversine_matrix(students, schools)
    capacity_study(dist, schools, students).to_csv(OUT / "sensitivity_capacity.csv", index=False)
    access_study(dist, schools).to_csv(OUT / "sensitivity_access.csv", index=False)
    priority_study(dist, schools).to_csv(OUT / "sensitivity_priority.csv", index=False)
    print("written to", OUT)


if __name__ == "__main__":
    main()
