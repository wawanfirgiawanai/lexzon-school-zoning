"""R1.3: audit the published numbers from the released artifacts alone.

The script uses only docs/artifacts/, that is the eligibility structure, the
two-sided rankings, the rounded distances and the capacities. No applicant
coordinates are required. It recomputes

  maximum enrollment N*          from the eligibility structure (max flow),
  deferred-acceptance enrollment from the published rankings,
  priority violations            of the published LEXZON assignment,

and compares them with the values recorded in instances.json.

Output: docs/artifacts/verification.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lexzon import baselines, solver  # noqa: E402
from lexzon.fairness import priority_violations  # noqa: E402

ART = ROOT / "docs" / "artifacts"
RADIUS = 1.0
FAR = 10.0          # stand-in distance for pairs outside the radius


def load(scen):
    caps = pd.read_csv(ART / f"capacities_{scen}.csv")
    pairs = pd.read_csv(ART / f"eligible_pairs_{scen}.csv.gz")
    assign = pd.read_csv(ART / f"assignments_{scen}.csv")
    schools = list(caps["School"])
    students = list(assign["Student"])
    s_index = {s: i for i, s in enumerate(students)}
    j_index = {s: j for j, s in enumerate(schools)}
    dist = np.full((len(students), len(schools)), FAR)
    rank = np.full((len(students), len(schools)), np.nan)
    ii = np.array([s_index[s] for s in pairs["Student"]])
    jj = np.array([j_index[s] for s in pairs["School"]])
    rank[ii, jj] = pairs["Student rank of school"].to_numpy()
    prio = np.full((len(students), len(schools)), np.nan)
    prio[ii, jj] = pairs["School rank of student"].to_numpy()
    dcol = [c for c in pairs.columns if c.startswith("Distance")]
    if dcol:
        dist[ii, jj] = pairs[dcol[0]].to_numpy()
    quotas = caps["Capacity"].to_numpy(dtype=int)
    lex = np.array([j_index.get(s, -1) if isinstance(s, str) and s else -1 for s in assign["LEXZON"]])
    return dist, rank, prio, quotas, lex, students, schools


def main():
    summary = json.loads((ART / "instances.json").read_text(encoding="utf-8"))
    rows = []
    for entry in summary["scenarios"]:
        scen = entry["scenario"]
        dist, rank, prio, quotas, lex, students, schools = load(scen)
        eligible = np.isfinite(rank)
        # Enrollment, deferred acceptance and violations depend on the order structure only,
        # so the exact published rankings are used; the rounded distances are not needed.
        BIG = 1e6
        pref_rank = np.where(eligible, np.nan_to_num(rank, nan=BIG), BIG)
        prio_rank = np.where(eligible, np.nan_to_num(prio, nan=BIG), BIG)
        cap_rank = float(np.nanmax(rank[eligible])) + 0.5
        n_star = solver.max_flow(np.where(eligible, 1.0, FAR), quotas, 1.0)
        da = baselines.deferred_acceptance(pref_rank, quotas, cap_rank, priority=prio_rank)
        viol = priority_violations(lex, pref_rank, quotas, cap_rank, priority=prio_rank)
        rows.append({
            "Scenario": scen,
            "N* published": entry["lexzon"]["n_star"], "N* recomputed": int(n_star),
            "DA published": entry["deferred_acceptance_assigned"],
            "DA recomputed": int((da.assignment >= 0).sum()),
            "Violations published": entry["lexzon_students_with_violations"],
            "Violations recomputed": int(viol["envy_students"]),
            "Artifacts used": "eligibility and rankings", "Distance mode": summary["privacy_mode"],
        })
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ART / "verification.csv", index=False)
    ok = (df["N* published"] == df["N* recomputed"]).all() and (df["DA published"] == df["DA recomputed"]).all()
    print("enrollment certificates reproduce exactly:", bool(ok))
    print("written", ART / "verification.csv")


if __name__ == "__main__":
    main()
