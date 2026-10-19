"""R2.4: the practical cost of the lexicographic hierarchy for individual students.

Enrollment gains are compared with the burden they impose. For every scenario
the LEXZON assignment is compared with the stable deferred-acceptance
assignment student by student:

  gained      admitted by LEXZON but not by deferred acceptance;
  moved       admitted by both, but to a different school;
  unchanged   admitted by both to the same school.

For the moved students the change in travel distance is reported as a
distribution, together with how many of them experience a priority violation.

Output: docs/results/revision/reassignment_cost.csv
        docs/results/revision/reassignment_extra_km.csv   (per-student values)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from zonasi import baselines, solver  # noqa: E402
from zonasi.distance import haversine_matrix  # noqa: E402
from zonasi.fairness import priority_violations  # noqa: E402
from zonasi.solver import UNASSIGNED  # noqa: E402

RADIUS = 1.0
SCENARIOS = {"S1": "skenario_1_kuota_pas", "S2": "skenario_2_kuota_lebih", "S3": "skenario_3_kuota_kurang"}
OUT = ROOT / "docs" / "results" / "revision"


def q(a, p):
    return round(float(np.percentile(a, p)), 3) if len(a) else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, per_student = [], []
    for scen, folder in SCENARIOS.items():
        schools = pd.read_csv(ROOT / "dataset" / folder / "sekolah.csv")
        students = pd.read_csv(ROOT / "dataset" / folder / "dataset_siswa.csv")
        dist = haversine_matrix(students, schools)
        quotas = schools["Quota"].to_numpy(dtype=int)
        lex = solver.proposed(dist, quotas, RADIUS).assignment
        da = baselines.deferred_acceptance(dist, quotas, RADIUS).assignment

        in_lex, in_da = lex != UNASSIGNED, da != UNASSIGNED
        gained = in_lex & ~in_da
        lost = in_da & ~in_lex
        both = in_lex & in_da
        moved = both & (lex != da)
        d_lex = np.where(in_lex, dist[np.arange(len(lex)), np.maximum(lex, 0)], np.nan)
        d_da = np.where(in_da, dist[np.arange(len(da)), np.maximum(da, 0)], np.nan)
        extra = (d_lex - d_da)[moved]

        envy = priority_violations(lex, dist, quotas, RADIUS)
        envious = np.zeros(len(lex), dtype=bool)
        # recompute the per-student indicator the same way the metric does
        cap = RADIUS + 1e-9
        used = np.bincount(lex[in_lex], minlength=len(quotas))
        tie = 1e-13
        skey = dist + np.arange(len(lex))[:, None] * tie
        worst = np.full(len(quotas), -np.inf)
        np.maximum.at(worst, lex[in_lex], skey[np.nonzero(in_lex)[0], lex[in_lex]])
        worst[used < quotas] = np.inf
        pkey = dist + np.arange(len(quotas))[None, :] * tie
        own = np.full(len(lex), np.inf)
        own[in_lex] = pkey[np.nonzero(in_lex)[0], lex[in_lex]]
        envious = ((dist <= cap) & (pkey < own[:, None]) & (skey < worst[None, :])).any(axis=1)

        rows.append({
            "Scenario": scen,
            "Assigned by LEXZON": int(in_lex.sum()), "Assigned by DA": int(in_da.sum()),
            "Gained by LEXZON": int(gained.sum()), "Lost by LEXZON": int(lost.sum()),
            "Same school": int((both & (lex == da)).sum()), "Moved to another school": int(moved.sum()),
            "Moved: mean extra km": round(float(extra.mean()), 3) if len(extra) else None,
            "Moved: median extra km": q(extra, 50), "Moved: p90 extra km": q(extra, 90),
            "Moved: max extra km": round(float(extra.max()), 3) if len(extra) else None,
            "Moved: farther (n)": int((extra > 1e-9).sum()), "Moved: closer (n)": int((extra < -1e-9).sum()),
            "Moved with violation (n)": int((moved & envious).sum()),
            "Gained with violation (n)": int((gained & envious).sum()),
            "Unchanged with violation (n)": int((both & (lex == da) & envious).sum()),
            "Total students with violations": int(envy["envy_students"]),
            "Mean distance LEXZON (km)": round(float(np.nanmean(d_lex)), 3),
            "Mean distance DA (km)": round(float(np.nanmean(d_da)), 3),
        })
        print(rows[-1], flush=True)
        for i in np.nonzero(moved)[0]:
            per_student.append({"Scenario": scen, "Student index": int(i),
                                "DA distance (km)": round(float(d_da[i]), 4),
                                "LEXZON distance (km)": round(float(d_lex[i]), 4),
                                "Extra distance (km)": round(float(d_lex[i] - d_da[i]), 4),
                                "Priority violation": bool(envious[i])})
    pd.DataFrame(rows).to_csv(OUT / "reassignment_cost.csv", index=False)
    pd.DataFrame(per_student).to_csv(OUT / "reassignment_extra_km.csv", index=False)
    print("written to", OUT)


if __name__ == "__main__":
    main()
