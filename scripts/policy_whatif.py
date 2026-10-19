"""
Where are students without a school, and what does one extra class change?

For each capacity scenario: (1) students without a school counted by their
nearest school, and (2) the proposed method re-run after adding one class of
28 seats to each school in turn.

Run from the project root:  python scripts/policy_whatif.py
Outputs: results/scenarios/unassigned_by_nearest_school.csv,
         results/scenarios/whatif_extra_class.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lexzon import data, solver  # noqa: E402
from lexzon.distance import haversine_matrix  # noqa: E402

OUT = ROOT / "results" / "scenarios"
RADIUS, CLASS_SIZE = 1.0, 28


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pressure, whatif = [], []
    for k, key in enumerate(data.SCENARIO_KEYS, start=1):
        code = f"S{k}"
        schools, students, _ = data.load_bundled(key)
        quotas = schools["Quota"].to_numpy(dtype=int)
        dist = haversine_matrix(students, schools)
        base = solver.proposed(dist, quotas, RADIUS)
        rejected = np.nonzero(base.assignment < 0)[0]
        counts = np.bincount(dist[rejected].argmin(axis=1), minlength=len(quotas))
        for j, name in enumerate(schools["School"]):
            pressure.append({"Scenario": code, "School": name, "Unassigned nearest to school": int(counts[j])})
            extra = quotas.copy()
            extra[j] += CLASS_SIZE
            alloc = solver.proposed(dist, extra, RADIUS)
            whatif.append({"Scenario": code, "School with extra class": name,
                           "Unassigned before": int(len(rejected)),
                           "Unassigned after": int((alloc.assignment < 0).sum()),
                           "z* after (km)": round(alloc.info["z_star"], 4)})
        print(code, "done", flush=True)
    pd.DataFrame(pressure).to_csv(OUT / "unassigned_by_nearest_school.csv", index=False)
    pd.DataFrame(whatif).to_csv(OUT / "whatif_extra_class.csv", index=False)


if __name__ == "__main__":
    main()
