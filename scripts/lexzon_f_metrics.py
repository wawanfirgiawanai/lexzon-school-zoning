"""
Metrics of the LEXZON-F allocations saved by pareto_experiments.py (fewest
students with justified envy among allocations admitting N* students).

Run from the project root after pareto_experiments.py:  python scripts/lexzon_f_metrics.py
Output: docs/results/pareto/lexzon_f_metrics.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from zonasi import data, fairness, solver  # noqa: E402
from zonasi.distance import haversine_matrix  # noqa: E402
from zonasi.metrics import evaluate  # noqa: E402

OUT = ROOT / "docs" / "results" / "pareto"
RADIUS = 1.0


def main():
    rows = []
    for s, key in enumerate(data.SCENARIO_KEYS, start=1):
        sc = f"S{s}"
        schools, students, _ = data.load_bundled(key)
        q = schools["Quota"].to_numpy(dtype=int)
        d = haversine_matrix(students, schools)
        a = np.load(OUT / f"{sc}_min_envy_assignment.npy")
        alloc = solver.Allocation(a, 0.0)
        m = evaluate(alloc, d, q, students, RADIUS, solver.max_flow(d, q, RADIUS))
        v = fairness.priority_violations(a, d, q, RADIUS)
        rows.append({"Scenario": sc, **m, **v})
    pd.DataFrame(rows).round(4).to_csv(OUT / "lexzon_f_metrics.csv", index=False)
    print(pd.DataFrame(rows)[["Scenario", "admitted", "rejected", "max_km", "total_km", "envy_students",
                              "envy_unassigned", "envy_share_pct"]].to_string())


if __name__ == "__main__":
    main()
