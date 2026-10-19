"""R2.3: how the LEXZON-F objective gap changes with the solver time limit.

For every capacity scenario the minimum-violation model is solved at increasing
time limits. The feasible count, the solver lower bound, the relative gap and the
status are recorded, so the manuscript can state which conclusions are invariant
to the unfinished optimization.

Usage:  python scripts/revision/r3_lexzonf_time.py [limits in seconds...]
Output: results/revision/lexzonf_time.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lexzon import solver  # noqa: E402
from lexzon.distance import haversine_matrix  # noqa: E402
from lexzon.pareto import EnvyModel  # noqa: E402

RADIUS = 1.0
WORKERS = 12
SCENARIOS = {"S1": "scenario_1_matched", "S2": "scenario_2_surplus", "S3": "scenario_3_insufficient"}
OUT = ROOT / "results" / "revision"


def main(limits):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for scen, folder in SCENARIOS.items():
        schools = pd.read_csv(ROOT / "dataset" / folder / "schools.csv")
        students = pd.read_csv(ROOT / "dataset" / folder / "students.csv")
        dist = haversine_matrix(students, schools)
        quotas = schools["Quota"].to_numpy(dtype=int)
        lex = solver.proposed(dist, quotas, RADIUS)
        n_star = int((lex.assignment >= 0).sum())
        model = EnvyModel(dist, quotas, RADIUS)
        for limit in limits:
            t0 = time.perf_counter()
            res = model.min_envy(n_star, hint=lex.assignment, time_limit=limit, workers=WORKERS)
            runtime = time.perf_counter() - t0
            feasible = int(model.envious(res["assignment"]).sum()) if res.get("assignment") is not None else None
            bound = res.get("bound")
            gap = None if (feasible is None or bound in (None, 0)) else (feasible - bound) / feasible * 100
            rows.append({"Scenario": scen, "N_star": n_star, "Time limit (s)": limit,
                         "Feasible violations": feasible, "Lower bound": bound,
                         "Relative gap (%)": None if gap is None else round(gap, 2),
                         "Proven optimal": bool(res.get("optimal")), "Status": res.get("status"),
                         "Runtime (s)": round(runtime, 1)})
            print(f"{scen} limit={limit:>5}s  feasible={feasible}  bound={bound}  "
                  f"gap={'' if gap is None else round(gap, 1)}%  status={res.get('status')}", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "lexzonf_time.csv", index=False)
    print("written", OUT / "lexzonf_time.csv")


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]] or [900, 1800, 3600]
    main(args)
