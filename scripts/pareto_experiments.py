"""
Price of fairness curve for the three capacity scenarios (R = 1 km).

For each envy budget k, the most students that can be admitted when at most k
students have justified envy under distance priority; plus the fewest envious
students at the maximum number of admissions (LEXZON's N*). Each point reports
the best allocation found and the solver's proven bound.

Run from the project root:  python scripts/pareto_experiments.py
Output: results/pareto/pareto_points.csv (appended after every solve)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lexzon import baselines, data, fairness, pareto, solver  # noqa: E402
from lexzon.distance import haversine_matrix  # noqa: E402

OUT = ROOT / "results" / "pareto"
RADIUS = 1.0
BUDGETS = [0, 10, 25, 50, 100, 150, 200, 300]
LIMIT_K, LIMIT_MIN = 300, 900
WORKERS = 20


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def record(rows, **kw):
    rows.append(kw)
    pd.DataFrame(rows).to_csv(OUT / "pareto_points.csv", index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for s, key in enumerate(data.SCENARIO_KEYS, start=1):
        sc = f"S{s}"
        schools, students, _ = data.load_bundled(key)
        q = schools["Quota"].to_numpy(dtype=int)
        d = haversine_matrix(students, schools)
        n_star = solver.max_flow(d, q, RADIUS)
        model = pareto.EnvyModel(d, q, RADIUS)
        lex = solver.proposed(d, q, RADIUS).assignment
        da = baselines.deferred_acceptance(d, q, RADIUS).assignment

        def envy_of(a):
            return fairness.priority_violations(a, d, q, RADIUS)["envy_students"]

        for code, a in [("LEXZON", lex), ("DA", da)]:
            record(rows, Scenario=sc, Point=code, budget=np.nan, admitted=int((a >= 0).sum()), envy=envy_of(a),
                   bound=np.nan, optimal=True, runtime_s=0.0, N_star=n_star)

        log(f"{sc}: min envy at N* = {n_star}")
        r = model.min_envy(n_star, hint=lex, time_limit=LIMIT_MIN, workers=WORKERS)
        cap = int(r["objective"])
        record(rows, Scenario=sc, Point="min_envy_at_Nstar", budget=np.nan, admitted=int((r["assignment"] >= 0).sum()),
               envy=envy_of(r["assignment"]), bound=r["bound"], optimal=r["optimal"], runtime_s=r["runtime_s"],
               N_star=n_star)
        np.save(OUT / f"{sc}_min_envy_assignment.npy", r["assignment"])
        log(f"  envy {cap} (bound {r['bound']}, {r['status']})")

        hint = da
        for k in BUDGETS:
            if k >= cap:
                break
            r = model.max_admissions(k, hint=hint, time_limit=LIMIT_K, workers=WORKERS)
            a = r["assignment"]
            record(rows, Scenario=sc, Point="budget", budget=k, admitted=int((a >= 0).sum()), envy=envy_of(a),
                   bound=r["bound"], optimal=r["optimal"], runtime_s=r["runtime_s"], N_star=n_star)
            log(f"  k={k}: admitted {(a >= 0).sum()} (bound {r['bound']}, {r['status']}, {r['runtime_s']:.0f}s)")
            hint = a
    log("Done.")


if __name__ == "__main__":
    main()
