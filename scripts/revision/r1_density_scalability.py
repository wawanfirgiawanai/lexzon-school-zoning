"""R1.6: runtime as a function of eligibility density, not only of instance size.

The scalability instances of the manuscript keep an almost constant eligible
degree (about 5.4 schools per student). This script varies the degree while the
number of students is held fixed, so the scope of the runtime claim can be
stated explicitly. Density is varied in two independent ways:

  radius   the eligibility radius is swept, which changes the degree directly;
  schools  the school network is replicated with a small offset, which raises
           the degree without changing the radius.

Output: results/revision/density_scalability.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lexzon import baselines, data, solver  # noqa: E402
from lexzon.distance import haversine_matrix  # noqa: E402

SIZES = (5_000, 20_000)
RADII = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
REPLICAS = (1, 2, 3)          # copies of the 28-school network
SEED = 42
REPEATS = 3
OUT = ROOT / "results" / "revision"


def replicate(schools, copies):
    """Add shifted copies of the school network; quotas are split across copies."""
    if copies == 1:
        return schools.copy()
    out = []
    step = 0.0009                      # about 100 m, enough to separate the copies
    for c in range(copies):
        part = schools.copy()
        part["Latitude"] = part["Latitude"] + (c - (copies - 1) / 2) * step
        part["School"] = [f"{name} c{c + 1}" for name in part["School"]]
        part["Quota"] = np.maximum(1, (part["Quota"] / copies).round().astype(int))
        out.append(part)
    return pd.concat(out, ignore_index=True)


def timed(dist, quotas, radius):
    runs = [solver.proposed(dist, quotas, radius) for _ in range(REPEATS)]
    stages = {k: float(np.median([r.info["stage_times"][k] for r in runs])) for k in runs[0].info["stage_times"]}
    total = float(np.median([r.runtime_s for r in runs]))
    return runs[0], stages, total


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    schools0 = pd.read_csv(ROOT / "dataset" / "scenario_2_surplus" / "schools.csv")
    students0 = pd.read_csv(ROOT / "dataset" / "scenario_2_surplus" / "students.csv")
    rows = []
    for n in SIZES:
        rng = np.random.default_rng(SEED)
        schools, students = data.synthetic(schools0, students0, n, rng)
        for mode, values in (("radius", RADII), ("schools", REPLICAS)):
            for v in values:
                sch = schools if mode == "radius" else replicate(schools, v)
                radius = v if mode == "radius" else 1.0
                dist = haversine_matrix(students, sch)
                quotas = sch["Quota"].to_numpy(dtype=int)
                pairs = int((dist <= radius + 1e-9).sum())
                alloc, stages, total = timed(dist, quotas, radius)
                da = baselines.deferred_acceptance(dist, quotas, radius)
                rows.append({
                    "Students": n, "Varied": mode, "Setting": v,
                    "Schools": len(sch), "Radius (km)": radius,
                    "Eligible pairs": pairs, "Mean eligible degree": round(pairs / n, 2),
                    "Assigned": int((alloc.assignment >= 0).sum()),
                    **{k: round(v2, 4) for k, v2 in stages.items()},
                    "LEXZON total (s)": round(total, 3),
                    "Deferred acceptance (s)": round(da.runtime_s, 3),
                })
                print(rows[-1], flush=True)
                pd.DataFrame(rows).to_csv(OUT / "density_scalability.csv", index=False)
    print("written", OUT / "density_scalability.csv")


if __name__ == "__main__":
    main()
