"""R2.5: when does the full hierarchy improve on maximum enrollment plus minimum total distance?

Controlled instance families vary two properties of the geography while the
number of students, the number of schools and the seats/student ratio are held
fixed:

  concentration  how demand is distributed: clustered (resampled from the
                 recorded locations), uniform over the bounding box, or a
                 mixture of both;
  connectivity   the eligibility radius, which sets the mean eligible degree.

For every instance the bottleneck stage (LEXZON) is compared with the
minimum-total benchmark B5 at identical enrollment, which isolates the value of
Stage 2.

Output: docs/results/revision/instance_families.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from zonasi import solver  # noqa: E402
from zonasi.distance import haversine_matrix  # noqa: E402

N_STUDENTS = 1209
RADII = (0.75, 1.0, 1.5)
CONCENTRATIONS = ("clustered", "mixed", "uniform")
SEEDS = (1, 2, 3, 4, 5)
JITTER_KM = 0.15
OUT = ROOT / "docs" / "results" / "revision"


def make_students(base, kind, rng):
    lat0, lat1 = base["Latitude"].min(), base["Latitude"].max()
    lon0, lon1 = base["Longitude"].min(), base["Longitude"].max()
    n_uniform = {"clustered": 0, "mixed": N_STUDENTS // 2, "uniform": N_STUDENTS}[kind]
    n_cluster = N_STUDENTS - n_uniform
    lat, lon = [], []
    if n_cluster:
        idx = rng.choice(len(base), size=n_cluster, replace=True)
        la = base["Latitude"].to_numpy()[idx] + rng.normal(0, JITTER_KM / 110.574, n_cluster)
        lo = base["Longitude"].to_numpy()[idx] + rng.normal(
            0, JITTER_KM / (111.320 * np.cos(np.radians(la.mean()))), n_cluster)
        lat.append(la); lon.append(lo)
    if n_uniform:
        lat.append(rng.uniform(lat0, lat1, n_uniform))
        lon.append(rng.uniform(lon0, lon1, n_uniform))
    return pd.DataFrame({"Student": [f"S{i+1}" for i in range(N_STUDENTS)],
                         "Latitude": np.concatenate(lat), "Longitude": np.concatenate(lon)})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    schools = pd.read_csv(ROOT / "dataset" / "skenario_2_kuota_lebih" / "sekolah.csv")
    base = pd.read_csv(ROOT / "dataset" / "skenario_2_kuota_lebih" / "dataset_siswa.csv")
    quotas = schools["Quota"].to_numpy(dtype=int)
    rows = []
    for kind in CONCENTRATIONS:
        for radius in RADII:
            for seed in SEEDS:
                rng = np.random.default_rng(1000 * SEEDS.index(seed) + int(radius * 100) + len(kind))
                students = make_students(base, kind, rng)
                dist = haversine_matrix(students, schools)
                lex = solver.proposed(dist, quotas, radius)
                b5 = solver.min_sum_within_radius(dist, quotas, radius)
                def stats(a):
                    ok = a >= 0
                    d = dist[np.nonzero(ok)[0], a[ok]] if ok.any() else np.array([0.0])
                    return int(ok.sum()), float(d.max()), float(d.sum())
                n_lex, max_lex, tot_lex = stats(lex.assignment)
                n_b5, max_b5, tot_b5 = stats(b5.assignment)
                pairs = int((dist <= radius + 1e-9).sum())
                rows.append({
                    "Concentration": kind, "Radius (km)": radius, "Seed": seed,
                    "Mean eligible degree": round(pairs / N_STUDENTS, 2),
                    "Students without an eligible school": int((~(dist <= radius + 1e-9).any(axis=1)).sum()),
                    "Assigned (LEXZON)": n_lex, "Assigned (B5)": n_b5,
                    "Max distance LEXZON (km)": round(max_lex, 4), "Max distance B5 (km)": round(max_b5, 4),
                    "Bottleneck gain (km)": round(max_b5 - max_lex, 4),
                    "Bottleneck gain (%)": round((max_b5 - max_lex) / max_b5 * 100, 2) if max_b5 else 0.0,
                    "Total distance LEXZON (km)": round(tot_lex, 1), "Total distance B5 (km)": round(tot_b5, 1),
                    "Total distance cost (%)": round((tot_lex - tot_b5) / tot_b5 * 100, 2) if tot_b5 else 0.0,
                })
                print(rows[-1], flush=True)
                pd.DataFrame(rows).to_csv(OUT / "instance_families.csv", index=False)
    df = pd.DataFrame(rows)
    summary = df.groupby(["Concentration", "Radius (km)"]).agg(
        degree=("Mean eligible degree", "mean"),
        gain_km=("Bottleneck gain (km)", "mean"),
        gain_pct=("Bottleneck gain (%)", "mean"),
        cost_pct=("Total distance cost (%)", "mean"),
        improved=("Bottleneck gain (km)", lambda s: int((s > 1e-6).sum()))).round(3)
    print(summary.to_string())
    summary.to_csv(OUT / "instance_families_summary.csv")
    print("written to", OUT)


if __name__ == "__main__":
    main()
