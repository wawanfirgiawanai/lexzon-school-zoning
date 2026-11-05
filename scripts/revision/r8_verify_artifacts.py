"""R1.3: audit the published numbers from the released artifacts alone.

The script uses only artifacts/, that is the eligibility structure, the
two-sided rankings, the rounded distances and the capacities. No applicant
coordinates are required. It recomputes

  maximum enrollment N*          from the eligibility structure (max flow),
  deferred-acceptance enrollment from the published rankings,
  priority violations            of the published LEXZON assignment,

and compares them with the values recorded in instances.json. It then audits the
distance objectives from objective_certificates.csv, distance_bins_S*.csv and
zstar_certificate.csv: the released group subtotals must add up to the certified
total and the released counts to the certified number of assignments, and the
threshold certificate must hold on the rounded pair table, that is a flow of N*
at the stated threshold and a smaller flow one 50 m bin below it.

Output: artifacts/verification.csv
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

ART = ROOT / "artifacts"
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
    print()
    print("distance objectives:", "reproduce exactly" if audit_distances() else "MISMATCH")


def audit_distances():
    """Rebuild the certified totals from the released groups and check the threshold."""
    cert = pd.read_csv(ART / "objective_certificates.csv")
    zst = pd.read_csv(ART / "zstar_certificate.csv").set_index("Scenario")
    good = True
    for scen in sorted(cert["Scenario"].unique()):
        bins = pd.read_csv(ART / f"distance_bins_{scen}.csv")
        agg = bins.groupby("Method").agg(pairs=("Pairs", "sum"), total=("Exact subtotal (km)", "sum"))
        for _, row in cert[cert["Scenario"] == scen].iterrows():
            a = agg.loc[row["Method"]]
            same = int(a["pairs"]) == int(row["Assigned"]) and                 abs(a["total"] - row["Total distance (km)"]) < 5e-4
            good &= same
            print(f"  {scen} {row['Method']:22s} assigned {int(a['pairs']):4d} "
                  f"total {a['total']:9.4f} km  certified {row['Total distance (km)']:9.4f} km  "
                  f"{'ok' if same else 'MISMATCH'}")

        # the threshold certificate, recomputed from the rounded pair table
        dist, rank, _, quotas, _, _, _ = load(scen)
        eligible = np.isfinite(rank)
        z = zst.loc[scen]
        n_star = int(z["N*"])
        t = float(z["Smallest rounded threshold with flow N* (km)"])
        at = int(solver.max_flow(np.where(eligible & (dist <= t + 1e-9), 1.0, FAR), quotas, 1.0))
        below = int(solver.max_flow(np.where(eligible & (dist <= t - 0.05 + 1e-9), 1.0, FAR),
                                    quotas, 1.0))
        holds = at == n_star == int(z["Flow at that threshold"]) and below < n_star
        good &= holds
        print(f"  {scen} threshold {t:.2f} km: flow {at} (N* = {n_star}), one bin below {below} "
              f"{'ok' if holds else 'MISMATCH'}")
    return bool(good)


if __name__ == "__main__":
    main()
