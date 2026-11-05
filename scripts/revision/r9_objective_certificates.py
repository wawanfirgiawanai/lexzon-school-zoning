"""R1.3 (second round): exact objective-value certificates that disclose no residence.

The first artifact release rounded every distance to 50 m, so maximum enrollment,
deferred-acceptance enrollment and the violation counts reproduced exactly (they
depend only on the eligibility structure and the two rankings) while the distance
objectives did not. This script adds what is needed to audit those as well.

  objective_certificates.csv  the exact assigned count, maximum distance and
                              total distance of every published assignment;
  distance_bins_S*.csv        for each method, the exact subtotal of the
                              assigned distances inside consecutive distance
                              groups, so the total is recovered by addition;
  zstar_certificate.csv       the smallest rounded threshold whose eligibility
                              subgraph still carries a flow of N*, and the flow
                              one 50 m bin below it, both reproducible from the
                              published pair table alone.

Privacy. A released group never holds fewer than MIN_GROUP assigned pairs:
consecutive 50 m bins are merged until that count is reached. No individual
distance is released, so no released number can be attached to a single
applicant, and the finest published locator stays the 50 m bin of the pair table.

Output: artifacts/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lexzon import solver  # noqa: E402
from lexzon.distance import haversine_matrix  # noqa: E402

ART = ROOT / "artifacts"
DATA = ROOT / "dataset"
FOLDER = {"S1": "scenario_1_matched", "S2": "scenario_2_surplus", "S3": "scenario_3_insufficient"}
RADIUS = 1.0
BIN = 0.05          # the rounding step of the published pair table
MIN_GROUP = 5       # smallest number of assigned pairs behind a released subtotal


def school_codes(n):
    return [f"ES {i + 1:02d}" for i in range(n)]


def groups(distances):
    """Consecutive 50 m bins merged until each group holds MIN_GROUP pairs."""
    edges = np.round(distances / BIN).astype(int)
    out, start, count, total = [], edges[0], 0, 0.0
    for e, d in zip(edges, distances):
        count += 1
        total += float(d)
        if count >= MIN_GROUP and e != edges[-1]:
            out.append([start * BIN, (e + 1) * BIN, count, total])
            start, count, total = e + 1, 0, 0.0
    if count:                                   # the tail joins the preceding group
        if out and count < MIN_GROUP:
            out[-1][1] = (edges[-1] + 1) * BIN
            out[-1][2] += count
            out[-1][3] += total
        else:
            out.append([start * BIN, (edges[-1] + 1) * BIN, count, total])
    return out


def main():
    certs, zrows = [], []
    for scen, folder in FOLDER.items():
        schools = pd.read_csv(DATA / folder / "schools.csv")
        students = pd.read_csv(DATA / folder / "students.csv")
        dist = haversine_matrix(students, schools)
        quotas = schools["Quota"].to_numpy(int)
        index = {c: j for j, c in enumerate(school_codes(len(schools)))}

        published = pd.read_csv(ART / f"assignments_{scen}.csv")
        assert len(published) == len(students), (scen, len(published), len(students))

        rows = []
        for method in [c for c in published.columns if c != "Student"]:
            col = published[method].fillna("")
            assigned = np.array([index[c] if c else -1 for c in col])
            ok = assigned >= 0
            d = np.sort(dist[np.arange(len(assigned))[ok], assigned[ok]])
            for lo, hi, n, total in groups(d):
                rows.append({"Method": method, "From (km)": round(lo, 2), "To (km)": round(hi, 2),
                             "Pairs": n, "Exact subtotal (km)": round(total, 6)})
            certs.append({"Scenario": scen, "Method": method, "Assigned": int(ok.sum()),
                          "Maximum distance (km)": round(float(d.max()), 6),
                          "Total distance (km)": round(float(d.sum()), 4)})
        pd.DataFrame(rows).to_csv(ART / f"distance_bins_{scen}.csv", index=False)

        # z* bracket, reproducible from the rounded pair table alone
        n_star = int(solver.max_flow(dist, quotas, RADIUS))
        rounded = np.where(dist <= RADIUS, np.round(dist / BIN) * BIN, np.inf)
        flows = {round(float(t), 2): int(solver.max_flow(np.where(rounded <= t + 1e-9, dist, 1e3),
                                                         quotas, 1e2))
                 for t in np.arange(BIN, RADIUS + BIN / 2, BIN)}
        attained = min(t for t, f in flows.items() if f == n_star)
        zrows.append({"Scenario": scen, "N*": n_star,
                      "Smallest rounded threshold with flow N* (km)": attained,
                      "Flow at that threshold": flows[attained],
                      "Flow one bin below": flows[round(attained - BIN, 2)],
                      "Exact z* (km)": round(
                          float(solver.proposed(dist, quotas, RADIUS).info["z_star"]), 6)})

    cert = pd.DataFrame(certs)
    cert.to_csv(ART / "objective_certificates.csv", index=False)
    pd.DataFrame(zrows).to_csv(ART / "zstar_certificate.csv", index=False)
    print(cert.to_string(index=False))
    print()
    print(pd.DataFrame(zrows).to_string(index=False))
    smallest = min(pd.read_csv(ART / f"distance_bins_{s}.csv")["Pairs"].min() for s in FOLDER)
    print("\nsmallest released group:", smallest, "pairs")


if __name__ == "__main__":
    main()
