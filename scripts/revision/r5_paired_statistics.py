"""R1.5: the unit of statistical inference in the subsampling experiment.

The 30 subsamples per scenario are drawn from one applicant pool and overlap,
so the paired outcomes are dependent and the p-values are conditional on this
resampling design. This script reports the paired differences themselves, the
tie handling, and the expected overlap between subsamples, so the manuscript
can present the evidence without implying independent replication.

Output: docs/results/revision/paired_differences.csv
        docs/results/revision/subsample_overlap.csv
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RES = ROOT / "docs" / "results"
OUT = RES / "revision"
SCENARIOS = ("S1", "S2", "S3")
COMPARATORS = ("B2", "B3", "B4")
FRACTION = 0.8
N_STUDENTS = 1209
SEED = 42
RUNS = 30


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for scen in SCENARIOS:
        runs = pd.read_csv(RES / "scenarios" / f"{scen}_stat_runs.csv")
        wide = runs.pivot_table(index="Run", columns="Code", values="rejected")
        for code in COMPARATORS:
            if code not in wide:
                continue
            diff = (wide[code] - wide["P"]).to_numpy(dtype=float)   # positive: LEXZON leaves fewer unassigned
            ties = int((diff == 0).sum())
            nonzero = diff[diff != 0]
            stat, p = (np.nan, np.nan) if len(nonzero) == 0 else wilcoxon(nonzero, alternative="greater")
            rows.append({
                "Scenario": scen, "Comparator": code, "Runs": len(diff),
                "Mean paired difference": round(float(diff.mean()), 2),
                "Median paired difference": round(float(np.median(diff)), 1),
                "SD of paired difference": round(float(diff.std(ddof=1)), 2),
                "Min": int(diff.min()), "Max": int(diff.max()),
                "Ties (zero difference)": ties,
                "Pairs used by the test": len(nonzero),
                "Wilcoxon W": None if np.isnan(stat) else float(stat),
                "p-value (one-sided, conditional)": None if np.isnan(p) else float(p),
            })
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "paired_differences.csv", index=False)

    # expected overlap between two 80% subsamples drawn without replacement
    rng = np.random.default_rng(SEED)
    size = int(round(FRACTION * N_STUDENTS))
    samples = [rng.choice(N_STUDENTS, size=size, replace=False) for _ in range(RUNS)]
    overlaps = [len(np.intersect1d(a, b)) / size * 100 for a, b in combinations(samples, 2)]
    pd.DataFrame([{
        "Subsamples": RUNS, "Subsample size": size, "Fraction": FRACTION,
        "Mean pairwise overlap (%)": round(float(np.mean(overlaps)), 1),
        "Min overlap (%)": round(float(np.min(overlaps)), 1),
        "Max overlap (%)": round(float(np.max(overlaps)), 1),
        "Expected overlap (%)": round(FRACTION * 100, 1),
    }]).to_csv(OUT / "subsample_overlap.csv", index=False)
    print("mean pairwise overlap %.1f%%" % np.mean(overlaps))
    print("written to", OUT)


if __name__ == "__main__":
    main()
