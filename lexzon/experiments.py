import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from . import baselines, data, solver
from .distance import euclidean_degree_matrix, haversine_matrix
from .metrics import evaluate

PROPOSED_CODE = "P"
PROPOSED_NAME = "Proposed"
METHOD_NAMES = {PROPOSED_CODE: PROPOSED_NAME, **{k: v[0] for k, v in baselines.METHODS.items()}}


def _cap(radius):
    return np.inf if radius is None else radius


def run_methods(dist_km, quotas, radius, seeds):
    """Every method on the same instance and the same radius.
    Returns (allocations by code, list of B3 allocations, max admissible)."""
    quotas = np.asarray(quotas, dtype=int)
    allocations = {PROPOSED_CODE: solver.proposed(dist_km, quotas, radius)}
    for code, (_, fn) in baselines.METHODS.items():
        if fn is not None:
            allocations[code] = fn(dist_km, quotas, radius)
    b3 = [baselines.greedy_random_order(dist_km, quotas, radius, seed) for seed in range(seeds)]
    return allocations, b3, solver.max_flow(dist_km, quotas, _cap(radius))


def comparison_table(dist_km, school_df, student_df, radius, seeds):
    quotas = school_df["Quota"].to_numpy(dtype=int)
    allocations, b3, max_adm = run_methods(dist_km, quotas, radius, seeds)
    rows = []
    for code in [PROPOSED_CODE, "B1", "B2", "B3", "B4", "B5", "B6"]:
        if code == "B3":
            evals = [evaluate(a, dist_km, quotas, student_df, radius, max_adm) for a in b3]
            mean = {k: float(np.nanmean([e[k] for e in evals])) for k in evals[0]}
            rows.append({"Code": code, "Method": f"{METHOD_NAMES[code]} (mean of {seeds} seeds)", **mean,
                         "sd_rejected": float(np.std([e["rejected"] for e in evals]))})
        else:
            rows.append({"Code": code, "Method": METHOD_NAMES[code],
                         **evaluate(allocations[code], dist_km, quotas, student_df, radius, max_adm)})
    return pd.DataFrame(rows), allocations, max_adm


def ablation(school_df, student_df, radius):
    """A0 -> A3, adding one component at a time. Every allocation is measured in
    Haversine km, including A0 whose solver used raw-degree distance."""
    quotas = school_df["Quota"].to_numpy(dtype=int)
    dist_km = haversine_matrix(student_df, school_df)
    dist_deg = euclidean_degree_matrix(student_df, school_df)
    flow_free = solver.max_flow(dist_km, quotas, np.inf)
    flow_r = solver.max_flow(dist_km, quotas, _cap(radius))
    configs = [
        ("A0", "Original thesis model (Euclidean degrees, everyone must be placed)",
         solver.forced_full_assignment(dist_deg, quotas), None, flow_free),
        ("A1", "+ Haversine distance (km)", solver.forced_full_assignment(dist_km, quotas), None, flow_free),
        # The radius needs rejection to be allowed (some students have no school within R), so both enter
        # together: max admissions within R, then min total distance.
        ("A2", "+ zoning radius R with rejection allowed (max admissions, then min total)",
         solver.min_sum_within_radius(dist_km, quotas, radius), radius, flow_r),
        ("A3", "+ min-max stage (proposed method)", solver.proposed(dist_km, quotas, radius), radius, flow_r),
    ]
    rows = []
    cap = _cap(radius)
    for code, name, alloc, rule_radius, max_adm in configs:
        if not alloc.feasible:
            rows.append({"Code": code, "Configuration": name, "Status": "Infeasible (seats < students)"})
            continue
        m = evaluate(alloc, dist_km, quotas, student_df, rule_radius, max_adm)
        admitted = alloc.assignment >= 0
        beyond = int((dist_km[np.nonzero(admitted)[0], alloc.assignment[admitted]] > cap).sum())
        rows.append({"Code": code, "Configuration": name, "Status": alloc.status, **m, "beyond_radius": beyond})
    return pd.DataFrame(rows)


SWEEP_METHODS = [PROPOSED_CODE, "B5", "B2", "B4"]


def radius_sweep(dist_km, school_df, student_df, radii, progress=None):
    quotas = school_df["Quota"].to_numpy(dtype=int)
    fns = {PROPOSED_CODE: solver.proposed, "B5": solver.min_sum_within_radius,
           "B2": baselines.greedy_nearest_available, "B4": baselines.deferred_acceptance}
    rows = []
    total = len(radii) * len(fns)
    step = 0
    for r in radii:
        max_adm = solver.max_flow(dist_km, quotas, _cap(r))
        for code in SWEEP_METHODS:
            m = evaluate(fns[code](dist_km, quotas, r), dist_km, quotas, student_df, r, max_adm)
            rows.append({"Radius (km)": np.inf if r is None else r, "Code": code, "Method": METHOD_NAMES[code], **m})
            step += 1
            if progress:
                progress(step / total)
    return pd.DataFrame(rows)


def scenario_study(radius, seeds, progress=None):
    rows = []
    keys = data.SCENARIO_KEYS
    for k, key in enumerate(keys, start=1):
        school_df, student_df, _ = data.load_bundled(key)
        quotas = school_df["Quota"].to_numpy(dtype=int)
        dist_km = haversine_matrix(student_df, school_df)
        table, _, _ = comparison_table(dist_km, school_df, student_df, radius, seeds)
        original_ok = solver.forced_full_assignment(dist_km, quotas).feasible
        label, description = key.split(" — ")
        for _, r in table.iterrows():
            rows.append({"Scenario": label, "Description": description, "Students": len(student_df),
                         "Total seats": int(quotas.sum()), "Original model feasible": "Yes" if original_ok else "No",
                         **r.to_dict()})
        if progress:
            progress(k / len(keys))
    return pd.DataFrame(rows)


TEST_METHODS = ["B2", "B3", "B4", "B5", "B6"]


def resampling_test(school_df, student_df, radius, runs, fraction, seed, progress=None):
    """Repeated random subsamples (quotas scaled by the same fraction). The
    proposed method is paired with each feasible baseline run-by-run and the
    number of students without a school is compared with a one-sided Wilcoxon
    signed-rank test (H1: the baseline leaves more students without a school)."""
    records = []
    for r in range(runs):
        rng = np.random.default_rng(seed + r)
        schools, students = data.subsample(school_df, student_df, fraction, rng)
        quotas = schools["Quota"].to_numpy(dtype=int)
        dist_km = haversine_matrix(students, schools)
        max_adm = solver.max_flow(dist_km, quotas, _cap(radius))
        allocs = {
            PROPOSED_CODE: solver.proposed(dist_km, quotas, radius),
            "B2": baselines.greedy_nearest_available(dist_km, quotas, radius),
            "B3": baselines.greedy_random_order(dist_km, quotas, radius, seed + r),
            "B4": baselines.deferred_acceptance(dist_km, quotas, radius),
            "B5": solver.min_sum_within_radius(dist_km, quotas, radius),
            "B6": solver.min_max_within_radius(dist_km, quotas, radius),
        }
        for code, alloc in allocs.items():
            m = evaluate(alloc, dist_km, quotas, students, radius, max_adm)
            records.append({"Run": r + 1, "Code": code, "Method": METHOD_NAMES[code],
                            "rejected": m["rejected"], "max_km": m["max_km"], "total_km": m["total_km"]})
        if progress:
            progress((r + 1) / runs)
    runs_df = pd.DataFrame(records)

    wide = runs_df.pivot(index="Run", columns="Code", values="rejected")
    p = wide[PROPOSED_CODE]
    summary = []
    for code in TEST_METHODS:
        b = wide[code]
        diff = b - p
        if np.all(diff == 0):
            stat, pvalue, note = np.nan, np.nan, "Identical in every run"
        else:
            res = wilcoxon(b, p, alternative="greater", zero_method="wilcox")
            stat, pvalue = float(res.statistic), float(res.pvalue)
            note = "Significant (p < 0.05)" if pvalue < 0.05 else "Not significant"
        summary.append({
            "Baseline": f"{code} — {METHOD_NAMES[code]}",
            "✕ Proposed (mean ± sd)": f"{p.mean():.1f} ± {p.std(ddof=1):.1f}",
            "✕ Baseline (mean ± sd)": f"{b.mean():.1f} ± {b.std(ddof=1):.1f}",
            "Mean difference": float(diff.mean()),
            "Proposed better (runs)": int((diff > 0).sum()),
            "Tied (runs)": int((diff == 0).sum()),
            "Proposed worse (runs)": int((diff < 0).sum()),
            "W statistic": stat,
            "p-value": pvalue,
            "Conclusion": note,
        })
    return runs_df, pd.DataFrame(summary)


def scalability(school_df, student_df, sizes, radius, seed, progress=None):
    rows = []
    for k, n in enumerate(sizes, start=1):
        rng = np.random.default_rng(seed)
        schools, students = data.synthetic(school_df, student_df, n, rng)
        quotas = schools["Quota"].to_numpy(dtype=int)
        dist_km = haversine_matrix(students, schools)
        alloc = solver.proposed(dist_km, quotas, radius)
        da = baselines.deferred_acceptance(dist_km, quotas, radius)
        rows.append({
            "Students": n,
            "Total seats": int(quotas.sum()),
            "Pairs within radius": int((dist_km <= _cap(radius)).sum()),
            "Admitted": int((alloc.assignment >= 0).sum()),
            "Max-flow upper bound": alloc.info["n_star"],
            "Status": alloc.status,
            **alloc.info["stage_times"],
            "Proposed total runtime (s)": alloc.runtime_s,
            "Deferred Acceptance runtime (s)": da.runtime_s,
        })
        if progress:
            progress(k / len(sizes))
    return pd.DataFrame(rows)
