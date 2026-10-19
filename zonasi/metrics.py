import numpy as np
from sklearn.metrics import davies_bouldin_score

from .distance import planar_km
from .solver import UNASSIGNED

LABELS = {
    "admitted": "Admitted",
    "rejected": "No school (✕)",
    "rejected_no_school": "✕ no school within radius",
    "rejected_full": "✕ schools in radius full",
    "admission_rate": "Admission rate (%)",
    "gap_to_max": "Gap to max admissions",
    "seats_vacant": "Vacant seats",
    "overflow_students": "Quota overflow (students)",
    "overflow_schools": "Schools over quota",
    "total_km": "Total distance (km)",
    "mean_km": "Mean distance (km)",
    "p90_km": "90th percentile distance (km)",
    "max_km": "Max distance (km)",
    "gini": "Gini (admitted)",
    "dbi": "Davies-Bouldin Index",
    "runtime_s": "Runtime (s)",
    "beyond_radius": "Admitted beyond radius",
}

STATUS_ADMITTED = "Admitted"
STATUS_NO_SCHOOL = "✕ No school within radius"
STATUS_FULL = "✕ Schools in radius full"


def gini(values):
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum())) - (n + 1) / n)


def assigned_distances(assignment, dist):
    admitted = np.nonzero(assignment != UNASSIGNED)[0]
    return dist[admitted, assignment[admitted]]


def evaluate(allocation, dist_km, quotas, student_df, radius, max_admissible):
    """All distance metrics are measured in Haversine km, whatever distance the
    method used internally, so every method is compared on the same scale."""
    a = allocation.assignment
    n = len(a)
    quotas = np.asarray(quotas, dtype=int)
    cap = np.inf if radius is None else radius
    admitted_mask = a != UNASSIGNED
    d = assigned_distances(a, dist_km)
    used = np.bincount(a[admitted_mask], minlength=len(quotas))
    overflow = np.maximum(used - quotas, 0)
    no_school = dist_km.min(axis=1) > cap
    rejected_mask = ~admitted_mask

    labels = a[admitted_mask]
    dbi = np.nan
    if admitted_mask.sum() > 1 and len(np.unique(labels)) > 1:
        dbi = float(davies_bouldin_score(planar_km(student_df.iloc[np.nonzero(admitted_mask)[0]]), labels))

    return {
        "admitted": int(admitted_mask.sum()),
        "rejected": int(rejected_mask.sum()),
        "rejected_no_school": int((rejected_mask & no_school).sum()),
        "rejected_full": int((rejected_mask & ~no_school).sum()),
        "admission_rate": 100.0 * admitted_mask.sum() / n if n else 0.0,
        "gap_to_max": int(max_admissible - admitted_mask.sum()) if overflow.sum() == 0 else np.nan,
        "seats_vacant": int(np.maximum(quotas - used, 0).sum()),
        "overflow_students": int(overflow.sum()),
        "overflow_schools": int((overflow > 0).sum()),
        "total_km": float(d.sum()),
        "mean_km": float(d.mean()) if len(d) else 0.0,
        "p90_km": float(np.percentile(d, 90)) if len(d) else 0.0,
        "max_km": float(d.max()) if len(d) else 0.0,
        "gini": gini(d),
        "dbi": dbi,
        "runtime_s": float(allocation.runtime_s),
    }


def school_utilization(assignment, dist_km, school_df):
    quotas = school_df["Quota"].to_numpy(dtype=int)
    admitted = assignment != UNASSIGNED
    used = np.bincount(assignment[admitted], minlength=len(quotas))
    out = school_df[["School", "Quota"]].copy()
    out["Filled"] = used
    out["Vacant"] = np.maximum(quotas - used, 0)
    out["Fill rate (%)"] = np.where(quotas > 0, 100 * used / np.maximum(quotas, 1), 0.0)
    far = np.zeros(len(quotas))
    idx = np.nonzero(admitted)[0]
    np.maximum.at(far, assignment[idx], dist_km[idx, assignment[idx]])
    out["Farthest student (km)"] = far
    return out


def student_table(assignment, dist_km, school_df, student_df, radius):
    cap = np.inf if radius is None else radius
    names = school_df["School"].to_numpy()
    out = student_df.copy()
    admitted = assignment != UNASSIGNED
    safe = np.where(admitted, assignment, 0)
    out["Assigned school"] = np.where(admitted, names[safe], "—")
    out["Distance (km)"] = np.where(admitted, dist_km[np.arange(len(assignment)), safe], np.nan)
    nearest = dist_km.min(axis=1)
    out["Nearest school (km)"] = nearest
    out["Status"] = np.select([admitted, nearest > cap], [STATUS_ADMITTED, STATUS_NO_SCHOOL], default=STATUS_FULL)
    return out
