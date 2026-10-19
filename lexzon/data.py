from pathlib import Path

import numpy as np
import pandas as pd

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"

SCHOOL_COLS = ["School", "Latitude", "Longitude", "Quota"]
STUDENT_COLS = ["Student", "Latitude", "Longitude"]

# Accept both the original Indonesian headers and English ones.
ALIASES = {
    "School": ["School", "School Name", "Nama Sekolah"],
    "Student": ["Student", "Student Name", "Nama Siswa"],
    "Quota": ["Quota", "Capacity", "Jumlah Diterima"],
    "Latitude": ["Latitude", "Lat"],
    "Longitude": ["Longitude", "Lon", "Lng"],
}

BUNDLED_DATASETS = {
    "Main dataset (original quotas)": ("schools.csv", "students.csv"),
    "Scenario 1 — Exact fit (seats = students)": ("scenario_1_matched/schools.csv", "scenario_1_matched/students.csv"),
    "Scenario 2 — Surplus (seats > students)": ("scenario_2_surplus/schools.csv", "scenario_2_surplus/students.csv"),
    "Scenario 3 — Shortage (seats < students)": ("scenario_3_insufficient/schools.csv", "scenario_3_insufficient/students.csv"),
}
SCENARIO_KEYS = [k for k in BUNDLED_DATASETS if k.startswith("Scenario")]


class DataError(ValueError):
    pass


def _canonical_columns(df):
    df = df.rename(columns=lambda c: str(c).strip())
    lower = {c.lower(): c for c in df.columns}
    mapping = {}
    for canonical, names in ALIASES.items():
        for name in names:
            if name.lower() in lower:
                mapping[lower[name.lower()]] = canonical
                break
    return df.rename(columns=mapping)


def _clean(df, required, kind):
    df = _canonical_columns(df)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataError(f"The {kind} file is missing column(s): {', '.join(missing)}. "
                        f"Required: {', '.join(required)}.")
    df = df[required].copy()
    for col in ("Latitude", "Longitude") + (("Quota",) if "Quota" in required else ()):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    before = len(df)
    df = df.dropna()
    df = df[df["Latitude"].between(-90, 90) & df["Longitude"].between(-180, 180)]
    if "Quota" in required:
        df = df[df["Quota"] >= 0]
        df["Quota"] = df["Quota"].round().astype(int)
    if df.empty:
        raise DataError(f"No valid {kind} rows found.")
    return df.reset_index(drop=True), before - len(df)


def anonymize(school_df, student_df):
    """Replace names with stable IDs based on row order."""
    school_df = school_df.copy()
    student_df = student_df.copy()
    school_df["School"] = [f"School {i + 1:02d}" for i in range(len(school_df))]
    student_df["Student"] = [f"Student {i + 1:04d}" for i in range(len(student_df))]
    return school_df, student_df


def load_pair(school_src, student_src):
    """Load, validate and anonymize a school/student CSV pair (paths or file-like).
    Returns (school_df, student_df, warnings)."""
    school_df, d1 = _clean(pd.read_csv(school_src), SCHOOL_COLS, "school")
    student_df, d2 = _clean(pd.read_csv(student_src), STUDENT_COLS, "student")
    warnings = []
    if d1:
        warnings.append(f"{d1} school row(s) skipped (empty or invalid).")
    if d2:
        warnings.append(f"{d2} student row(s) skipped (empty or invalid).")
    school_df, student_df = anonymize(school_df, student_df)
    return school_df, student_df, warnings


def load_bundled(key):
    school_file, student_file = BUNDLED_DATASETS[key]
    return load_pair(DATASET_DIR / school_file, DATASET_DIR / student_file)


def scale_quotas(quotas, target_total):
    """Scale quotas proportionally to an exact integer total (largest remainder),
    keeping at least one seat in every school that originally had one."""
    q = np.asarray(quotas, dtype=float)
    if q.sum() == 0 or target_total <= 0:
        return np.zeros(len(q), dtype=int)
    raw = q * (target_total / q.sum())
    out = np.floor(raw).astype(int)
    out[(q > 0) & (out == 0)] = 1
    remainder = int(target_total - out.sum())
    if remainder > 0:
        order = np.argsort(-(raw - np.floor(raw)))
        for idx in order[:remainder]:
            out[idx] += 1
    elif remainder < 0:
        for idx in np.argsort(-out):
            if remainder == 0:
                break
            if out[idx] > 1:
                take = min(out[idx] - 1, -remainder)
                out[idx] -= take
                remainder += take
    return out


def subsample(school_df, student_df, fraction, rng):
    """Random subset of students; quotas scaled by the same fraction so the
    capacity pressure (quota/student ratio) is preserved."""
    n = max(1, int(round(len(student_df) * fraction)))
    idx = np.sort(rng.choice(len(student_df), size=n, replace=False))
    schools = school_df.copy()
    schools["Quota"] = scale_quotas(school_df["Quota"], int(round(school_df["Quota"].sum() * fraction)))
    return schools, student_df.iloc[idx].reset_index(drop=True)


def synthetic(school_df, student_df, n_students, rng, jitter_km=0.15):
    """Synthetic instance of a given size for scalability tests: students are
    resampled from the real locations plus Gaussian jitter; quotas are scaled
    so the original quota/student ratio stays the same."""
    idx = rng.choice(len(student_df), size=n_students, replace=True)
    base = student_df.iloc[idx]
    lat = base["Latitude"].to_numpy() + rng.normal(0, jitter_km / 110.574, n_students)
    lon = base["Longitude"].to_numpy() + rng.normal(0, jitter_km / (111.320 * np.cos(np.radians(lat.mean()))), n_students)
    students = pd.DataFrame({"Student": [f"Synthetic {i + 1}" for i in range(n_students)], "Latitude": lat, "Longitude": lon})
    ratio = school_df["Quota"].sum() / len(student_df)
    schools = school_df.copy()
    schools["Quota"] = scale_quotas(school_df["Quota"], int(round(ratio * n_students)))
    return schools, students
