import numpy as np

EARTH_RADIUS_KM = 6371.0


def haversine_matrix(student_df, school_df):
    """(n_students, n_schools) great-circle distance in km."""
    lat1 = np.radians(student_df["Latitude"].to_numpy(dtype=float))[:, None]
    lon1 = np.radians(student_df["Longitude"].to_numpy(dtype=float))[:, None]
    lat2 = np.radians(school_df["Latitude"].to_numpy(dtype=float))[None, :]
    lon2 = np.radians(school_df["Longitude"].to_numpy(dtype=float))[None, :]
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def euclidean_degree_matrix(student_df, school_df):
    """Raw lat/lon Euclidean distance -- only to reproduce the original thesis
    method, whose allocation is then re-measured in km for comparison."""
    s = student_df[["Latitude", "Longitude"]].to_numpy(dtype=float)[:, None, :]
    c = school_df[["Latitude", "Longitude"]].to_numpy(dtype=float)[None, :, :]
    return np.sqrt(((s - c) ** 2).sum(axis=2))


def planar_km(df):
    """Equirectangular projection to km, so clustering metrics are isotropic."""
    lat = df["Latitude"].to_numpy(dtype=float)
    lon = df["Longitude"].to_numpy(dtype=float)
    lat0 = np.radians(lat.mean()) if len(lat) else 0.0
    return np.column_stack([lon * 111.320 * np.cos(lat0), lat * 110.574])
