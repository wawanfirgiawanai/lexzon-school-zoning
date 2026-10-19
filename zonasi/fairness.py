"""
Distance-priority fairness and a clustering baseline from the zoning literature.

priority_violations(): counts justified envy under distance priority, the rule
used by distance-based admission. Student i has justified envy toward school j
when j lies within the radius, i prefers j to its own outcome (i is unassigned,
or j is closer than i's school), and j either has a free seat or admitted some
student who lives farther from j than i does. Ties are broken by the lower
index, exactly as in deferred_acceptance, so a stable allocation scores zero.

constrained_kmeans(): constrained k-means (Bradley, Bennett and Demiriz, 2000),
as used for school zoning by Zainuddin and Risal (2024): cluster centres start
at the schools, each assignment step solves a capacitated transportation
problem on squared planar distance (cluster size <= school quota), centres move
to the mean of their members, and the loop stops when the assignment repeats.
Cluster j is served by school j; students farther than R from that school are
then left without a school, so the result is judged under the same rules as
every other method.
"""
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack

from .distance import planar_km
from .solver import UNASSIGNED, Allocation, _radius

TIE = 1e-13  # index tie-break, far below any real distance difference


def priority_violations(assignment, dist, quotas, radius, priority=None):
    """priority: optional n x m matrix of school priority keys (smaller = higher
    priority); it defaults to dist, the distance-based rule."""
    n, m = dist.shape
    priority = dist if priority is None else np.asarray(priority, dtype=float)
    cap = _radius(radius)
    quotas = np.asarray(quotas, dtype=int)
    admitted = assignment != UNASSIGNED
    idx = np.nonzero(admitted)[0]
    used = np.bincount(assignment[idx], minlength=m)

    # Priority key of the weakest admitted student at each school (inf if a seat is free).
    student_key = priority + np.arange(n)[:, None] * TIE
    worst = np.full(m, -np.inf)
    np.maximum.at(worst, assignment[idx], student_key[idx, assignment[idx]])
    worst[used < quotas] = np.inf

    pref_key = dist + np.arange(m)[None, :] * TIE
    own = np.full(n, np.inf)
    own[idx] = pref_key[idx, assignment[idx]]

    envy = (dist <= cap) & (pref_key < own[:, None]) & (student_key < worst[None, :])
    envious = envy.any(axis=1)
    via_vacancy = (envy & np.isinf(worst)[None, :]).any(axis=1)
    # For admitted students with envy: how much closer the best envied school is than their own.
    extra = []
    for i in np.nonzero(envious & admitted)[0]:
        extra.append(dist[i, assignment[i]] - dist[i, envy[i]].min())
    extra = np.asarray(extra)
    return {
        "envy_extra_km_mean": float(extra.mean()) if len(extra) else 0.0,
        "envy_extra_km_median": float(np.median(extra)) if len(extra) else 0.0,
        "envy_extra_km_max": float(extra.max()) if len(extra) else 0.0,
        "envy_students": int(envious.sum()),
        "envy_unassigned": int((envious & ~admitted).sum()),
        "envy_assigned": int((envious & admitted).sum()),
        "envy_pairs": int(envy.sum()),
        "envy_via_vacant_seat": int(via_vacancy.sum()),
        "envy_share_pct": 100.0 * envious.sum() / n if n else 0.0,
    }


def _capacitated_assignment(cost, quotas, n_admit):
    n, m = cost.shape
    k = n * m
    rows = np.repeat(np.arange(n), m)
    cols = np.tile(np.arange(m), n)
    var = np.arange(k)
    ones = np.ones(k)
    per_student = coo_matrix((ones, (rows, var)), shape=(n, k))
    per_school = coo_matrix((ones, (cols, var)), shape=(m, k))
    total = coo_matrix((ones, (np.zeros(k, int), var)), shape=(1, k))
    res = linprog(cost.ravel(), A_ub=vstack([per_student, per_school]).tocsr(),
                  b_ub=np.concatenate([np.ones(n), quotas]), A_eq=total.tocsr(), b_eq=[n_admit],
                  bounds=(0, 1), method="highs-ds")
    if res.status != 0:
        raise RuntimeError(res.message)
    x = res.x.reshape(n, m)
    assignment = np.full(n, UNASSIGNED, dtype=int)
    chosen = x.max(axis=1) > 0.5
    assignment[chosen] = x[chosen].argmax(axis=1)
    return assignment


def constrained_kmeans(student_df, school_df, dist, quotas, radius, max_iter=50):
    t0 = time.perf_counter()
    quotas = np.asarray(quotas, dtype=int)
    n, m = dist.shape
    both = planar_km(__import__("pandas").concat([student_df[["Latitude", "Longitude"]],
                                                  school_df[["Latitude", "Longitude"]]], ignore_index=True))
    points, centres = both[:n], both[n:].copy()
    n_admit = int(min(n, quotas.sum()))
    previous, iterations = None, 0
    for iterations in range(1, max_iter + 1):
        cost = ((points[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        assignment = _capacitated_assignment(cost, quotas, n_admit)
        if previous is not None and np.array_equal(assignment, previous):
            break
        previous = assignment
        for j in range(m):
            members = assignment == j
            if members.any():
                centres[j] = points[members].mean(axis=0)
    clustered = assignment.copy()
    cap = _radius(radius)
    ok = clustered != UNASSIGNED
    beyond = np.zeros(n, dtype=bool)
    beyond[ok] = dist[np.nonzero(ok)[0], clustered[ok]] > cap
    assignment = np.where(beyond, UNASSIGNED, clustered)
    return Allocation(assignment, time.perf_counter() - t0,
                      info={"iterations": iterations, "beyond_radius_removed": int(beyond.sum())})
