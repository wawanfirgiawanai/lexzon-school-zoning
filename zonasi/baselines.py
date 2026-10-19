"""
Comparison methods. All respect the same zoning radius R as the proposed
method, so every difference in results comes from HOW students are allocated,
not from different rules.
"""
import heapq
import time

import numpy as np

from .solver import UNASSIGNED, Allocation, _radius, min_max_within_radius, min_sum_within_radius


def nearest_no_quota(dist, quotas, radius):
    """B1: nearest school within R, ignoring capacity (infeasible reference)."""
    t0 = time.perf_counter()
    cap = _radius(radius)
    nearest = np.argmin(dist, axis=1)
    assignment = np.where(dist[np.arange(len(dist)), nearest] <= cap, nearest, UNASSIGNED)
    return Allocation(assignment, time.perf_counter() - t0)


def greedy_nearest_available(dist, quotas, radius, order=None):
    """B2: students processed by their own nearest distance (closest first),
    each placed in the nearest school within R that still has a seat."""
    t0 = time.perf_counter()
    cap = _radius(radius)
    n = len(dist)
    remaining = np.asarray(quotas, dtype=int).copy()
    assignment = np.full(n, UNASSIGNED, dtype=int)
    if order is None:
        order = np.argsort(dist.min(axis=1), kind="stable")
    for i in order:
        for j in np.argsort(dist[i], kind="stable"):
            if dist[i, j] > cap:
                break
            if remaining[j] > 0:
                assignment[i] = j
                remaining[j] -= 1
                break
    return Allocation(assignment, time.perf_counter() - t0)


def greedy_random_order(dist, quotas, radius, seed):
    """B3: same greedy rule, students processed in a random order."""
    order = np.random.default_rng(seed).permutation(len(dist))
    return greedy_nearest_available(dist, quotas, radius, order=order)


def deferred_acceptance(dist, quotas, radius, priority=None):
    """B4: student-proposing Deferred Acceptance (Gale-Shapley). Students rank
    schools within R by distance; schools prioritise the closest applicants
    (ties broken by lower student index). Mirrors distance-based PPDB selection.
    priority: optional n x m matrix of school priority keys (smaller = higher
    priority). It defaults to dist, which gives the distance-based rule; another
    matrix lets the school ranking differ from the student ranking, for example
    under a lottery."""
    t0 = time.perf_counter()
    priority = dist if priority is None else np.asarray(priority, dtype=float)
    cap = _radius(radius)
    n, m = dist.shape
    q = np.asarray(quotas, dtype=int)
    prefs = [[j for j in np.argsort(dist[i], kind="stable") if dist[i, j] <= cap] for i in range(n)]
    next_choice = np.zeros(n, dtype=int)
    held = [[] for _ in range(m)]
    free = list(range(n))
    while free:
        i = free.pop()
        if next_choice[i] >= len(prefs[i]):
            continue
        j = prefs[i][next_choice[i]]
        next_choice[i] += 1
        if q[j] == 0:
            free.append(i)
            continue
        heapq.heappush(held[j], (-priority[i, j], -i, i))
        if len(held[j]) > q[j]:
            _, _, rejected = heapq.heappop(held[j])
            free.append(rejected)
    assignment = np.full(n, UNASSIGNED, dtype=int)
    for j in range(m):
        for _, _, i in held[j]:
            assignment[i] = j
    return Allocation(assignment, time.perf_counter() - t0)


METHODS = {
    "B1": ("Nearest, quota ignored", nearest_no_quota),
    "B2": ("Greedy nearest", greedy_nearest_available),
    "B3": ("Greedy random order", None),
    "B4": ("Deferred Acceptance", deferred_acceptance),
    "B5": ("Exact min-total", min_sum_within_radius),
    "B6": ("Exact min-max", min_max_within_radius),
}
