"""
Exact solvers.

Proposed method (lexicographic, all stages exact):
  1. Maximise the number of admitted students using only student-school pairs
     within the zoning radius R  -> N*   (max-flow; its value is an upper bound
     that no allocation respecting R and the quotas can exceed).
  2. Among allocations admitting N*, minimise the largest travel distance
     -> z*  (bisection over the sorted distinct distances; feasibility of a
     threshold t is "max-flow using pairs <= t equals N*", monotone in t).
  3. Among those, minimise total travel distance. This is a min-cost flow
     problem, whose constraint matrix is totally unimodular, so the LP optimum
     found by the HiGHS dual simplex is already integer. Integrality is checked
     on every solve (falling back to MILP otherwise); an integer LP optimum is
     itself a proof of optimality (gap 0).

A student left unassigned (-1) means no seat is available for them within R.
"""
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix, csr_matrix, vstack
from scipy.sparse.csgraph import maximum_flow

INTEGRALITY_TOL = 1e-6

UNASSIGNED = -1


@dataclass
class Allocation:
    assignment: np.ndarray
    runtime_s: float
    feasible: bool = True
    status: str = "Optimal"
    info: dict = field(default_factory=dict)


def _radius(radius):
    return np.inf if radius is None else float(radius)


def _flow_graph(dist, quotas, threshold):
    n, m = dist.shape
    rows, cols = np.nonzero(dist <= threshold)
    source, sink = 0, n + m + 1
    heads = np.concatenate([np.zeros(n, int), rows + 1, np.arange(m) + n + 1])
    tails = np.concatenate([np.arange(n) + 1, cols + n + 1, np.full(m, sink)])
    caps = np.concatenate([np.ones(n, int), np.ones(len(rows), int), np.asarray(quotas, int)])
    graph = csr_matrix((caps.astype(np.int32), (heads, tails)), shape=(n + m + 2, n + m + 2))
    return graph, source, sink


def max_flow(dist, quotas, threshold, with_assignment=False):
    """Maximum number of students placeable using pairs with distance <= threshold."""
    n, m = dist.shape
    graph, source, sink = _flow_graph(dist, quotas, threshold)
    result = maximum_flow(graph, source, sink, method="dinic")
    if not with_assignment:
        return int(result.flow_value)
    flow = result.flow.tocoo()
    mask = (flow.data > 0) & (flow.row >= 1) & (flow.row <= n) & (flow.col > n) & (flow.col <= n + m)
    assignment = np.full(n, UNASSIGNED, dtype=int)
    assignment[flow.row[mask] - 1] = flow.col[mask] - n - 1
    return int(result.flow_value), assignment


def bottleneck(dist, quotas, radius, target):
    """Smallest threshold t <= radius at which `target` students can still be placed."""
    cap = _radius(radius)
    candidates = np.unique(dist[dist <= cap])
    if target == 0 or len(candidates) == 0:
        return 0.0, 0
    lo, hi, iterations = 0, len(candidates) - 1, 0
    while lo < hi:
        mid = (lo + hi) // 2
        iterations += 1
        if max_flow(dist, quotas, candidates[mid]) >= target:
            hi = mid
        else:
            lo = mid + 1
    return float(candidates[lo]), iterations


def _min_cost_assignment(dist, quotas, threshold, n_admit=None, force_all=False):
    """Min total distance over pairs <= threshold. Either exactly n_admit
    students are placed, or (force_all) every student must be placed.
    Returns (assignment, status, solver_used)."""
    n, m = dist.shape
    rows, cols = np.nonzero(dist <= threshold)
    k = len(rows)
    if k == 0 or (n_admit == 0 and not force_all):
        empty = np.full(n, UNASSIGNED, dtype=int)
        return empty, ("Optimal" if not force_all or n == 0 else "Infeasible"), "—"
    var = np.arange(k)
    ones = np.ones(k)
    per_student = coo_matrix((ones, (rows, var)), shape=(n, k)).tocsr()
    per_school = coo_matrix((ones, (cols, var)), shape=(m, k)).tocsr()
    q = np.asarray(quotas, dtype=float)
    cost = dist[rows, cols]
    assignment = np.full(n, UNASSIGNED, dtype=int)

    if force_all:
        lp = linprog(cost, A_ub=per_school, b_ub=q, A_eq=per_student, b_eq=np.ones(n), bounds=(0, 1), method="highs-ds")
    else:
        lp = linprog(cost, A_ub=vstack([per_student, per_school]).tocsr(), b_ub=np.concatenate([np.ones(n), q]),
                     A_eq=csr_matrix(ones.reshape(1, -1)), b_eq=[n_admit], bounds=(0, 1), method="highs-ds")
    if lp.status == 2:
        return assignment, "Infeasible", "LP (HiGHS dual simplex)"
    if lp.status == 0 and np.abs(lp.x - np.round(lp.x)).max() < INTEGRALITY_TOL:
        chosen = lp.x > 0.5
        assignment[rows[chosen]] = cols[chosen]
        return assignment, "Optimal", "LP (HiGHS dual simplex), integer solution verified"

    constraints = [LinearConstraint(per_school, -np.inf, q)]
    if force_all:
        constraints.append(LinearConstraint(per_student, 1, 1))
    else:
        constraints.append(LinearConstraint(per_student, -np.inf, 1))
        constraints.append(LinearConstraint(csr_matrix(ones.reshape(1, -1)), n_admit, n_admit))
    res = milp(cost, integrality=ones, bounds=Bounds(0, 1), constraints=constraints)
    if res.status != 0 or res.x is None:
        return assignment, ("Infeasible" if res.status == 2 else f"Failed ({res.message})"), "MILP (HiGHS)"
    chosen = res.x > 0.5
    assignment[rows[chosen]] = cols[chosen]
    return assignment, "Optimal", "MILP (HiGHS)"


def proposed(dist, quotas, radius):
    """Lexicographic: max admissions within R -> min max distance -> min total distance."""
    t0 = time.perf_counter()
    cap = _radius(radius)
    n_star = max_flow(dist, quotas, cap)
    t1 = time.perf_counter()
    z_star, iterations = bottleneck(dist, quotas, radius, n_star)
    t2 = time.perf_counter()
    assignment, status, solver_used = _min_cost_assignment(dist, quotas, z_star, n_admit=n_star)
    t3 = time.perf_counter()

    distinct = np.unique(dist[dist <= cap])
    below = distinct[distinct < z_star]
    certificate = {
        "flow_at_radius": n_star,
        "flow_at_z_star": max_flow(dist, quotas, z_star) if n_star else 0,
        "flow_just_below_z_star": max_flow(dist, quotas, below[-1]) if n_star and len(below) else 0,
    }
    return Allocation(
        assignment=assignment,
        runtime_s=t3 - t0,
        status=status,
        info={
            "n_star": n_star,
            "z_star": z_star,
            "bisection_iterations": iterations,
            "solver": solver_used,
            "stage_times": {"Stage 1 (max admissions)": t1 - t0, "Stage 2 (min max distance)": t2 - t1,
                            "Stage 3 (min total distance)": t3 - t2},
            "certificate": certificate,
        },
    )


def min_sum_within_radius(dist, quotas, radius):
    """Max admissions within R, then min total distance (no min-max stage)."""
    t0 = time.perf_counter()
    cap = _radius(radius)
    n_star = max_flow(dist, quotas, cap)
    assignment, status, solver_used = _min_cost_assignment(dist, quotas, cap, n_admit=n_star)
    return Allocation(assignment, time.perf_counter() - t0, status=status, info={"n_star": n_star, "solver": solver_used})


def min_max_within_radius(dist, quotas, radius):
    """Max admissions within R, then min max distance; any such allocation is
    returned (no total-distance tie-break)."""
    t0 = time.perf_counter()
    n_star = max_flow(dist, quotas, _radius(radius))
    z_star, _ = bottleneck(dist, quotas, radius, n_star)
    _, assignment = max_flow(dist, quotas, z_star, with_assignment=True)
    return Allocation(assignment, time.perf_counter() - t0, info={"n_star": n_star, "z_star": z_star})


def forced_full_assignment(dist, quotas):
    """Original thesis formulation: every student MUST be placed (sum_j x_ij = 1),
    no radius, minimise total distance. Infeasible when seats < students."""
    t0 = time.perf_counter()
    assignment, status, solver_used = _min_cost_assignment(dist, quotas, np.inf, force_all=True)
    return Allocation(assignment, time.perf_counter() - t0, feasible=(status == "Optimal"), status=status,
                      info={"solver": solver_used})
