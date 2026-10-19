"""
Price of fairness: the exact trade-off between admissions and distance priority.

Distance priority is the rule of distance-based admission: a school prefers the
applicant who lives closer (ties broken by the lower index, as in
deferred_acceptance). A student has justified envy when a school within the
radius that the student prefers to their own outcome has a free seat or holds a
student who lives farther from it (see fairness.priority_violations).

Model (x_ij = student i admitted to school j, only pairs within R; y_i = student
i is allowed to have justified envy). For every admissible pair (i, j):

    sum_{i' ahead of i at j} x_i'j + q_j * sum_{j' at least as good as j for i} x_ij' >= q_j * (1 - y_i)

i.e. unless i sits at j or at a school i likes at least as much, school j must be
full with students who have priority over i. With y = 0 the constraints describe
exactly the stable allocations (and, by the rural hospitals theorem, every one of
them admits the same number of students as deferred acceptance).

Solved with OR-Tools CP-SAT, warm-started from a feasible allocation:
- max_admissions(k): most admitted students with at most k envious students;
- min_envy(n): fewest envious students among allocations admitting n students.
Both return the allocation, the best bound and whether optimality was proven.
"""
import time

import numpy as np
from ortools.sat.python import cp_model

from .solver import UNASSIGNED, _radius

TIE = 1e-13


class EnvyModel:
    def __init__(self, dist, quotas, radius):
        self.dist = dist
        self.n, self.m = dist.shape
        self.quotas = np.asarray(quotas, dtype=int)
        cap = _radius(radius)
        self.rows, self.cols = np.nonzero(dist <= cap)
        self.prio = dist[self.rows, self.cols] + self.rows * TIE
        self.pref = dist[self.rows, self.cols] + self.cols * TIE
        order = np.argsort(self.rows, kind="stable")
        starts = np.searchsorted(self.rows[order], np.arange(self.n + 1))
        self.by_student = [order[starts[i]:starts[i + 1]] for i in range(self.n)]
        self.by_school = [np.nonzero(self.cols == j)[0] for j in range(self.m)]

    def _build(self):
        md = cp_model.CpModel()
        x = [md.NewBoolVar(f"x{p}") for p in range(len(self.rows))]
        y = [md.NewBoolVar(f"y{i}") for i in range(self.n)]
        for i in range(self.n):
            if len(self.by_student[i]):
                md.Add(sum(x[p] for p in self.by_student[i]) <= 1)
            else:
                md.Add(y[i] == 0)
        for j in range(self.m):
            idx = self.by_school[j]
            md.Add(sum(x[p] for p in idx) <= int(self.quotas[j]))
            order = idx[np.argsort(self.prio[idx], kind="stable")]
            qj = int(self.quotas[j])
            prefix = 0
            for rank, p in enumerate(order):
                i = self.rows[p]
                own = self.by_student[i]
                better = own[self.pref[own] <= self.pref[p]]
                md.Add(prefix + qj * sum(x[b] for b in better) + qj * y[i] >= qj)
                prefix = prefix + x[p] if rank else x[p]
        return md, x, y

    def _hint(self, md, x, y, assignment):
        if assignment is None:
            return
        from .fairness import priority_violations  # noqa: F401  (documentation of the envy notion)
        chosen = set(zip(np.nonzero(assignment >= 0)[0], assignment[assignment >= 0]))
        for p, (i, j) in enumerate(zip(self.rows, self.cols)):
            md.AddHint(x[p], (i, j) in chosen)
        envious = self.envious(assignment)
        for i in range(self.n):
            md.AddHint(y[i], bool(envious[i]))

    def envious(self, assignment):
        cap_ok = np.zeros(self.dist.shape, dtype=bool)
        cap_ok[self.rows, self.cols] = True
        n, m = self.dist.shape
        admitted = assignment != UNASSIGNED
        idx = np.nonzero(admitted)[0]
        used = np.bincount(assignment[idx], minlength=m)
        key = self.dist + np.arange(n)[:, None] * TIE
        worst = np.full(m, -np.inf)
        np.maximum.at(worst, assignment[idx], key[idx, assignment[idx]])
        worst[used < self.quotas] = np.inf
        pref = self.dist + np.arange(m)[None, :] * TIE
        own = np.full(n, np.inf)
        own[idx] = pref[idx, assignment[idx]]
        return (cap_ok & (pref < own[:, None]) & (key < worst[None, :])).any(axis=1)

    def _solve(self, md, x, y, time_limit, workers, seed=None):
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.num_workers = workers
        if seed is not None:                      # reported runs use the OR-Tools default
            solver.parameters.random_seed = int(seed)
        t0 = time.perf_counter()
        status = solver.Solve(md)
        runtime = time.perf_counter() - t0
        assignment = np.full(self.n, UNASSIGNED, dtype=int)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for p in range(len(x)):
                if solver.Value(x[p]):
                    assignment[self.rows[p]] = self.cols[p]
        return {"assignment": assignment, "objective": solver.ObjectiveValue(), "bound": solver.BestObjectiveBound(),
                "optimal": status == cp_model.OPTIMAL, "status": solver.StatusName(status), "runtime_s": runtime,
                "workers": workers, "time_limit_s": time_limit, "seed": seed}

    def max_admissions(self, k, hint=None, time_limit=300, workers=16, seed=None):
        md, x, y = self._build()
        md.Add(sum(y) <= int(k))
        md.Maximize(sum(x))
        self._hint(md, x, y, hint)
        return self._solve(md, x, y, time_limit, workers, seed)

    def min_envy(self, n_admit, hint=None, time_limit=300, workers=16, seed=None):
        md, x, y = self._build()
        md.Add(sum(x) >= int(n_admit))
        md.Minimize(sum(y))
        self._hint(md, x, y, hint)
        return self._solve(md, x, y, time_limit, workers, seed)
