import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lexzon import baselines, data, solver  # noqa: E402
from lexzon.distance import haversine_matrix  # noqa: E402


def brute_force_lexicographic(dist, quotas, radius):
    """Exhaustive (admitted desc, max distance asc, total distance asc) optimum."""
    n, m = dist.shape
    best = None
    for choice in itertools.product(range(-1, m), repeat=n):
        a = np.array(choice)
        if any(j >= 0 and dist[i, j] > radius for i, j in enumerate(a)):
            continue
        if any((a == j).sum() > quotas[j] for j in range(m)):
            continue
        d = np.array([dist[i, j] for i, j in enumerate(a) if j >= 0])
        key = (-len(d), d.max() if len(d) else 0.0, d.sum() if len(d) else 0.0)
        if best is None or key < best:
            best = key
    return best


def key_of(assignment, dist):
    d = np.array([dist[i, j] for i, j in enumerate(assignment) if j >= 0])
    return (-len(d), d.max() if len(d) else 0.0, d.sum() if len(d) else 0.0)


@pytest.mark.parametrize("seed", range(25))
def test_proposed_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    n, m = rng.integers(3, 7), rng.integers(2, 4)
    dist = np.round(rng.uniform(0.1, 2.0, size=(n, m)), 3)
    quotas = rng.integers(0, 3, size=m)
    radius = float(rng.uniform(0.6, 1.8))
    expected = brute_force_lexicographic(dist, quotas, radius)
    got = key_of(solver.proposed(dist, quotas, radius).assignment, dist)
    assert got[0] == expected[0]
    assert got[1] == pytest.approx(expected[1])
    assert got[2] == pytest.approx(expected[2])


@pytest.mark.parametrize("seed", range(10))
def test_baselines_feasible_and_bounded(seed):
    rng = np.random.default_rng(seed)
    dist = rng.uniform(0.05, 2.0, size=(40, 5))
    quotas = rng.integers(2, 10, size=5)
    radius = 1.0
    bound = solver.max_flow(dist, quotas, radius)
    for fn in (baselines.greedy_nearest_available, baselines.deferred_acceptance,
               solver.min_sum_within_radius, solver.min_max_within_radius):
        a = fn(dist, quotas, radius).assignment
        placed = a >= 0
        assert np.all(dist[np.nonzero(placed)[0], a[placed]] <= radius)
        assert np.all(np.bincount(a[placed], minlength=5) <= quotas)
        assert placed.sum() <= bound
    assert (solver.proposed(dist, quotas, radius).assignment >= 0).sum() == bound


@pytest.mark.parametrize("seed", range(10))
def test_deferred_acceptance_is_stable(seed):
    rng = np.random.default_rng(seed)
    dist = rng.uniform(0.05, 2.0, size=(30, 4))
    quotas = rng.integers(1, 8, size=4)
    radius = 1.2
    a = baselines.deferred_acceptance(dist, quotas, radius).assignment
    for i in range(len(dist)):
        for j in range(dist.shape[1]):
            if dist[i, j] > radius or a[i] == j:
                continue
            student_prefers = a[i] < 0 or dist[i, j] < dist[i, a[i]]
            members = np.nonzero(a == j)[0]
            school_wants = len(members) < quotas[j] or any(
                (dist[i, j], i) < (dist[k, j], k) for k in members)
            assert not (student_prefers and school_wants and quotas[j] > 0), f"blocking pair ({i}, {j})"


def test_scale_quotas_exact_total():
    for target in (1, 100, 1000, 1209, 1428):
        out = data.scale_quotas([56, 28, 84, 56, 0, 28], target)
        assert out.sum() == target or target < 5
        assert out[4] == 0


def test_real_data_regression():
    schools, students, _ = data.load_bundled("Main dataset (original quotas)")
    dist = haversine_matrix(students, schools)
    quotas = schools["Quota"].to_numpy()
    alloc = solver.proposed(dist, quotas, 1.0)
    assert (alloc.assignment >= 0).sum() == 1180
    assert alloc.info["z_star"] == pytest.approx(0.9954, abs=1e-4)
    cert = alloc.info["certificate"]
    assert cert["flow_just_below_z_star"] < cert["flow_at_z_star"] == alloc.info["n_star"]
    assert alloc.info["solver"].startswith("LP")
    placed = alloc.assignment >= 0
    assert dist[np.nonzero(placed)[0], alloc.assignment[placed]].sum() == pytest.approx(323.9314, abs=1e-3)


def test_forced_assignment_infeasible_when_seats_short():
    schools, students, _ = data.load_bundled("Scenario 3 — Shortage (seats < students)")
    dist = haversine_matrix(students, schools)
    assert not solver.forced_full_assignment(dist, schools["Quota"].to_numpy()).feasible


def test_bordering_schools_never_share_confusable_colors():
    from lexzon import maps

    schools, _, _ = data.load_bundled("Main dataset (original quotas)")
    assert maps.neighbour_clashes(schools, maps.school_colors(schools)) == []


def test_names_are_anonymized_on_load():
    schools, students, _ = data.load_bundled("Main dataset (original quotas)")
    assert schools["School"].iloc[0] == "School 01"
    assert students["Student"].iloc[0] == "Student 0001"
