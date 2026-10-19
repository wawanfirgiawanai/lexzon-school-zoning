import numpy as np
import pytest

from lexzon import baselines, data, fairness, solver
from lexzon.distance import haversine_matrix


def random_instance(seed, n=40, m=5):
    rng = np.random.default_rng(seed)
    dist = rng.uniform(0.05, 2.0, size=(n, m))
    quotas = rng.integers(3, 10, size=m)
    return dist, quotas


@pytest.mark.parametrize("seed", range(25))
def test_deferred_acceptance_has_no_justified_envy(seed):
    dist, quotas = random_instance(seed)
    alloc = baselines.deferred_acceptance(dist, quotas, 1.0)
    assert fairness.priority_violations(alloc.assignment, dist, quotas, 1.0)["envy_students"] == 0


@pytest.mark.parametrize("seed", range(25))
def test_lexzon_never_rejects_a_student_with_priority(seed):
    """Stage 1 rules out envy toward a free seat, Stage 3 rules out a rejected
    student being closer than an admitted one (a swap would cut total distance)."""
    dist, quotas = random_instance(seed)
    alloc = solver.proposed(dist, quotas, 1.0)
    v = fairness.priority_violations(alloc.assignment, dist, quotas, 1.0)
    assert v["envy_unassigned"] == 0
    assert v["envy_via_vacant_seat"] == 0


def test_violation_counter_on_a_hand_made_case():
    # Two students, one school with one seat: the farther student holds the seat.
    dist = np.array([[0.2], [0.5]])
    v = fairness.priority_violations(np.array([-1, 0]), dist, np.array([1]), 1.0)
    assert v["envy_unassigned"] == 1 and v["envy_pairs"] == 1


def test_constrained_kmeans_respects_quota_and_radius():
    school_df, student_df, _ = data.load_bundled("Main dataset (original quotas)")
    quotas = school_df["Quota"].to_numpy(int)
    dist = haversine_matrix(student_df, school_df)
    alloc = fairness.constrained_kmeans(student_df, school_df, dist, quotas, 1.0)
    a = alloc.assignment
    ok = a >= 0
    assert (np.bincount(a[ok], minlength=len(quotas)) <= quotas).all()
    assert (dist[np.nonzero(ok)[0], a[ok]] <= 1.0).all()
    assert ok.sum() <= solver.max_flow(dist, quotas, 1.0)
