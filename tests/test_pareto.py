import numpy as np
import pytest

from lexzon import baselines, fairness, pareto, solver


def instance(seed, n=30, m=4):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.05, 1.5, size=(n, m)), rng.integers(3, 8, size=m)


@pytest.mark.parametrize("seed", range(8))
def test_zero_budget_equals_deferred_acceptance(seed):
    """Rural hospitals theorem: every stable allocation admits as many students as DA."""
    dist, quotas = instance(seed)
    r = pareto.EnvyModel(dist, quotas, 1.0).max_admissions(0, time_limit=30, workers=4)
    da = baselines.deferred_acceptance(dist, quotas, 1.0).assignment
    assert r["optimal"]
    assert (r["assignment"] >= 0).sum() == (da >= 0).sum()
    assert fairness.priority_violations(r["assignment"], dist, quotas, 1.0)["envy_students"] == 0


@pytest.mark.parametrize("seed", range(8))
def test_unlimited_budget_reaches_max_flow(seed):
    dist, quotas = instance(seed)
    r = pareto.EnvyModel(dist, quotas, 1.0).max_admissions(len(dist), time_limit=30, workers=4)
    assert (r["assignment"] >= 0).sum() == solver.max_flow(dist, quotas, 1.0)


@pytest.mark.parametrize("seed", range(8))
def test_min_envy_at_max_admissions_never_worse_than_lexzon(seed):
    dist, quotas = instance(seed)
    model = pareto.EnvyModel(dist, quotas, 1.0)
    lex = solver.proposed(dist, quotas, 1.0).assignment
    n_star = solver.max_flow(dist, quotas, 1.0)
    r = model.min_envy(n_star, hint=lex, time_limit=30, workers=4)
    a = r["assignment"]
    assert (a >= 0).sum() == n_star
    envy = fairness.priority_violations(a, dist, quotas, 1.0)["envy_students"]
    assert envy == int(r["objective"])
    assert envy <= fairness.priority_violations(lex, dist, quotas, 1.0)["envy_students"]
    assert (model.envious(a) == 0).sum() + envy == len(dist)
