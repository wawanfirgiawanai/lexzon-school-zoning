# Dataset

This folder holds the **public** part of the input data: the locations and
capacities of the 28 public elementary schools, for each capacity scenario.

The applicant home locations are personal data and are deliberately absent. The
scripts that need them (`scenario_experiments.py`, `fairness_experiments.py`,
`pareto_experiments.py`, the map figures and the Streamlit app) therefore do not
run from this repository alone.

What can be verified without them:

* `scripts/revision/r8_verify_artifacts.py` recomputes maximum enrollment,
  deferred-acceptance enrollment and the priority-violation counts from
  `artifacts/`, which contain the eligibility structure and both rankings;
* `results/` contains every number reported in the paper;
* `tests/` runs on synthetic fixtures and covers the solver, the baselines, the
  fairness metric and the violation models.

Requests for access to the applicant data should be addressed to the
corresponding author; release is subject to the data-protection rules that
apply to admission records.
