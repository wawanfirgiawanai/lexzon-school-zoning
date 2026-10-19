# LEXZON: lexicographic network-flow optimization for school zoning

Reference implementation and reproducibility package for the paper

> *LEXZON: Lexicographic Network-Flow Optimization for Maximum-Enrollment School
> Zoning under Capacity and Distance Constraints*, International Journal of
> Intelligent Engineering and Systems.

LEXZON assigns applicants to schools inside an eligibility radius with a strict
objective hierarchy: first maximum enrollment, then the minimum of the longest
assigned distance, and finally the minimum total distance. Students may remain
unassigned, and each unassigned student is labelled with the binding cause.

## What is in this repository

| Path | Contents |
|---|---|
| `lexzon/` | The method. `solver.py` holds the three stages, `pareto.py` the violation-budget model and LEXZON-F, `fairness.py` the priority-violation metric and the constrained k-means baseline, `baselines.py` the comparison rules. |
| `scripts/` | Experiment drivers. `scenario_experiments.py`, `fairness_experiments.py` and `pareto_experiments.py` produce the main results; `revision/` holds the additional studies requested in review. |
| `tests/` | 126 unit tests and 22 end-to-end checks, including optimality checks against exhaustive solutions on small instances. |
| `artifacts/` | Machine-readable description of every experimental instance and output. |
| `results/` | Every CSV behind the tables and figures of the paper. |
| `figures/` | The figure scripts and the SVG files used in the paper. |
| `app.py` | Streamlit interface: zoning map, school highlighting, result tables. |

## Data availability

The applicant locations are personal data and are **not** published. The
repository releases instead, for each capacity scenario:

* `artifacts/capacities_S*.csv` — the capacity of every school;
* `artifacts/eligible_pairs_S*.csv.gz` — every eligible applicant–school pair
  with both rankings (the applicant's rank of the school and the school's
  priority rank of the applicant) and the distance **rounded to 50 m**;
* `artifacts/assignments_S*.csv` — the assignment returned by each method;
* `artifacts/subsample_membership.csv.gz` — the members of all 30 subsamples
  per scenario;
* `artifacts/instances.json` — all solver settings, seeds, tie-breaking rules
  and generation parameters.

School locations and capacities are public information and are included in
`dataset/`. Applicant coordinates are withheld, so the figures that plot
applicant positions cannot be regenerated from this repository.

## Auditing the reported numbers

Maximum enrollment, deferred-acceptance enrollment and the priority-violation
counts depend only on the eligibility structure and the two rankings, not on
the metric values. They can therefore be verified from the released artifacts:

```bash
python scripts/revision/r8_verify_artifacts.py
```

The script recomputes the three quantities and compares them with
`artifacts/instances.json`. Expected output:

| Scenario | N* | Deferred acceptance | Students with violations |
|---|---|---|---|
| S1 | 1141 | 1026 | 376 |
| S2 | 1180 | 1107 | 281 |
| S3 | 986 | 940 | 193 |

## Reproducing the experiments

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
pytest tests -q                                      # 126 tests
python scripts/scenario_experiments.py               # Tables 3, 4, 5 and Figures 2, 5
python scripts/fairness_experiments.py               # Table 6
python scripts/pareto_experiments.py                 # Figure 6
python scripts/revision/r1_density_scalability.py    # runtime versus eligibility density
python scripts/revision/r2_sensitivity.py            # capacity, access and priority sensitivity
python scripts/revision/r3_lexzonf_time.py           # LEXZON-F gap versus solver time
python scripts/revision/r4_reassignment_cost.py      # burden on reassigned students
python scripts/revision/r5_paired_statistics.py      # paired differences and subsample overlap
python scripts/revision/r6_instance_families.py      # when the bottleneck stage matters
python scripts/revision/r7_artifacts.py rounded      # regenerate the artifact bundle
```

## Solver settings

| Component | Setting |
|---|---|
| Stage 1, maximum flow | Dinic's algorithm (`scipy.sparse.csgraph.maximum_flow`) |
| Stage 2, bottleneck | binary search over the distinct eligible distances, one max-flow call per threshold |
| Stage 3, minimum cost | linear program solved by HiGHS dual simplex (`scipy.optimize.linprog`, `method="highs-ds"`), default HiGHS tolerances |
| Integrality check | `max abs(x - round(x)) < 1e-6`; if it fails the instance is re-solved with `scipy.optimize.milp` (HiGHS branch and bound) |
| Tie-breaking | lower applicant index and lower school index, applied as a 1e-13 key offset |
| LEXZON-F and violation budgets | CP-SAT (OR-Tools), 12 workers, wall-clock limit per point, warm start from the LEXZON or deferred-acceptance assignment; solutions that are not proven optimal are reported with the solver bound |
| Distances | Haversine, Earth radius 6371 km |

Portfolio search in CP-SAT with several workers is time dependent, so a feasible
solution may differ between machines. Every reported point carries its status
and bound, and the conclusions drawn in the paper are stated for the bounds
rather than for the feasible values alone.

## Environment

Python 3.11.9 with the versions pinned in `requirements.txt` (NumPy 1.26.4,
SciPy 1.17.1, OR-Tools 9.11.4210). Results were produced on an Intel Core
i7-14650HX with 16 GB of RAM under Windows 11.

## Citation

```bibtex
@article{lexzon2026,
  title   = {LEXZON: Lexicographic Network-Flow Optimization for Maximum-Enrollment
             School Zoning under Capacity and Distance Constraints},
  author  = {Nirwana, Hafsah and Firgiawan, Wawan and Arifin, Nurhikmah and
             Saidy, Hamdy Nur and Sulfayanti and Nur, Nahya and
             Zainuddin, Zahir and Amalia, Bilqis},
  journal = {International Journal of Intelligent Engineering and Systems},
  year    = {2026}
}
```

## License

MIT, see `LICENSE`.
