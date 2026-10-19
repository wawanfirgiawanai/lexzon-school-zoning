# LEXZON: results, formulas, and the materials and methods section

This document holds the result tables (ready to plot), the formulas, and the materials and methods section. Every number is read directly from the experiment files in `results/`. The zoning radius is R = 1 km unless stated otherwise.

**Abbreviations:** UA = unassigned applicants; N* = the enrollment upper bound (maximum flow); z* = the minimum maximum distance; S1/S2/S3 = the matched/surplus/insufficient capacity scenarios.


---

# Part A. Materials and methods

## A.1 Materials (data)

The data hold the home locations of **1,209 applicants** together with the locations and capacities of **28 public elementary schools** in the urban area of Majene Regency, West Sulawesi (about 5 km x 6 km). Applicant and school names are replaced by codes (Student 0001, School 01, ...); only the coordinates (latitude, longitude) and the capacities are used.

### Table A1. Characteristics of the data

| Quantity | Value |
|---|---|
| Applicants | 1,209 |
| Elementary schools | 28 |
| Applicant-school pairs in total | 33,852 |
| Eligible pairs (distance <= 1 km) | 6,641 |
| Mean schools within the radius per applicant | 5.5 (median 5, max 10) |
| Applicants with no school within 1 km | 3 |
| Median distance to the nearest school | 0.185 km |
| Largest nearest-school distance | 1.105 km |

### Table A2. Capacity scenarios

| Scenario | Description | Total seats | Seats per applicant | Capacity per school | Source |
|---|---|---|---|---|---|
| S1 | Matched capacity (seats = applicants) | 1,209 | 1 | 24-47-71-74 | Approximately proportional scaling of the recorded quotas |
| S2 | Surplus capacity (recorded quotas) | 1,428 | 1.18 | 28 / 56 / 84 | Recorded data (1-3 classes x 28) |
| S3 | Insufficient capacity | 1,000 | 0.83 | 20-39-57-59 | Approximately proportional scaling of the recorded quotas |

## A.2 System framework

| Layer | Component | Output |
|---|---|---|
| 1. Data and preprocessing | CSV validation, anonymization, Haversine distances, eligible pairs E_R | Distance matrix d_ij, E_R |
| 2. LEXZON optimization | Stage 1 maximum flow -> Stage 2 bisection -> Stage 3 minimum-cost flow | Assignment x, N*, z* |
| 3. Verification and audit | Optimality certificates, cause of each unassigned applicant, priority violations, price-of-fairness curve, LEXZON-F | Proof of optimality, fairness audit |
| 4. Decision support | Interactive map, radius/capacity/extra-class simulation, export | Map, tables, files |

## A.3 Proposed method: LEXZON (Lexicographic Exact Zoning)

LEXZON solves three objectives in sequence (lexicographically); the output of each stage becomes a constraint for the next one:

1. **Stage 1 - Maximum enrollment.** Maximum flow (Dinic's algorithm) on the network source -> applicants (capacity 1) -> schools within the radius (capacity 1) -> sink (capacity q_j). Result: N*.
2. **Stage 2 - Minimum maximum distance.** Bisection over the distinct distances; each step tests whether the maximum flow restricted to pairs at distance <= t still equals N*. Result: z*.
3. **Stage 3 - Minimum total distance.** A minimum-cost flow that routes exactly N* applicants over pairs at distance <= z*, solved with the HiGHS dual simplex method; the constraint matrix is totally unimodular, so the LP solution is integral.
4. **Optimality certificates.** F(R) = F(z*) = N* and F(z-) < N*, together with the integrality of Stage 3.
5. **Labelling the unassigned.** Every unassigned applicant is labelled with its cause: no school within the radius, or every school within the radius is full.

### Algorithm 1. LEXZON

```text
Input : distances d_ij, capacities q_j, radius R
Output: assignment x, N*, z*, certificates
 1: E_R <- {(i, j) : d_ij <= R}
 2: N* <- MaxFlow(E_R, q)                            // Stage 1
 3: D  <- sorted distinct distances over E_R; lo <- 1; hi <- |D|
 4: while lo < hi do                                 // Stage 2
 5:     mid <- floor((lo + hi) / 2)
 6:     if MaxFlow(E_D[mid], q) = N* then hi <- mid else lo <- mid + 1
 7: z* <- D[lo]
 8: x <- MinCostFlow(E_z*, q, flow value N*)          // Stage 3
 9: for every applicant i left without a pair do mark unassigned
10:     cause <- no school within R, or every school within R is full
11: return x, N*, z*, (F(R), F(z*), F(z-))
```

### Algorithm 2. Price-of-fairness curve and LEXZON-F

```text
Input : d_ij, q_j, R, violation budgets K = {k1 < k2 < ...}, time limit tau
Output: curve {(k, A(k))}, LEXZON-F assignment
1: mu_L <- LEXZON(d, q, R);  mu_D <- DeferredAcceptance(d, q, R)
2: build constraints (7), (8), (15) for every (i, j) in E_R
3: (mu_F, eps_F) <- solve (17) warm-started at mu_L, time limit tau
4: mu <- mu_D; record (0, |mu_D|)                     // Proposition 2
5: for k in K with 0 < k < eps_F do
6:     mu <- solve (16) warm-started at mu, time limit tau
7:     record (k, |mu|, solver bound)
8: record (eps_F, N*); return the curve and mu_F
```

## A.4 Comparison methods

| Code | Method | How it works | Role |
|---|---|---|---|
| B1 | Nearest school | Every applicant to the nearest school within the radius, capacity ignored | Reference (infeasible) |
| B2 | Distance-ordered greedy | The nearest applicant is served first and takes the nearest school with a free seat | Common manual rule |
| B3 | Random-order greedy | The B2 rule in random order, averaged over 30 seeds | First come, first served |
| B4 | Deferred acceptance | Applicants propose to their nearest school; schools hold the nearest applicants | Distance-based admission (stable) |
| B5 | Exact minimum total distance | Stage 1 + Stage 3 (without Stage 2) | Tests whether Stage 2 is needed |
| B6 | Exact minimum maximum distance | Stage 1 + Stage 2 (without Stage 3) | Tests whether Stage 3 is needed |
| B7 | Constrained k-means | Balanced capacitated clustering (Zainuddin and Risal, 2024) | Method used in Indonesian zoning studies |

## A.5 Ablation study

Each configuration adds one component. The radius and the option to leave applicants unassigned are not introduced separately in A2, because three applicants have no school within 1 km, so a radius without that option has no solution. The objectives of A2 and A3 are **lexicographic**: enrollment is maximized first, so an empty assignment can never be optimal.

| Code | Placement constraint | Distance | Radius | Objectives (in order) | Equivalent to |
|---|---|---|---|---|---|
| A0 | Every applicant must be placed (sum_j x_ij = 1) | Euclidean (degrees) | None | min total distance | Initial thesis model |
| A1 | Every applicant must be placed (sum_j x_ij = 1) | Haversine (km) | None | min total distance | – |
| A2 | Applicants may stay unassigned (sum_j x_ij <= 1) | Haversine (km) | R = 1 km | (1) max enrollment -> (2) min total distance | B5 |
| A3 | Applicants may stay unassigned (sum_j x_ij <= 1) | Haversine (km) | R = 1 km | (1) max enrollment -> (2) min maximum distance -> (3) min total distance | LEXZON |

## A.6 Evaluation metrics

| Metric | Definition |
|---|---|
| Unassigned (UA) | Applicants left without a school, split into no school within the radius / every school full |
| Gap to N* | N* minus the number of assigned applicants (0 = optimal) |
| Capacity overflow | Applicants placed beyond the capacity of a school (feasibility) |
| Mean / max / total distance | Haversine distance from an assigned applicant to its school (km) |
| Priority violations (justified envy) | Applicants who prefer a school within the radius that still has a free seat or admits a farther applicant |
| Runtime | Seconds (median of three repetitions for the scalability study) |

## A.7 Experimental protocol

| No | Experiment | Setting |
|---|---|---|
| 1 | Method comparison | 3 scenarios, R = 1 km, 8 methods |
| 2 | Ablation study | S1, S2, S3; A0-A3 |
| 3 | Radius sensitivity | R = 0.5, 0.75, 1, 1.25, 1.5, 2, 3 km; unrestricted |
| 4 | Statistical test | 30 subsamples (80% of the applicants, seed 42 + index), capacities scaled; one-sided Wilcoxon, alpha = 0.05 |
| 5 | Scalability | 1,000-50,000 synthetic applicants (recorded locations plus Gaussian noise of 150 m), median of 3 repetitions |
| 6 | Distance-priority fairness | Justified envy for every method, on the full data and on 90 subsamples |
| 7 | Price of fairness | k in {0, 10, 25, 50, 100, 150, 200}; 300 s per point; LEXZON-F 900 s (CP-SAT) |
| 8 | Policy simulation | One extra class (28 seats) at each school in turn |
| 9 | Software testing | 126 unit tests (pytest) + 22 end-to-end tests (Playwright) |

## A.8 Implementation environment

| Component | Version / specification |
|---|---|
| Language | Python 3.11.9 |
| Numerics | NumPy 1.26.4, SciPy 1.17.1, pandas 3.0.2 |
| Maximum flow | scipy.sparse.csgraph.maximum_flow (Dinic) |
| LP/MILP | HiGHS (dual simplex) through SciPy |
| Price-of-fairness model | Google OR-Tools CP-SAT |
| Application | Streamlit 1.57.0, Folium 0.20.0 |
| Hardware | Intel Core i7-14650HX, 16 GB RAM, Windows 11 |


---

# Part B. Formulas

**(1) Haversine distance**

$$
d_{ij} = 2\rho \arcsin\sqrt{a_{ij}}
$$

**(2) Haversine component**

$$
a_{ij} = \sin^2\!\left(\frac{\Delta\varphi}{2}\right) + \cos\varphi_i \cos\varphi_j \sin^2\!\left(\frac{\Delta\lambda}{2}\right),\quad \rho = 6371\ \text{km}
$$

**(3) Eligible pairs**

$$
E_R = \{(i,j) \in S \times C : d_{ij} \le R\}
$$

**(4) Lexicographic objective**

$$
\operatorname{lex\,max}_x \big(f_1(x),\ -f_2(x),\ -f_3(x)\big)
$$

**(5) Objectives 1 and 3**

$$
f_1(x) = \sum_{(i,j)\in E_R} x_{ij},\qquad f_3(x) = \sum_{(i,j)\in E_R} d_{ij}\,x_{ij}
$$

**(6) Objective 2**

$$
f_2(x) = \max_{(i,j)\in E_R} d_{ij}\,x_{ij}
$$

**(7) Applicant constraint**

$$
\sum_{j:(i,j)\in E_R} x_{ij} \le 1,\quad \forall i \in S
$$

**(8) Capacity constraint**

$$
\sum_{i:(i,j)\in E_R} x_{ij} \le q_j,\quad \forall j \in C,\quad x_{ij}\in\{0,1\}
$$

**(9) Thresholded maximum flow**

$$
F(t) = \max\Big\{\sum_{d_{ij}\le t} x_{ij} : x \text{ feasible}\Big\},\qquad N^* = F(R)
$$

**(10) Minimum maximum distance**

$$
z^* = \min\{t \in D_R : F(t) = N^*\}
$$

**(11) Minimum total distance**

$$
\min \sum_{d_{ij}\le z^*} d_{ij}\,x_{ij}\quad \text{s.t. } \sum x_{ij} = N^*,\ (7),(8)
$$

**(12) Optimality certificate**

$$
F(z^*) = N^* \ \text{and}\ F(z^-) < N^*
$$

**(13) Distance priorities**

$$
i \succ_j i' \Leftrightarrow d_{ij} < d_{i'j},\qquad j \succ_i j' \Leftrightarrow d_{ij} < d_{ij'}
$$

**(14) Priority violation**

$$
\varepsilon_i = 1 \Leftrightarrow \exists (i,j)\in E_R : j \succ_i \mu(i) \ \wedge\ \big(|\mu^{-1}(j)| < q_j \ \vee\ \exists i'\in\mu^{-1}(j): i \succ_j i'\big)
$$

**(15) Weighted stability**

$$
\sum_{i' \succ_j i} x_{i'j} + q_j \sum_{j' \succeq_i j} x_{ij'} \ge q_j\,(1 - y_i)
$$

**(16) Price-of-fairness curve**

$$
A(k) = \max \sum_{(i,j)\in E_R} x_{ij}\quad \text{s.t. } (7),(8),(15),\ \sum_i y_i \le k
$$

**(17) LEXZON-F**

$$
\varepsilon_F = \min \sum_i y_i\quad \text{s.t. } (7),(8),(15),\ \sum_{(i,j)\in E_R} x_{ij} = N^*
$$

**Additional metrics**

$$
\text{Reduction in UA}(\%) = \frac{UA_{\text{comparator}} - UA_{\text{LEXZON}}}{UA_{\text{comparator}}}\times 100
$$

$$
\text{Gap to } N^* = N^* - \sum_{(i,j)} x_{ij}
$$

**Proposition 1.** Every LEXZON assignment (a) leaves no unassigned applicant who has a school within the radius with a free seat, and (b) rejects no applicant who is closer to a school than an applicant admitted to that school.

**Proposition 2.** A(0) equals the enrollment of deferred acceptance (the rural hospitals theorem), and A(k) = N* for k >= eps_F.


---

# Part C. Results

## C.1 Software testing

| Test type | Count | Passed | Notes |
|---|---|---|---|
| Solver unit tests (brute-force optimum, bounds, feasibility) | 50 | 50 | test_solver.py |
| Fairness and constrained k-means unit tests | 52 | 52 | test_fairness.py |
| Price-of-fairness model unit tests (Proposition 2) | 24 | 24 | test_pareto.py |
| End-to-end application tests (browser) | 22 | 22 | TC-01 to TC-22 |
| Reproducibility (29 result files rerun) | 29 | 29 | Identical, apart from runtimes |

## C.2 LEXZON stage results per scenario

| Scenario | Seats | N* | z* (km) | Bisection steps | F(z-) | Schools full | Forced model A0: beyond R |
|---|---|---|---|---|---|---|---|
| S1 | 1,209 | 1,141 | 0.995 | 13 | 1,140 | 22 | 121 |
| S2 | 1,428 | 1,180 | 0.995 | 13 | 1,179 | 18 | 47 |
| S3 | 1,000 | 986 | 0.859 | 13 | 985 | 27 | infeasible |

## C.3 Method comparison (R = 1 km)

### Table C3-S1. Method comparison - S1: matched capacity (seats = applicants)

| Method | Assigned | UA | UA: no school | UA: schools full | Gap to N* | Capacity overflow | Mean (km) | Max (km) | Total (km) | Gini | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LEXZON | 1,141 | 68 | 3 | 65 | 0 | 0 | 0.321 | 0.995 | 366.1 | 0.414 | 0.047 |
| B1 Nearest school (capacity ignored) | 1,206 | 3 | 3 | 0 | – | 246 | 0.215 | 0.963 | 259.3 | 0.347 | 0.000 |
| B2 Distance-ordered greedy | 1,057 | 152 | 3 | 149 | 84 | 0 | 0.239 | 0.990 | 252.5 | 0.424 | 0.002 |
| B3 Random-order greedy (mean of 30 seeds) | 1,043.1 | 165.9 | 3 | 162.9 | 97.9 | 0 | 0.267 | 0.991 | 278.7 | 0.386 | 0.002 |
| B4 Deferred acceptance | 1,026 | 183 | 3 | 180 | 115 | 0 | 0.200 | 0.990 | 205.1 | 0.360 | 0.007 |
| B5 Exact minimum total distance | 1,141 | 68 | 3 | 65 | 0 | 0 | 0.321 | 0.995 | 366.1 | 0.414 | 0.038 |
| B6 Exact minimum maximum distance | 1,141 | 68 | 3 | 65 | 0 | 0 | 0.582 | 0.995 | 663.8 | 0.253 | 0.009 |
| B7 Constrained k-means | 1,066 | 143 | 3 | 140 | 75 | 0 | 0.339 | 0.995 | 361.6 | 0.363 | 12.715 |

### Table C3-S2. Method comparison - S2: surplus capacity (recorded quotas)

| Method | Assigned | UA | UA: no school | UA: schools full | Gap to N* | Capacity overflow | Mean (km) | Max (km) | Total (km) | Gini | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LEXZON | 1,180 | 29 | 3 | 26 | 0 | 0 | 0.275 | 0.995 | 323.9 | 0.401 | 0.058 |
| B1 Nearest school (capacity ignored) | 1,206 | 3 | 3 | 0 | – | 184 | 0.215 | 0.963 | 259.3 | 0.347 | 0 |
| B2 Distance-ordered greedy | 1,117 | 92 | 3 | 89 | 63 | 0 | 0.230 | 0.970 | 256.9 | 0.399 | 0.002 |
| B3 Random-order greedy (mean of 30 seeds) | 1,113.1 | 95.9 | 3 | 92.9 | 66.9 | 0 | 0.254 | 0.990 | 282.9 | 0.379 | 0.003 |
| B4 Deferred acceptance | 1,107 | 102 | 3 | 99 | 73 | 0 | 0.213 | 0.854 | 235.9 | 0.369 | 0.010 |
| B5 Exact minimum total distance | 1,180 | 29 | 3 | 26 | 0 | 0 | 0.275 | 0.995 | 323.9 | 0.401 | 0.053 |
| B6 Exact minimum maximum distance | 1,180 | 29 | 3 | 26 | 0 | 0 | 0.625 | 0.995 | 737.9 | 0.248 | 0.011 |
| B7 Constrained k-means | 1,147 | 62 | 3 | 59 | 33 | 0 | 0.322 | 0.995 | 369.5 | 0.377 | 13.305 |

### Table C3-S3. Method comparison - S3: insufficient capacity

| Method | Assigned | UA | UA: no school | UA: schools full | Gap to N* | Capacity overflow | Mean (km) | Max (km) | Total (km) | Gini | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LEXZON | 986 | 223 | 3 | 220 | 0 | 0 | 0.230 | 0.859 | 226.6 | 0.401 | 0.046 |
| B1 Nearest school (capacity ignored) | 1,206 | 3 | 3 | 0 | – | 353 | 0.215 | 0.963 | 259.3 | 0.347 | 0.000 |
| B2 Distance-ordered greedy | 976 | 233 | 3 | 230 | 10 | 0 | 0.265 | 0.988 | 258.5 | 0.466 | 0.003 |
| B3 Random-order greedy (mean of 30 seeds) | 952.9 | 256.1 | 3 | 253.1 | 33.1 | 0 | 0.290 | 0.994 | 276.0 | 0.394 | 0.003 |
| B4 Deferred acceptance | 940 | 269 | 3 | 266 | 46 | 0 | 0.203 | 0.957 | 191.3 | 0.393 | 0.010 |
| B5 Exact minimum total distance | 986 | 223 | 3 | 220 | 0 | 0 | 0.227 | 0.975 | 223.4 | 0.403 | 0.052 |
| B6 Exact minimum maximum distance | 986 | 223 | 3 | 220 | 0 | 0 | 0.488 | 0.859 | 481.5 | 0.284 | 0.018 |
| B7 Constrained k-means | 984 | 225 | 3 | 222 | 2 | 0 | 0.344 | 0.995 | 338.3 | 0.358 | 17.616 |

### Table C3-R. Reduction in unassigned applicants achieved by LEXZON (%)

| Comparator | S1 | S2 | S3 |
|---|---|---|---|
| B2 Distance-ordered greedy | 55.3 | 68.5 | 4.3 |
| B3 Random-order greedy (mean of 30 seeds) | 59.0 | 69.8 | 12.9 |
| B4 Deferred acceptance | 62.8 | 71.6 | 17.1 |
| B7 Constrained k-means | 52.4 | 53.2 | 0.9 |

## C.4 Ablation study

| Scenario | Code | Configuration | Status | Assigned | UA | Beyond R | Max (km) | Total (km) |
|---|---|---|---|---|---|---|---|---|
| S1 | A0 | Initial model (Euclidean degrees, every applicant placed) | Feasible | 1,209 | 0 | 121 | 2.230 | 489.2 |
| S1 | A1 | + Haversine (km) | Feasible | 1,209 | 0 | 121 | 2.230 | 489.2 |
| S1 | A2 | + radius R = 1 km with applicants allowed to stay unassigned | Feasible | 1,141 | 68 | 0 | 0.995 | 366.1 |
| S1 | A3 | + minimum maximum distance stage (LEXZON) | Feasible | 1,141 | 68 | 0 | 0.995 | 366.1 |
| S2 | A0 | Initial model (Euclidean degrees, every applicant placed) | Feasible | 1,209 | 0 | 47 | 2.163 | 378.3 |
| S2 | A1 | + Haversine (km) | Feasible | 1,209 | 0 | 47 | 2.163 | 378.3 |
| S2 | A2 | + radius R = 1 km with applicants allowed to stay unassigned | Feasible | 1,180 | 29 | 0 | 0.995 | 323.9 |
| S2 | A3 | + minimum maximum distance stage (LEXZON) | Feasible | 1,180 | 29 | 0 | 0.995 | 323.9 |
| S3 | A0 | Initial model (Euclidean degrees, every applicant placed) | Infeasible (seats < applicants) | – | – | – | – | – |
| S3 | A1 | + Haversine (km) | Infeasible (seats < applicants) | – | – | – | – | – |
| S3 | A2 | + radius R = 1 km with applicants allowed to stay unassigned | Feasible | 986 | 223 | 0 | 0.975 | 223.4 |
| S3 | A3 | + minimum maximum distance stage (LEXZON) | Feasible | 986 | 223 | 0 | 0.859 | 226.6 |

## C.5 Radius sensitivity

### Table C5-a. Unassigned applicants by radius

| Scenario | Radius (km) | LEXZON | B5 | B2 | B4 |
|---|---|---|---|---|---|
| S1 | 0.50 | 177 | 177 | 217 | 218 |
| S1 | 0.75 | 85 | 85 | 177 | 199 |
| S1 | 1 | 68 | 68 | 152 | 183 |
| S1 | 1.25 | 52 | 52 | 142 | 177 |
| S1 | 1.50 | 49 | 49 | 123 | 153 |
| S1 | 2 | 0 | 0 | 48 | 86 |
| S1 | 3 | 0 | 0 | 12 | 12 |
| S1 | inf | 0 | 0 | 0 | 0 |
| S2 | 0.50 | 110 | 110 | 160 | 163 |
| S2 | 0.75 | 53 | 53 | 112 | 122 |
| S2 | 1 | 29 | 29 | 92 | 102 |
| S2 | 1.25 | 13 | 13 | 89 | 98 |
| S2 | 1.50 | 10 | 10 | 78 | 90 |
| S2 | 2 | 0 | 0 | 22 | 23 |
| S2 | 3 | 0 | 0 | 0 | 0 |
| S2 | inf | 0 | 0 | 0 | 0 |
| S3 | 0.50 | 277 | 277 | 330 | 331 |
| S3 | 0.75 | 255 | 255 | 283 | 290 |
| S3 | 1 | 223 | 223 | 233 | 269 |
| S3 | 1.25 | 223 | 223 | 223 | 256 |
| S3 | 1.50 | 209 | 209 | 223 | 232 |
| S3 | 2 | 209 | 209 | 209 | 209 |
| S3 | 3 | 209 | 209 | 209 | 209 |
| S3 | inf | 209 | 209 | 209 | 209 |

### Table C5-b. Maximum distance (km) by radius

| Scenario | Radius (km) | LEXZON | B5 | B2 | B4 |
|---|---|---|---|---|---|
| S1 | 0.50 | 0.499 | 0.499 | 0.499 | 0.499 |
| S1 | 0.75 | 0.748 | 0.749 | 0.744 | 0.733 |
| S1 | 1 | 0.995 | 0.995 | 0.990 | 0.990 |
| S1 | 1.25 | 1.241 | 1.247 | 1.218 | 1.218 |
| S1 | 1.50 | 1.319 | 1.474 | 1.490 | 1.490 |
| S1 | 2 | 1.993 | 1.998 | 1.987 | 1.998 |
| S1 | 3 | 1.993 | 2.230 | 2.839 | 2.816 |
| S1 | inf | 1.993 | 2.230 | 3.693 | 4.321 |
| S2 | 0.50 | 0.499 | 0.499 | 0.499 | 0.499 |
| S2 | 0.75 | 0.744 | 0.746 | 0.750 | 0.746 |
| S2 | 1 | 0.995 | 0.995 | 0.970 | 0.854 |
| S2 | 1.25 | 1.241 | 1.241 | 1.198 | 1.218 |
| S2 | 1.50 | 1.319 | 1.403 | 1.482 | 1.493 |
| S2 | 2 | 1.901 | 1.987 | 1.987 | 1.987 |
| S2 | 3 | 1.901 | 2.163 | 2.516 | 2.516 |
| S2 | inf | 1.901 | 2.163 | 2.516 | 2.516 |
| S3 | 0.50 | 0.499 | 0.500 | 0.499 | 0.499 |
| S3 | 0.75 | 0.733 | 0.733 | 0.749 | 0.733 |
| S3 | 1 | 0.859 | 0.975 | 0.988 | 0.957 |
| S3 | 1.25 | 0.859 | 1.115 | 1.241 | 1.240 |
| S3 | 1.50 | 1.478 | 1.500 | 1.498 | 1.345 |
| S3 | 2 | 1.478 | 1.802 | 1.992 | 1.923 |
| S3 | 3 | 1.478 | 1.802 | 2.918 | 1.923 |
| S3 | inf | 1.478 | 1.802 | 5.216 | 1.923 |

### Table C5-c. Total distance (km) by radius

| Scenario | Radius (km) | LEXZON | B5 | B2 | B4 |
|---|---|---|---|---|---|
| S1 | 0.50 | 217.1 | 217.1 | 184.5 | 180.5 |
| S1 | 0.75 | 367.3 | 367.1 | 221.4 | 191.8 |
| S1 | 1 | 366.1 | 366.1 | 252.5 | 205.1 |
| S1 | 1.25 | 386.1 | 386.1 | 267.9 | 211.9 |
| S1 | 1.50 | 390.1 | 386.9 | 299.6 | 244.9 |
| S1 | 2 | 506.4 | 504.8 | 436.2 | 357.9 |
| S1 | 3 | 506.4 | 489.2 | 532.5 | 531.8 |
| S1 | inf | 506.4 | 489.2 | 572.1 | 576.1 |
| S2 | 0.50 | 232.8 | 232.8 | 197.9 | 194.2 |
| S2 | 0.75 | 288.6 | 288.5 | 234.9 | 220.0 |
| S2 | 1 | 323.9 | 323.9 | 256.9 | 235.9 |
| S2 | 1.25 | 356.5 | 356.5 | 260.9 | 240.6 |
| S2 | 1.50 | 362.9 | 362.7 | 281.5 | 251.8 |
| S2 | 2 | 379.6 | 379.4 | 379.8 | 367.9 |
| S2 | 3 | 379.6 | 378.3 | 428.0 | 418.4 |
| S2 | inf | 379.6 | 378.3 | 428.0 | 418.4 |
| S3 | 0.50 | 184.6 | 184.6 | 154.7 | 148.9 |
| S3 | 0.75 | 192.6 | 192.6 | 199.3 | 173.5 |
| S3 | 1 | 226.6 | 223.4 | 258.5 | 191.3 |
| S3 | 1.25 | 226.6 | 223.1 | 285.5 | 206.5 |
| S3 | 1.50 | 251.6 | 251.3 | 310.7 | 237.5 |
| S3 | 2 | 251.6 | 246.3 | 379.9 | 275.8 |
| S3 | 3 | 251.6 | 246.3 | 434.9 | 275.8 |
| S3 | inf | 251.6 | 246.3 | 454.1 | 275.8 |

## C.6 One-sided Wilcoxon test (30 subsamples, 80% of the applicants)

| Scenario | Comparator | UA LEXZON (mean ± sd) | UA comparator (mean ± sd) | Mean difference | Better/tied/worse | W | p-value | Conclusion |
|---|---|---|---|---|---|---|---|---|
| S1 | B2 | 54.5 ± 6.5 | 123.4 ± 6.0 | 69.0 | 30/0/0 | 465 | 8.5e-07 | Significant |
| S1 | B3 | 54.5 ± 6.5 | 134.3 ± 5.9 | 79.8 | 30/0/0 | 465 | 8.3e-07 | Significant |
| S1 | B4 | 54.5 ± 6.5 | 147.7 ± 6.9 | 93.3 | 30/0/0 | 465 | 8.5e-07 | Significant |
| S1 | B5 | 54.5 ± 6.5 | 54.5 ± 6.5 | 0 | 0/30/0 | – | – | Identical |
| S1 | B6 | 54.5 ± 6.5 | 54.5 ± 6.5 | 0 | 0/30/0 | – | – | Identical |
| S1 | B7 | 54.5 ± 6.5 | 103.7 ± 17.9 | 49.2 | 30/0/0 | 465 | 8.6e-07 | Significant |
| S2 | B2 | 23.2 ± 6.0 | 71.4 ± 6.2 | 48.2 | 30/0/0 | 465 | 8.4e-07 | Significant |
| S2 | B3 | 23.2 ± 6.0 | 76.6 ± 6.0 | 53.4 | 30/0/0 | 465 | 8.5e-07 | Significant |
| S2 | B4 | 23.2 ± 6.0 | 80.8 ± 6.8 | 57.6 | 30/0/0 | 465 | 8.5e-07 | Significant |
| S2 | B5 | 23.2 ± 6.0 | 23.2 ± 6.0 | 0 | 0/30/0 | – | – | Identical |
| S2 | B6 | 23.2 ± 6.0 | 23.2 ± 6.0 | 0 | 0/30/0 | – | – | Identical |
| S2 | B7 | 23.2 ± 6.0 | 59.8 ± 15.2 | 36.7 | 30/0/0 | 465 | 8.6e-07 | Significant |
| S3 | B2 | 178.2 ± 2.6 | 185.7 ± 5.0 | 7.5 | 28/2/0 | 406 | 1.9e-06 | Significant |
| S3 | B3 | 178.2 ± 2.6 | 205.4 ± 4.4 | 27.2 | 30/0/0 | 465 | 8.4e-07 | Significant |
| S3 | B4 | 178.2 ± 2.6 | 215.0 ± 7.0 | 36.8 | 30/0/0 | 465 | 8.6e-07 | Significant |
| S3 | B5 | 178.2 ± 2.6 | 178.2 ± 2.6 | 0 | 0/30/0 | – | – | Identical |
| S3 | B6 | 178.2 ± 2.6 | 178.2 ± 2.6 | 0 | 0/30/0 | – | – | Identical |
| S3 | B7 | 178.2 ± 2.6 | 185.8 ± 12.9 | 7.6 | 27/3/0 | 378 | 2.7e-06 | Significant |

## C.7 Scalability (synthetic data, median of 3 repetitions)

| Applicants | Seats | Eligible pairs | Assigned | N* | Stage 1 (s) | Stage 2 (s) | Stage 3 (s) | LEXZON total (s) | Deferred acceptance (s) |
|---|---|---|---|---|---|---|---|---|---|
| 1,000 | 1,181 | 5,435 | 984 | 984 | 0.0008 | 0.0083 | 0.0356 | 0.044 | 0.009 |
| 2,000 | 2,362 | 10,850 | 1,971 | 1,971 | 0.0014 | 0.0169 | 0.0789 | 0.097 | 0.014 |
| 5,000 | 5,906 | 26,961 | 4,946 | 4,946 | 0.0029 | 0.0396 | 0.3056 | 0.346 | 0.037 |
| 10,000 | 11,811 | 53,742 | 9,813 | 9,813 | 0.0054 | 0.0761 | 1.1766 | 1.257 | 0.077 |
| 20,000 | 23,623 | 107,970 | 19,490 | 19,490 | 0.0117 | 0.1827 | 3.8214 | 4.015 | 0.196 |
| 50,000 | 59,057 | 270,097 | 48,769 | 48,769 | 0.0397 | 0.6186 | 11.2154 | 11.867 | 0.480 |

## C.8 Distance-priority fairness (R = 1 km, full data)

| Scenario | Method | UA | Applicants with a violation | Rejected although closer | Assigned away from the nearest school | Through a vacant seat | % of applicants | Median extra distance (km) |
|---|---|---|---|---|---|---|---|---|
| S1 | LEXZON | 68 | 376 | 0 | 376 | 0 | 31.1 | 0.317 |
| S1 | B2 Distance-ordered greedy | 152 | 278 | 135 | 143 | 0 | 23.0 | 0.280 |
| S1 | B3 Random-order greedy (mean of 30 seeds) | 165.9 | 328.5 | 161.8 | 166.7 | 0 | 27.2 | 0.304 |
| S1 | B4 Deferred acceptance | 183 | 0 | 0 | 0 | 0 | 0 | 0 |
| S1 | B5 Exact minimum total distance | 68 | 376 | 0 | 376 | 0 | 31.1 | 0.317 |
| S1 | B6 Exact minimum maximum distance | 68 | 1,001 | 64 | 937 | 161 | 82.8 | 0.453 |
| S1 | B7 Constrained k-means | 143 | 653 | 127 | 526 | 124 | 54.0 | 0.227 |
| S1 | LEXZON-F | 68 | 250 | 0 | 250 | 26 | 20.7 | 0.599 |
| S2 | LEXZON | 29 | 281 | 0 | 281 | 0 | 23.2 | 0.185 |
| S2 | B2 Distance-ordered greedy | 92 | 153 | 78 | 75 | 0 | 12.7 | 0.325 |
| S2 | B3 Random-order greedy (mean of 30 seeds) | 95.9 | 232.5 | 92.4 | 140.1 | 0 | 19.2 | 0.290 |
| S2 | B4 Deferred acceptance | 102 | 0 | 0 | 0 | 0 | 0 | 0 |
| S2 | B5 Exact minimum total distance | 29 | 281 | 0 | 281 | 0 | 23.2 | 0.185 |
| S2 | B6 Exact minimum maximum distance | 29 | 971 | 25 | 946 | 476 | 80.3 | 0.529 |
| S2 | B7 Constrained k-means | 62 | 552 | 59 | 493 | 372 | 45.7 | 0.171 |
| S2 | LEXZON-F | 29 | 158 | 0 | 158 | 21 | 13.1 | 0.656 |
| S3 | LEXZON | 223 | 193 | 0 | 193 | 0 | 16.0 | 0.152 |
| S3 | B2 Distance-ordered greedy | 233 | 405 | 208 | 197 | 0 | 33.5 | 0.379 |
| S3 | B3 Random-order greedy (mean of 30 seeds) | 256.1 | 455.5 | 251 | 204.5 | 0 | 37.7 | 0.330 |
| S3 | B4 Deferred acceptance | 269 | 0 | 0 | 0 | 0 | 0 | 0 |
| S3 | B5 Exact minimum total distance | 223 | 173 | 0 | 173 | 0 | 14.3 | 0.138 |
| S3 | B6 Exact minimum maximum distance | 223 | 957 | 217 | 740 | 0 | 79.2 | 0.359 |
| S3 | B7 Constrained k-means | 225 | 636 | 113 | 523 | 28 | 52.6 | 0.320 |
| S3 | LEXZON-F | 223 | 73 | 0 | 73 | 0 | 6.0 | 0.523 |

### Table C8-R. Fairness over 30 subsamples (means)

| Scenario | Method | UA | Applicants with a violation | Rejected although closer | % of applicants |
|---|---|---|---|---|---|
| S1 | B2 Distance-ordered greedy | 123.4 | 216.7 | 108.9 | 22.4 |
| S1 | B3 Random-order greedy (mean of 30 seeds) | 134.3 | 265.5 | 130.8 | 27.5 |
| S1 | B4 Deferred acceptance | 147.7 | 0 | 0 | 0 |
| S1 | B5 Exact minimum total distance | 54.5 | 303.6 | 0 | 31.4 |
| S1 | B6 Exact minimum maximum distance | 54.5 | 798.7 | 51.6 | 82.6 |
| S1 | B7 Constrained k-means | 103.7 | 516.8 | 90.9 | 53.4 |
| S1 | LEXZON | 54.5 | 303.8 | 0 | 31.4 |
| S2 | B2 Distance-ordered greedy | 71.4 | 132.8 | 60.9 | 13.7 |
| S2 | B3 Random-order greedy (mean of 30 seeds) | 76.6 | 187.2 | 73.7 | 19.4 |
| S2 | B4 Deferred acceptance | 80.8 | 0 | 0 | 0 |
| S2 | B5 Exact minimum total distance | 23.2 | 225.9 | 0 | 23.4 |
| S2 | B6 Exact minimum maximum distance | 23.2 | 775.1 | 20.3 | 80.2 |
| S2 | B7 Constrained k-means | 59.8 | 471.1 | 57.6 | 48.7 |
| S2 | LEXZON | 23.2 | 225.9 | 0 | 23.4 |
| S3 | B2 Distance-ordered greedy | 185.7 | 309.3 | 162.5 | 32.0 |
| S3 | B3 Random-order greedy (mean of 30 seeds) | 205.4 | 363.6 | 201.1 | 37.6 |
| S3 | B4 Deferred acceptance | 215 | 0 | 0 | 0 |
| S3 | B5 Exact minimum total distance | 178.2 | 142.1 | 0 | 14.7 |
| S3 | B6 Exact minimum maximum distance | 178.2 | 764.2 | 173.2 | 79.0 |
| S3 | B7 Constrained k-means | 185.8 | 424.1 | 88.9 | 43.9 |
| S3 | LEXZON | 178.2 | 156.2 | 0 | 16.2 |

## C.9 Price-of-fairness curve

| Scenario | Point | k | Assigned | Applicants with a violation | Solver bound* | Proved optimal | Runtime (s) | N* |
|---|---|---|---|---|---|---|---|---|
| S1 | LEXZON | – | 1,141 | 376 | – | Yes | 0 | 1,141 |
| S1 | Deferred acceptance | – | 1,026 | 0 | – | Yes | 0 | 1,141 |
| S1 | LEXZON-F (minimum violations at N*) | – | 1,141 | 250 | 227 | No | 900.6 | 1,141 |
| S1 | Violation budget k | 0 | 1,026 | 0 | 1,026 | Yes | 0.4 | 1,141 |
| S1 | Violation budget k | 10 | 1,036 | 10 | 1,036 | Yes | 46.8 | 1,141 |
| S1 | Violation budget k | 25 | 1,051 | 25 | 1,053 | No | 300.5 | 1,141 |
| S1 | Violation budget k | 50 | 1,072 | 50 | 1,077 | No | 304.2 | 1,141 |
| S1 | Violation budget k | 100 | 1,099 | 100 | 1,107 | No | 302.1 | 1,141 |
| S1 | Violation budget k | 150 | 1,110 | 150 | 1,124 | No | 300.8 | 1,141 |
| S1 | Violation budget k | 200 | 1,124 | 200 | 1,138 | No | 300.8 | 1,141 |
| S2 | LEXZON | – | 1,180 | 281 | – | Yes | 0 | 1,180 |
| S2 | Deferred acceptance | – | 1,107 | 0 | – | Yes | 0 | 1,180 |
| S2 | LEXZON-F (minimum violations at N*) | – | 1,180 | 158 | 133 | No | 901.2 | 1,180 |
| S2 | Violation budget k | 0 | 1,107 | 0 | 1,107 | Yes | 0.9 | 1,180 |
| S2 | Violation budget k | 10 | 1,117 | 10 | 1,117 | Yes | 25.1 | 1,180 |
| S2 | Violation budget k | 25 | 1,132 | 25 | 1,132 | Yes | 39.1 | 1,180 |
| S2 | Violation budget k | 50 | 1,148 | 50 | 1,151 | No | 301.3 | 1,180 |
| S2 | Violation budget k | 100 | 1,163 | 100 | 1,173 | No | 300.9 | 1,180 |
| S2 | Violation budget k | 150 | 1,177 | 150 | 1,180 | No | 300.9 | 1,180 |
| S3 | LEXZON | – | 986 | 193 | – | Yes | 0 | 986 |
| S3 | Deferred acceptance | – | 940 | 0 | – | Yes | 0 | 986 |
| S3 | LEXZON-F (minimum violations at N*) | – | 986 | 73 | 50 | No | 900.7 | 986 |
| S3 | Violation budget k | 0 | 940 | 0 | 940 | Yes | 0.5 | 986 |
| S3 | Violation budget k | 10 | 950 | 10 | 950 | Yes | 120.8 | 986 |
| S3 | Violation budget k | 25 | 962 | 25 | 964 | No | 300.8 | 986 |
| S3 | Violation budget k | 50 | 976 | 50 | 986 | No | 300.9 | 986 |

*For a budget point: the upper bound on enrollment; for LEXZON-F: the lower bound on the number of violations.

### Table C9-R. Price-of-fairness summary

| Scenario | Assigned by DA (k = 0) | N* | Additional enrollment | % of the gain reached at k = 50 | LEXZON violations | LEXZON-F violations | Reduction in violations (%) | Increase in LEXZON-F total distance (%) |
|---|---|---|---|---|---|---|---|---|
| S1 | 1,026 | 1,141 | 115 | 40 | 376 | 250 | 33.5 | 8.2 |
| S2 | 1,107 | 1,180 | 73 | 56.2 | 281 | 158 | 43.8 | 10.7 |
| S3 | 940 | 986 | 46 | 78.3 | 193 | 73 | 62.2 | 8.6 |

## C.10 Policy simulation: adding one class (28 seats)

| School | UA nearest S1 | UA nearest S2 | UA nearest S3 | Reduction with +1 class S1 | Reduction with +1 class S2 | Reduction with +1 class S3 |
|---|---|---|---|---|---|---|
| ES 01 | 0 | 0 | 5 | 0 | 0 | 28 |
| ES 02 | 0 | 0 | 0 | 0 | 0 | 28 |
| ES 03 | 0 | 0 | 1 | 0 | 0 | 28 |
| ES 04 | 0 | 0 | 34 | 0 | 0 | 28 |
| ES 05 | 0 | 0 | 5 | 0 | 0 | 28 |
| ES 06 | 0 | 0 | 15 | 0 | 0 | 28 |
| ES 07 | 45 | 26 | 55 | 28 | 23 | 28 |
| ES 08 | 0 | 0 | 0 | 0 | 0 | 28 |
| ES 09 | 0 | 0 | 0 | 0 | 0 | 28 |
| ES 10 | 0 | 0 | 0 | 0 | 0 | 28 |
| ES 11 | 8 | 0 | 17 | 28 | 23 | 28 |
| ES 13 | 0 | 0 | 0 | 0 | 0 | 28 |
| ES 14 | 0 | 0 | 0 | 0 | 0 | 28 |
| ES 15 | 0 | 0 | 0 | 0 | 0 | 28 |
| ES 16 | 0 | 0 | 0 | 0 | 0 | 28 |
| ES 17 | 2 | 0 | 5 | 28 | 23 | 28 |
| ES 18 | 0 | 0 | 0 | 0 | 0 | 28 |
| ES 19 | 0 | 0 | 0 | 0 | 0 | 28 |
| ES 20 | 0 | 0 | 2 | 0 | 0 | 28 |
| ES 21 | 7 | 3 | 11 | 7 | 3 | 11 |
| ES 22 | 6 | 0 | 33 | 28 | 23 | 28 |
| ES 23 | 0 | 0 | 0 | 12 | 3 | 28 |
| ES 24 | 0 | 0 | 0 | 0 | 0 | 28 |
| ES 25 | 0 | 0 | 0 | 0 | 0 | 28 |
| ES 26 | 0 | 0 | 36 | 0 | 0 | 28 |
| ES 27 | 0 | 0 | 0 | 0 | 0 | 28 |
| ES 28 | 0 | 0 | 4 | 0 | 0 | 28 |

| Scenario | UA before | Best UA after +1 class | Schools with the largest reduction | Schools with no reduction |
|---|---|---|---|---|
| S1 | 68 | 40 | 4 | 22 |
| S2 | 29 | 6 | 4 | 22 |
| S3 | 223 | 195 | 26 | 1 |
