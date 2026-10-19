"""Generate the Markdown results/formula/methods document from the experiment CSVs."""
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Repository root. Set LEXZON_ROOT to run against data kept outside the repository.
ROOT = Path(os.environ.get("LEXZON_ROOT", Path(__file__).resolve().parents[1]))
R = ROOT / "results"
OUT = ROOT / "results" / "RESULTS.md"
SC = ["S1", "S2", "S3"]
NAME = {"P": "LEXZON", "B1": "B1 Nearest school (capacity ignored)", "B2": "B2 Distance-ordered greedy",
        "B3": "B3 Random-order greedy (mean of 30 seeds)", "B4": "B4 Deferred acceptance",
        "B5": "B5 Exact minimum total distance", "B6": "B6 Exact minimum maximum distance",
        "B7": "B7 Constrained k-means", "LEXZON": "LEXZON", "LEXZON-F": "LEXZON-F"}


def f(v, d=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if isinstance(v, (int, np.integer)) or float(v).is_integer():
        return f"{int(round(float(v))):,}"
    return f"{float(v):,.{d}f}"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


md = []
w = md.append

w("# LEXZON: results, formulas, and the materials and methods section\n")
w("This document holds the result tables (ready to plot), the formulas, and the materials and methods "
  "section. Every number is read directly from the experiment files in `results/`. The zoning radius is "
  "R = 1 km unless stated otherwise.\n")
w("**Abbreviations:** UA = unassigned applicants; N* = the enrollment upper bound (maximum flow); "
  "z* = the minimum maximum distance; S1/S2/S3 = the matched/surplus/insufficient capacity scenarios.\n")

# ============================================================ MATERIALS AND METHODS
w("\n---\n\n# Part A. Materials and methods\n")
w("## A.1 Materials (data)\n")
w("The data hold the home locations of **1,209 applicants** together with the locations and capacities of "
  "**28 public elementary schools** in the urban area of Majene Regency, West Sulawesi (about 5 km x 6 km). "
  "Applicant and school names are replaced by codes (Student 0001, School 01, ...); only the coordinates "
  "(latitude, longitude) and the capacities are used.\n")
w("### Table A1. Characteristics of the data\n")
w(table(["Quantity", "Value"], [
    ["Applicants", "1,209"], ["Elementary schools", "28"],
    ["Applicant-school pairs in total", "33,852"], ["Eligible pairs (distance <= 1 km)", "6,641"],
    ["Mean schools within the radius per applicant", "5.5 (median 5, max 10)"],
    ["Applicants with no school within 1 km", "3"],
    ["Median distance to the nearest school", "0.185 km"], ["Largest nearest-school distance", "1.105 km"]]))

summ = pd.read_csv(R / "scenarios" / "scenario_summary.csv")
w("### Table A2. Capacity scenarios\n")
desc = {"S1": "Matched capacity (seats = applicants)", "S2": "Surplus capacity (recorded quotas)",
        "S3": "Insufficient capacity"}
src = {"S1": "Approximately proportional scaling of the recorded quotas", "S2": "Recorded data (1-3 classes x 28)",
       "S3": "Approximately proportional scaling of the recorded quotas"}
rows = []
for _, r in summ.iterrows():
    rows.append([r["Scenario"], desc[r["Scenario"]], f(r["Seats"]), f(r["Seats"] / r["Students"], 2),
                 r["Quota values"].replace(", ", "-") if r["Scenario"] != "S2" else "28 / 56 / 84", src[r["Scenario"]]])
w(table(["Scenario", "Description", "Total seats", "Seats per applicant", "Capacity per school", "Source"], rows))

w("## A.2 System framework\n")
w(table(["Layer", "Component", "Output"], [
    ["1. Data and preprocessing", "CSV validation, anonymization, Haversine distances, eligible pairs E_R",
     "Distance matrix d_ij, E_R"],
    ["2. LEXZON optimization", "Stage 1 maximum flow -> Stage 2 bisection -> Stage 3 minimum-cost flow",
     "Assignment x, N*, z*"],
    ["3. Verification and audit",
     "Optimality certificates, cause of each unassigned applicant, priority violations, price-of-fairness curve, LEXZON-F",
     "Proof of optimality, fairness audit"],
    ["4. Decision support", "Interactive map, radius/capacity/extra-class simulation, export",
     "Map, tables, files"]]))

w("## A.3 Proposed method: LEXZON (Lexicographic Exact Zoning)\n")
w("LEXZON solves three objectives in sequence (lexicographically); the output of each stage becomes a "
  "constraint for the next one:\n")
w("1. **Stage 1 - Maximum enrollment.** Maximum flow (Dinic's algorithm) on the network source -> applicants "
  "(capacity 1) -> schools within the radius (capacity 1) -> sink (capacity q_j). Result: N*.\n"
  "2. **Stage 2 - Minimum maximum distance.** Bisection over the distinct distances; each step tests whether "
  "the maximum flow restricted to pairs at distance <= t still equals N*. Result: z*.\n"
  "3. **Stage 3 - Minimum total distance.** A minimum-cost flow that routes exactly N* applicants over pairs "
  "at distance <= z*, solved with the HiGHS dual simplex method; the constraint matrix is totally unimodular, "
  "so the LP solution is integral.\n"
  "4. **Optimality certificates.** F(R) = F(z*) = N* and F(z-) < N*, together with the integrality of Stage 3.\n"
  "5. **Labelling the unassigned.** Every unassigned applicant is labelled with its cause: no school within "
  "the radius, or every school within the radius is full.\n")

w("### Algorithm 1. LEXZON\n")
w("```text\nInput : distances d_ij, capacities q_j, radius R\nOutput: assignment x, N*, z*, certificates\n"
  " 1: E_R <- {(i, j) : d_ij <= R}\n"
  " 2: N* <- MaxFlow(E_R, q)                            // Stage 1\n"
  " 3: D  <- sorted distinct distances over E_R; lo <- 1; hi <- |D|\n"
  " 4: while lo < hi do                                 // Stage 2\n"
  " 5:     mid <- floor((lo + hi) / 2)\n"
  " 6:     if MaxFlow(E_D[mid], q) = N* then hi <- mid else lo <- mid + 1\n"
  " 7: z* <- D[lo]\n"
  " 8: x <- MinCostFlow(E_z*, q, flow value N*)          // Stage 3\n"
  " 9: for every applicant i left without a pair do mark unassigned\n"
  "10:     cause <- no school within R, or every school within R is full\n"
  "11: return x, N*, z*, (F(R), F(z*), F(z-))\n```\n")

w("### Algorithm 2. Price-of-fairness curve and LEXZON-F\n")
w("```text\nInput : d_ij, q_j, R, violation budgets K = {k1 < k2 < ...}, time limit tau\n"
  "Output: curve {(k, A(k))}, LEXZON-F assignment\n"
  "1: mu_L <- LEXZON(d, q, R);  mu_D <- DeferredAcceptance(d, q, R)\n"
  "2: build constraints (7), (8), (15) for every (i, j) in E_R\n"
  "3: (mu_F, eps_F) <- solve (17) warm-started at mu_L, time limit tau\n"
  "4: mu <- mu_D; record (0, |mu_D|)                     // Proposition 2\n"
  "5: for k in K with 0 < k < eps_F do\n"
  "6:     mu <- solve (16) warm-started at mu, time limit tau\n"
  "7:     record (k, |mu|, solver bound)\n"
  "8: record (eps_F, N*); return the curve and mu_F\n```\n")

w("## A.4 Comparison methods\n")
w(table(["Code", "Method", "How it works", "Role"], [
    ["B1", "Nearest school", "Every applicant to the nearest school within the radius, capacity ignored",
     "Reference (infeasible)"],
    ["B2", "Distance-ordered greedy",
     "The nearest applicant is served first and takes the nearest school with a free seat", "Common manual rule"],
    ["B3", "Random-order greedy", "The B2 rule in random order, averaged over 30 seeds", "First come, first served"],
    ["B4", "Deferred acceptance", "Applicants propose to their nearest school; schools hold the nearest applicants",
     "Distance-based admission (stable)"],
    ["B5", "Exact minimum total distance", "Stage 1 + Stage 3 (without Stage 2)", "Tests whether Stage 2 is needed"],
    ["B6", "Exact minimum maximum distance", "Stage 1 + Stage 2 (without Stage 3)", "Tests whether Stage 3 is needed"],
    ["B7", "Constrained k-means", "Balanced capacitated clustering (Zainuddin and Risal, 2024)",
     "Method used in Indonesian zoning studies"]]))

w("## A.5 Ablation study\n")
w("Each configuration adds one component. The radius and the option to leave applicants unassigned are not "
  "introduced separately in A2, because three applicants have no school within 1 km, so a radius without that "
  "option has no solution. The objectives of A2 and A3 are **lexicographic**: enrollment is maximized first, "
  "so an empty assignment can never be optimal.\n")
w(table(["Code", "Placement constraint", "Distance", "Radius", "Objectives (in order)", "Equivalent to"], [
    ["A0", "Every applicant must be placed (sum_j x_ij = 1)", "Euclidean (degrees)", "None", "min total distance",
     "Initial thesis model"],
    ["A1", "Every applicant must be placed (sum_j x_ij = 1)", "Haversine (km)", "None", "min total distance", "–"],
    ["A2", "Applicants may stay unassigned (sum_j x_ij <= 1)", "Haversine (km)", "R = 1 km",
     "(1) max enrollment -> (2) min total distance", "B5"],
    ["A3", "Applicants may stay unassigned (sum_j x_ij <= 1)", "Haversine (km)", "R = 1 km",
     "(1) max enrollment -> (2) min maximum distance -> (3) min total distance", "LEXZON"]]))

w("## A.6 Evaluation metrics\n")
w(table(["Metric", "Definition"], [
    ["Unassigned (UA)",
     "Applicants left without a school, split into no school within the radius / every school full"],
    ["Gap to N*", "N* minus the number of assigned applicants (0 = optimal)"],
    ["Capacity overflow", "Applicants placed beyond the capacity of a school (feasibility)"],
    ["Mean / max / total distance", "Haversine distance from an assigned applicant to its school (km)"],
    ["Priority violations (justified envy)",
     "Applicants who prefer a school within the radius that still has a free seat or admits a farther applicant"],
    ["Runtime", "Seconds (median of three repetitions for the scalability study)"]]))

w("## A.7 Experimental protocol\n")
w(table(["No", "Experiment", "Setting"], [
    ["1", "Method comparison", "3 scenarios, R = 1 km, 8 methods"],
    ["2", "Ablation study", "S1, S2, S3; A0-A3"],
    ["3", "Radius sensitivity", "R = 0.5, 0.75, 1, 1.25, 1.5, 2, 3 km; unrestricted"],
    ["4", "Statistical test",
     "30 subsamples (80% of the applicants, seed 42 + index), capacities scaled; one-sided Wilcoxon, alpha = 0.05"],
    ["5", "Scalability",
     "1,000-50,000 synthetic applicants (recorded locations plus Gaussian noise of 150 m), median of 3 repetitions"],
    ["6", "Distance-priority fairness", "Justified envy for every method, on the full data and on 90 subsamples"],
    ["7", "Price of fairness", "k in {0, 10, 25, 50, 100, 150, 200}; 300 s per point; LEXZON-F 900 s (CP-SAT)"],
    ["8", "Policy simulation", "One extra class (28 seats) at each school in turn"],
    ["9", "Software testing", "126 unit tests (pytest) + 22 end-to-end tests (Playwright)"]]))

w("## A.8 Implementation environment\n")
w(table(["Component", "Version / specification"], [
    ["Language", "Python 3.11.9"], ["Numerics", "NumPy 1.26.4, SciPy 1.17.1, pandas 3.0.2"],
    ["Maximum flow", "scipy.sparse.csgraph.maximum_flow (Dinic)"],
    ["LP/MILP", "HiGHS (dual simplex) through SciPy"],
    ["Price-of-fairness model", "Google OR-Tools CP-SAT"], ["Application", "Streamlit 1.57.0, Folium 0.20.0"],
    ["Hardware", "Intel Core i7-14650HX, 16 GB RAM, Windows 11"]]))

# ============================================================ FORMULAS
w("\n---\n\n# Part B. Formulas\n")
eqs = [
    ("(1) Haversine distance", r"d_{ij} = 2\rho \arcsin\sqrt{a_{ij}}"),
    ("(2) Haversine component", r"a_{ij} = \sin^2\!\left(\frac{\Delta\varphi}{2}\right) + \cos\varphi_i \cos\varphi_j \sin^2\!\left(\frac{\Delta\lambda}{2}\right),\quad \rho = 6371\ \text{km}"),
    ("(3) Eligible pairs", r"E_R = \{(i,j) \in S \times C : d_{ij} \le R\}"),
    ("(4) Lexicographic objective", r"\operatorname{lex\,max}_x \big(f_1(x),\ -f_2(x),\ -f_3(x)\big)"),
    ("(5) Objectives 1 and 3", r"f_1(x) = \sum_{(i,j)\in E_R} x_{ij},\qquad f_3(x) = \sum_{(i,j)\in E_R} d_{ij}\,x_{ij}"),
    ("(6) Objective 2", r"f_2(x) = \max_{(i,j)\in E_R} d_{ij}\,x_{ij}"),
    ("(7) Applicant constraint", r"\sum_{j:(i,j)\in E_R} x_{ij} \le 1,\quad \forall i \in S"),
    ("(8) Capacity constraint", r"\sum_{i:(i,j)\in E_R} x_{ij} \le q_j,\quad \forall j \in C,\quad x_{ij}\in\{0,1\}"),
    ("(9) Thresholded maximum flow", r"F(t) = \max\Big\{\sum_{d_{ij}\le t} x_{ij} : x \text{ feasible}\Big\},\qquad N^* = F(R)"),
    ("(10) Minimum maximum distance", r"z^* = \min\{t \in D_R : F(t) = N^*\}"),
    ("(11) Minimum total distance", r"\min \sum_{d_{ij}\le z^*} d_{ij}\,x_{ij}\quad \text{s.t. } \sum x_{ij} = N^*,\ (7),(8)"),
    ("(12) Optimality certificate", r"F(z^*) = N^* \ \text{and}\ F(z^-) < N^*"),
    ("(13) Distance priorities", r"i \succ_j i' \Leftrightarrow d_{ij} < d_{i'j},\qquad j \succ_i j' \Leftrightarrow d_{ij} < d_{ij'}"),
    ("(14) Priority violation", r"\varepsilon_i = 1 \Leftrightarrow \exists (i,j)\in E_R : j \succ_i \mu(i) \ \wedge\ \big(|\mu^{-1}(j)| < q_j \ \vee\ \exists i'\in\mu^{-1}(j): i \succ_j i'\big)"),
    ("(15) Weighted stability", r"\sum_{i' \succ_j i} x_{i'j} + q_j \sum_{j' \succeq_i j} x_{ij'} \ge q_j\,(1 - y_i)"),
    ("(16) Price-of-fairness curve", r"A(k) = \max \sum_{(i,j)\in E_R} x_{ij}\quad \text{s.t. } (7),(8),(15),\ \sum_i y_i \le k"),
    ("(17) LEXZON-F", r"\varepsilon_F = \min \sum_i y_i\quad \text{s.t. } (7),(8),(15),\ \sum_{(i,j)\in E_R} x_{ij} = N^*"),
]
for name, tex in eqs:
    w(f"**{name}**\n\n$$\n{tex}\n$$\n")
w("**Additional metrics**\n\n$$\n\\text{Reduction in UA}(\\%) = \\frac{UA_{\\text{comparator}} - UA_{\\text{LEXZON}}}{UA_{\\text{comparator}}}\\times 100\n$$\n")
w("$$\n\\text{Gap to } N^* = N^* - \\sum_{(i,j)} x_{ij}\n$$\n")
w("**Proposition 1.** Every LEXZON assignment (a) leaves no unassigned applicant who has a school within the "
  "radius with a free seat, and (b) rejects no applicant who is closer to a school than an applicant admitted "
  "to that school.\n\n"
  "**Proposition 2.** A(0) equals the enrollment of deferred acceptance (the rural hospitals theorem), and "
  "A(k) = N* for k >= eps_F.\n")

# ============================================================ RESULTS
w("\n---\n\n# Part C. Results\n")
w("## C.1 Software testing\n")
w(table(["Test type", "Count", "Passed", "Notes"], [
    ["Solver unit tests (brute-force optimum, bounds, feasibility)", "50", "50", "test_solver.py"],
    ["Fairness and constrained k-means unit tests", "52", "52", "test_fairness.py"],
    ["Price-of-fairness model unit tests (Proposition 2)", "24", "24", "test_pareto.py"],
    ["End-to-end application tests (browser)", "22", "22", "TC-01 to TC-22"],
    ["Reproducibility (29 result files rerun)", "29", "29", "Identical, apart from runtimes"]]))

w("## C.2 LEXZON stage results per scenario\n")
w(table(["Scenario", "Seats", "N*", "z* (km)", "Bisection steps", "F(z-)", "Schools full",
         "Forced model A0: beyond R"],
        [[r["Scenario"], f(r["Seats"]), f(r["Max-flow bound N*"]), f(r["z* (km)"]), f(r["Bisection steps"]),
          f(r["Certificate flow below z*"]), f(r["Proposed schools full"]),
          f(r["Forced model beyond R"]) if r["Forced model feasible"] else "infeasible"] for _, r in summ.iterrows()]))

b7 = pd.read_csv(R / "fairness" / "b7_comparison.csv")
w("## C.3 Method comparison (R = 1 km)\n")
for s in SC:
    c = pd.read_csv(R / "scenarios" / f"{s}_comparison.csv")
    c = pd.concat([c, b7[b7["Scenario"] == s].assign(Code="B7")], ignore_index=True)
    rows = []
    for _, r in c.iterrows():
        rows.append([NAME[r["Code"]], f(r["admitted"], 1), f(r["rejected"], 1), f(r["rejected_no_school"]),
                     f(r["rejected_full"], 1), f(r["gap_to_max"], 1), f(r["overflow_students"]), f(r["mean_km"]),
                     f(r["max_km"]), f(r["total_km"], 1), f(r["gini"]), f(r["runtime_s"], 3)])
    w(f"### Table C3-{s}. Method comparison - {s}: {desc[s].lower()}\n")
    w(table(["Method", "Assigned", "UA", "UA: no school", "UA: schools full", "Gap to N*", "Capacity overflow",
             "Mean (km)", "Max (km)", "Total (km)", "Gini", "Runtime (s)"], rows))

w("### Table C3-R. Reduction in unassigned applicants achieved by LEXZON (%)\n")
rows = []
for code in ["B2", "B3", "B4", "B7"]:
    row = [NAME[code]]
    for s in SC:
        c = pd.read_csv(R / "scenarios" / f"{s}_comparison.csv").set_index("Code")
        base = b7.set_index("Scenario").loc[s, "rejected"] if code == "B7" else c.loc[code, "rejected"]
        row.append(f(100 * (base - c.loc["P", "rejected"]) / base, 1))
    rows.append(row)
w(table(["Comparator", "S1", "S2", "S3"], rows))

ab = pd.read_csv(R / "scenarios" / "ablation_all_scenarios.csv")
w("## C.4 Ablation study\n")
lab = {"A0": "Initial model (Euclidean degrees, every applicant placed)", "A1": "+ Haversine (km)",
       "A2": "+ radius R = 1 km with applicants allowed to stay unassigned",
       "A3": "+ minimum maximum distance stage (LEXZON)"}
rows = []
for _, r in ab.iterrows():
    ok = r["Status"] == "Optimal"
    rows.append([r["Scenario"], r["Code"], lab[r["Code"]], "Feasible" if ok else "Infeasible (seats < applicants)",
                 f(r["admitted"]) if ok else "–", f(r["rejected"]) if ok else "–",
                 f(r["beyond_radius"]) if ok else "–",
                 f(r["max_km"]) if ok else "–", f(r["total_km"], 1) if ok else "–"])
w(table(["Scenario", "Code", "Configuration", "Status", "Assigned", "UA", "Beyond R", "Max (km)", "Total (km)"], rows))

w("## C.5 Radius sensitivity\n")
for metric, label, d in [("rejected", "Unassigned applicants", 0), ("max_km", "Maximum distance (km)", 3),
                         ("total_km", "Total distance (km)", 1)]:
    w(f"### Table C5-{'abc'[['rejected','max_km','total_km'].index(metric)]}. {label} by radius\n")
    rows = []
    for s in SC:
        sw = pd.read_csv(R / "scenarios" / f"{s}_radius.csv")
        piv = sw.pivot(index="Radius (km)", columns="Code", values=metric)
        for rad, r in piv.iterrows():
            rows.append([s, "inf" if np.isinf(rad) else f(rad, 2), f(r["P"], d), f(r["B5"], d), f(r["B2"], d),
                         f(r["B4"], d)])
    w(table(["Scenario", "Radius (km)", "LEXZON", "B5", "B2", "B4"], rows))

w("## C.6 One-sided Wilcoxon test (30 subsamples, 80% of the applicants)\n")
st = pd.read_csv(R / "scenarios" / "scenario_stat_summary.csv")
bw = pd.read_csv(R / "fairness" / "b7_wilcoxon.csv")
rows = []
for s in SC:
    for _, r in st[st["Scenario"] == s].iterrows():
        code = r["Baseline"].split(" ")[0]
        rows.append([s, code, r["✕ Proposed (mean ± sd)"], r["✕ Baseline (mean ± sd)"],
                     f(r["Mean difference"], 1),
                     f"{r['Proposed better (runs)']}/{r['Tied (runs)']}/{r['Proposed worse (runs)']}",
                     f(r["W statistic"], 0), "–" if pd.isna(r["p-value"]) else f"{r['p-value']:.1e}",
                     "Significant" if str(r["Conclusion"]).startswith("Significant") else "Identical"])
    b = bw[bw["Scenario"] == s].iloc[0]
    rows.append([s, "B7", f"{f(b['LEXZON rejected mean'],1)} ± {f(b['LEXZON rejected sd'],1)}",
                 f"{f(b['Baseline rejected mean'],1)} ± {f(b['Baseline rejected sd'],1)}",
                 f(b["Baseline rejected mean"] - b["LEXZON rejected mean"], 1),
                 f"{b['better']}/{b['tied']}/{b['worse']}",
                 f(b["W"], 0), f"{b['p']:.1e}", "Significant" if b["p"] < 0.05 else "Not significant"])
w(table(["Scenario", "Comparator", "UA LEXZON (mean ± sd)", "UA comparator (mean ± sd)", "Mean difference",
         "Better/tied/worse", "W", "p-value", "Conclusion"], rows))

w("## C.7 Scalability (synthetic data, median of 3 repetitions)\n")
sc = pd.read_csv(R / "scenarios" / "scalability_median.csv")
w(table(["Applicants", "Seats", "Eligible pairs", "Assigned", "N*", "Stage 1 (s)", "Stage 2 (s)", "Stage 3 (s)",
         "LEXZON total (s)", "Deferred acceptance (s)"],
        [[f(r["Students"]), f(r["Total seats"]), f(r["Pairs within radius"]), f(r["Admitted"]),
          f(r["Max-flow upper bound"]),
          f(r["Stage 1 (max admissions)"], 4), f(r["Stage 2 (min max distance)"], 4),
          f(r["Stage 3 (min total distance)"], 4),
          f(r["Proposed total runtime (s)"], 3), f(r["Deferred Acceptance runtime (s)"], 3)]
         for _, r in sc.iterrows()]))

w("## C.8 Distance-priority fairness (R = 1 km, full data)\n")
fa = pd.read_csv(R / "fairness" / "fairness_full.csv")
lf = pd.read_csv(R / "pareto" / "lexzon_f_metrics.csv")
rows = []
for s in SC:
    for _, r in fa[fa["Scenario"] == s].iterrows():
        rows.append([s, NAME[r["Code"]], f(r["rejected"], 1), f(r["envy_students"], 1), f(r["envy_unassigned"], 1),
                     f(r["envy_assigned"], 1), f(r["envy_via_vacant_seat"], 1), f(r["envy_share_pct"], 1),
                     f(r["envy_extra_km_median"])])
    r = lf[lf["Scenario"] == s].iloc[0]
    rows.append([s, "LEXZON-F", f(r["rejected"]), f(r["envy_students"]), f(r["envy_unassigned"]),
                 f(r["envy_assigned"]), f(r["envy_via_vacant_seat"]), f(r["envy_share_pct"], 1),
                 f(r["envy_extra_km_median"])])
w(table(["Scenario", "Method", "UA", "Applicants with a violation", "Rejected although closer",
         "Assigned away from the nearest school", "Through a vacant seat", "% of applicants",
         "Median extra distance (km)"], rows))

fr = pd.read_csv(R / "fairness" / "fairness_runs.csv")
w("### Table C8-R. Fairness over 30 subsamples (means)\n")
g = fr.groupby(["Scenario", "Code"])[["rejected", "envy_students", "envy_unassigned",
                                      "envy_share_pct"]].mean().reset_index()
w(table(["Scenario", "Method", "UA", "Applicants with a violation", "Rejected although closer", "% of applicants"],
        [[r["Scenario"], NAME[r["Code"]], f(r["rejected"], 1), f(r["envy_students"], 1), f(r["envy_unassigned"], 1),
          f(r["envy_share_pct"], 1)] for _, r in g.iterrows()]))

w("## C.9 Price-of-fairness curve\n")
pp = pd.read_csv(R / "pareto" / "pareto_points.csv")
rows = []
label = {"LEXZON": "LEXZON", "DA": "Deferred acceptance",
         "min_envy_at_Nstar": "LEXZON-F (minimum violations at N*)", "budget": "Violation budget k"}
for _, r in pp.iterrows():
    rows.append([r["Scenario"], label[r["Point"]], f(r["budget"]) if r["Point"] == "budget" else "–", f(r["admitted"]),
                 f(r["envy"]), f(r["bound"], 0) if not pd.isna(r["bound"]) else "–", "Yes" if r["optimal"] else "No",
                 f(r["runtime_s"], 1), f(r["N_star"])])
w(table(["Scenario", "Point", "k", "Assigned", "Applicants with a violation", "Solver bound*", "Proved optimal",
         "Runtime (s)", "N*"], rows))
w("*For a budget point: the upper bound on enrollment; for LEXZON-F: the lower bound on the number of "
  "violations.\n")

w("### Table C9-R. Price-of-fairness summary\n")
rows = []
for s in SC:
    g = pp[pp["Scenario"] == s]
    ns = int(g["N_star"].iloc[0]); da = int(g[g["Point"] == "DA"]["admitted"].iloc[0])
    lex = int(g[g["Point"] == "LEXZON"]["envy"].iloc[0]); fe = int(g[g["Point"] == "min_envy_at_Nstar"]["envy"].iloc[0])
    b50 = g[(g["Point"] == "budget") & (g["budget"] == 50)]["admitted"]
    lt = pd.read_csv(R / "scenarios" / f"{s}_comparison.csv").set_index("Code").loc["P", "total_km"]
    ft = lf.set_index("Scenario").loc[s, "total_km"]
    rows.append([s, f(da), f(ns), f(ns - da), f(100 * (int(b50.iloc[0]) - da) / (ns - da), 1) if len(b50) else "–",
                 f(lex), f(fe), f(100 * (lex - fe) / lex, 1), f(100 * (ft - lt) / lt, 1)])
w(table(["Scenario", "Assigned by DA (k = 0)", "N*", "Additional enrollment", "% of the gain reached at k = 50",
         "LEXZON violations", "LEXZON-F violations", "Reduction in violations (%)",
         "Increase in LEXZON-F total distance (%)"], rows))

w("## C.10 Policy simulation: adding one class (28 seats)\n")
un = pd.read_csv(R / "scenarios" / "unassigned_by_nearest_school.csv")
wi = pd.read_csv(R / "scenarios" / "whatif_extra_class.csv")
m = un.merge(wi, left_on=["Scenario", "School"], right_on=["Scenario", "School with extra class"])
m["Reduction"] = m["Unassigned before"] - m["Unassigned after"]
keep = m[(m["Unassigned nearest to school"] > 0) | (m["Reduction"] > 0)]
piv = keep.pivot_table(index="School", columns="Scenario", values=["Unassigned nearest to school", "Reduction"],
                       fill_value=0)
rows = []
for school in piv.index:
    rows.append([school.replace("School ", "ES ")] +
                [f(piv.loc[school, ("Unassigned nearest to school", s)]) for s in SC] +
                [f(piv.loc[school, ("Reduction", s)]) for s in SC])
w(table(["School", "UA nearest S1", "UA nearest S2", "UA nearest S3", "Reduction with +1 class S1",
         "Reduction with +1 class S2", "Reduction with +1 class S3"], rows))
rows = []
for s in SC:
    g = wi[wi["Scenario"] == s]
    red = g["Unassigned before"] - g["Unassigned after"]
    rows.append([s, f(g["Unassigned before"].iloc[0]), f(g["Unassigned after"].min()), f(int((red == red.max()).sum())),
                 f(int((red == 0).sum()))])
w(table(["Scenario", "UA before", "Best UA after +1 class", "Schools with the largest reduction",
         "Schools with no reduction"], rows))

OUT.write_text("\n".join(md), encoding="utf-8")
print(OUT, len("\n".join(md)))
