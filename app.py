"""
Zonasi 2 — exact student-to-school allocation within a zoning radius.

Run:  streamlit run app.py
"""
import hashlib
import io
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from zonasi import charts, data, maps, metrics, solver
from zonasi import experiments as ex
from zonasi.distance import haversine_matrix

st.set_page_config(page_title="Zonasi 2 — School Zoning Allocation", page_icon="🏫", layout="wide")

P = ex.PROPOSED_CODE
NAMES = ex.METHOD_NAMES


# ------------------------------------------------------------------ cached computations
@st.cache_data(show_spinner=False)
def cached_bundled(key):
    return data.load_bundled(key)


@st.cache_data(show_spinner=False)
def cached_uploaded(school_bytes, student_bytes):
    return data.load_pair(io.BytesIO(school_bytes), io.BytesIO(student_bytes))


@st.cache_data(show_spinner=False)
def cached_distance(school_df, student_df):
    return haversine_matrix(student_df, school_df)


@st.cache_data(show_spinner=False)
def cached_proposed(school_df, student_df, radius):
    dist = cached_distance(school_df, student_df)
    quotas = school_df["Quota"].to_numpy(dtype=int)
    alloc = solver.proposed(dist, quotas, radius)
    return alloc, metrics.evaluate(alloc, dist, quotas, student_df, radius, alloc.info["n_star"])


@st.cache_data(show_spinner=False)
def cached_comparison(school_df, student_df, radius, seeds):
    table, _, max_adm = ex.comparison_table(cached_distance(school_df, student_df), school_df, student_df, radius, seeds)
    return table, max_adm


@st.cache_data(show_spinner=False)
def cached_ablation(school_df, student_df, radius):
    return ex.ablation(school_df, student_df, radius)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_map_html(school_df, student_df, assignment, radius, highlight):
    # Rendered to standalone HTML: streamlit-folium fails to re-add layer groups on rerun.
    fmap = maps.zoning_map(school_df, student_df, assignment, cached_distance(school_df, student_df), radius, highlight)
    return fmap.get_root().render()


@st.cache_data(show_spinner=False)
def cached_static_map(school_df, student_df, assignment, radius, title):
    return maps.static_map_png(school_df, student_df, assignment, radius, title)


# ------------------------------------------------------------------ helpers
def fmt_radius(radius):
    return "no limit" if radius is None else f"{radius:g} km"


def show_chart(fig):
    st.plotly_chart(fig, width="stretch", theme=None, config=charts.PLOTLY_CONFIG)


def download(df, name, label="Download table (CSV)"):
    st.session_state.setdefault("exports", {})[name] = df
    st.download_button(label, df.to_csv(index=False).encode("utf-8-sig"), file_name=f"{name}.csv", mime="text/csv",
                       on_click="ignore", key=f"dl_{name}")


def _cell(value):
    if pd.isna(value):
        return "—"
    if isinstance(value, (int, np.integer)) or (isinstance(value, (float, np.floating)) and float(value).is_integer()):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    return str(value)


def metric_table(df, lead_cols, keys):
    out = df[lead_cols + [c for c in keys if c in df.columns]].rename(columns=metrics.LABELS)
    # Columns that mix numbers with a not-applicable dash (e.g. gap to the upper bound for a method that breaks
    # quotas) are rendered as text, so every column has a single type for Arrow.
    for col in out.columns:
        if out[col].isna().any():
            out[col] = out[col].map(_cell)
    return out.round(4)


def show_html(html, height):
    """st.iframe on newer Streamlit (components.v1.html is deprecated there), components.html otherwise."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        components.html(html, height=height)


def run_state(name, context):
    key = f"ran_{name}_{context}"
    return key, st.session_state.get(key, False)


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🏫 Zonasi 2")
st.sidebar.caption("Exact student-to-school allocation within a zoning radius")

source = st.sidebar.radio("Data source", ["Bundled dataset", "Upload CSV"], horizontal=True)
school_df = student_df = None
load_warnings = []
dataset_id = None
if source == "Bundled dataset":
    dataset_key = st.sidebar.selectbox("Dataset", list(data.BUNDLED_DATASETS.keys()))
    school_df, student_df, load_warnings = cached_bundled(dataset_key)
    dataset_id = dataset_key
else:
    up_school = st.sidebar.file_uploader("Schools CSV", type="csv", help="Columns: School, Latitude, Longitude, Quota")
    up_student = st.sidebar.file_uploader("Students CSV", type="csv", help="Columns: Student, Latitude, Longitude")
    if up_school and up_student:
        sb, tb = up_school.getvalue(), up_student.getvalue()
        try:
            school_df, student_df, load_warnings = cached_uploaded(sb, tb)
            dataset_id = "upload-" + hashlib.md5(sb + tb).hexdigest()[:10]
        except data.DataError as exc:
            st.sidebar.error(str(exc))

st.sidebar.divider()
limit = st.sidebar.toggle("Limit by zoning radius", value=True)
radius = st.sidebar.number_input("Zoning radius (km)", min_value=0.1, max_value=20.0, value=1.0, step=0.1,
                                 disabled=not limit, help="A student may only be placed in a school within this distance.")
radius = float(radius) if limit else None
seeds = st.sidebar.number_input("Seeds for B3 (random greedy)", min_value=5, max_value=200, value=30, step=5)

st.title("Student-to-School Allocation within a Zoning Radius")

if school_df is None:
    st.info("Upload two CSV files (schools and students) in the sidebar to start.")
    st.stop()

for w in load_warnings:
    st.sidebar.warning(w)

context = f"{dataset_id}|{radius}|{seeds}"
if st.session_state.get("export_context") != context:
    st.session_state["export_context"] = context
    st.session_state["exports"] = {}

quotas = school_df["Quota"].to_numpy(dtype=int)
dist = cached_distance(school_df, student_df)
st.caption(f"{len(student_df):,} students · {len(school_df)} schools · {quotas.sum():,} seats · radius {fmt_radius(radius)} · "
           "names anonymized")

tabs = st.tabs(["📖 About", "🗂️ Data", "📍 Allocation", "⚖️ Comparison", "🧪 Ablation",
                "📏 Radius Sensitivity", "🎬 Capacity Scenarios", "📊 Statistical Test", "⏱️ Scalability"])

# ------------------------------------------------------------------ About
with tabs[0]:
    st.markdown(f"""
### Problem
Under school zoning, students are placed by distance from home. When the nearest school is full, common placement
rules (closest student first, or distance-based selection as in PPDB) can leave **students who could still be seated**
at another school within the radius without any school at all.

### Proposed method (lexicographic, every stage exact)
1. **Maximize the number of admitted students**, using only student–school pairs within the zoning radius.
   It is computed as a *maximum flow*, so the value is also an **upper bound** no method can exceed.
2. **Minimize the longest trip** among all allocations that admit that many students (bisection over distances).
3. **Minimize total distance** among those. This stage is a *min-cost flow* problem whose constraint matrix is totally
   unimodular, so the *linear programming* optimum (HiGHS) is already integer; integrality is checked on every solve,
   with *mixed-integer programming* as a fallback.

Students who still have no school are marked **✕**, for one of two reasons:
- **no school within the radius**, or
- **every school within the radius is full** — and provably no reallocation could seat them without displacing another student.

### Baselines (same radius rule)
| Code | Method | Description |
|---|---|---|
| B1 | Nearest, quota ignored | Nearest school regardless of capacity — an infeasible reference |
| B2 | Greedy nearest | Closest students first, each to the nearest school with a free seat |
| B3 | Greedy random order | B2's rule with students in random order ({seeds} seeds) |
| B4 | Deferred Acceptance | Gale–Shapley; students prefer nearer schools, schools prioritize nearer applicants |
| B5 | Exact min-total | Max admissions, then min total distance (no min-max stage) |
| B6 | Exact min-max | Max admissions, then min longest trip (no min-total stage) |

### Validation
Multi-criteria comparison, ablation study, radius sensitivity, three capacity scenarios, a resampling test
(Wilcoxon signed-rank), and scalability on synthetic data.

### Limitations
- Distance is **straight-line (Haversine)** between points; roads, rivers and terrain are not modelled.
- Selection criteria other than distance (age, affirmative quotas, parental job transfer) and family preferences are not modelled.
- Quotas are static for a single intake, with no projection of the school-age population.
- Results depend on the accuracy of students' home coordinates.
- The data cover one region (Majene Regency) and one level (primary school).

### Privacy
School and student names are replaced with IDs (`School 01`, `Student 0001`) whenever data are loaded, including
uploaded files. Coordinates are kept because the method needs them; home locations remain personal data.

### Map colors
Each school and its admitted students share a color. There are more schools than clearly distinguishable colors,
so colors repeat, but schools whose zones border each other never share a color or a colorblind-confusable pair.
Hover over any point to see its school.

### CSV format
- **Schools:** `School, Latitude, Longitude, Quota`
- **Students:** `Student, Latitude, Longitude`

Indonesian headers (`Nama Sekolah`, `Jumlah Diterima`, `Nama Siswa`) are accepted too.
Every chart can be saved as a high-resolution PNG with the camera icon at its top right.
""")

# ------------------------------------------------------------------ Data
with tabs[1]:
    cap = np.inf if radius is None else radius
    nearest = dist.min(axis=1)
    c = st.columns(5)
    c[0].metric("Students", f"{len(student_df):,}")
    c[1].metric("Schools", len(school_df))
    c[2].metric("Seats", f"{quotas.sum():,}")
    c[3].metric("Seats per student", f"{quotas.sum() / len(student_df):.2f}")
    c[4].metric("No school within radius", int((nearest > cap).sum()),
                help="Students whose nearest school is already outside the radius.")
    show_chart(charts.distance_histogram(nearest, radius, "Distance from each student to the nearest school (quotas ignored)"))
    left, right = st.columns(2)
    with left:
        st.subheader("Schools")
        st.dataframe(school_df, hide_index=True, width="stretch", height=360)
    with right:
        st.subheader("Students")
        st.dataframe(student_df, hide_index=True, width="stretch", height=360)

# ------------------------------------------------------------------ Allocation
with tabs[2]:
    with st.spinner("Computing the optimal allocation…"):
        alloc, m = cached_proposed(school_df, student_df, radius)
    assignment = alloc.assignment
    n = len(student_df)
    c = st.columns(5)
    c[0].metric("Admitted", f"{m['admitted']:,}")
    c[0].caption(f"{m['admission_rate']:.2f}% of {n:,} students")
    c[1].metric("No school ✕", m["rejected"])
    c[1].caption(f"{m['rejected_no_school']} with no school in radius · {m['rejected_full']} because schools are full")
    c[2].metric("Longest trip", f"{m['max_km']:.3f} km")
    c[3].metric("Mean distance", f"{m['mean_km']:.3f} km")
    c[4].metric("Vacant seats", m["seats_vacant"])

    info = alloc.info
    cert = info["certificate"]
    with st.expander("Proof of optimality"):
        st.markdown(f"""
- **Stage 1:** maximum flow with radius {fmt_radius(radius)} = **{info['n_star']}** students. This is an upper bound; this allocation admits **{m['admitted']}**, a gap of **{m['gap_to_max']}**.
- **Stage 2:** smallest possible longest trip z* = **{info['z_star']:.4f} km** ({info['bisection_iterations']} bisection steps).
  At z* the maximum flow is {cert['flow_at_z_star']}; at the next smaller distance it drops to {cert['flow_just_below_z_star']}, so z* cannot be lowered without admitting fewer students.
- **Stage 3:** status **{alloc.status}** — {info['solver']}. The LP optimum is already integer, so it equals the LP lower bound: the optimality gap is 0.
""")
        st.dataframe(pd.DataFrame({"Stage": list(info["stage_times"].keys()),
                                   "Runtime (s)": list(info["stage_times"].values())}).round(4),
                     hide_index=True, width="content")

    st.subheader("Zoning map")
    names = school_df["School"].tolist()
    focus = st.selectbox("Highlight a school", ["All schools"] + names,
                         help="Show one school's students and its zoning radius.")
    highlight = None if focus == "All schools" else names.index(focus)
    show_html(cached_map_html(school_df, student_df, assignment, radius, highlight), height=620)
    st.caption("Squares are schools; circles are students in their school's color; red ✕ marks a student with no school. "
               "Hover for details; switch basemap and layers with the icon at the top right.")

    if st.button("Prepare map figure for the paper (PNG, 300 DPI)"):
        st.session_state[f"static_map_{context}"] = True
    if st.session_state.get(f"static_map_{context}"):
        with st.spinner("Rendering map…"):
            png = cached_static_map(school_df, student_df, assignment, radius,
                                    f"Allocation by the proposed method (radius {fmt_radius(radius)})")
        st.download_button("Download map (PNG)", png, file_name="zoning_map.png", mime="image/png", on_click="ignore")

    left, right = st.columns(2)
    with left:
        show_chart(charts.distance_histogram(metrics.assigned_distances(assignment, dist), radius))
    util = metrics.school_utilization(assignment, dist, school_df)
    with right:
        show_chart(charts.utilization(util))
    with st.expander("Seats per school (table)"):
        st.dataframe(util.round(3), hide_index=True, width="stretch")
        download(util.round(4), "school_seats")

    st.subheader("Students")
    table = metrics.student_table(assignment, dist, school_df, student_df, radius)
    status_filter = st.segmented_control("Show", ["All", "Admitted", "✕ No school"], default="All")
    shown = table
    if status_filter == "Admitted":
        shown = table[table["Status"] == metrics.STATUS_ADMITTED]
    elif status_filter == "✕ No school":
        shown = table[table["Status"] != metrics.STATUS_ADMITTED]
    st.dataframe(shown.round(4), hide_index=True, width="stretch", height=380)
    download(table.round(4), "student_allocation", "Download full list (CSV)")

# ------------------------------------------------------------------ Comparison
with tabs[3]:
    st.markdown(f"Every method uses the **{fmt_radius(radius)} radius** and the same quotas, "
                "so differences come only from how students are placed.")
    with st.spinner("Running all methods…"):
        comp, max_adm = cached_comparison(school_df, student_df, radius, int(seeds))
    st.info(f"Upper bound (maximum flow): at most **{max_adm}** students can be admitted under this rule. "
            "Methods that reach it have a gap of 0.")
    show_chart(charts.rejected_by_method(comp, P))
    shown = metric_table(comp, ["Code", "Method"], ["admitted", "rejected", "rejected_full", "rejected_no_school", "gap_to_max",
                                                  "overflow_students", "seats_vacant", "mean_km", "p90_km", "max_km",
                                                  "total_km", "gini", "dbi", "runtime_s"])
    if "sd_rejected" in comp.columns:
        shown.insert(4, "SD of ✕ (B3)", comp["sd_rejected"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—"))
    shown = shown.map(lambda v: round(v, 3) if isinstance(v, float) else v)
    st.dataframe(shown, hide_index=True, width="stretch")
    download(shown, "method_comparison")
    st.warning("Read distance metrics **together with** the number of students ✕. A method that turns away more distant "
               "students will show a smaller total distance and Gini — that does not make it fairer. B1 ignores quotas, "
               "so its gap to the upper bound is not shown.")
    left, right = st.columns(2)
    with left:
        show_chart(charts.metric_by_method(comp, "max_km", "Longest trip of an admitted student", "km", P))
    with right:
        show_chart(charts.metric_by_method(comp, "total_km", "Total distance of admitted students", "km", P, fmt=".1f"))

# ------------------------------------------------------------------ Ablation
with tabs[4]:
    st.markdown("Components of the proposed method are added **one at a time** on top of the original thesis model. "
                "All distances are re-measured in Haversine km so the rows are comparable.")
    key, ran = run_state("ablation", context)
    if st.button("Run ablation study", key="btn_ablation"):
        st.session_state[key] = ran = True
    if ran:
        with st.spinner("Running ablation…"):
            abl = cached_ablation(school_df, student_df, radius)
        shown = abl[["Code", "Configuration", "Status"]].copy()
        for col in ["admitted", "rejected", "beyond_radius", "max_km", "mean_km", "total_km", "gini", "dbi"]:
            if col in abl.columns:
                shown[metrics.LABELS[col]] = abl[col]
        st.dataframe(shown.round(4), hide_index=True, width="stretch")
        download(shown.round(4), "ablation")
        if (abl["Status"] != "Optimal").any():
            st.info("The original model requires every student to be placed, so it has no solution when there are fewer seats than students.")
        left, right = st.columns(2)
        with left:
            show_chart(charts.ablation_steps(abl))
        with right:
            show_chart(charts.ablation_distance(abl))

# ------------------------------------------------------------------ Radius sensitivity
with tabs[5]:
    st.markdown("How the number of students ✕ and travel distances change as the zoning radius changes.")
    raw = st.text_input("Radii (km, comma-separated)", "0.5, 0.75, 1, 1.25, 1.5, 2, 3")
    add_inf = st.checkbox("Include the no-radius case (∞)", value=True)
    try:
        radii = sorted({float(x.strip()) for x in raw.split(",") if x.strip()})
    except ValueError:
        st.error("Invalid radius list.")
        radii = []
    sweep_ctx = f"{dataset_id}|{radii}|{add_inf}"
    key, ran = run_state("sweep", sweep_ctx)
    if st.button("Run sensitivity analysis", key="btn_sweep", disabled=not radii):
        st.session_state[key] = ran = True
    if ran and radii:
        store = f"sweep_df_{sweep_ctx}"
        if store not in st.session_state:
            bar = st.progress(0.0, text="Computing…")
            st.session_state[store] = ex.radius_sweep(dist, school_df, student_df, radii + ([None] if add_inf else []),
                                                      progress=lambda p: bar.progress(p, text=f"Computing… {p:.0%}"))
            bar.empty()
        sweep = st.session_state[store].copy()
        top, labels = charts.sweep_x_labels(sweep["Radius (km)"].unique())
        sweep["x"] = sweep["Radius (km)"].replace(np.inf, top)
        show_chart(charts.lines_by_method(sweep, "x", "rejected", ex.SWEEP_METHODS, NAMES,
                                          "Students without a school (✕) vs. radius", "Radius (km)", "Students", labels))
        left, right = st.columns(2)
        with left:
            show_chart(charts.lines_by_method(sweep, "x", "max_km", [P, "B5"], NAMES,
                                              "Longest trip: proposed vs. without min-max stage", "Radius (km)", "km", labels))
        with right:
            show_chart(charts.lines_by_method(sweep, "x", "total_km", [P, "B5"], NAMES,
                                              "Total distance: proposed vs. without min-max stage", "Radius (km)", "km", labels))
        shown = metric_table(sweep, ["Radius (km)", "Code", "Method"], ["admitted", "rejected", "max_km", "mean_km", "total_km", "gini"])
        st.dataframe(shown.round(4), hide_index=True, width="stretch")
        download(shown.round(4), "radius_sensitivity")

# ------------------------------------------------------------------ Scenarios
with tabs[6]:
    st.markdown("Three bundled datasets with identical coordinates; only the total number of seats differs. "
                f"All use the {fmt_radius(radius)} radius from the sidebar.")
    key, ran = run_state("scenario", f"{radius}|{seeds}")
    if st.button("Run scenario study", key="btn_scenario"):
        st.session_state[key] = ran = True
    if ran:
        store = f"scenario_df_{radius}_{seeds}"
        if store not in st.session_state:
            bar = st.progress(0.0, text="Computing…")
            st.session_state[store] = ex.scenario_study(radius, int(seeds), progress=lambda p: bar.progress(p, text=f"Computing… {p:.0%}"))
            bar.empty()
        sc = st.session_state[store]
        show_chart(charts.scenario_grouped(sc, [P, "B2", "B4", "B5"], NAMES))
        shown = metric_table(sc, ["Scenario", "Description", "Students", "Total seats", "Original model feasible", "Code", "Method"],
                             ["admitted", "rejected", "gap_to_max", "seats_vacant", "max_km", "total_km"])
        st.dataframe(shown.round(3), hide_index=True, width="stretch")
        download(shown.round(4), "capacity_scenarios")

# ------------------------------------------------------------------ Statistical test
with tabs[7]:
    st.markdown("Students are resampled many times (quotas scaled by the same fraction). In each run the number of "
                "students ✕ under the proposed method is paired with each baseline and compared with a **one-sided "
                "Wilcoxon signed-rank test** (H₁: the baseline leaves more students without a school). "
                "B1 is excluded because it violates quotas.")
    c = st.columns(3)
    runs = c[0].number_input("Runs", 10, 200, 30, 5)
    fraction = c[1].slider("Share of students per run", 0.5, 0.95, 0.8, 0.05)
    base_seed = c[2].number_input("Base seed", 0, 10_000, 42)
    stat_ctx = f"{dataset_id}|{radius}|{runs}|{fraction}|{base_seed}"
    key, ran = run_state("stats", stat_ctx)
    if st.button("Run statistical test", key="btn_stats"):
        st.session_state[key] = ran = True
    if ran:
        store = f"stats_{stat_ctx}"
        if store not in st.session_state:
            bar = st.progress(0.0, text="Resampling…")
            st.session_state[store] = ex.resampling_test(school_df, student_df, radius, int(runs), float(fraction), int(base_seed),
                                                         progress=lambda p: bar.progress(p, text=f"Resampling… {p:.0%}"))
            bar.empty()
        runs_df, summary = st.session_state[store]
        st.dataframe(summary, hide_index=True, width="stretch",
                     column_config={"p-value": st.column_config.NumberColumn(format="%.2e"),
                                    "W statistic": st.column_config.NumberColumn(format="%.1f"),
                                    "Mean difference": st.column_config.NumberColumn(format="%.2f")})
        download(summary, "statistical_test_summary")
        show_chart(charts.resampling_box(runs_df, [P] + ex.TEST_METHODS, NAMES, P))
        with st.expander("Per-run data"):
            st.dataframe(runs_df.round(4), hide_index=True, width="stretch")
            download(runs_df.round(4), "statistical_test_runs")

# ------------------------------------------------------------------ Scalability
with tabs[8]:
    st.markdown("Synthetic data are built from the real student locations plus small random offsets; quotas are scaled "
                "to keep the loaded dataset's seats-per-student ratio.")
    sizes = st.multiselect("Dataset sizes (students)", [1000, 2000, 5000, 10000, 20000, 50000], default=[1000, 5000, 10000, 20000])
    scale_ctx = f"{dataset_id}|{radius}|{sorted(sizes)}"
    key, ran = run_state("scale", scale_ctx)
    if st.button("Run scalability test", key="btn_scale", disabled=not sizes):
        st.session_state[key] = ran = True
    if ran and sizes:
        store = f"scale_{scale_ctx}"
        if store not in st.session_state:
            bar = st.progress(0.0, text="Running…")
            st.session_state[store] = ex.scalability(school_df, student_df, sorted(sizes), radius, 7,
                                                     progress=lambda p: bar.progress(p, text=f"Running… {p:.0%}"))
            bar.empty()
        scale = st.session_state[store]
        show_chart(charts.runtime_line(scale))
        st.dataframe(scale.round(4), hide_index=True, width="stretch")
        download(scale.round(4), "scalability")

# ------------------------------------------------------------------ export all
exports = st.session_state.get("exports", {})
st.sidebar.divider()
if exports:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, df in exports.items():
            zf.writestr(f"{name}.csv", df.to_csv(index=False).encode("utf-8-sig"))
    st.sidebar.download_button(f"⬇ Download all tables ({len(exports)} files, ZIP)", buffer.getvalue(),
                               file_name="zonasi_2_results.zip", mime="application/zip", on_click="ignore")
else:
    st.sidebar.caption("Tables you compute can be downloaded together here.")
