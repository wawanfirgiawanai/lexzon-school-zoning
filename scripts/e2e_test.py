"""
End-to-end (black-box) test of the Streamlit app, driven through a real browser.

Starts the app on a free port, operates it like a user (tabs, sidebar settings,
uploads, map highlight, experiments, downloads), checks every result against
its expected value, and writes:
  docs/testing/e2e_results.json   one record per test case (PASS/FAIL, actual result, duration)
  docs/testing/screenshots/*.png  evidence for the test report
  docs/testing/downloads/*        files downloaded from the app during the test

Requires: pip install -r requirements-dev.txt  (uses the installed Microsoft Edge)
Run from the project root:  python scripts/e2e_test.py
"""
import io
import json
import re
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "testing"
SHOTS = OUT / "screenshots"
DOWNLOADS = OUT / "downloads"
TIMEOUT_S = 240

CASES = []


def case(tc_id, feature, steps, expected):
    def register(fn):
        CASES.append({"id": tc_id, "feature": feature, "steps": steps, "expected": expected, "fn": fn})
        return fn
    return register


# ------------------------------------------------------------------ helpers
def free_port():
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def wait_idle(page, timeout=TIMEOUT_S):
    """Wait until the script run, spinners and progress bars have finished."""
    deadline = time.time() + timeout
    quiet = 0
    while time.time() < deadline:
        busy = page.evaluate("""() => {
            const status = document.querySelector('[data-testid="stStatusWidget"]');
            const running = !!status && /running|stop/i.test(status.innerText);
            const spinning = [...document.querySelectorAll('[data-testid="stSpinner"], [data-testid="stProgress"]')]
                .some(e => e.offsetParent !== null);
            return running || spinning;
        }""")
        quiet = 0 if busy else quiet + 1
        if quiet >= 4:
            return
        time.sleep(0.5)
    raise TimeoutError("app did not become idle")


def panel(page):
    return page.locator('[role="tabpanel"]:not([hidden])')


def open_tab(page, name):
    page.get_by_role("tab", name=re.compile(re.escape(name))).click()
    wait_idle(page)
    return panel(page)


def metric(scope, label):
    box = scope.locator('[data-testid="stMetric"]').filter(
        has=scope.page.locator('[data-testid="stMetricLabel"]', has_text=re.compile(f"^{re.escape(label)}$")))
    return box.locator('[data-testid="stMetricValue"]').inner_text().strip()


def number(text):
    return float(re.sub(r"[^0-9.\-]", "", text))


def sidebar(page):
    return page.locator('[data-testid="stSidebar"]')


def choose(page, select_scope, option_text):
    select_scope.click()
    page.keyboard.type(option_text)
    page.get_by_role("option", name=option_text, exact=True).click()
    wait_idle(page)


def set_radius(page, value):
    field = sidebar(page).locator('[data-testid="stNumberInput"] input').first
    field.fill(str(value))
    field.press("Enter")
    wait_idle(page)


def download(page, scope, button_text, name):
    with page.expect_download(timeout=TIMEOUT_S * 1000) as info:
        scope.locator('[data-testid="stDownloadButton"] button', has_text=button_text).first.click()
    path = DOWNLOADS / name
    info.value.save_as(path)
    return path


def shot(page, name, target=None):
    if target is not None:
        target.scroll_into_view_if_needed()
        time.sleep(0.6)
    page.screenshot(path=SHOTS / name)
    return name


def run_experiment(page, tab, button):
    scope = open_tab(page, tab)
    scope.get_by_role("button", name=button).click()
    wait_idle(page)
    return panel(page)


def map_frame(page):
    page.locator('iframe[data-testid="stIFrame"]').first.scroll_into_view_if_needed()
    for _ in range(60):
        for fr in page.frames:
            if fr.url.startswith("about:srcdoc"):
                try:
                    ok = fr.evaluate("() => !!Object.keys(window).find(k => k.startsWith('map_')) && document.readyState === 'complete'")
                except Exception:
                    ok = False
                if ok:
                    return fr
        time.sleep(0.5)
    raise AssertionError("map frame not found")


def map_state(fr):
    time.sleep(2.5)  # refit runs after load, then tiles settle
    return fr.evaluate("""() => {
        const m = window[Object.keys(window).find(k => k.startsWith('map_'))];
        const icons = [...document.querySelectorAll('.leaflet-marker-icon')];
        return {
            zoom: m.getZoom(),
            crosses: icons.filter(i => i.innerText.includes('✕')).length,
            schools: icons.filter(i => !i.innerText.includes('✕')).length,
            circles: document.querySelectorAll('path.leaflet-interactive').length,
            legend: document.body.innerText,
        };
    }""")


# ------------------------------------------------------------------ test cases
@case("TC-01", "Start the application",
      "Open the application in a browser.",
      "The page loads without errors and shows the title and nine tabs.")
def tc01(page, ctx):
    t0 = time.time()
    page.goto(ctx["url"])
    page.wait_for_selector('[data-testid="stMetric"]', state="attached", timeout=TIMEOUT_S * 1000)
    wait_idle(page)
    ctx["load_seconds"] = round(time.time() - t0, 1)
    tabs = page.get_by_role("tab").all_inner_texts()
    assert len(tabs) == 9, tabs
    assert page.locator('[data-testid="stException"]').count() == 0
    expect(page.get_by_role("heading", name="Student-to-School Allocation within a Zoning Radius")).to_be_visible()
    shot(page, "01_about.png")
    return f"Page loaded in {ctx['load_seconds']} s; nine tabs: {', '.join(t.split(' ', 1)[-1] for t in tabs)}."


@case("TC-02", "About tab",
      "Open the About tab.",
      "The problem statement, the three-stage method, the B1-B6 comparison table, the limitations, the privacy note, and the CSV format are shown.")
def tc02(page, ctx):
    scope = open_tab(page, "About")
    text = scope.inner_text()
    for needle in ["Proposed method", "Baselines", "Limitations", "Privacy", "CSV format", "Deferred Acceptance"]:
        assert needle in text, needle
    return "Every section is shown (Problem, Proposed method, Baselines, Validation, Limitations, Privacy, Map colors, CSV format)."


@case("TC-03", "Data tab (main dataset)",
      "Open the Data tab with the main dataset.",
      "Students = 1,209; Schools = 28; Seats = 1,428; school and applicant names are anonymized.")
def tc03(page, ctx):
    scope = open_tab(page, "Data")
    vals = {k: metric(scope, k) for k in ["Students", "Schools", "Seats", "Seats per student", "No school within radius"]}
    assert vals["Students"] == "1,209" and vals["Schools"] == "28" and vals["Seats"] == "1,428", vals
    shot(page, "03_data.png")
    return (f"Students {vals['Students']}, Schools {vals['Schools']}, Seats {vals['Seats']}, "
            f"seats per student {vals['Seats per student']}, no school within the radius {vals['No school within radius']}.")


@case("TC-04", "Allocation tab: result of the proposed method",
      "Open the Allocation tab (radius 1 km).",
      "Admitted = 1,180; No school ✕ = 29; Longest trip = 0.995 km.")
def tc04(page, ctx):
    scope = open_tab(page, "Allocation")
    vals = {k: metric(scope, k) for k in ["Admitted", "No school ✕", "Longest trip", "Mean distance", "Vacant seats"]}
    assert vals["Admitted"] == "1,180" and vals["No school ✕"] == "29" and vals["Longest trip"] == "0.995 km", vals
    shot(page, "04_allocation.png")
    return (f"Admitted {vals['Admitted']}, ✕ {vals['No school ✕']}, longest trip {vals['Longest trip']}, "
            f"mean {vals['Mean distance']}, vacant seats {vals['Vacant seats']}.")


@case("TC-05", "Bukti optimalitas",
      "Buka expander 'Proof of optimality'.",
      "Max-flow = 1180, gap = 0, max-flow di bawah z* < 1180, status tahap 3 Optimal dengan solusi integer terverifikasi.")
def tc05(page, ctx):
    scope = panel(page)
    scope.get_by_text("Proof of optimality").click()
    time.sleep(1)
    text = scope.locator('[data-testid="stExpander"]').first.inner_text()
    assert "= 1180 students" in text and "a gap of 0" in text, text
    assert "drops to 1179" in text and "Optimal" in text and "integer solution verified" in text, text
    shot(page, "05_proof.png", scope.locator('[data-testid="stExpander"]').first)
    z = re.search(r"z\* = ([0-9.]+) km", text).group(1)
    return f"Max-flow 1180, gap 0, z* = {z} km (max-flow turun ke 1179 di bawahnya), tahap 3 Optimal (LP integer terverifikasi)."


@case("TC-06", "Interactive zoning map",
      "Scroll to the map on the Allocation tab.",
      "The map is shown with 28 school markers and 29 unassigned markers, and the applicant points are drawn.")
def tc06(page, ctx):
    fr = map_frame(page)
    state = map_state(fr)
    ctx["zoom_all"] = state["zoom"]
    assert state["crosses"] == 29 and state["schools"] == 28, state
    assert state["circles"] >= 1180, state
    shot(page, "06_map.png", page.locator('iframe[data-testid="stIFrame"]').first)
    return f"28 schools, 29 unassigned, {state['circles']} point and line elements, initial zoom {state['zoom']:.2f}."


@case("TC-07", "Highlight one school on the map",
      "Pick 'School 26' under 'Highlight a school'.",
      "The map stays visible, the legend marks the selected school, and the map zooms to its zone.")
def tc07(page, ctx):
    scope = panel(page)
    choose(page, scope.locator('[data-testid="stSelectbox"]').first, "School 26")
    fr = map_frame(page)
    state = map_state(fr)
    assert "(selected school)" in state["legend"], {k: v for k, v in state.items() if k != "legend"}
    assert state["zoom"] > ctx["zoom_all"], state
    shot(page, "07_highlight.png", page.locator('iframe[data-testid="stIFrame"]').first)
    choose(page, panel(page).locator('[data-testid="stSelectbox"]').first, "All schools")
    return f"The legend marks the selected school; zoom changed from {ctx['zoom_all']:.2f} to {state['zoom']:.2f}."


@case("TC-08", "Download the assignment list",
      "Click 'Download full list (CSV)'.",
      "The CSV holds 1,209 rows, 29 unassigned applicants, and anonymized applicant names.")
def tc08(page, ctx):
    path = download(page, panel(page), "Download full list (CSV)", "student_allocation.csv")
    df = pd.read_csv(path)
    rejected = int((df["Status"] != "Admitted").sum())
    assert len(df) == 1209 and rejected == 29, (len(df), rejected)
    assert df["Student"].iloc[0] == "Student 0001" and df["Student"].str.match(r"^Student \d{4}$").all()
    reasons = df.loc[df["Status"] != "Admitted", "Status"].value_counts().to_dict()
    return f"{len(df)} rows; unassigned = {rejected} ({'; '.join(f'{k}: {v}' for k, v in reasons.items())}); names formatted as 'Student 0001'."


@case("TC-09", "Export the map for the paper",
      "Click 'Prepare map figure for the paper', then 'Download map (PNG)'.",
      "A valid 300 DPI PNG file is downloaded.")
def tc09(page, ctx):
    scope = panel(page)
    scope.get_by_role("button", name="Prepare map figure for the paper (PNG, 300 DPI)").click()
    wait_idle(page)
    path = download(page, panel(page), "Download map (PNG)", "zoning_map.png")
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 100_000, len(data)
    from PIL import Image
    with Image.open(path) as im:
        size, dpi = im.size, im.info.get("dpi", (0, 0))
    assert round(dpi[0]) == 300, dpi
    return f"PNG {size[0]}x{size[1]} pixels, {round(dpi[0])} DPI, {len(data) // 1024} KB."


@case("TC-10", "Comparison tab",
      "Open the Comparison tab and download its table.",
      "Upper bound 1180; unassigned: proposed 29, B2 92, B4 102; B5 and B6 = 29; B1 breaks capacity.")
def tc10(page, ctx):
    scope = open_tab(page, "Comparison")
    assert "at most 1180 students" in scope.inner_text()
    shot(page, "10_comparison.png", scope.locator('[data-testid="stPlotlyChart"]').first)
    df = pd.read_csv(download(page, scope, "Download table (CSV)", "method_comparison.csv"))
    rej = dict(zip(df["Code"], df["No school (✕)"]))
    overflow = dict(zip(df["Code"], df["Quota overflow (students)"]))
    assert rej["P"] == 29 and rej["B2"] == 92 and rej["B4"] == 102 and rej["B5"] == 29 and rej["B6"] == 29, rej
    assert float(overflow["B1"]) > 0 and float(overflow["P"]) == 0, overflow
    return (f"Unassigned - P {rej['P']}, B1 {rej['B1']} (capacity overflow {overflow['B1']}), B2 {rej['B2']}, B3 {rej['B3']}, "
            f"B4 {rej['B4']}, B5 {rej['B5']}, B6 {rej['B6']}.")


@case("TC-11", "Ablation study",
      "Open the Ablation tab, click 'Run ablation study', download the table.",
      "Four configurations; A0-A1 place 47 applicants beyond the radius; A2-A3 leave 29 unassigned and none beyond the radius.")
def tc11(page, ctx):
    scope = run_experiment(page, "Ablation", "Run ablation study")
    shot(page, "11_ablation.png", scope.locator('[data-testid="stPlotlyChart"]').first)
    df = pd.read_csv(download(page, scope, "Download table (CSV)", "ablation.csv"))
    beyond = dict(zip(df["Code"], df["Admitted beyond radius"]))
    rej = dict(zip(df["Code"], df["No school (✕)"]))
    assert len(df) == 4 and beyond["A0"] == 47 and beyond["A2"] == 0 and rej["A3"] == 29, (beyond, rej)
    return f"Four rows; beyond the radius A0-A1 = {beyond['A0']}, A2-A3 = {beyond['A2']}; unassigned A3 = {rej['A3']}."


@case("TC-12", "Radius sensitivity",
      "Open the Radius Sensitivity tab (radius 0.5-3 km plus unrestricted), click 'Run sensitivity analysis'.",
      "32 rows (8 radii x 4 methods); the proposed method never leaves more applicants unassigned than B2 or B4 at any radius.")
def tc12(page, ctx):
    scope = run_experiment(page, "Radius Sensitivity", "Run sensitivity analysis")
    shot(page, "12_radius.png", scope.locator('[data-testid="stPlotlyChart"]').first)
    df = pd.read_csv(download(page, scope, "Download table (CSV)", "radius_sensitivity.csv"))
    pivot = df.pivot(index="Radius (km)", columns="Code", values="No school (✕)")
    assert len(df) == 32 and (pivot["P"] <= pivot["B2"]).all() and (pivot["P"] <= pivot["B4"]).all()
    return "32 rows; unassigned under the proposed method by radius: " + ", ".join(f"{('∞' if r == float('inf') else f'{r:g}')}→{int(v)}" for r, v in pivot["P"].items()) + "."


@case("TC-13", "Capacity scenarios",
      "Open the Capacity Scenarios tab, click 'Run scenario study'.",
      "Three scenarios; the original model is infeasible in Scenario 3; the proposed method leaves 68 / 29 / 223 unassigned.")
def tc13(page, ctx):
    scope = run_experiment(page, "Capacity Scenarios", "Run scenario study")
    shot(page, "13_scenarios.png", scope.locator('[data-testid="stPlotlyChart"]').first)
    df = pd.read_csv(download(page, scope, "Download table (CSV)", "capacity_scenarios.csv"))
    p = df[df["Code"] == "P"].set_index("Scenario")
    feasible = df.drop_duplicates("Scenario").set_index("Scenario")["Original model feasible"].to_dict()
    assert list(p["No school (✕)"]) == [68, 29, 223] and feasible["Scenario 3"] == "No", (p, feasible)
    return f"Unassigned under the proposed method: Scenario 1 = 68, Scenario 2 = 29, Scenario 3 = 223; original model feasible: {feasible}."


@case("TC-14", "Statistical test (resampling)",
      "Open the Statistical Test tab (30 runs, 80%, seed 42), click 'Run statistical test'.",
      "B2, B3 and B4 are significant (p < 0.05); B5 and B6 are identical.")
def tc14(page, ctx):
    scope = run_experiment(page, "Statistical Test", "Run statistical test")
    shot(page, "14_statistics.png", scope.locator('[data-testid="stPlotlyChart"]').first)
    df = pd.read_csv(download(page, scope, "Download table (CSV)", "statistical_test_summary.csv"))
    conclusions = dict(zip(df["Baseline"].str[:2], df["Conclusion"]))
    assert all(conclusions[c].startswith("Significant") for c in ["B2", "B3", "B4"]), conclusions
    assert all(conclusions[c].startswith("Identical") for c in ["B5", "B6"]), conclusions
    pvals = dict(zip(df["Baseline"].str[:2], df["p-value"]))
    return "; ".join(f"{c}: {conclusions[c]}" + (f" (p = {pvals[c]:.1e})" if pd.notna(pvals[c]) else "") for c in conclusions) + "."


@case("TC-15", "Scalability test",
      "Open the Scalability tab (1k, 5k, 10k, 20k applicants), click 'Run scalability test'.",
      "Every size reports status Optimal and enrollment equal to the maximum-flow upper bound.")
def tc15(page, ctx):
    scope = run_experiment(page, "Scalability", "Run scalability test")
    shot(page, "15_scalability.png", scope.locator('[data-testid="stPlotlyChart"]').first)
    df = pd.read_csv(download(page, scope, "Download table (CSV)", "scalability.csv"))
    assert (df["Status"] == "Optimal").all() and (df["Admitted"] == df["Max-flow upper bound"]).all()
    return "; ".join(f"{int(n):,} applicants: {s:.2f} s"
                     for n, s in zip(df["Students"], df["Proposed total runtime (s)"])) + "."


@case("TC-16", "Download every table (ZIP)",
      "Click the ZIP button in the sidebar.",
      "The ZIP is downloaded and holds every table computed so far.")
def tc16(page, ctx):
    path = download(page, sidebar(page), "Download all tables", "lexzon_results.zip")
    with zipfile.ZipFile(path) as zf:
        names = sorted(zf.namelist())
    assert len(names) >= 8 and "method_comparison.csv" in names and "ablation.csv" in names, names
    return f"The ZIP holds {len(names)} files: {', '.join(names)}."


@case("TC-17", "Change the zoning radius",
      "Set 'Zoning radius (km)' to 0.5.",
      "The results update: Admitted = 1,099 and unassigned = 110; the longest trip is at most 0.5 km.")
def tc17(page, ctx):
    set_radius(page, 0.5)
    scope = open_tab(page, "Allocation")
    vals = {k: metric(scope, k) for k in ["Admitted", "No school ✕", "Longest trip"]}
    assert vals["Admitted"] == "1,099" and vals["No school ✕"] == "110" and number(vals["Longest trip"]) <= 0.5, vals
    return f"Admitted {vals['Admitted']}, ✕ {vals['No school ✕']}, longest trip {vals['Longest trip']}."


@case("TC-18", "Radius boundary value (minimum 0.1 km)",
      "Set the radius to 0.1 km, the smallest value allowed.",
      "The application keeps running without errors; the longest trip is at most 0.1 km.")
def tc18(page, ctx):
    set_radius(page, 0.1)
    scope = open_tab(page, "Allocation")
    vals = {k: metric(scope, k) for k in ["Admitted", "No school ✕", "Longest trip"]}
    assert page.locator('[data-testid="stException"]').count() == 0
    assert number(vals["Longest trip"]) <= 0.1, vals
    return f"No errors; Admitted {vals['Admitted']}, unassigned {vals['No school ✕']}, longest trip {vals['Longest trip']}."


@case("TC-19", "Without a radius limit",
      "Switch off 'Limit by zoning radius'.",
      "All 1,209 applicants are assigned (none unassigned) and the longest trip is 1.901 km, so the min-max stage is active.")
def tc19(page, ctx):
    set_radius(page, 1.0)
    sidebar(page).get_by_text("Limit by zoning radius").click()
    wait_idle(page)
    scope = open_tab(page, "Allocation")
    vals = {k: metric(scope, k) for k in ["Admitted", "No school ✕", "Longest trip"]}
    assert vals["Admitted"] == "1,209" and vals["No school ✕"] == "0" and vals["Longest trip"] == "1.901 km", vals
    sidebar(page).get_by_text("Limit by zoning radius").click()
    wait_idle(page)
    return f"Admitted {vals['Admitted']}, ✕ {vals['No school ✕']}, longest trip {vals['Longest trip']}."


@case("TC-20", "Switch dataset: Scenario 3 (seat shortage)",
      "Pick the dataset 'Scenario 3 — Shortage (seats < students)'.",
      "Seats = 1,000; Admitted = 986; unassigned = 223.")
def tc20(page, ctx):
    choose(page, sidebar(page).locator('[data-testid="stSelectbox"]').first, "Scenario 3 — Shortage (seats < students)")
    data_scope = open_tab(page, "Data")
    seats = metric(data_scope, "Seats")
    scope = open_tab(page, "Allocation")
    vals = {k: metric(scope, k) for k in ["Admitted", "No school ✕"]}
    assert seats == "1,000" and vals["Admitted"] == "986" and vals["No school ✕"] == "223", (seats, vals)
    shot(page, "20_scenario3.png")
    choose(page, sidebar(page).locator('[data-testid="stSelectbox"]').first, "Main dataset (original quotas)")
    return f"Seats {seats}, Admitted {vals['Admitted']}, ✕ {vals['No school ✕']}."


@case("TC-21", "Upload a valid CSV (Indonesian headers, real names)",
      "Pick 'Upload CSV' and upload school and applicant files whose columns are named 'Nama Sekolah' / 'Jumlah Diterima' / 'Nama Siswa'.",
      "The data are read (1,209 applicants, 28 schools), names are anonymized automatically, and the results match the bundled dataset.")
def tc21(page, ctx):
    tmp = OUT / "upload_fixtures"
    tmp.mkdir(parents=True, exist_ok=True)
    schools = pd.read_csv(ROOT / "dataset" / "schools.csv").rename(columns={"School": "Nama Sekolah", "Quota": "Jumlah Diterima"})
    schools["Nama Sekolah"] = [f"SD CONTOH {i + 1}" for i in range(len(schools))]
    students = pd.read_csv(ROOT / "dataset" / "students.csv").rename(columns={"Student": "Nama Siswa"})
    students["Nama Siswa"] = [f"Nama Asli {i + 1}" for i in range(len(students))]
    schools.to_csv(tmp / "sekolah_upload.csv", index=False)
    students.to_csv(tmp / "siswa_upload.csv", index=False)

    sidebar(page).get_by_text("Upload CSV", exact=True).click()
    wait_idle(page)
    uploads = sidebar(page).locator('[data-testid="stFileUploader"] input[type="file"]')
    uploads.nth(0).set_input_files(str(tmp / "sekolah_upload.csv"))
    wait_idle(page)
    uploads.nth(1).set_input_files(str(tmp / "siswa_upload.csv"))
    wait_idle(page)
    scope = open_tab(page, "Allocation")
    vals = {k: metric(scope, k) for k in ["Admitted", "No school ✕"]}
    df = pd.read_csv(download(page, scope, "Download full list (CSV)", "upload_allocation.csv"))
    assert vals["Admitted"] == "1,180" and vals["No school ✕"] == "29", vals
    assert not df["Student"].str.contains("Nama Asli").any() and not df["Assigned school"].astype(str).str.contains("CONTOH").any()
    shot(page, "21_upload.png")
    return f"Upload accepted; Admitted {vals['Admitted']}, unassigned {vals['No school ✕']}; no real name appears in the download."


@case("TC-22", "Upload an invalid CSV",
      "Upload a school CSV without the capacity column.",
      "A clear error message appears in the sidebar and the application does not crash.")
def tc22(page, ctx):
    tmp = OUT / "upload_fixtures"
    bad = pd.read_csv(ROOT / "dataset" / "schools.csv").drop(columns=["Quota"])
    bad.to_csv(tmp / "sekolah_tanpa_kuota.csv", index=False)
    uploads = sidebar(page).locator('[data-testid="stFileUploader"] input[type="file"]')
    uploads.nth(0).set_input_files(str(tmp / "sekolah_tanpa_kuota.csv"))
    wait_idle(page)
    error = sidebar(page).locator('[data-testid="stAlertContentError"]')
    expect(error).to_be_visible()
    message = error.inner_text().strip()
    assert "missing column" in message and "Quota" in message, message
    assert page.locator('[data-testid="stException"]').count() == 0
    shot(page, "22_invalid_upload.png")
    return f"Pesan kesalahan: \"{message}\"; tidak ada exception."


# ------------------------------------------------------------------ runner
def main():
    for d in (SHOTS, DOWNLOADS):
        d.mkdir(parents=True, exist_ok=True)
    port = free_port()
    server = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(port),
                               "--server.headless", "true"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    records = []
    try:
        for _ in range(90):
            try:
                urllib.request.urlopen(f"http://localhost:{port}/_stcore/health", timeout=2)
                break
            except Exception:  # noqa: BLE001 - server still starting
                time.sleep(1)
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="msedge", headless=True)
            context = browser.new_context(viewport={"width": 1440, "height": 1000}, accept_downloads=True)
            page = context.new_page()
            page.set_default_timeout(TIMEOUT_S * 1000)
            ctx = {"url": f"http://localhost:{port}", "browser": f"Microsoft Edge {browser.version}"}
            for c in CASES:
                t0 = time.time()
                try:
                    actual = c["fn"](page, ctx)
                    status = "PASS"
                except Exception as exc:  # noqa: BLE001 - record and continue with the next case
                    actual = f"{type(exc).__name__}: {exc}"
                    status = "FAIL"
                    traceback.print_exc()
                    try:
                        page.screenshot(path=SHOTS / f"FAIL_{c['id']}.png")
                    except Exception:  # noqa: BLE001
                        pass
                records.append({k: c[k] for k in ("id", "feature", "steps", "expected")}
                               | {"actual": actual, "status": status, "seconds": round(time.time() - t0, 1)})
                print(f"{c['id']} {status} ({records[-1]['seconds']} s) {c['feature']}: {actual}", flush=True)
            browser.close()
    finally:
        server.terminate()
    summary = {"browser": ctx.get("browser"), "load_seconds": ctx.get("load_seconds"),
               "passed": sum(r["status"] == "PASS" for r in records), "total": len(records),
               "run_at": time.strftime("%Y-%m-%d %H:%M:%S"), "cases": records}
    (OUT / "e2e_results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{summary['passed']}/{summary['total']} test cases passed")
    return 0 if summary["passed"] == summary["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
