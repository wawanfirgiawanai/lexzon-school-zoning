"""Generate the Markdown results/formula/methods document from the experiment CSVs."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\Wawan Firgiawan\OneDrive\Documents\PROJECT\PENELITIAN\Zonasi_2")
R = ROOT / "docs" / "results"
OUT = ROOT / "Document" / "Hasil_Pengujian_Rumus_Metode_LEXZON.md"
SC = ["S1", "S2", "S3"]
NAME = {"P": "LEXZON", "B1": "B1 Sekolah terdekat (kuota diabaikan)", "B2": "B2 Greedy terdekat",
        "B3": "B3 Greedy urutan acak (rata-rata 30 seed)", "B4": "B4 Deferred acceptance",
        "B5": "B5 Eksak total jarak minimum", "B6": "B6 Eksak jarak terjauh minimum",
        "B7": "B7 Constrained k-means", "LEXZON": "LEXZON", "LEXZON-F": "LEXZON-F"}


def f(v, d=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if isinstance(v, (int, np.integer)) or float(v).is_integer():
        return f"{int(round(float(v))):,}".replace(",", ".")
    return f"{float(v):,.{d}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


md = []
w = md.append

w("# LEXZON: Hasil Pengujian, Rumus, dan Bab Material dan Metode\n")
w("Dokumen ini berisi tabel hasil pengujian (siap divisualisasikan), rumus, dan bab Material dan Metode. "
  "Semua angka diambil langsung dari berkas hasil eksperimen di `docs/results/`. Radius zonasi R = 1 km "
  "kecuali disebutkan lain. Format angka: titik = pemisah ribuan, koma = desimal.\n")
w("**Singkatan:** TT = siswa tidak tertampung; N* = batas atas penerimaan (aliran maksimum); "
  "z* = jarak terjauh minimum; S1/S2/S3 = skenario kuota pas/lebih/kurang.\n")

# ============================================================ MATERIAL & METODE
w("\n---\n\n# Bagian A. Material dan Metode\n")
w("## A.1 Material (data)\n")
w("Data berisi lokasi rumah **1.209 calon siswa** serta lokasi dan kuota **28 sekolah dasar negeri** di wilayah "
  "perkotaan Kabupaten Majene, Sulawesi Barat (cakupan ±5 km × 6 km). Nama siswa dan sekolah dianonimkan "
  "(Student 0001, School 01, ...); hanya koordinat (lintang, bujur) dan kuota yang digunakan.\n")
w("### Tabel A1. Karakteristik data\n")
w(table(["Besaran", "Nilai"], [
    ["Jumlah calon siswa", "1.209"], ["Jumlah sekolah dasar", "28"],
    ["Pasangan siswa–sekolah seluruhnya", "33.852"], ["Pasangan layak (jarak ≤ 1 km)", "6.641"],
    ["Rata-rata sekolah dalam radius per siswa", "5,5 (median 5, maks. 10)"],
    ["Siswa tanpa sekolah dalam radius 1 km", "3"],
    ["Median jarak ke sekolah terdekat", "0,185 km"], ["Jarak terdekat terbesar", "1,105 km"]]))

summ = pd.read_csv(R / "scenarios" / "scenario_summary.csv")
w("### Tabel A2. Skenario kuota\n")
desc = {"S1": "Kuota pas (kursi = siswa)", "S2": "Kuota lebih (kuota asli)", "S3": "Kuota kurang"}
src = {"S1": "Penskalaan mendekati proporsional kuota asli", "S2": "Data asli (1–3 rombel × 28)",
       "S3": "Penskalaan mendekati proporsional kuota asli"}
rows = []
for _, r in summ.iterrows():
    rows.append([r["Scenario"], desc[r["Scenario"]], f(r["Seats"]), f(r["Seats"] / r["Students"], 2),
                 r["Quota values"].replace(", ", "–") if r["Scenario"] != "S2" else "28 / 56 / 84", src[r["Scenario"]]])
w(table(["Skenario", "Keterangan", "Total kursi", "Kursi/siswa", "Kuota per sekolah", "Sumber"], rows))

w("## A.2 Kerangka kerja sistem\n")
w(table(["Lapisan", "Komponen", "Keluaran"], [
    ["1. Data dan praproses", "Validasi CSV, anonimisasi, jarak Haversine, pasangan layak E_R", "Matriks jarak d_ij, E_R"],
    ["2. Optimasi LEXZON", "Tahap 1 aliran maksimum → Tahap 2 bisection → Tahap 3 aliran biaya minimum", "Alokasi x, N*, z*"],
    ["3. Verifikasi dan audit", "Sertifikat optimalitas, penyebab siswa ×, pelanggaran prioritas, kurva harga keadilan, LEXZON-F", "Bukti optimal, audit keadilan"],
    ["4. Pendukung keputusan", "Peta interaktif, simulasi radius/kuota/+1 rombel, ekspor", "Peta, tabel, berkas"]]))

w("## A.3 Metode usulan: LEXZON (Lexicographic Exact Zoning)\n")
w("LEXZON menyelesaikan tiga tujuan secara berurutan (leksikografis); keluaran setiap tahap menjadi kendala "
  "bagi tahap berikutnya:\n")
w("1. **Tahap 1 – Penerimaan maksimum.** Aliran maksimum (algoritma Dinic) pada jaringan sumber → siswa "
  "(kapasitas 1) → sekolah dalam radius (kapasitas 1) → tujuan (kapasitas q_j). Hasil: N*.\n"
  "2. **Tahap 2 – Jarak terjauh minimum.** Pencarian biner (bisection) atas jarak unik; setiap langkah menguji "
  "apakah aliran maksimum dengan pasangan berjarak ≤ t masih bernilai N*. Hasil: z*.\n"
  "3. **Tahap 3 – Total jarak minimum.** Aliran biaya minimum yang mengalirkan tepat N* siswa pada pasangan "
  "berjarak ≤ z*, diselesaikan dengan dual simplex HiGHS (matriks unimodular total ⇒ solusi LP integral).\n"
  "4. **Sertifikat optimalitas.** F(R) = F(z*) = N* dan F(z⁻) < N*, ditambah integralitas Tahap 3.\n"
  "5. **Penandaan ×.** Siswa tidak tertampung diberi penyebab: tidak ada sekolah dalam radius, atau semua "
  "sekolah dalam radius penuh.\n")

w("### Algoritma 1. LEXZON\n")
w("```text\nMasukan : jarak d_ij, kuota q_j, radius R\nKeluaran: alokasi x, N*, z*, sertifikat\n"
  " 1: E_R ← {(i, j) : d_ij ≤ R}\n"
  " 2: N* ← AliranMaks(E_R, q)                         // Tahap 1\n"
  " 3: D  ← jarak unik terurut pada E_R; lo ← 1; hi ← |D|\n"
  " 4: selama lo < hi lakukan                           // Tahap 2\n"
  " 5:     mid ← ⌊(lo + hi) / 2⌋\n"
  " 6:     jika AliranMaks(E_D[mid], q) = N* maka hi ← mid selainnya lo ← mid + 1\n"
  " 7: z* ← D[lo]\n"
  " 8: x ← AliranBiayaMin(E_z*, q, nilai aliran N*)       // Tahap 3\n"
  " 9: untuk setiap siswa i tanpa pasangan terpilih lakukan tandai ×\n"
  "10:     penyebab ← tidak ada sekolah dalam R, atau semua sekolah dalam R penuh\n"
  "11: kembalikan x, N*, z*, (F(R), F(z*), F(z⁻))\n```\n")

w("### Algoritma 2. Kurva harga keadilan dan LEXZON-F\n")
w("```text\nMasukan : d_ij, q_j, R, batas pelanggaran K = {k1 < k2 < …}, batas waktu τ\n"
  "Keluaran: kurva {(k, A(k))}, alokasi LEXZON-F\n"
  "1: μ_L ← LEXZON(d, q, R);  μ_D ← DeferredAcceptance(d, q, R)\n"
  "2: bangun kendala (7), (8), (15) untuk semua (i, j) ∈ E_R\n"
  "3: (μ_F, ε_F) ← selesaikan (17) dengan solusi awal μ_L, batas waktu τ\n"
  "4: μ ← μ_D; catat (0, |μ_D|)                         // Proposisi 2\n"
  "5: untuk k ∈ K dengan 0 < k < ε_F lakukan\n"
  "6:     μ ← selesaikan (16) dengan solusi awal μ, batas waktu τ\n"
  "7:     catat (k, |μ|, batas atas solver)\n"
  "8: catat (ε_F, N*); kembalikan kurva dan μ_F\n```\n")

w("## A.4 Metode pembanding\n")
w(table(["Kode", "Metode", "Cara kerja", "Peran"], [
    ["B1", "Sekolah terdekat", "Setiap siswa ke sekolah terdekat dalam radius, kuota diabaikan", "Acuan (tidak layak)"],
    ["B2", "Greedy terdekat", "Siswa terdekat diproses dahulu, masuk sekolah terdekat yang masih ada kursi", "Aturan manual umum"],
    ["B3", "Greedy acak", "Aturan B2 dengan urutan acak, rata-rata 30 seed", "Siapa cepat dia dapat"],
    ["B4", "Deferred acceptance", "Siswa melamar sekolah terdekat; sekolah menahan pendaftar terdekat", "Seleksi jarak PPDB/SPMB (stabil)"],
    ["B5", "Eksak total jarak minimum", "Tahap 1 + Tahap 3 (tanpa Tahap 2)", "Uji perlunya Tahap 2"],
    ["B6", "Eksak jarak terjauh minimum", "Tahap 1 + Tahap 2 (tanpa Tahap 3)", "Uji perlunya Tahap 3"],
    ["B7", "Constrained k-means", "Klaster seimbang berkapasitas (Zainuddin dan Risal, 2024)", "Metode studi zonasi Indonesia"]]))

w("## A.5 Studi ablasi\n")
w("Setiap konfigurasi menambahkan satu komponen. Radius dan izin tidak ditempatkan masuk bersamaan pada A2, karena "
  "tiga siswa tidak memiliki sekolah dalam 1 km sehingga radius tanpa izin tidak ditempatkan tidak memiliki solusi. "
  "Objektif A2 dan A3 bersifat **leksikografis**: jumlah siswa diterima dimaksimalkan terlebih dahulu, sehingga "
  "alokasi kosong tidak mungkin menjadi optimum.\n")
w(table(["Kode", "Kendala penempatan", "Jarak", "Radius", "Objektif (berurutan)", "Setara dengan"], [
    ["A0", "Semua siswa wajib ditempatkan (Σ_j x_ij = 1)", "Euclidean (derajat)", "Tidak ada", "min total jarak", "Model awal tesis"],
    ["A1", "Semua siswa wajib ditempatkan (Σ_j x_ij = 1)", "Haversine (km)", "Tidak ada", "min total jarak", "–"],
    ["A2", "Boleh tidak ditempatkan (Σ_j x_ij ≤ 1)", "Haversine (km)", "R = 1 km", "(1) maks. diterima → (2) min total jarak", "B5"],
    ["A3", "Boleh tidak ditempatkan (Σ_j x_ij ≤ 1)", "Haversine (km)", "R = 1 km", "(1) maks. diterima → (2) min jarak terjauh → (3) min total jarak", "LEXZON"]]))

w("## A.6 Metrik evaluasi\n")
w(table(["Metrik", "Definisi"], [
    ["Tidak tertampung (TT)", "Jumlah siswa tanpa sekolah; dipisah: tanpa sekolah dalam radius / sekolah penuh"],
    ["Selisih terhadap N*", "N* − jumlah siswa diterima (0 = optimal)"],
    ["Melebihi kuota", "Jumlah siswa yang ditempatkan di atas kuota (kelayakan)"],
    ["Rata-rata / maks / total jarak", "Jarak Haversine siswa diterima ke sekolahnya (km)"],
    ["Pelanggaran prioritas (justified envy)", "Siswa yang lebih memilih sekolah dalam radius yang masih ada kursi atau menerima siswa lebih jauh"],
    ["Waktu komputasi", "Detik (median tiga pengulangan untuk skalabilitas)"]]))

w("## A.7 Protokol pengujian\n")
w(table(["No", "Pengujian", "Pengaturan"], [
    ["1", "Perbandingan metode", "3 skenario, R = 1 km, 8 metode"],
    ["2", "Studi ablasi", "S1, S2, S3; A0–A3"],
    ["3", "Sensitivitas radius", "R = 0,5; 0,75; 1; 1,25; 1,5; 2; 3 km; tanpa radius"],
    ["4", "Uji statistik", "30 sampel ulang (80% siswa, seed 42 + nomor), kuota diskalakan; Wilcoxon satu sisi, α = 0,05"],
    ["5", "Skalabilitas", "1.000–50.000 siswa sintetis (lokasi nyata + gangguan Gaussian 150 m), median 3 pengulangan"],
    ["6", "Keadilan prioritas jarak", "Justified envy untuk semua metode, data penuh dan 90 sampel ulang"],
    ["7", "Harga keadilan", "k ∈ {0, 10, 25, 50, 100, 150, 200}; 300 detik per titik; LEXZON-F 900 detik (CP-SAT)"],
    ["8", "Simulasi kebijakan", "+1 rombel (28 kursi) di setiap sekolah bergantian"],
    ["9", "Pengujian perangkat lunak", "126 uji unit (pytest) + 22 uji end-to-end (Playwright)"]]))

w("## A.8 Lingkungan implementasi\n")
w(table(["Komponen", "Versi / spesifikasi"], [
    ["Bahasa", "Python 3.11.9"], ["Numerik", "NumPy 1.26.4, SciPy 1.17.1, pandas 3.0.2"],
    ["Aliran maksimum", "scipy.sparse.csgraph.maximum_flow (Dinic)"], ["LP/MILP", "HiGHS (dual simplex) via SciPy"],
    ["Model harga keadilan", "Google OR-Tools CP-SAT"], ["Aplikasi", "Streamlit 1.57.0, Folium 0.20.0"],
    ["Perangkat keras", "Intel Core i7-14650HX, RAM 16 GB, Windows 11"]]))

# ============================================================ RUMUS
w("\n---\n\n# Bagian B. Rumus\n")
eqs = [
    ("(1) Jarak Haversine", r"d_{ij} = 2\rho \arcsin\sqrt{a_{ij}}"),
    ("(2) Komponen Haversine", r"a_{ij} = \sin^2\!\left(\frac{\Delta\varphi}{2}\right) + \cos\varphi_i \cos\varphi_j \sin^2\!\left(\frac{\Delta\lambda}{2}\right),\quad \rho = 6.371\ \text{km}"),
    ("(3) Pasangan layak", r"E_R = \{(i,j) \in S \times C : d_{ij} \le R\}"),
    ("(4) Tujuan leksikografis", r"\operatorname{lex\,max}_x \big(f_1(x),\ -f_2(x),\ -f_3(x)\big)"),
    ("(5) Tujuan 1 dan 3", r"f_1(x) = \sum_{(i,j)\in E_R} x_{ij},\qquad f_3(x) = \sum_{(i,j)\in E_R} d_{ij}\,x_{ij}"),
    ("(6) Tujuan 2", r"f_2(x) = \max_{(i,j)\in E_R} d_{ij}\,x_{ij}"),
    ("(7) Kendala siswa", r"\sum_{j:(i,j)\in E_R} x_{ij} \le 1,\quad \forall i \in S"),
    ("(8) Kendala kuota", r"\sum_{i:(i,j)\in E_R} x_{ij} \le q_j,\quad \forall j \in C,\quad x_{ij}\in\{0,1\}"),
    ("(9) Aliran maksimum berambang", r"F(t) = \max\Big\{\sum_{d_{ij}\le t} x_{ij} : x \text{ layak}\Big\},\qquad N^* = F(R)"),
    ("(10) Jarak terjauh minimum", r"z^* = \min\{t \in D_R : F(t) = N^*\}"),
    ("(11) Total jarak minimum", r"\min \sum_{d_{ij}\le z^*} d_{ij}\,x_{ij}\quad \text{s.t. } \sum x_{ij} = N^*,\ (7),(8)"),
    ("(12) Sertifikat optimalitas", r"F(z^*) = N^* \ \text{dan}\ F(z^-) < N^*"),
    ("(13) Prioritas jarak", r"i \succ_j i' \Leftrightarrow d_{ij} < d_{i'j},\qquad j \succ_i j' \Leftrightarrow d_{ij} < d_{ij'}"),
    ("(14) Pelanggaran prioritas", r"\varepsilon_i = 1 \Leftrightarrow \exists (i,j)\in E_R : j \succ_i \mu(i) \ \wedge\ \big(|\mu^{-1}(j)| < q_j \ \vee\ \exists i'\in\mu^{-1}(j): i \succ_j i'\big)"),
    ("(15) Kestabilan berbobot", r"\sum_{i' \succ_j i} x_{i'j} + q_j \sum_{j' \succeq_i j} x_{ij'} \ge q_j\,(1 - y_i)"),
    ("(16) Kurva harga keadilan", r"A(k) = \max \sum_{(i,j)\in E_R} x_{ij}\quad \text{s.t. } (7),(8),(15),\ \sum_i y_i \le k"),
    ("(17) LEXZON-F", r"\varepsilon_F = \min \sum_i y_i\quad \text{s.t. } (7),(8),(15),\ \sum_{(i,j)\in E_R} x_{ij} = N^*"),
]
for name, tex in eqs:
    w(f"**{name}**\n\n$$\n{tex}\n$$\n")
w("**Metrik tambahan**\n\n$$\n\\text{Pengurangan TT}(\\%) = \\frac{TT_{\\text{pembanding}} - TT_{\\text{LEXZON}}}{TT_{\\text{pembanding}}}\\times 100\n$$\n")
w("$$\n\\text{Selisih terhadap } N^* = N^* - \\sum_{(i,j)} x_{ij}\n$$\n")
w("**Proposisi 1.** Setiap alokasi LEXZON (a) tidak menyisakan siswa tidak tertampung yang memiliki sekolah "
  "dalam radius dengan kursi kosong, dan (b) tidak menolak siswa yang lebih dekat ke suatu sekolah daripada "
  "siswa yang diterima di sekolah tersebut.\n\n"
  "**Proposisi 2.** A(0) = jumlah siswa diterima deferred acceptance (teorema rural hospitals), dan "
  "A(k) = N* untuk k ≥ ε_F.\n")

# ============================================================ HASIL
w("\n---\n\n# Bagian C. Hasil Pengujian\n")
w("## C.1 Pengujian perangkat lunak\n")
w(table(["Jenis uji", "Jumlah", "Lulus", "Keterangan"], [
    ["Uji unit solver (optimum brute force, batas, kelayakan)", "50", "50", "test_solver.py"],
    ["Uji unit keadilan dan constrained k-means", "52", "52", "test_fairness.py"],
    ["Uji unit model harga keadilan (Proposisi 2)", "24", "24", "test_pareto.py"],
    ["Uji end-to-end aplikasi (peramban)", "22", "22", "TC-01 s.d. TC-22"],
    ["Reproduktibilitas (29 berkas hasil dijalankan ulang)", "29", "29", "Identik, kecuali waktu komputasi"]]))

w("## C.2 Hasil tiap tahap LEXZON per skenario\n")
w(table(["Skenario", "Kursi", "N*", "z* (km)", "Langkah bisection", "F(z⁻)", "Sekolah penuh", "Model paksa A0: di luar R"],
        [[r["Scenario"], f(r["Seats"]), f(r["Max-flow bound N*"]), f(r["z* (km)"]), f(r["Bisection steps"]),
          f(r["Certificate flow below z*"]), f(r["Proposed schools full"]),
          f(r["Forced model beyond R"]) if r["Forced model feasible"] else "tidak layak"] for _, r in summ.iterrows()]))

b7 = pd.read_csv(R / "fairness" / "b7_comparison.csv")
w("## C.3 Perbandingan metode (R = 1 km)\n")
for s in SC:
    c = pd.read_csv(R / "scenarios" / f"{s}_comparison.csv")
    c = pd.concat([c, b7[b7["Scenario"] == s].assign(Code="B7")], ignore_index=True)
    rows = []
    for _, r in c.iterrows():
        rows.append([NAME[r["Code"]], f(r["admitted"], 1), f(r["rejected"], 1), f(r["rejected_no_school"]),
                     f(r["rejected_full"], 1), f(r["gap_to_max"], 1), f(r["overflow_students"]), f(r["mean_km"]),
                     f(r["max_km"]), f(r["total_km"], 1), f(r["gini"]), f(r["runtime_s"], 3)])
    w(f"### Tabel C3-{s}. Perbandingan metode – {s}: {desc[s].lower()}\n")
    w(table(["Metode", "Diterima", "TT", "TT: tanpa sekolah", "TT: sekolah penuh", "Selisih N*", "Melebihi kuota",
             "Rata² (km)", "Maks (km)", "Total (km)", "Gini", "Waktu (s)"], rows))

w("### Tabel C3-R. Ringkasan pengurangan siswa tidak tertampung oleh LEXZON (%)\n")
rows = []
for code in ["B2", "B3", "B4", "B7"]:
    row = [NAME[code]]
    for s in SC:
        c = pd.read_csv(R / "scenarios" / f"{s}_comparison.csv").set_index("Code")
        base = b7.set_index("Scenario").loc[s, "rejected"] if code == "B7" else c.loc[code, "rejected"]
        row.append(f(100 * (base - c.loc["P", "rejected"]) / base, 1))
    rows.append(row)
w(table(["Pembanding", "S1", "S2", "S3"], rows))

ab = pd.read_csv(R / "scenarios" / "ablation_all_scenarios.csv")
w("## C.4 Studi ablasi\n")
lab = {"A0": "Model awal (Euclidean derajat, semua wajib ditempatkan)", "A1": "+ Haversine (km)",
       "A2": "+ radius R = 1 km dengan izin tidak ditempatkan", "A3": "+ tahap jarak terjauh minimum (LEXZON)"}
rows = []
for _, r in ab.iterrows():
    ok = r["Status"] == "Optimal"
    rows.append([r["Scenario"], r["Code"], lab[r["Code"]], "Layak" if ok else "Tidak layak (kursi < siswa)",
                 f(r["admitted"]) if ok else "–", f(r["rejected"]) if ok else "–", f(r["beyond_radius"]) if ok else "–",
                 f(r["max_km"]) if ok else "–", f(r["total_km"], 1) if ok else "–"])
w(table(["Skenario", "Kode", "Konfigurasi", "Status", "Diterima", "TT", "Di luar R", "Maks (km)", "Total (km)"], rows))

w("## C.5 Sensitivitas radius\n")
for metric, label, d in [("rejected", "Siswa tidak tertampung", 0), ("max_km", "Jarak terjauh (km)", 3),
                         ("total_km", "Total jarak (km)", 1)]:
    w(f"### Tabel C5-{'abc'[['rejected','max_km','total_km'].index(metric)]}. {label} menurut radius\n")
    rows = []
    for s in SC:
        sw = pd.read_csv(R / "scenarios" / f"{s}_radius.csv")
        piv = sw.pivot(index="Radius (km)", columns="Code", values=metric)
        for rad, r in piv.iterrows():
            rows.append([s, "∞" if np.isinf(rad) else f(rad, 2), f(r["P"], d), f(r["B5"], d), f(r["B2"], d), f(r["B4"], d)])
    w(table(["Skenario", "Radius (km)", "LEXZON", "B5", "B2", "B4"], rows))

w("## C.6 Uji statistik Wilcoxon satu sisi (30 sampel ulang, 80% siswa)\n")
st = pd.read_csv(R / "scenarios" / "scenario_stat_summary.csv")
bw = pd.read_csv(R / "fairness" / "b7_wilcoxon.csv")
rows = []
for s in SC:
    for _, r in st[st["Scenario"] == s].iterrows():
        code = r["Baseline"].split(" ")[0]
        rows.append([s, code, r["✕ Proposed (mean ± sd)"].replace(".", ","), r["✕ Baseline (mean ± sd)"].replace(".", ","),
                     f(r["Mean difference"], 1), f"{r['Proposed better (runs)']}/{r['Tied (runs)']}/{r['Proposed worse (runs)']}",
                     f(r["W statistic"], 0), "–" if pd.isna(r["p-value"]) else f"{r['p-value']:.1e}",
                     "Signifikan" if str(r["Conclusion"]).startswith("Significant") else "Identik"])
    b = bw[bw["Scenario"] == s].iloc[0]
    rows.append([s, "B7", f"{f(b['LEXZON rejected mean'],1)} ± {f(b['LEXZON rejected sd'],1)}",
                 f"{f(b['Baseline rejected mean'],1)} ± {f(b['Baseline rejected sd'],1)}",
                 f(b["Baseline rejected mean"] - b["LEXZON rejected mean"], 1), f"{b['better']}/{b['tied']}/{b['worse']}",
                 f(b["W"], 0), f"{b['p']:.1e}", "Signifikan" if b["p"] < 0.05 else "Tidak"])
w(table(["Skenario", "Pembanding", "TT LEXZON (rata² ± sd)", "TT pembanding (rata² ± sd)", "Selisih rata²",
         "Lebih baik/seri/lebih buruk", "W", "p-value", "Kesimpulan"], rows))

w("## C.7 Skalabilitas (data sintetis, median 3 pengulangan)\n")
sc = pd.read_csv(R / "scenarios" / "scalability_median.csv")
w(table(["Siswa", "Kursi", "Pasangan layak", "Diterima", "N*", "Tahap 1 (s)", "Tahap 2 (s)", "Tahap 3 (s)",
         "LEXZON total (s)", "Deferred acceptance (s)"],
        [[f(r["Students"]), f(r["Total seats"]), f(r["Pairs within radius"]), f(r["Admitted"]), f(r["Max-flow upper bound"]),
          f(r["Stage 1 (max admissions)"], 4), f(r["Stage 2 (min max distance)"], 4), f(r["Stage 3 (min total distance)"], 4),
          f(r["Proposed total runtime (s)"], 3), f(r["Deferred Acceptance runtime (s)"], 3)] for _, r in sc.iterrows()]))

w("## C.8 Keadilan prioritas jarak (R = 1 km, data penuh)\n")
fa = pd.read_csv(R / "fairness" / "fairness_full.csv")
lf = pd.read_csv(R / "pareto" / "lexzon_f_metrics.csv")
rows = []
for s in SC:
    for _, r in fa[fa["Scenario"] == s].iterrows():
        rows.append([s, NAME[r["Code"]], f(r["rejected"], 1), f(r["envy_students"], 1), f(r["envy_unassigned"], 1),
                     f(r["envy_assigned"], 1), f(r["envy_via_vacant_seat"], 1), f(r["envy_share_pct"], 1),
                     f(r["envy_extra_km_median"])])
    r = lf[lf["Scenario"] == s].iloc[0]
    rows.append([s, "LEXZON-F", f(r["rejected"]), f(r["envy_students"]), f(r["envy_unassigned"]), f(r["envy_assigned"]),
                 f(r["envy_via_vacant_seat"]), f(r["envy_share_pct"], 1), f(r["envy_extra_km_median"])])
w(table(["Skenario", "Metode", "TT", "Siswa dengan pelanggaran", "Ditolak padahal lebih dekat",
         "Diterima bukan di sekolah terdekat", "Via kursi kosong", "% siswa", "Median selisih jarak (km)"], rows))

fr = pd.read_csv(R / "fairness" / "fairness_runs.csv")
w("### Tabel C8-R. Keadilan pada 30 sampel ulang (rata-rata)\n")
g = fr.groupby(["Scenario", "Code"])[["rejected", "envy_students", "envy_unassigned", "envy_share_pct"]].mean().reset_index()
w(table(["Skenario", "Metode", "TT", "Siswa dengan pelanggaran", "Ditolak padahal lebih dekat", "% siswa"],
        [[r["Scenario"], NAME[r["Code"]], f(r["rejected"], 1), f(r["envy_students"], 1), f(r["envy_unassigned"], 1),
          f(r["envy_share_pct"], 1)] for _, r in g.iterrows()]))

w("## C.9 Kurva harga keadilan\n")
pp = pd.read_csv(R / "pareto" / "pareto_points.csv")
rows = []
label = {"LEXZON": "LEXZON", "DA": "Deferred acceptance", "min_envy_at_Nstar": "LEXZON-F (min. pelanggaran pada N*)",
         "budget": "Batas pelanggaran k"}
for _, r in pp.iterrows():
    rows.append([r["Scenario"], label[r["Point"]], f(r["budget"]) if r["Point"] == "budget" else "–", f(r["admitted"]),
                 f(r["envy"]), f(r["bound"], 0) if not pd.isna(r["bound"]) else "–", "Ya" if r["optimal"] else "Tidak",
                 f(r["runtime_s"], 1), f(r["N_star"])])
w(table(["Skenario", "Titik", "k", "Diterima", "Siswa dengan pelanggaran", "Batas solver*", "Terbukti optimal",
         "Waktu (s)", "N*"], rows))
w("*Untuk titik k: batas atas jumlah diterima; untuk LEXZON-F: batas bawah jumlah pelanggaran.\n")

w("### Tabel C9-R. Ringkasan harga keadilan\n")
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
w(table(["Skenario", "Diterima DA (k = 0)", "N*", "Tambahan penerimaan", "% tambahan tercapai pada k = 50",
         "Pelanggaran LEXZON", "Pelanggaran LEXZON-F", "Penurunan pelanggaran (%)", "Kenaikan total jarak LEXZON-F (%)"], rows))

w("## C.10 Simulasi kebijakan: penambahan satu rombongan belajar (28 kursi)\n")
un = pd.read_csv(R / "scenarios" / "unassigned_by_nearest_school.csv")
wi = pd.read_csv(R / "scenarios" / "whatif_extra_class.csv")
m = un.merge(wi, left_on=["Scenario", "School"], right_on=["Scenario", "School with extra class"])
m["Pengurangan"] = m["Unassigned before"] - m["Unassigned after"]
keep = m[(m["Unassigned nearest to school"] > 0) | (m["Pengurangan"] > 0)]
piv = keep.pivot_table(index="School", columns="Scenario", values=["Unassigned nearest to school", "Pengurangan"], fill_value=0)
rows = []
for school in piv.index:
    rows.append([school.replace("School ", "SD ")] +
                [f(piv.loc[school, ("Unassigned nearest to school", s)]) for s in SC] +
                [f(piv.loc[school, ("Pengurangan", s)]) for s in SC])
w(table(["Sekolah", "TT terdekat S1", "TT terdekat S2", "TT terdekat S3", "Pengurangan +1 rombel S1",
         "Pengurangan +1 rombel S2", "Pengurangan +1 rombel S3"], rows))
rows = []
for s in SC:
    g = wi[wi["Scenario"] == s]
    red = g["Unassigned before"] - g["Unassigned after"]
    rows.append([s, f(g["Unassigned before"].iloc[0]), f(g["Unassigned after"].min()), f(int((red == red.max()).sum())),
                 f(int((red == 0).sum()))])
w(table(["Skenario", "TT awal", "TT terbaik setelah +1 rombel", "Jumlah sekolah dengan pengurangan terbesar",
         "Jumlah sekolah tanpa pengurangan"], rows))

OUT.write_text("\n".join(md), encoding="utf-8")
print(OUT, len("\n".join(md)))
