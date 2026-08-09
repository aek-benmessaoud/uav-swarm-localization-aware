"""
analysis/paper_build.py — renders the ver0 paper (Projet08) to PDF.

Reads the raw campaign CSVs to compute every table number, embeds the
quantitative figures from analysis/fig_paper.py and the qualitative figures
from experiments/fig_qualitative.py, and assembles a complete manuscript
(abstract, related work, formulation, methods, preregistered protocol,
results, discussion, scope and boundaries, future work, references).

Usage:
  python analysis/fig_paper.py       # (already run) quantitative figures
  python analysis/paper_build.py     # -> results/paper_ver0.pdf
"""

import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats

import matplotlib
fontdir = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")

# ----------------------------------------------------------------------
# ReportLab setup (full-Unicode DejaVu from matplotlib's bundled fonts)
# ----------------------------------------------------------------------
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)

_F = lambda n: os.path.join(fontdir, n)
pdfmetrics.registerFont(TTFont("DVS", _F("DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DVSB", _F("DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DVSO", _F("DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DVSBO", _F("DejaVuSans-BoldOblique.ttf")))
pdfmetrics.registerFontFamily("DVS", normal="DVS", bold="DVSB",
                              italic="DVSO", boldItalic="DVSBO")

RESULTS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results")
FIG = os.path.join(RESULTS, "figures")


def _next_paper_path():
    """paper_ver0X.pdf with X auto-incremented (never overwrite an existing
    version). The current version is the highest existing paper_ver0*.pdf."""
    import re as _re
    vers = []
    if os.path.isdir(RESULTS):
        for name in os.listdir(RESULTS):
            m = _re.fullmatch(r"paper_ver0(\d+)\.pdf", name)
            if m:
                vers.append(int(m.group(1)))
    nxt = (max(vers) + 1) if vers else 1
    return os.path.join(RESULTS, f"paper_ver0{nxt}.pdf")


OUT_PDF = _next_paper_path()

REGIMES = ["A2_obs005", "A3_obs005", "A6_obs005", "A6_obs020"]
REGIME_LABEL = {"A2_obs005": "A2 (2 UAVs)", "A3_obs005": "A3 (3 UAVs)",
                "A6_obs005": "A6 (6 UAVs)", "A6_obs020": "A6, 20% obs"}
AGENTS = {"A2_obs005": 2, "A3_obs005": 3, "A6_obs005": 6, "A6_obs020": 6}
BUDGET = {"A2_obs005": 4200, "A3_obs005": 3200, "A6_obs005": 1600,
          "A6_obs020": 1750}

# ----------------------------------------------------------------------
# data helpers (mirror analysis/budget_stats.py)
# ----------------------------------------------------------------------
def sanitize(method):
    return re.sub(r"[^\w\-]", "_", method)


def load(dir_path, method, tag=None):
    fname = f"raw_comm_limited__{sanitize(method)}.csv"
    if tag:
        fname = f"raw_comm_limited__{sanitize(method)}__{tag}.csv"
    path = os.path.join(dir_path, fname)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: int(r.get("run", 0)))
    return rows


def vec(rows, key):
    out = []
    for r in rows or []:
        try:
            out.append(float(r.get(key)))
        except (TypeError, ValueError):
            out.append(np.nan)
    return np.array(out, dtype=float)


def med_iqr(x):
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    return np.median(x), np.percentile(x, 25), np.percentile(x, 75)


def wilcox(a, b):
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) != len(b) or len(a) == 0:
        return 1.0
    if np.all(np.abs(a - b) < 1e-12):
        return 1.0
    try:
        return stats.wilcoxon(a, b, zero_method="wilcox").pvalue
    except ValueError:
        return 1.0


def holm(ps):
    n = len(ps)
    order = np.argsort(ps)
    out = [None] * n
    for rank, idx in enumerate(order):
        out[idx] = min(1.0, ps[idx] * (n - rank))
    for i in range(n - 1, 0, -1):
        out[order[i - 1]] = min(out[order[i - 1]], out[order[i]])
    return out


def fmt_p(p):
    return "&lt;0.0001" if p < 0.0001 else f"{p:.4f}"


def budget_rows(rg):
    d = os.path.join(RESULTS, f"budget_{rg}")
    return load(d, "Frontier-Bounded"), load(d, "Coverage-U")


def e5_table():
    """E5 ladder rows: FB / Coverage-U / Central-Config / Central-CRLB."""
    rows = []
    for rg in ("A3_obs005", "A6_obs005"):
        d = os.path.join(RESULTS, f"budget_{rg}")
        fb = load(d, "Frontier-Bounded")
        cu = load(d, "Coverage-U")
        cc = load(d, "CentralOracle-Config")
        cr = load(d, "CentralOracle-CRLB")
        vfb = vec(fb, "mean_bound_final")
        r = [rg]
        for rows_ in (fb, cu, cc, cr):
            r.append(np.nanmedian(vec(rows_, "mean_bound_final")))
            r.append(np.nanmedian(vec(rows_, "final_coverage")))
        def _red(x):
            return (np.median(vfb) - np.nanmedian(vec(x, "mean_bound_final"))) \
                / np.median(vfb) * 100.0
        r += [round(_red(cu), 1), round(_red(cc), 1), round(_red(cr), 1)]
        rows.append(r)
    return rows, e5_diag_table()


def e5_diag_table():
    """E5-DIAG guard rows: cov (ε=0.05) and cov2 (ε=0.30) vs unguarded CRLB."""
    rows = []
    for rg in ("A3_obs005", "A6_obs005"):
        d = os.path.join(RESULTS, f"budget_{rg}")
        cr = vec(load(d, "CentralOracle-CRLB"), "mean_bound_final")
        cvc = vec(load(d, "CentralOracle-CRLB"), "final_coverage")
        for m, eps in (("CentralOracle-CRLB-cov", "ε=0.05"),
                       ("CentralOracle-CRLB-cov2", "ε=0.30")):
            rows_c = load(d, m)
            r = [f"{rg} · {eps}"]
            r.append(np.nanmedian(vec(rows_c, "mean_bound_final")))
            r.append(np.nanmedian(vec(rows_c, "final_coverage")))
            r.append(round(wilcox(vec(rows_c, "mean_bound_final"), cr), 4))
            r.append(round(wilcox(vec(rows_c, "final_coverage"), cvc), 4))
            rows.append(r)
    return rows


def e5_corrected_table():
    """E5-CORRECTED rows: local-frame oracles vs Coverage-U / FB.

    Columns per regime: FB, CU, CRLB (global frame), CRLB-local,
    Config-local median mean_bound_final and final_coverage, plus the paired
    p-values CRLB-local vs CU on both metrics.
    """
    rows = []
    for rg in ("A3_obs005", "A6_obs005"):
        d = os.path.join(RESULTS, f"budget_{rg}")
        fb = vec(load(d, "Frontier-Bounded"), "mean_bound_final")
        cu = vec(load(d, "Coverage-U"), "mean_bound_final")
        cr = vec(load(d, "CentralOracle-CRLB"), "mean_bound_final")
        crl = vec(load(d, "CentralOracle-CRLB-local"), "mean_bound_final")
        cfl = vec(load(d, "CentralOracle-Config-local"), "mean_bound_final")
        cvc = vec(load(d, "Coverage-U"), "final_coverage")
        crlc = vec(load(d, "CentralOracle-CRLB-local"), "final_coverage")
        r = [f"{rg}",
             f"{np.nanmedian(fb):.4f}", f"{np.nanmedian(cu):.4f}",
             f"{np.nanmedian(cr):.4f}", f"{np.nanmedian(crl):.4f}",
             f"{np.nanmedian(cfl):.4f}",
             f"{np.nanmedian(cvc):.1f}", f"{np.nanmedian(crlc):.1f}"]
        p_mb = wilcox(crl, cu)
        p_cv = wilcox(crlc, cvc)
        r += [fmt_p(p_mb), fmt_p(p_cv)]
        rows.append(r)
    return rows


def e4_confirm_table():
    rows = []
    for rg in REGIMES:
        fb, cu = budget_rows(rg)
        va, vb = vec(cu, "mean_bound_final"), vec(fb, "mean_bound_final")
        p = wilcox(va, vb)
        rel = (np.median(vb) - np.median(va)) / np.median(vb) * 100.0
        rows.append([rg, np.median(vb), np.median(va), rel, p])
    holm_ps = holm([r[4] for r in rows])
    for r, hp in zip(rows, holm_ps):
        r.append(hp)
    med_rel = float(np.nanmedian([r[3] for r in rows]))
    chi2 = -2.0 * np.sum(np.log(np.maximum([r[4] for r in rows], 1e-300)))
    f_p = float(1.0 - stats.chi2.cdf(chi2, 2 * len(rows)))
    return rows, med_rel, f_p


def coverage_table():
    rows = []
    for rg in REGIMES:
        fb, cu = budget_rows(rg)
        va, vb = vec(cu, "final_coverage"), vec(fb, "final_coverage")
        p = wilcox(va, vb)
        rows.append([rg, np.median(vb), np.median(va), p])
    holm_ps = holm([r[3] for r in rows])
    for r, hp in zip(rows, holm_ps):
        r.append(hp)
    return rows


def undetermined_table():
    rows = []
    for rg in REGIMES:
        fb, cu = budget_rows(rg)
        va, vb = vec(cu, "undetermined_final"), vec(fb, "undetermined_final")
        p = wilcox(va, vb)
        g = 100.0 * (vb - va) / vb
        rows.append([rg, np.median(vb), np.median(va), np.nanmedian(g), p])
    holm_ps = holm([r[4] for r in rows])
    for r, hp in zip(rows, holm_ps):
        r.append(hp)
    return rows


def pareto_table():
    lams = [0.25, 0.5, 1.0, 2.0]
    out = {}
    for rg in ("A3_obs005", "A6_obs005"):
        d = os.path.join(RESULTS, f"budget_{rg}")
        p = os.path.join(RESULTS, f"pareto_{rg}")
        fb = vec(load(d, "Frontier-Bounded"), "mean_bound_final")
        row = []
        for lam in lams:
            rows = load(d, "Coverage-U") if lam == 0.5 else load(
                p, "Coverage-U", tag=f"lam{lam}")
            cu = vec(rows, "mean_bound_final")
            rel = (np.median(fb) - np.median(cu)) / np.median(fb) * 100.0
            pw = wilcox(cu, fb)
            cov = med_iqr(vec(rows, "final_coverage"))[0]
            row.append((rel, pw, cov))
        out[rg] = row
    return lams, out


def e3_table():
    rows = []
    for rg in REGIMES:
        d = os.path.join(RESULTS, f"deploy_{rg}")
        fb, du = load(d, "Frontier-Bounded"), load(d, "Deploy-U")
        va, vb = vec(du, "steps_dual"), vec(fb, "steps_dual")
        p = wilcox(va, vb)
        g = 100.0 * (vb - va) / vb
        rows.append([rg, np.median(vb), np.median(va), np.nanmedian(g), p])
    holm_ps = holm([r[4] for r in rows])
    for r, hp in zip(rows, holm_ps):
        r.append(hp)
    return rows


def e1_table():
    d = os.path.join(RESULTS, "phase1_S1_fov5")
    out = []
    for m in ("Random", "Frontier-Bounded", "Richness-Angular"):
        r = load(d, m)
        qa = med_iqr(vec(r, "quality_auc"))
        qf = med_iqr(vec(r, "quality_final"))
        ttq = med_iqr(vec(r, "time_to_quality"))
        und = med_iqr(vec(r, "undetermined_final"))
        out.append((m, qa[0], qa[1], qa[2], qf[0], ttq[0], und[0]))
    return out


def phase1a_gate():
    with open(os.path.join(RESULTS, "..", "gates", "phase1_GO.txt"),
              encoding="utf-8") as fh:
        txt = fh.read()
    import re as _re
    rho = float(_re.search(r"rho\(bound,error\)\|localiz = ([\-\d.]+)",
                           txt).group(1))
    rho_u = float(_re.search(r"rho\(U_local,error\)\s+= ([\-\d.]+)",
                             txt).group(1))
    return rho, rho_u


BASELINE_METHODS = ["Random", "Frontier-Bounded", "Richness-Angular",
                    "Entropy-Frac", "Frontier+Entropy", "Coverage-U"]


def baseline_budget_rows():
    """Baseline battery: Random / FB / RA / Entropy-Frac /
    Frontier+Entropy / Coverage-U on the three 40-run budget regimes
    (A3_obs005, A6_obs005, A6_obs020). Methods not run in a regime (Random,
    Frontier+Entropy missing at 20% obstacles) are skipped. Returns per
    regime: [method, mean_bound med, coverage med, quality_auc med,
    p_bound vs FB (Holm-corrected), p_cov vs FB (Holm-corrected),
    rel-bound% vs FB]."""
    rows = []
    for rg in ("A3_obs005", "A6_obs005", "A6_obs020"):
        d = os.path.join(RESULTS, f"budget_{rg}")
        fb = vec(load(d, "Frontier-Bounded"), "mean_bound_final")
        fbc = vec(load(d, "Frontier-Bounded"), "final_coverage")
        for m in BASELINE_METHODS:
            r = load(d, m)
            if not r:
                continue
            mb = np.nanmedian(vec(r, "mean_bound_final"))
            cov = np.nanmedian(vec(r, "final_coverage"))
            qa = np.nanmedian(vec(r, "quality_auc"))
            rel = (np.median(fb) - mb) / np.median(fb) * 100.0
            rows.append([rg, m, mb, cov, qa, wilcox(vec(r, "mean_bound_final"), fb),
                         wilcox(vec(r, "final_coverage"), fbc), rel])
    for met_idx in (5, 6):
        for rg in ("A3_obs005", "A6_obs005", "A6_obs020"):
            idx = [i for i, row in enumerate(rows) if row[0] == rg]
            ps = [rows[i][met_idx] for i in idx]
            corr = holm(ps)
            for i, c in zip(idx, corr):
                rows[i][met_idx] = c
    return rows


def cpu_table():
    """ms_per_decision from the CPU benchmark."""
    path = os.path.join(RESULTS, "benchmark_cpu.csv")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        r = list(csv.DictReader(fh))
    out = {}
    for row in r:
        out[row["method"]] = float(row["ms_per_decision_median"])
    return out


def _sweep_cu_fb(rg, variant):
    """Returns (fb, cu) row-lists for a sweep variant (None = confirmed base)."""
    d = os.path.join(RESULTS, f"budget_{rg}")
    base = os.path.join(RESULTS, f"budget_{rg}__{variant}") if variant else d
    return load(base, "Coverage-U", tag=variant), \
        load(base, "Frontier-Bounded", tag=variant)


SWEEP_LABEL = {"s05": "0.5", "s10": "1.0",
               "b03": "0.3", "b05": "0.5", "b09": "0.9",
               "r25": "2.5", "r125": "1.25",
               "be2": "2.0"}


def sweep_table(variants, metric="mean_bound_final"):
    """Rows: [regime, variant, med_FB, med_CU, rel-red%, p, p_Holm]."""
    rows = []
    for rg in ("A3_obs005", "A6_obs005"):
        for variant in variants:
            cu, fb = _sweep_cu_fb(rg, variant)
            va, vb = vec(cu, metric), vec(fb, metric)
            p = wilcox(va, vb)
            rel = (np.median(vb) - np.median(va)) / np.median(vb) * 100.0
            rows.append([rg, variant, np.median(vb), np.median(va), rel, p])
    ps = [r[5] for r in rows]
    corr = holm(ps)
    for r, pc in zip(rows, corr):
        r.append(pc)
    return rows


def sweep_guard_table(variants):
    """Coverage guard rows: [regime, variant, med_FB, med_CU, p, p_Holm]."""
    rows = []
    for rg in ("A3_obs005", "A6_obs005"):
        for variant in variants:
            cu, fb = _sweep_cu_fb(rg, variant)
            va, vb = vec(cu, "final_coverage"), vec(fb, "final_coverage")
            p = wilcox(va, vb)
            rows.append([rg, variant, np.median(vb), np.median(va), p])
    ps = [r[4] for r in rows]
    corr = holm(ps)
    for r, pc in zip(rows, corr):
        r.append(pc)
    return rows


def sweep_guard_combined():
    """Assemble the per-sweep guard tables (Holm family = one sweep's cells)
    into a single listing: [regime, label, med_FB cov, med_CU cov,
    delta pp, p, p_Holm]."""
    labels = {"s05": "σ=0.5", "s10": "σ=1.0",
              "b03": "T=0.3", "b05": "T=0.5", "b09": "T=0.9",
              "r25": "R=2.5", "r125": "R=1.25",
              "be2": "σθ=2°"}
    out = []
    for tbl, prefix in ((sweep_sigma_guard, "B"),
                        (sweep_budget_guard, "C"),
                        (sweep_comm_guard, "D"),
                        (sweep_bearing_guard, "E")):
        for rg, v, mf, mc, p, pc in tbl:
            out.append([rg, f"{prefix}:{labels[v]}", mf, mc, mc - mf, p, pc])
    return out
def st(name, **kw):
    base = dict(fontName="DVS", fontSize=10, leading=14.5, textColor=colors.HexColor("#1a1a1a"))
    base.update(kw)
    return ParagraphStyle(name, **base)


S = {
    "title": st("t", fontName="DVSB", fontSize=21, leading=26,
                alignment=TA_CENTER, textColor=colors.HexColor("#0b3d63")),
    "alt": st("alt", fontSize=10.5, leading=14, alignment=TA_CENTER,
              textColor=colors.HexColor("#444444")),
    "meta": st("meta", fontSize=10, leading=14, alignment=TA_CENTER,
               textColor=colors.HexColor("#333333")),
    "h1": st("h1", fontName="DVSB", fontSize=14, leading=18,
             spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#0b3d63")),
    "h2": st("h2", fontName="DVSB", fontSize=11.5, leading=15,
             spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#17456b")),
    "h3": st("h3", fontName="DVSB", fontSize=10, leading=14,
             spaceBefore=7, spaceAfter=3),
    "body": st("b", alignment=TA_JUSTIFY),
    "cap": st("cap", fontSize=8.6, leading=11.4, textColor=colors.HexColor("#555555"),
              spaceBefore=2, spaceAfter=10),
    "tbl": st("tbl", fontSize=8.2, leading=10.2),
    "ref": st("ref", fontSize=8.6, leading=11.6, leftIndent=18,
              firstLineIndent=-18, spaceAfter=2),
    "bullet": st("bul", alignment=TA_JUSTIFY, leftIndent=14, bulletIndent=4,
                 spaceAfter=2),
}


def P(txt, style="body"):
    return Paragraph(txt, S[style])


def BUL(txt):
    return Paragraph(txt, S["bullet"], bulletText="•")


def img(path, width_cm):
    from PIL import Image as PILImage
    w_px, h_px = PILImage.open(path).size
    return Image(path, width=width_cm * cm,
                 height=width_cm * cm * h_px / w_px)


def make_table(header, data, widths, aligns=None, font=8.2):
    rows = [[Paragraph(f"<b>{c}</b>", ParagraphStyle(
        "th", fontName="DVSB", fontSize=font, leading=font + 2,
        alignment=TA_CENTER, textColor=colors.white)) for c in header]]
    for row in data:
        rows.append([Paragraph(str(c), ParagraphStyle(
            "td", fontName="DVS", fontSize=font, leading=font + 2,
            alignment=TA_CENTER)) for c in row])
    t = Table(rows, colWidths=[w * cm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17456b")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#eef3f8")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b9c8d6")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    return t


def make_table_grouped(header, data, widths, aligns=None, font=8.2):
    """Like make_table, but each data row starts with a group label (e.g.
    'A3 (3 UAVs)'). The label is rendered once as a bold spanning sub-header
    and the column is dropped, giving one table split into regime blocks."""
    ncols = len(header)
    rows = [[Paragraph(f"<b>{c}</b>", ParagraphStyle(
        "th", fontName="DVSB", fontSize=font, leading=font + 2,
        alignment=TA_CENTER, textColor=colors.white)) for c in header]]
    r = 1
    spans = []
    seen = []
    for rec in data:
        label, vals = rec[0], rec[1:]
        if label not in seen:
            seen.append(label)
            rows.append([Paragraph(f"<b>{label}</b>", ParagraphStyle(
                "gh", fontName="DVSB", fontSize=font, leading=font + 2,
                alignment=TA_CENTER, textColor=colors.HexColor("#0f3b5e")))])
            rows[-1] += [Paragraph("", ParagraphStyle(
                "gp", fontName="DVS", fontSize=font, leading=font + 2))
                for _ in range(ncols - 1)]
            spans.append(("SPAN", (0, r), (-1, r)))
            r += 1
        rows.append([Paragraph(str(c), ParagraphStyle(
            "td", fontName="DVS", fontSize=font, leading=font + 2,
            alignment=TA_CENTER)) for c in vals])
        r += 1
    t = Table(rows, colWidths=[w * cm for w in widths], repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17456b")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#eef3f8")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b9c8d6")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]
    for cmd, a, b in spans:
        style.append(("SPAN", a, b))
        style.append(("BACKGROUND", a, b, colors.HexColor("#dce8f2")))
        style.append(("LINEBELOW", a, b, 0.4, colors.HexColor("#b9c8d6")))
    t.setStyle(TableStyle(style))
    return t


_fig_n = [0]


def fig_cap(text):
    _fig_n[0] += 1
    story.append(Paragraph(f"<b>Figure {_fig_n[0]}.</b> {text}", S["cap"]))


_tbl_n = [0]


def tbl_cap(text):
    _tbl_n[0] += 1
    story.append(Paragraph(f"<b>Table {_tbl_n[0]}.</b> {text}", S["cap"]))


def img_keep(path, width_cm, cap_text):
    """Append an image and its caption together so they never split across
    a page boundary."""
    from reportlab.platypus import KeepTogether
    _fig_n[0] += 1
    story.append(KeepTogether([
        img(path, width_cm),
        Paragraph(f"<b>Figure {_fig_n[0]}.</b> {cap_text}", S["cap"]),
    ]))


def tbl_keep(table, cap_text):
    """Append a table and its caption together so they never split across
    a page boundary."""
    from reportlab.platypus import KeepTogether
    _tbl_n[0] += 1
    story.append(KeepTogether([
        table,
        Paragraph(f"<b>Table {_tbl_n[0]}.</b> {cap_text}", S["cap"]),
    ]))


# ----------------------------------------------------------------------
# document
# ----------------------------------------------------------------------
story = []


def sec(txt):
    story.append(P(txt, "h1"))


def sub(txt):
    story.append(P(txt, "h2"))


def body(txt):
    story.append(P(txt, "body"))


def spacer(h=0.3):
    story.append(Spacer(1, h * cm))


def page_break():
    story.append(Spacer(1, 0.1 * cm))
    from reportlab.platypus import PageBreak
    story.append(PageBreak())


def hr():
    from reportlab.platypus import HRFlowable
    story.append(HRFlowable(width="100%", thickness=0.6,
                            color=colors.HexColor("#17456b"),
                            spaceBefore=6, spaceAfter=6))


AUTHORS = "Abdelkader Benmessaoud, Nabil Abdelkader Nouri, Belkacem Mostefai"
AFFILIATION = "LASER & CSAIL, Ziane Achour University of Djelfa, Algeria"


def build():
    doc = BaseDocTemplate(OUT_PDF, pagesize=A4,
                          leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                          topMargin=2.0 * cm, bottomMargin=1.8 * cm,
                          title="U-Prioritized Coverage under Finite Mission Budgets",
                          author=AUTHORS)

    def header_footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("DVS", 8)
        canvas.setFillColor(colors.HexColor("#777777"))
        canvas.drawCentredString(A4[0] / 2, 1.0 * cm,
                                 f"— {doc_.page} —")
        if doc_.page > 1:
            canvas.setFont("DVS", 7.5)
            canvas.setFillColor(colors.HexColor("#555555"))
            canvas.drawString(doc_.leftMargin, A4[1] - 1.5 * cm, AUTHORS)
            canvas.drawRightString(A4[0] - doc_.rightMargin,
                                   A4[1] - 1.5 * cm, AFFILIATION)
        canvas.restoreState()

    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame],
                                       onPage=header_footer)])
    doc.build(story)


# ======================================================================
# TITLE PAGE
# ======================================================================
story.append(Spacer(1, 1.6 * cm))
story.append(P("U-Prioritized Coverage under Finite Mission Budgets: "
               "Config-Count Richness Reduces Residual Bearing-Only "
               "Localization Error in UAV Swarms at Equal Coverage", "title"))
spacer(0.8)
story.append(P("Working title (ver0). Alternative titles proposed:", "alt"))
story.append(P("<i>Alt. A —</i> “Config-Count Richness as a Decision Signal for "
               "UAV Swarm Localization: Three Preregistered Falsifications and "
               "One Confirmed Effect under a Finite Mission Budget.”", "alt"))
story.append(P("<i>Alt. B —</i> “Spending Coverage Where Accuracy Is at Risk: "
               "Continuous Prioritization of Angularly Under-Determined Cells "
               "under Fixed Mission Time.”", "alt"))
spacer(1.0)
story.append(P(AUTHORS, "meta"))
story.append(P(AFFILIATION, "meta"))
spacer(0.3)
hr()
story.append(P("Abstract", "h1"))

body("We study multi-agent bearing-only localization under limited-range "
     "communication and a <b>finite mission budget</b> — the setting where "
     "coverage and localization accuracy genuinely compete. Building on the "
     "finding that bounded-horizon geometric movement (Frontier-Bounded, FB) "
     "captures most of the coverage gain attributed to sophisticated "
     "uncertainty signals, we ask whether the <i>statistical richness</i> of "
     "independent angular observations — Chao-type estimators transposed to "
     "per-cell angular configuration counts — can still <i>drive</i> accuracy "
     "when the decision axis is <i>which frontier target to spend remaining "
     "coverage on</i>, rather than how to move or when to switch modes.")
body("Following a strict preregistered protocol (paired seeds, Wilcoxon + "
     "Holm-Bonferroni, coverage guard, hard validation gate), we report three "
     "documented falsifications and one confirmed effect. Richness as a direct "
     "target-selection signal (Phase-1/E1) and as a mode-switching signal "
     "(Deploy-U, E2/E3) do not beat the FB control, and the E4 primary "
     "quality_auc — the binary fraction of cells localized above threshold — "
     "is null as well: that fraction saturates under finite budgets. The "
     "confirmed effect is on a distinct pre-specified secondary metric, "
     "mean_bound_final (the continuous residual CRLB bound at mission end, "
     "tracked since E2/E3): a coherent n = 10 signal, reported explicitly as "
     "a discovery rather than a verdict, motivated a separate higher-power "
     "preregistration (E4-CONFIRM, protocol locked before its runs) that "
     "promoted it to primary with pre-specified success criteria. There, "
     "<b>continuous U-prioritized coverage</b> (Coverage-U: an FB target-score "
     "weighted by the local count of angularly under-determined cells, "
     "λ = 0.5) reduces the residual oracle CRLB bound at mission end by a "
     "median of "
     "<b>20.9%</b> relative to FB across four regimes (Holm-significant in "
      "three of four, Fisher combined p ≈ 0), <b>with no coverage regression</b> "
      "and a corroborating reduction in never-observed cells. Anchored against "
      "the Random floor and the classical occupancy-entropy baselines "
      "Entropy-Frac and Frontier+Entropy (n = 40 paired), the "
      "config-count family is the only signal family that beats the FB "
      "movement frame on residual bound at equal-or-better coverage: "
      "Coverage-U reduces it by a median 20.9% and Richness-Angular by "
      "23–32% across regimes, and Richness-Angular is the only method whose "
      "advantage survives 20% obstacle density. "
      "The effect is "
      "robust across λ ∈ [0.25, 2.0] (a plateau, not a knife-edge), "
      "conditional on sparse-obstacle environments, and specific to the "
      "budget-limited "
      "regime: in unbounded episodes final localization quality is at parity. "
     "We interpret the results as evidence that config-count richness is a "
      "reliable state witness that becomes an operating lever precisely when "
      "mission time — not distance — is the scarce resource. A centralized "
       "perfect oracle (E5, n = 40 × 2) regresses both metrics, and the "
       "local-vs-global ladder isolates why: the same config-count signal "
       "reduces the residual bound ~+30% when scored locally and regresses "
       "when the same under-set is fused globally. Re-running both oracle "
       "signals in Coverage-U’s own local movement frame (E5-CORRECTED, "
        "n = 40) confirms the effect is the <i>signal</i>, not the oracle’s "
        "global movement frame — evidence that <b>locality, not signal "
        "strength</b>, is what makes the accuracy/coverage trade-off "
        "attainable.")
body("<b>Index Terms</b> — multi-robot exploration, bearing-only "
     "localization, CRLB, Chao estimators, decentralized coverage, finite "
     "mission budget.")
page_break()

# ======================================================================
# 1. INTRODUCTION
# ======================================================================
sec("1. Introduction")
body("Multi-robot exploration has long balanced two objectives: covering an "
      "environment and acquiring information that is useful downstream "
      "[Yamauchi1997, Burgard2005, Lauri2023]. In a growing class of missions — "
      "localization of RF/audio emitters, passive sensing, structure-from-"
      "motion, cooperative positioning — the downstream task is itself "
      "<i>localization</i>: each cell must be observed from enough independent "
      "viewing directions that a bearing-only estimator becomes well "
      "conditioned. In that setting the geometric spread of observations "
      "(angular diversity) is the quantity that matters, classically measured "
      "by GDOP (Geometric Dilution of Precision) or the Fisher information "
      "matrix [Yarlagadda2000, "
      "Bishop2004].")
body("A separate line of work has proposed <i>information-driven</i> "
     "exploration signals — map entropy [Bourgault2002, Stachniss2005], "
     "expected information gain [Julian2014, Charrow2015], POMDP-style "
     "value [Bai2014] — as a replacement or complement to purely geometric "
      "frontier control [Yamauchi1997]. In our internal validation campaign we "
      "found that "
     "when these signals are evaluated against a carefully matched "
     "<i>bounded-horizon geometric control</i>, most of their apparent gain "
     "on pure coverage is explained by the receding-horizon movement frame "
     "itself, not by the signal. This paper continues that thread: we keep "
     "the honest control, and ask the harder question of whether a "
     "localization-relevant uncertainty signal can add value <i>within</i> "
     "that control, under a finite mission budget.")
body("The signal we study is unusual: it is borrowed from ecology. The "
     "Chao-type estimators [Chao1984, ChaoLee1992, ChaoYang1993] and related "
     "richness measures estimate the number of unseen species from the counts "
     "of singletons and doubletons. We transpose these estimators to a "
     "per-cell <i>angular configuration count</i>: two observations of a cell "
     "contribute independent configurations when their bearing directions "
     "differ by more than a tolerance (greedy clustering), and a cell is "
     "under-determined when it has ≤ 1 configuration. The transposed "
     "singleton/doubleton signal U then behaves as a local, cheap proxy for "
     "“how much localization work remains here.”")
body("The contribution of this paper is a <b>preregistered, negative-first "
     "evaluation</b> of whether this signal can be an operating lever, and "
     "the isolation of the one setting where it is:")
story.append(BUL("<b>C1.</b> A hard validation gate (Phase 1a): the oracle "
                 "CRLB bound correlates with empirical estimator error "
                 "(ρ = 0.638 pooled, gate GO), and the local richness signal "
                 "correlates with residual work (E1, ρ = 0.73–0.82)."))
story.append(BUL("<b>C2.</b> Three documented falsifications: richness as "
                 "target-selection (E1) and as mode-switching (E2/E3) does not "
                 "beat the FB control on the preregistered primaries."))
story.append(BUL("<b>C3.</b> A confirmed positive effect: continuous "
                 "U-prioritized coverage (Coverage-U, λ = 0.5) reduces "
                 "residual oracle CRLB bound at mission end by a median "
                 "20.9% at equal coverage (E4-CONFIRM, n = 40), robust to λ "
                 "(E4-PARETO), specific to budget-limited missions in "
                 "sparse-obstacle environments."))
story.append(BUL("<b>C4.</b> A clean experimental protocol and tooling: "
                  "paired seeds, Holm-corrected Wilcoxon tests, coverage "
                  "guards, resumable campaigns, and an evaluation decoupled "
                  "from the decision signal (global oracle CRLB), anchored "
                  "against the classical occupancy-entropy baselines "
                  "Entropy-Frac and Frontier+Entropy (Section 6.6). The full "
                  "source code and experiment scripts are publicly available "
                  "at [repository URL] to ensure reproducibility."))
body("The decisive isolation is that <b>locality, not signal strength</b>, is "
     "what makes the accuracy/coverage trade-off attainable: the same "
     "config-count under-set reduces the residual bound by ~+30% when scored "
     "locally and regresses both metrics when the same under-set is fused "
     "globally by a centralized oracle (E5, Section 6.9).")

# ======================================================================
# 2. RELATED WORK
# ======================================================================
sec("2. Related Work")
body("This section reviews the main research areas relevant to our work: "
     "frontier-based exploration, information-driven exploration, "
     "localization-aware planning, cooperative localization, and statistical "
     "richness estimation. We position our contribution relative to each.")
sub("2.1 Multi-robot exploration and frontier methods")
body("Frontier-based exploration [Yamauchi1997, Yamauchi1998] and its "
     "coordinated extensions [Burgard2005, Franchi2009] select targets on the "
     "boundary between explored and unexplored space. Multi-objective and "
     "multi-criteria variants weight frontier targets by distance, utility, "
     "or information [Gonzalez2002, BasilicoAmigoni2011]. Receding-horizon "
     "next-best-view planning [Bircher2016, Bircher2018] formulates target "
     "selection as short-horizon optimization and is the modern state of the "
     "art for geometric exploration; our FB control instantiates exactly this "
     "principle and serves as the movement baseline throughout. More recent "
     "decentralized multi-UAV systems such as RACER [Zhou2023] dispatch large "
     "teams under asynchronous, bandwidth-limited communication, but their "
     "objective remains rapid spatial coverage and workload balance rather "
     "than the geometry of the acquired observations — the decision axis of "
     "the present work.")
sub("2.2 Information-driven exploration")
body("Information-theoretic exploration maximizes expected information gain "
     "or entropy reduction of the map [Bourgault2002, Stachniss2005, "
     "Julian2014, Charrow2015]; in cooperative settings, decentralized "
     "approximations trade optimality for scalability [Grocholsky2002, "
     "Ponda2012]. Communication-constrained entropy-field exploration "
     "shares this idea in fully distributed settings [Pongsirijinda2025], "
     "ranking targets by frontier and robot entropy while "
     "merging maps only when robots are within range — close to our "
      "proximity-fusion model. These methods are computationally heavier than "
      "frontier selection [Grocholsky2002, Ponda2012, Lauri2023] and quantify "
      "probabilistic map uncertainty; our work "
      "differs in that the operating signal is neither occupancy entropy nor "
      "information gain but the <i>angular configuration count</i> of a "
      "bearing-only observation model.")
sub("2.3 Localization-aware planning: GDOP, CRLB, FIM")
body("Sensor-placement and path-planning for localization are classically "
     "cast as optimizing a scalar function of the Fisher information matrix — "
     "D-optimality (determinant), A-optimality (trace of the inverse), or "
     "GDOP [Kaplan2017, Ucinski2005, Martinez2006, Krause2008]. For "
     "bearing-only problems the observability conditions [NardoneAidala1981] "
     "and optimal-observer-maneuver results [Passerieux1998, "
     "Oshman1999, Dogancay2012] show that accuracy is governed by "
     "baseline geometry, which is precisely why our CRLB-based evaluation "
      "[Cramér1946, Rao1945] and our angular-diversity signal are principled. "
     "At fleet scale, cooperative dilution-of-precision analysis of UAV "
     "swarms [Chen2020] shows the same geometric principle: the relative "
     "configuration of the swarm directly governs cooperative positioning "
     "accuracy, and increasing the number of agents does not automatically "
     "improve localization.")
sub("2.4 Cooperative localization and communication")
body("Cooperative localization in wireless networks [Patwari2005, "
     "Wymeersch2009] and multi-robot localization [Ristic2004] emphasize the "
     "role of inter-agent measurement fusion. Distributed estimators now "
     "operate on the sensing graph directly: DCL-Sparse [Sagale2024] improves "
     "range-only cooperative localization in noisy, sparse graphs; "
     "GNSS-denied UAV swarms rely on coalition-based relative localization "
     "[Ruan2022] and formation-constrained geometry [Li2026] to bound "
     "accuracy; and a high-precision airborne study reports significant "
     "vertical error when the swarm's relative baselines lack diversity "
     "[Liu2026]. Communication disruptions motivate predictive bidding, "
     "where robots estimate the missing task-allocation information of "
     "disconnected teammates [Woosley2021], and low-overhead decentralized "
     "strategies that exchange only positions and current target points "
     "[Batinovic2020]. We adopt a limited-range proximity-fusion model "
     "(agents exchange maps when within range), which keeps the decision "
     "signal local and makes the centralized oracle (E5) a genuinely "
     "informative upper bound.")
sub("2.5 Statistical richness estimators")
body("Chao1 and related estimators [Chao1984, ChaoLee1992, ChaoYang1993, "
     "BurnhamOverton1978] estimate species richness from abundance data and "
     "are standard in ecology. Their use as <i>decision signals for robot "
     "exploration</i> — rather than as post-hoc analytics — is, to the best "
     "of our knowledge, novel; this paper is the first to evaluate them "
     "transposed to angular configurations and under a preregistered "
     "falsification protocol.")
sub("2.6 Positioning")
body("Relative to this literature, the paper makes three moves. First, it "
     "evaluates information signals against a <i>matched receding-horizon "
     "control</i>, not against a weak or absent baseline (the dominant "
     "failure mode of information-driven exploration claims). Second, it "
     "uses a global oracle CRLB bound decoupled from the local decision "
     "signal, so accuracy claims are not circular. Third, it is organized "
     "around preregistered falsifications: we report honestly which "
     "operating levers failed before claiming the one that succeeded.")

# ======================================================================
# 3. PROBLEM FORMULATION
# ======================================================================
sec("3. Problem Formulation")
sub("3.1 System and observation model")
body("We consider <i>m</i> agents moving on a 100 × 100 grid with randomly "
     "placed obstacles (ratio <i>q</i> ∈ {0.05, 0.20}); agents do not know "
     "the map. At each time step, agent <i>i</i> at pose <i>p<sub>i</sub></i> "
     "obtains, for every traversable cell in a Chebyshev sensing footprint of "
     "radius <i>F</i> = 5, a <i>bearing-only</i> observation toward the cell "
     "center: a direction θ<sub>k</sub> from the true geometry. This is the "
     "minimal model under which angular diversity is the currency of "
     "localization, and it lets us compute an exact oracle CRLB (Section 3.3).")
sub("3.2 Independent angular configurations and the richness signal")
body("For each cell, observations are summarized by a greedy clustering of "
     "their bearing directions: a new direction joins an existing cluster if "
     "it lies within ANG_TOL = 15° (circular) of the nearest center, "
     "otherwise it starts a new cluster, capped at CLUSTER_CAP = 8. Each "
     "cluster center is an <i>independent angular configuration</i>. A cell "
     "with one configuration is geometrically under-determined (a single "
     "bearing gives a rank-deficient Fisher matrix); a cell with two or more "
     "well-separated configurations is localizable. We transpose the Chao "
     "richness vocabulary to these counts: F1/F2 are the numbers of cells "
     "with exactly one/two configurations, and the decision signal is")
story.append(P("U = min( F1·(F1−1) / (2·(F2+1)), cap ), &nbsp; α = U / (U + K),", "body"))
body("with the same bias-corrected form and normalization cap used in the "
     "coverage setting [Chao1984]. All decision signals used by policies are "
     "computed from <i>local</i> counts (own observations, augmented only by "
     "proximity fusion); the global count and the CRLB oracle are never fed "
     "to any policy (no leakage, enforced by interface and test).")
sub("3.3 Oracle CRLB evaluation metric")
body("Localization quality is scored by a global oracle: for every traversable "
     "cell, the information matrix from the true observation geometry is "
     "J = Σ<sub>k</sub> u<sub>k</sub>u<sub>k</sub><sup>⊤</sup>/(σ²d<sub>k</sub>²) "
     "(u<sub>k</sub> the unit bearing vector to the k-th observing pose, "
     "d<sub>k</sub> the distance, σ the nominal bearing precision of 1°). The "
     "per-cell bound is b = sqrt(trace(J⁻¹)) in grid-cell units. A cell is "
     "<i>well-localized</i> when b ≤ QUALITY_THRESHOLD = 1.5 cells. The "
     "primary continuous accuracy metric is the <i>mean residual bound</i> "
     "across traversable cells at mission end, mean_bound_final (lower is "
     "better); the binary fraction of well-localized cells is "
     "quality(t) = fraction with b ≤ threshold, sampled every 25 steps "
     "(normalized AUC = quality_auc).")
sub("3.4 Communication model")
body("Agents use limited-range communication: two agents within COMM_RANGE = "
      "F exchange their maps each step (rendezvous-triggered fusion "
      "[Pongsirijinda2025, Batinovic2020]). The "
      "decision signal of every policy is therefore strictly local and "
      "temporally stale relative to the true map — the honest distributed "
      "setting. Evaluation uses the global accumulator, which is never "
      "revealed to policies.")
sub("3.5 Metrics")
body("We report the preregistered metrics inline:")
story.append(BUL("<i>Coverage</i>: final_coverage (%) of traversable cells "
                 "visited, coverage_auc over the episode."))
story.append(BUL("<i>Localization</i>: quality_auc, time_to_quality, "
                 "mean_bound_final, undetermined_final (fraction of "
                 "traversable cells never observed)."))
story.append(BUL("<i>Dual objective (E2/E3)</i>: steps_dual = first step with "
                 "coverage ≥ 90% AND quality ≥ 0.9."))
body("All tests are paired by environment seed; gains are median relative "
     "differences; p-values are Holm-Bonferroni corrected across regimes.")

# ======================================================================
# 4. METHODS
# ======================================================================
sec("4. Methods")
sub("4.1 Frontier-Bounded control (FB)")
body("FB selects a target among frontier cells reachable within a bounded "
      "BFS horizon (H = 8), preferring the frontier cell that maximizes the "
      "remaining-exploration potential, with deterministic tie-breaking by "
      "per-agent scatter noise; movement is receding-horizon with an "
      "exploration fallback. H = 8 was fixed empirically before the "
      "preregistered campaign, and the 90% coverage threshold is standard in "
      "exploration benchmarks [Yamauchi1997, Burgard2005]. FB uses <i>no</i> "
      "uncertainty signal: it is the "
      "validated geometric control and the reference against which every "
      "candidate is tested. Its steps_90 matches the geometric baseline "
      "validated in our internal validation campaign.")
sub("4.2 Richness-Angular (E1, target-selection falsification)")
body("RA scores frontier targets by the transposed richness signal U over "
     "the local config-count map (frontier × richness weighting) inside the "
     "same bounded frame. It tests whether richness as a <i>direct target "
     "selection</i> signal beats FB on localization quality.")
sub("4.3 Deploy-U (E2/E3, mode-switching falsification)")
body("Deploy-U keeps the FB coverage mode while the known local map is mostly "
     "under-localized (fraction of known cells with ≤ 1 configuration above "
     "0.30), then switches to a <i>deploy</i> mode that orbits the worst "
     "known under-determined cell (approach by bounded BFS, then orbit with "
     "viewing-angle variation to add independent configurations). It tests "
     "richness as a <i>mode</i> signal.")
sub("4.4 Coverage-U (E4, continuous prioritization — proposed method)")
body("Coverage-U does not change mode. Inside the FB frame it replaces the "
     "frontier utility by a continuous target score")
story.append(P("score(target) = D / H − λ · under_count_FOV(target) / FOV_area,", "body"))
body("where D is the bounded-BFS distance to the target, H the horizon, and "
     "under_count_FOV counts the <i>known-free cells with ≤ 1 angular "
     "configuration</i> inside the target’s sensing footprint, computed in "
     "O(1) per candidate by an integral image. λ = 0.5 was fixed before any "
     "campaign (no tuning); λ = 0 is exactly FB (verified by an "
     "action-identical test). The hypothesis: with a finite mission time, "
     "standard coverage spends the remaining budget on frontier cells that "
     "are cheap to reach but leave residual localization error; biasing "
      "target choice toward under-observed regions buys residual accuracy at "
      "the same coverage cost.")
sub("4.4.1 Dynamic normalization for dense environments (proposed variant)")
body("One boundary condition found in Section 6.6 motivates a proposed variant "
     "of the score: the Coverage-U advantage vanishes at 20% obstacle density, "
     "and part of the mechanism is signal dilution. The score above "
     "normalizes under_count_FOV by the constant square area FOV_area; in a "
     "fragmented environment a large fraction of that window is blocked, so "
     "the same absolute under-count yields a smaller bonus and the signal is "
     "attenuated precisely where it is needed. The variant replaces the "
      "constant denominator by the number of <i>traversable</i> cells (free + "
      "unknown) actually inside the footprint:")
story.append(P("score(t) = D / H − λ · under_count_FOV(t) / free_count_FOV(t),", "body"))
body("which keeps the bonus normalized by what the sensor can actually "
     "observe. It is a two-line change to the integral-image computation and "
     "is the first natural extension to test in dense terrain; it is not run "
     "here (it would be a new preregistered variant) and is returned to in "
     "Section 9.")
sub("4.5 Centralized oracle (E5, control bound)")
body("E5 defines an infeasible centralized control with perfect map knowledge "
      "that maximizes the same score form using global under-sets — either "
      "the config-count signal under perfect fusion (CentralOracle-Config) or "
      "the true CRLB bottleneck (CentralOracle-CRLB). It anchors the "
      "transposition ratio ρ = reduction(Coverage-U) / reduction(CentralCRLB). "
       "Section 6.9 reports the completed campaign, its diagnostic extension "
       "(coverage-guarded oracle variants, which cannot bind), and the "
       "frame-matched re-run (E5-CORRECTED) that isolates the signal effect.")
sub("4.6 Comparison baselines (occupancy entropy)")
body("We also compare against classical occupancy-entropy baselines: "
     "Entropy-Frac (fractional entropy gain over the footprint) and "
     "Frontier+Entropy (frontier selection weighted by map entropy) "
     "[Bourgault2002, Stachniss2005]. They run inside the same Frontier-Bounded "
     "frame and are introduced here so that the battery of Section 6.6 reads as "
     "a planned comparison rather than a post-hoc addition.")

# ======================================================================
# 5. EXPERIMENTAL PROTOCOL
# ======================================================================
sec("5. Experimental Protocol")
sub("5.1 Preregistration and gates")
body("Every experiment is preregistered in the repository (PRE_REG_*.md) "
     "before execution, with locked metrics, thresholds, and verdict rules; "
     "the analysis scripts implement the verdicts exactly and are run "
     "unchanged on the final data. A hard gate (Phase 1a) must pass before "
     "any campaign runs: the oracle CRLB bound must correlate with empirical "
     "estimator error, and the local richness signal must correlate with "
     "residual work (Section 6.1).")
sub("5.2 Regimes and budgets")
body("Regimes vary the team size and obstacle ratio: A2 (2 UAVs), A3 "
     "(3 UAVs), A6 (6 UAVs) at 5% obstacles, and A6 at 20% obstacles "
     "(A6_obs020). The finite budget T per regime is 0.7 × the FB median "
     "steps_90 measured in E3: A2 = 4200, A3 = 3200, A6 = 1600, "
     "A6_obs020 = 1750. At these budgets coverage is partial (66–76%), so "
     "coverage and accuracy genuinely compete.")
sub("5.3 Paired design and statistics")
body("All comparisons are paired at the map level: run index r uses "
     "env_seed = 0 + 1000·r for every method. Significance is assessed by "
     "the paired Wilcoxon signed-rank test; p-values are Holm-Bonferroni "
     "corrected across regimes; the matched-pairs delta m/n² is reported. "
     "Primary/guard/secondary metrics and verdict rules are fixed "
     "per-stage. n = 10 pairs for discovery (E1/E2/E3/E4), n = 40 for the "
     "confirmation campaign (E4-CONFIRM, E4-PARETO).")
rho_be, rho_u = phase1a_gate()
e1_rows = e1_table()
e3_rows = e3_table()
e4c_rows, e4c_med, e4c_fisher = e4_confirm_table()
cov_rows = coverage_table()
und_rows = undetermined_table()
lams, pareto = pareto_table()
baseline_rows = baseline_budget_rows()
cpu_ms = cpu_table()
sweep_sigma_rows = sweep_table(["s05", "s10"])
sweep_sigma_guard = sweep_guard_table(["s05", "s10"])
sweep_budget_rows = sweep_table(["b03", "b05", "b09"])
sweep_budget_guard = sweep_guard_table(["b03", "b05", "b09"])
sweep_comm_rows = sweep_table(["r25", "r125"])
sweep_comm_guard = sweep_guard_table(["r25", "r125"])
sweep_bearing_rows = sweep_table(["be2"])
sweep_bearing_guard = sweep_guard_table(["be2"])
sweep_guard_all = sweep_guard_combined()

# ----------------------------------------------------------------------
# Topology campaigns (post-submission, 2026-08-08/09) — maze + clusters.
# Same protocol family: paired seeds 0..39000, budget T = 0.7 x FB median
# steps_90 (measured per topology by probe_budget_maze.py), Wilcoxon +
# Holm across the 3 pairwise comparisons on quality_auc (primary) with the
# final_coverage guard.
# ----------------------------------------------------------------------
TOPO_METHODS = ["Frontier-Bounded", "Coverage-U", "Richness-Angular"]
TOPO_CAMPAIGNS = [
    # (dir, tag, label, budget_T)
    ("budget_A6_maze__maze", "maze", "maze perfect (A6)", 1323),
    ("budget_A6_maze__loops10", "loops10", "maze +10% loops (A6)", 1323),
    ("budget_A6_maze__h16", "h16", "maze perfect, H=16 (A6)", 1323),
    ("budget_A6_cluster020__cluster020", "cluster020",
     "clusters 20% (A6)", 1354),
    ("budget_A3_cluster020__cluster020_A3", "cluster020_A3",
     "clusters 20% (A3)", 2710),
]
TOPO_PAIRS = [
    ("Coverage-U", "Frontier-Bounded"),
    ("Richness-Angular", "Frontier-Bounded"),
    ("Richness-Angular", "Coverage-U"),
]


def _topo_rows(campaign, metric, lower_better=False, guard=False):
    """Per-campaign paired stats: rows [A, B, med_A, med_B, gain%, p] with
    Holm across the 3 pairs. gain% uses the paper's rel-red convention
    (denominator = baseline B): lower_better -> (vb-va)/vb, else (va-vb)/vb.
    """
    dir_name, tag, label, budget = campaign
    d = os.path.join(RESULTS, dir_name)
    data = {}
    for m in TOPO_METHODS:
        data[m] = load(d, m, tag)
    out = []
    for a, b in TOPO_PAIRS:
        va, vb = vec(data[a], metric), vec(data[b], metric)
        p = wilcox(va, vb)
        denom = np.where(vb == 0, np.nan, vb)
        num = (vb - va) if lower_better else (va - vb)
        g = np.nanmedian(num / denom) * 100.0
        out.append([a, b, np.nanmedian(va), np.nanmedian(vb), g, p])
    for r, pc in zip(out, holm([r[5] for r in out])):
        r.append(pc)
    return out


def _topo_guard(campaign):
    return _topo_rows(campaign, "final_coverage", guard=True)


topo_primary = {label: _topo_rows(c, "quality_auc")
                for c in TOPO_CAMPAIGNS
                for label in [c[2]]}
topo_guard = {label: _topo_rows(c, "final_coverage")
              for c in TOPO_CAMPAIGNS for label in [c[2]]}

TOPO_VERDICT = {
    "maze perfect (A6)":
        ("RA and CU both significantly LOWER quality_auc than FB (−3.0% / "
         "−2.1%, Holm-sig) with a significant coverage regression → clean "
         "negative."),
    "maze +10% loops (A6)":
        ("penalty neutralized (ns in all three comparisons), but no advantage "
         "restored → neutral."),
    "maze perfect, H=16 (A6)":
        ("bit-identical to H=8 for CU (−2.1%, Holm-sig); RA improves to ns → "
         "horizon is not the lever."),
    "clusters 20% (A6)":
        ("RA advantage INTACT (+3.0% vs FB, Holm-sig) with no coverage "
         "regression; directional RA≥CU confirmed → routing, not occlusion."),
    "clusters 20% (A3)":
        ("RA advantage REPRODUCED at 3 UAVs (+2.6% vs FB, Holm-sig); RA vs CU "
         "+6.3% (strongest delta) → swarm-size robust."),
}
TOPO_VERDICT_N = {  # stable short verdicts for the summary table
    "maze perfect (A6)": "NEGATIVE",
    "maze +10% loops (A6)": "NEUTRAL",
    "maze perfect, H=16 (A6)": "NEGATIVE (H inert)",
    "clusters 20% (A6)": "RA POSITIVE",
    "clusters 20% (A3)": "RA POSITIVE (A3)",
}

# --- Table 1: experiment summary ---
sub("5.4 Experiment map")
t1 = [
    ["Stage", "Question", "Primary metric", "n", "Verdict"],
    ["Phase 1a", "CRLB valid? U predicts residual work?", "ρ(bound, error)", "10", "GO (gate)"],
    ["E1", "Richness as target selection?", "quality_auc", "10", "FAIL (parity)"],
    ["E2/E3", "Richness as mode switch?", "steps_dual", "10", "FAIL (parity)"],
    ["E4", "U-prioritization under budget?", "quality_auc", "10", "FAIL (parity) → discovery on mean_bound"],
    ["E4-CONFIRM", "Confirmation, higher power?", "mean_bound_final", "40", "PASS"],
    ["E4-PARETO", "λ robustness?", "mean_bound_final", "40", "PASS (plateau)"],
    ["E5", "Centralized oracle bound?", "mean_bound_final", "40", "FAIL (regression)"],
    ["E5-DIAG", "Oracle failure: calibration or structure?", "mean_bound_final", "40", "NEGATIVE STRUCTURE (guard inert)"],
    ["E5-CORRECTED", "Local frame + global signal?", "mean_bound_final", "40", "FAIL (locality confirmed)"],
]
tbl_keep(make_table(["Stage", "Question", "Primary", "n", "Verdict"],
                    t1[1:], [2.4, 5.4, 2.6, 1.0, 2.6]),
         "Experiment map. All stages preregistered; E4-CONFIRM and E4-PARETO "
         "fixed λ = 0.5 and budgets before any data were seen.")
page_break()

# ======================================================================
# 6. RESULTS
# ======================================================================
sec("6. Results")
sub("6.1 Phase 1a: metric validation (gate)")
body(f"The oracle CRLB bound is a valid accuracy proxy: pooled across "
     f"95,000 tested cells (91,530 localizable), ρ(bound, empirical error) = "
     f"{rho_be:.3f} on localizable cells (p &lt; 0.05, min 0.5), and the "
     f"local richness signal correlates negatively with empirical error "
     f"(ρ(U_local, error) = {rho_u:.3f}, max −0.4, p &lt; 0.05). Both "
     "conditions pass and the gate is GO. This is the evidence that the "
     "evaluation metric and the decision signal are both meaningful "
     "indicators of localization work.")
sub("6.2 E1: richness as target selection — falsified")
body("In the base scenario (S1: 6 UAVs, FOV 5, unbounded 4500-step "
      "episodes), the three-phase-1 methods differ sharply on the movement "
      "frame and barely on the signal. Both FB (p = 0.002 vs Random) and RA "
      "(p = 0.002 vs Random) far exceed the Random floor, while RA is not "
      "significantly better than FB (p = 0.084, −1.5%): richness as a "
      "target-selection signal is falsified.")
t3 = [["Method", "quality_auc", "quality_final", "time_to_quality", "undetermined"]]
for m, qa, lo, hi, qf, ttq, und in e1_rows:
    t3.append([m, f"{qa:.3f} [{lo:.3f}–{hi:.3f}]", f"{qf:.3f}", f"{ttq:.0f}",
               f"{und:.4f}"])
story.append(make_table(["Method", "quality_auc (med [IQR])", "quality_final",
                         "time_to_quality", "undetermined"],
                        t3[1:], [2.9, 3.6, 2.2, 2.6, 2.2]))
tbl_cap("Phase-1 (E1), 6 UAVs, FOV 5, n = 10.")
story.append(img(os.path.join(FIG, "fig_paper_phase1.png"), 8.4))
fig_cap("Phase-1 quality AUC for Random, Frontier-Bounded (FB), and "
        "Richness-Angular (RA) on the unbounded 4500-step scenario "
        "(n = 10; † = p &lt; 0.05 vs Random).")
sub("6.3 E2/E3: richness as a mode switch — falsified")
body("Deploy-U orbits under-determined cells once the known map is mostly "
      "localized. Its deploy mode fires only 2–6% of decisions (orbit "
      "0.5–1%) and never changes the dual-objective outcome: the median gain "
      "on steps_dual is −1.1% (threshold +8%), with no Holm significance — "
      "richness as a mode-switching signal is falsified, and final "
      "localization quality is at parity everywhere (quality_final = 1.0).")
t4 = [["Regime", "FB steps_dual", "Deploy-U steps_dual", "gain%", "p", "p_Holm"]]
for rg, mf, md, g, p, hp in e3_rows:
    t4.append([REGIME_LABEL[rg], f"{mf:.0f}", f"{md:.0f}", f"{g:+.1f}",
               fmt_p(p), fmt_p(hp)])
story.append(make_table_grouped(["FB steps_dual", "Deploy-U steps_dual",
                                 "gain%", "p", "p_Holm"],
                                t4[1:], [2.3, 2.6, 1.6, 1.8, 1.8]))
tbl_cap("E2/E3: steps_dual (first step with coverage ≥ 90% and quality ≥ 0.9). "
        "Gain threshold +8%.")
story.append(img(os.path.join(FIG, "fig_paper_e3.png"), 12.6))
fig_cap("E2/E3: dual-objective completion time (log scale).")
sub("6.4 E4: preregistered primary fails; a consistent secondary discovery")
body("Under finite budgets, Coverage-U (λ = 0.5) does not move the "
     "preregistered primary quality_auc (median gain +0.4%, no Holm "
     "significance): the binary fraction of well-localized cells saturates "
     "and cannot see improvements below the threshold. However, on the "
     "continuous accuracy metric mean_bound_final — a secondary metric "
     "pre-specified since E2/E3 — the effect is coherent "
     "across all four regimes at n = 10 (relative reduction vs FB "
     "13.6–28.3%, three of four raw p &lt; 0.05, best Holm p = 0.098) with "
     "no coverage regression. This directional discovery — not a verdict — "
     "motivated the higher-power preregistered confirmation (E4-CONFIRM, "
     "protocol locked before its runs).")
sub("6.5 E4-CONFIRM: confirmed reduction at equal coverage (PASS)")
body("With n = 40 paired runs per regime, the residual oracle CRLB bound at "
      "mission end is significantly lower for Coverage-U in three of four "
      "regimes, with a median relative reduction of 20.9% and Fisher combined "
      "p ≈ 0. The corroborating reduction in never-observed cells is also "
      "Holm-significant in three of four regimes; A6_obs020 is the single "
      "non-significant, mildly reversed regime:")
t5 = [["Regime", "FB bound", "CU bound", "rel-red%", "p", "p_Holm"]]
for rg, mf, mc, rel, p, hp in e4c_rows:
    t5.append([REGIME_LABEL[rg], f"{mf:.4f}", f"{mc:.4f}", f"{rel:+.1f}",
               fmt_p(p), fmt_p(hp)])
story.append(make_table_grouped(["FB bound", "CU bound", "rel-red%", "p",
                                 "p_Holm"], t5[1:], [2.0, 2.0, 2.0, 2.2, 2.2]))
tbl_cap("E4-CONFIRM: mean_bound_final at mission end (budget T), n = 40 "
        "paired.")
story.append(img(os.path.join(FIG, "fig_paper_e4confirm.png"), 13.2))
fig_cap("E4-CONFIRM: residual CRLB bound at mission end (* = Holm-significant "
        "Wilcoxon, p &lt; 0.05).")
story.append(img(os.path.join(FIG, "fig_paper_coverage.png"), 13.2))
fig_cap("Coverage guard: final coverage at mission end for Frontier-Bounded "
        "(FB) and Coverage-U (CU) under finite budgets (n = 40).")
t6 = [["Regime", "FB undet.", "CU undet.", "gain%", "p", "p_Holm"]]
for rg, mf, mc, g, p, hp in und_rows:
    t6.append([REGIME_LABEL[rg], f"{mf:.4f}", f"{mc:.4f}", f"{g:+.1f}",
               fmt_p(p), fmt_p(hp)])
story.append(make_table_grouped(["FB undet.", "CU undet.", "gain%", "p",
                                 "p_Holm"], t6[1:], [2.0, 2.0, 2.0, 2.2, 2.2]))
tbl_cap("E4-CONFIRM corroboration: fraction of traversable cells never "
        "observed (lower is better).")
story.append(img(os.path.join(FIG, "fig_paper_undetermined.png"), 13.2))
fig_cap("Never-observed cell fraction at mission end (log scale). Error "
        "bars: Holm-significant differences vs FB (p &lt; 0.05).")
sub("6.6 Baseline battery: random floor and classical signals")
body("The finite-budget table is anchored against the Random floor and "
     "classical information signals, not only against the FB movement frame. "
     "We completed n = 40 paired budget runs for Random, "
     "Richness-Angular, Entropy-Frac, and Frontier+Entropy on the two "
     "5%-obstacle regimes (A3, A6), appended to the existing FB / Coverage-U "
      "data, and extended the same battery to the 20%-obstacle stress regime "
      "A6_obs020 (Random and Frontier+Entropy not run there). Table 6 reports "
      "median mean_bound_final at mission end (budget T) (lower is better), "
      "final coverage, and "
      "quality_auc at mission end (budget T), with paired Holm-corrected "
      "p-values against FB.")
bl = [["Regime", "Method", "bound (med)", "cov (med)", "qual_auc",
       "p_b vs FB", "p_cov vs FB", "rel-bound vs FB"]]
for rg, m, mb, cov, qa, p_b, p_c, rel in baseline_rows:
    star_b = " †" if p_b < 0.05 else ""
    star_c = " †" if p_c < 0.05 else ""
    bl.append([REGIME_LABEL[rg], m, f"{mb:.4f}", f"{cov:.1f}",
               f"{qa:.4f}", f"{fmt_p(p_b)}{star_b}",
               f"{fmt_p(p_c)}{star_c}", f"{rel:+.1f}%"])
story.append(make_table_grouped(
    ["Method", "bound", "cov%", "q_auc", "p_b", "p_cov", "rel-b"],
    bl[1:], [2.3, 1.5, 1.3, 1.4, 1.4, 1.4, 1.6], font=7.4))
tbl_cap("Baseline battery, n = 40 paired per cell († = Holm-sig vs FB, "
        "Wilcoxon). bound = mean_bound_final (lower is better), cov = final "
        "coverage (%), q_auc = quality_auc (normalized AUC of the "
        "well-localized fraction).")
body("<b>Interpretation (preregistered falsification).</b> Random "
     "sits at ≈20% coverage and bound ≈ 0.05 — the floor against which all "
     "claims are measured. Classical occupancy entropy (Entropy-Frac, "
     "Frontier+Entropy) is at parity with FB on coverage and slightly better "
      "on mean_bound_final at 5% obstacles, replicating the finding of our "
      "internal validation campaign that the "
      "movement frame already captures most of the coverage gain. The "
     "config-count signal confirms its specificity — it targets the "
     "diversity of angular observation configurations, whereas occupancy "
     "entropy targets probabilistic map uncertainty: Richness-Angular and "
     "Coverage-U are the two methods that beat FB on the residual bound at "
     "equal-or-better coverage at 5% obstacles, and their accuracy gain over "
     "Random (~65–70% bound reduction) is comparable in size to the coverage "
     "gain itself. The dense stress regime A6_obs020 sharpens the picture: "
     "only Richness-Angular keeps a significant edge (+24%, Holm-sig) when "
      "obstacles fragment the known-free space, whereas Coverage-U and "
      "Entropy-Frac fall to +5–6% (ns = not significant, p ≥ 0.05) — the "
      "angular-selection component of "
      "the config-count signal is the part that is robust to density, while "
      "the plain under-coverage component is the part that saturates.")
if cpu_ms is not None:
    body("<b>Compute cost per decision.</b> Serial benchmark "
         "(A6 regime, n = 10, time.process_time inside select_action). "
         "The claim that Coverage-U is comparable to FB and cheaper than "
         "classical information scorers is now supported by measured CPU "
         "timings (Table 7):")
    cpu_rows = [["Method", "ms/decision (median)"],
                ["Random", f"{cpu_ms['Random']:.2f}"],
                ["Frontier-Bounded", f"{cpu_ms['Frontier-Bounded']:.2f}"],
                ["Coverage-U", f"{cpu_ms['Coverage-U']:.2f}"],
                ["Entropy-Frac", f"{cpu_ms['Entropy-Frac']:.2f}"]]
    story.append(make_table(["Method", "ms/decision"],
                            cpu_rows[1:], [4.0, 3.0], font=8.0))
    tbl_cap("CPU/decision, A6_obs005, n = 10, single process.")
sub("6.7 E4-PARETO: the effect is a plateau, not a knife-edge")
body("Sweeping λ ∈ {0.25, 0.5, 1.0, 2.0} on the two confirmed regimes "
     "(A3, A6) at n = 40 shows the reduction is stable — even slightly "
     "increasing — and never at the cost of coverage:")
t7 = [["Regime", "λ=0.25", "λ=0.5", "λ=1.0", "λ=2.0"]]
for rg in ("A3_obs005", "A6_obs005"):
    cells = []
    for rel, pw, cov in pareto[rg]:
        cells.append(f"{rel:+.1f}%* ({cov:.0f}% cov)")
    t7.append([REGIME_LABEL[rg]] + cells)
story.append(make_table_grouped(["λ=0.25", "λ=0.5", "λ=1.0", "λ=2.0"],
                                t7[1:], [3.0, 3.0, 3.0, 3.0]))
tbl_cap("E4-PARETO: relative reduction of mean_bound_final vs FB at each λ "
        "(n = 40 paired), with median final coverage in parentheses. "
        "* Holm-significant vs FB (Wilcoxon, p &lt; 0.05).")
story.append(img(os.path.join(FIG, "fig_paper_lambda.png"), 14.6))
fig_cap("λ-sweep: residual-bound reduction vs λ (annotated with final "
        "coverage).")
sub("6.8 Qualitative illustration")
body("Figure 7 contrasts FB and Coverage-U trajectories on representative "
     "runs (median and 75th-percentile residual-bound reduction): CU keeps "
     "the bounded-exploration frame but concentrates revisits around "
      "angularly under-determined regions. Figure 8 shows the fraction of "
      "traversable cells left in the rank-deficient single-configuration "
      "state over time — the ambiguous residue that Coverage-U resolves more "
      "completely by mission end.")
story.append(img(os.path.join(FIG, "fig_qualitative_traj.png"), 14.0))
fig_cap("Representative trajectories (rows: A3 median, A3 p75, A6 median, "
        "A6 p75; columns: FB, CU). Obstacles in grey, final UAV positions in "
        "black. CU re-observes under-determined cells within the same bounded "
        "frame, concentrating revisits around angularly sparse regions.")
story.append(img(os.path.join(FIG, "fig_qualitative_f1.png"), 13.6))
fig_cap("Ambiguous fraction vs step: traversable cells with exactly one "
        "angular configuration (observed but rank-deficient).")
sub("6.9 E5: centralized oracle — the localization value of locality")
body("The E5 campaign (2 regimes × 2 oracle methods × 40 paired runs) asks "
     "how much of Coverage-U's accuracy gain a centralized perfect oracle "
     "would capture on the same dual objective. The answer is sharp: the "
     "oracle does not merely fail to improve on FB — it regresses the "
     "primary metric and destroys the coverage guard at the same time.")
e5_rows, e5_diag_rows = e5_table()
story.append(make_table_grouped(
    ["FB mb", "CU mb", "Conf mb", "CRLB mb",
     "FB cov", "CU cov", "Conf cov", "CRLB cov",
     "red_CU%", "red_Conf%", "red_CRLB%"],
    [[f"{REGIME_LABEL[r[0]]}", f"{r[1]:.4f}", f"{r[3]:.4f}", f"{r[5]:.4f}",
      f"{r[7]:.4f}", f"{r[2]:.1f}", f"{r[4]:.1f}", f"{r[6]:.1f}",
      f"{r[8]:.1f}", f"{r[9]:+.1f}", f"{r[10]:+.1f}", f"{r[11]:+.1f}"]
     for r in e5_rows],
     [1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.5, 1.8, 1.8], font=7.4))
tbl_cap("E5 ladder (n = 40 paired): mean_bound_final and final coverage for "
        "Frontier-Bounded (FB), Coverage-U (CU), and the two centralized "
        "oracles (Config, CRLB).")
body("The ladder isolates the decisive variable. CentralOracle-Config uses "
     "exactly the same config-count under-set as Coverage-U, only under "
     "perfect global fusion; CentralOracle-CRLB uses the true global CRLB "
     "bottleneck instead. Both collapse to ≈42% coverage (A3) / 36% (A6) and "
     "a residual bound ≈2× worse than FB (Holm-significant, p&lt;0.0001, "
     "both regimes, both oracle rows), while Coverage-U holds coverage "
     "within noise of FB and reduces the residual bound by ≈+30% median. "
     "Config vs CRLB are statistically indistinguishable (A3 p=0.29, A6 "
     "p=0.48): the signal choice is not the problem — the global frame is.")
body("<b>Diagnostic: is the failure calibration or structure?</b> The "
     "preregistered E5-DIAG adds coverage-guarded oracle variants "
     "CentralOracle-CRLB-cov (ε=0.05) and -cov2 (ε=0.30) that cap the "
     "accuracy bonus by a (1−ε) fraction of the coverage term, so accuracy "
     "can reorder equal-coverage targets but can no longer make a far target "
     "win over a near one. The ε=0.30 cap provably binds only when a target "
     "has box-sum bonus &gt; 21.2·D; instrumenting real trajectories (4,800 "
     "reachable-target evaluations) shows the bonus never exceeds the cap "
     "even at ε=0.30 (0 bindings), because the global CRLB under-set is too "
     "sparse for any FOV to reach the threshold. The two guarded variants "
     "are therefore bit-identical to the unguarded oracle on all 40 seeds "
     "(residual bound 0.0445 vs 0.0445 in A3, 0.0487 vs 0.0487 in A6).")
story.append(make_table(
    ["Regime · ε", "mb (med)", "cov (med)", "p_mb", "p_cov"],
    e5_diag_rows,
    [3.0, 2.0, 2.0, 2.0, 2.0], font=7.6))
tbl_cap("E5-DIAG: coverage-guarded oracle variants vs unguarded "
        "CentralOracle-CRLB (n = 40 paired).")
body("<b>Interpretation (preregistered diagnostic).</b> The guard never binding "
      "(0 bindings across 4,800 reachable-target evaluations) is a negative "
      "structural result rather than a dead end: the global CRLB under-set "
      "is so sparse that no FOV can approach the cap, so this "
      "coverage-guard mechanism is structurally incapable of arbitrating "
      "the calibration-vs-structure question here — the calibration "
      "hypothesis is therefore not supported by the guard, but also not "
      "definitively disproven by it. The diagnostic's value is that it "
      "narrowed the search and forced the local-vs-global comparison that "
      "follows. The locality result does not depend on this diagnostic: "
      "the config-count signal that reduces the residual bound by ~+30% "
      "when scored locally (Coverage-U) regresses both metrics when the "
      "same under-set is fused globally (CentralOracle-Config) — the "
      "decisive comparison is the ladder above, not the guard.")
body("<b>E5-CORRECTED: removing the movement-frame confound.</b> The ladder "
     "above mixes two differences: the under-set SIGNAL (local vs global "
     "fusion) and the movement FRAME (the oracle’s global BFS excludes every "
     "cell visited by any agent from the path, so its reachable targets are "
     "always far, D ≥ 5, while Coverage-U/FB move on the local bounded_bfs "
     "frame). A follow-up campaign (PRE_REG_E5_CORRECTED.md, n = 40 paired, "
     "same seeds) re-runs both oracle signals in the byte-identical local "
     "bounded_bfs frame of Coverage-U, isolating the signal effect.")
e5_corr_rows = e5_corrected_table()
story.append(make_table_grouped(
    ["FB mb", "CU mb", "CRLB mb", "CRLB-L mb", "Conf-L mb",
     "CU cov", "CRLB-L cov", "p_mb", "p_cov"],
    [[REGIME_LABEL[r[0]]] + r[1:] for r in e5_corr_rows],
    [1.3, 1.3, 1.3, 1.5, 1.5, 1.4, 1.4, 1.3, 1.3], font=7.2))
tbl_cap("E5-CORRECTED (n = 40 paired, local movement frame for every row): "
        "mean_bound_final and final coverage for FB, Coverage-U, and the two "
        "global-fusion oracles (CRLB, Config) in the byte-identical local "
        "frame.")
body("<b>Interpretation (preregistered, outcome B).</b> The global movement "
      "frame was a real but secondary confound (CRLB 0.049 → 0.032 median "
      "bound in A6), and the dominant effect is the signal: the local "
      "Coverage-U signal beats both global-fusion oracles by ≈2× on "
      "mean_bound_final and ≈30 pp on coverage with the frame held identical "
      "(p &lt; 10⁻⁸ both regimes). Even with the movement "
      "frame held byte-identical, the perfect-fusion global under-set "
      "regresses both metrics: it is dense across the whole map, so the "
      "accuracy bonus dominates the distance term and agents over-chase far "
      "under-determined cells instead of discovering. The local proxy "
      "under-counts (each agent only sees its own observations), which keeps "
      "the coverage term in the tradeoff — the mechanism that makes "
      "Coverage-U work. The qualitative conclusion of E5 survives the "
      "confound fix: what matters is the <b>locality of the decision "
      "signal</b>, not its strength.")
sub("6.10 Robustness sweeps (post-preregistration, confirmatory)")
body("After the preregistered campaign we ran four confirmatory robustness "
     "sweeps on the two confirmed regimes (A3, A6; n = 40 paired each, same "
     "seed ladder, same Wilcoxon + Holm protocol) to map the boundary "
     "conditions of the Coverage-U effect. <i>Self-localization noise.</i> "
     "The local decision signal is corrupted by zero-mean Gaussian noise with "
     "std σ ∈ {0.5, 1.0} grid cells, while the oracle CRLB evaluation keeps "
     "the true geometry (the noise decoupling is by design, env.py). "
     "<i>Bearing noise.</i> The angular observations that feed the "
     "config-count signal carry zero-mean measurement noise of std σ<sub>θ</sub> "
     "= 2° (the level a MEMS magnetometer/gyroscope combo exhibits in the "
     "field); the oracle CRLB again keeps the true geometry, so the metric "
     "stays decoupled from the injected error. <i>Budget.</i> The mission "
     "budget is scaled to {0.3, 0.5, 0.9} × FB "
     "steps_90 (0.7 is the confirmed operating point). <i>Fusion range.</i> "
     "The proximity-fusion communication range is reduced to {2.5, 1.25} "
      "cells (default = FOV = 5). All tables report Coverage-U vs FB on "
      "mean_bound_final at mission end (budget T) with Holm-corrected paired "
      "Wilcoxon p-values and "
      "the final-coverage guard.")
body("<b>B. Noise: the effect survives moderate self-localization error.</b> "
     "At σ = 0.5 the residual-bound reduction remains Holm-significant in both "
     "regimes (+14.8% A3, +20.8% A6) with no coverage regression; at σ = 1.0 "
     "it attenuates — in A3 the reduction disappears (point estimate slightly "
     "negative, ns) while in A6 it remains Holm-significant (+14.9%). The "
      "degradation tracks the decision signal's own noise — expected, since "
      "Coverage-U's bonus is computed from the noisy local map — and the coverage "
     "guard never regresses at either level.")
sweep_b_tbl = [["Regime", "σ", "FB bound", "CU bound", "rel-red%", "p", "p_Holm"]]
for rg, v, mf, mc, rel, p, pc in sweep_sigma_rows:
    sweep_b_tbl.append([REGIME_LABEL[rg], SWEEP_LABEL[v], f"{mf:.4f}",
                        f"{mc:.4f}", f"{rel:+.1f}", fmt_p(p), fmt_p(pc)])
story.append(make_table_grouped(["σ", "FB bound", "CU bound", "rel-red%",
                                 "p", "p_Holm"], sweep_b_tbl[1:],
                                [1.2, 1.8, 1.8, 1.8, 1.8, 1.8], font=7.6))
tbl_cap("Sweep B (n = 40 paired): Coverage-U vs FB on mean_bound_final at "
        "mission end (budget T) under self-localization noise σ (cells); "
        "oracle keeps true geometry.")
story.append(img(os.path.join(FIG, "fig_paper_sweep_sigma.png"), 15.2))
fig_cap("Self-localization noise sweep: residual-bound reduction of "
        "Coverage-U vs FB at σ ∈ {0, 0.5, 1.0} cells.")
body("<b>C. Budget: the advantage concentrates at tight-to-moderate budgets — "
     "the core ‘budget effect’ claim, confirmed in direction.</b> In A3 the "
     "reduction is monotonically decreasing in T: +29.2% at 0.3 × FB, +20.1% "
     "at 0.5 × FB, +17.0% at the confirmed 0.7 operating point, and "
     "non-significant (+9.1%, ns) at 0.9 × FB. In A6 it is roughly flat over "
     "T ∈ [0.3, 0.7] (+19.5% / +23.8% / +25.7%) and shrinks at 0.9 × FB "
     "(+11.1%, still sig) as coverage approaches saturation and the "
     "residual-bound gap closes. The cost side is visible at the extreme: at "
     "0.3 × FB the accuracy gain carries a small coverage regression in A6 "
     "(−1.5 pp, Holm-sig) and a −1.7 pp trend in A3 that does not survive "
     "Holm; no regression at 0.5–0.9 × FB.")
sweep_c_tbl = [["Regime", "T×FB", "FB bound", "CU bound", "rel-red%", "p", "p_Holm"]]
for rg, v, mf, mc, rel, p, pc in sweep_budget_rows:
    sweep_c_tbl.append([REGIME_LABEL[rg], SWEEP_LABEL[v], f"{mf:.4f}",
                        f"{mc:.4f}", f"{rel:+.1f}", fmt_p(p), fmt_p(pc)])
story.append(make_table_grouped(["T×FB", "FB bound", "CU bound", "rel-red%",
                                 "p", "p_Holm"], sweep_c_tbl[1:],
                                [1.2, 1.8, 1.8, 1.8, 1.8, 1.8], font=7.6))
tbl_cap("Sweep C (n = 40 paired): Coverage-U vs FB across mission budgets "
        "T = {0.3, 0.5, 0.7, 0.9} × FB steps_90.")
story.append(img(os.path.join(FIG, "fig_paper_sweep_budget.png"), 15.2))
fig_cap("Budget sweep: residual-bound reduction vs budget (annotated with CU "
        "final coverage).")
body("<b>D. Fusion range: the effect is robust to much shorter-range "
     "communication — locality, not range, is the active ingredient.</b> "
     "At R = 2.5 (half FOV) the reduction is +19.2% (A3) / +21.6% (A6), both "
     "Holm-sig with no coverage regression; at R = 1.25 (a quarter of FOV) it "
     "is +26.2% (A3) / +15.7% (A6), both sig, with a small Holm-significant "
     "coverage regression in A6 (−2.6 pp). The apparent regime reversal (A3 "
     "favouring the shorter range, A6 the longer) is not statistically real: "
     "CU's bound at R = 1.25 vs R = 2.5 is indistinguishable within each "
     "regime on the same paired seeds (A3 p = 0.97, A6 p = 0.40), so the two "
      "range levels are equivalent for accuracy. Because Coverage-U's decision "
      "signal "
      "is computed from the agent's own (proximity-fused) map, shrinking the "
     "fusion range does not remove the signal — it only delays the spread of "
     "under-determined-cell knowledge, and the local re-observation "
     "mechanism still resolves it.")
sweep_d_tbl = [["Regime", "R", "FB bound", "CU bound", "rel-red%", "p", "p_Holm"]]
for rg, v, mf, mc, rel, p, pc in sweep_comm_rows:
    sweep_d_tbl.append([REGIME_LABEL[rg], SWEEP_LABEL[v], f"{mf:.4f}",
                        f"{mc:.4f}", f"{rel:+.1f}", fmt_p(p), fmt_p(pc)])
story.append(make_table_grouped(["R", "FB bound", "CU bound", "rel-red%",
                                 "p", "p_Holm"], sweep_d_tbl[1:],
                                [1.2, 1.8, 1.8, 1.8, 1.8, 1.8], font=7.6))
tbl_cap("Sweep D (n = 40 paired): Coverage-U vs FB under reduced "
        "proximity-fusion range R ∈ {2.5, 1.25} cells (default R = FOV = 5).")
story.append(img(os.path.join(FIG, "fig_paper_sweep_comm.png"), 15.2))
fig_cap("Fusion-range sweep: residual-bound reduction of Coverage-U vs FB at "
        "R ∈ {2.5, 1.25} cells.")
body("<b>E. Bearing noise: the effect survives measurement noise on the "
     "bearing itself.</b> The config-count signal is built from angular "
     "observations; here each recorded bearing carries independent zero-mean "
     "Gaussian noise of std σ<sub>θ</sub> = 2° (well below the 15° angular "
     "tolerance that defines a new configuration). The residual-bound "
     "reduction remains Holm-significant in both regimes, and the coverage "
     "guard does not regress in either — the effect is unchanged in direction "
     "and magnitude by measurement noise on the very quantity the signal "
     "counts.")
sweep_e_tbl = [["Regime", "σθ (deg)", "FB bound", "CU bound", "rel-red%", "p", "p_Holm"]]
for rg, v, mf, mc, rel, p, pc in sweep_bearing_rows:
    sweep_e_tbl.append([REGIME_LABEL[rg], SWEEP_LABEL[v], f"{mf:.4f}",
                        f"{mc:.4f}", f"{rel:+.1f}", fmt_p(p), fmt_p(pc)])
story.append(make_table_grouped(["σθ (deg)", "FB bound", "CU bound",
                                 "rel-red%", "p", "p_Holm"], sweep_e_tbl[1:],
                                [1.2, 1.8, 1.8, 1.8, 1.8, 1.8], font=7.6))
tbl_cap("Sweep E (n = 40 paired): Coverage-U vs FB on mean_bound_final at "
        "mission end (budget T) under bearing-measurement noise "
        "σ<sub>θ</sub> = 2° applied to the local angular observations; "
        "oracle keeps true geometry.")
story.append(img(os.path.join(FIG, "fig_paper_sweep_bearing.png"), 15.2))
fig_cap("Bearing-noise sweep: residual-bound reduction of Coverage-U vs FB "
        "when the config-count signal is computed from noisy bearings "
        "(σ<sub>θ</sub> = 2°).")
body("<b>Interpretation.</b> The four sweeps jointly delimit the effect: it is "
     "robust to moderate self-localization noise (σ = 0.5 in both regimes; at "
     "σ = 1.0 it vanishes in A3 and survives in A6), to bearing-measurement "
     "noise on the signal itself (σ<sub>θ</sub> = 2°, both regimes), and to "
     "short-range "
     "communication (down to R = 1.25) — the three departures from the "
     "idealized setting that matter for real hardware — and it is strongest "
     "precisely where the paper claims it operates (tight-to-moderate "
     "budgets). The strict preregistration guard (no significant coverage "
     "regression) holds in every sweep cell except the 0.3 × FB / A6 cell "
     "and the R = 1.25 / A6 cell, each failure small (1.5–2.6 pp; Table G); "
     "the 0.3 × FB / A3 cell is a −1.7 pp trend that does not survive Holm. "
     "The 0.3 × FB extreme therefore marks the "
     "boundary of the free lunch — at such tight budgets an accuracy gain "
      "must cost coverage, exactly the trade-off the paper predicts — and "
      "the 0.5–0.7 operating window is where "
      "Coverage-U improves accuracy at no significant coverage cost, the "
      "regime the "
      "paper's claim targets. These are "
     "post-preregistration confirmatory follow-ups and are labeled as such.")
sweep_guard_tbl = [["Regime", "Sweep", "FB cov", "CU cov", "Δ (pp)", "p",
                    "p_Holm"]]
for rg, lab, mf, mc, dlt, p, pc in sweep_guard_all:
    reg = "regress" if (pc < 0.05 and mc < mf) else ""
    sweep_guard_tbl.append([REGIME_LABEL[rg], lab, f"{mf:.1f}", f"{mc:.1f}",
                            f"{dlt:+.1f}", fmt_p(p), fmt_p(pc), reg])
story.append(make_table_grouped(["Sweep", "FB cov", "CU cov", "Δ (pp)", "p",
                                 "p_Holm", ""], sweep_guard_tbl[1:],
                                [1.5, 1.7, 1.7, 1.6, 1.7, 1.7, 1.6],
                                font=7.4))
tbl_cap("Coverage guard across all sweep cells (n = 40 paired; Holm applied "
        "within each sweep). Δ = CU cov − FB cov (pp); Δ &gt; 0 favours "
        "Coverage-U; guard regression = CU significantly below FB.")


# ======================================================================
# 6.11 TOPOLOGY GENERALIZATION (post-submission, confirmatory)
# ======================================================================
sub("6.11 Topology generalization: maze and cluster environments "
    "(post-submission, confirmatory)")
body("The campaign above is bounded to the open-space grid (random obstacle "
     "topology, Section 8). After submission we ran a post-submission "
     "topology dossier to map the boundary of the confirmed effects, "
     "keeping the manuscript frozen: the number and narrative below are "
     "confirmatory extensions, not part of the preregistered protocol. "
     "Two new environment families extend <i>GridEnv</i> with the same "
     "line-of-sight FOV occlusion but different routing structure. "
     "<i>Mazes</i> (env.py <i>_build_maze_map</i>): a Kruskal randomized "
     "perfect maze on a C=49 cell grid — ONE connected free component, "
     "exactly one path between any two free cells (E = V−1, no closed "
     "loops), wall fraction ~52%. <i>Clusters</i> (env.py "
     "<i>_build_cluster_map</i>): contiguous obstacle blocks (axis-aligned "
     "rectangles 1–5 cells wide) at 20% density with the outer edge ring "
     "free — real occlusion (unlike i.i.d. single cells) while the free "
     "space keeps multiple routing paths. The random topology keeps the "
     "historical square-FOV model (non-regression). Every campaign follows "
     "the same protocol family as the main budget campaigns: n = 40 paired "
     "seeds (0…39000), methods {FB, Coverage-U, Richness-Angular}, "
     "Wilcoxon + Holm across the three pairwise comparisons on quality_auc "
     "@ T (primary) with the final-coverage guard, and budget T = 0.7 × FB "
     "median steps_90 measured per topology by a probe (maze 1323, "
     "cluster A6 1354, cluster A3 2710 steps).")
body("<b>A. Perfect maze: the open-space advantage flips sign under "
     "single-path routing.</b> In the perfect maze (no loops), both "
     "candidates significantly <i>lose</i> quality_auc relative to FB "
     "(RA −3.0%, CU −2.1%, both Holm-sig) and both regress final coverage "
     "(RA 68.7 vs 71.3, p &lt; 0.0004; CU 69.5 vs 71.3, p &lt; 0.005). "
     "The effect is a coverage regression, not a precision regression: "
     "mean_bound at T is near-identical across methods and if anything "
     "slightly lower (better) for the candidates, while the loss sits in "
     "undetermined_final and final_coverage. In open space the same "
     "bonuses are accuracy-positive; in corridors (52% walls, LOS "
     "occlusion) the under-observed/richness bias over-fits the open-space "
     "signal — it steers agents back into already-covered corridors "
     "(overlap 0.580/0.585 vs FB 0.569, both p &lt; 0.01; visual traces "
     "show heavy corridor backtracking) instead of pushing dead-end "
     "progress. The directional MAexp-coherent check (RA ≥ CU) is not "
     "confirmed here (medians equal).")
def _topo_row(campaign_label, row):
    a, b, ma, mb, g, p, pc = row
    _an = a.split("-")[-1].upper() if "-" in a else a
    _bn = b.split("-")[-1].upper() if "-" in b else b
    return [campaign_label, f"{_an} vs {_bn}", f"{ma:.3f}", f"{mb:.3f}",
            f"{g:+.1f}", fmt_p(p), fmt_p(pc)]

topo_tbl_a = [["Campaign", "A vs B", "med_A", "med_B", "gain%", "p", "p_Holm"]]
topo_tbl_a += [_topo_row("maze perfect (A6)", row)
               for row in topo_primary["maze perfect (A6)"]]
topo_tbl_a += [_topo_row("maze perfect, H=16 (A6)", row)
               for row in topo_primary["maze perfect, H=16 (A6)"]]
story.append(make_table(["Campaign", "A vs B", "med_A", "med_B", "gain%", "p",
                        "p_Holm"], topo_tbl_a[1:], [2.9, 1.8, 1.5, 1.5, 1.5,
                                                    1.5, 1.5], font=7.6))
tbl_cap("Maze family, quality_auc @ T (higher better), n = 40 paired per "
        "campaign; rel-gain% vs baseline B per the paper's convention; "
        "Holm across the 3 pairs. Perfect maze (A6): both candidates lose "
        "significantly. H=16: CU is bit-identical to H=8 (−2.1%, Holm-sig), "
        "RA improves to ns.")
body("<b>B. Loops at 10%: the penalty is neutralized but no advantage is "
     "restored.</b> Re-opening ~10% of the non-tree walls (→ ~206 added "
     "cycles, wall fraction 52.0%→49.9%, still one free component) removes "
     "the significant penalty: all three comparisons are non-significant "
     "(RA vs FB −3.3% ns, CU vs FB −0.9% ns, RA vs CU −2.2% ns after "
     "Holm) and the coverage guard passes in every cell. The maze loss is "
     "therefore a single-path artifact: partial routing neutralizes it, "
     "but no candidate recovers an edge. Overlap for RA drops with loops "
     "(0.585 → 0.550, p &lt; 0.001), consistent with loops diluting the "
     "corridor over-lingering.")
img_keep(os.path.join(FIG, "fig_maze_maps.png"), 15.2,
         "Sixteen Kruskal perfect mazes (topology=maze, 100×100, C=49, wall "
         "fraction ~52%); the free space is a single connected component "
         "with exactly one path between any two cells.")
img_keep(os.path.join(FIG, "fig_maze_los.png"), 15.2,
         "Line-of-sight FOV occlusion in the maze: the same agent FOV "
         "(r = 5) with (right) and without (left) the wall-blocking rule. "
         "In maze and cluster topologies a cell is visible iff the "
         "supercover line crosses no obstacle.")
body("<b>C. Horizon-16 is inert — the maze penalty is structural, not a "
     "lookahead artifact.</b> A longer planning horizon (H=16 vs H=8) was "
     "tested on the same perfect-maze seeds to rule out the confound "
     "'H=8 is too short for long corridors'. CU is bit-identical to H=8 "
     "(−2.1%, Holm-sig; same median, same p, same coverage, same "
     "undetermined): <i>bounded_bfs</i> stops at the first reachable-unknown "
     "target and blocks visited cells, so in corridors the nearest unknown "
     "cell is always within ≤ 8 layers and a longer horizon never changes "
     "the target. RA improves from −3.0% (H=8) to neutral (−1.1% ns), but "
     "no candidate gains. The horizon is not the lever.")
body("<b>D. Clusters 20% (A6): RA's advantage survives occlusion when "
     "routing is preserved.</b> The cluster campaign is the "
     "occlusion-vs-routing decomposition test: the same LOS filter as the "
     "maze, but contiguous blocks at 20% density keep multiple routing "
     "paths (largest free component ≥ 99.9% on all probe seeds). The maze "
     "penalty is gone for CU (−0.4%, ns, no coverage regression), and "
     "Richness-Angular keeps its edge: RA vs FB <b>+3.0% (p = 0.0147, "
     "Holm-sig)</b> with no coverage regression (69.2 vs 67.9, ns; RA is "
     "if anything above FB), and the directional RA ≥ CU comparison is "
     "<b>confirmed for the first time in the topology dossier</b> "
     "(+1.7%, p = 0.020, Holm-sig). The maze failure is therefore caused "
     "by the loss of routing (single-path spanning tree), not by occlusion "
     "itself.")
topo_tbl_b = [["Campaign", "A vs B", "med_A", "med_B", "gain%", "p", "p_Holm"]]
for label, row in zip(["clusters 20% (A6)", "clusters 20% (A3)"],
                      [topo_primary["clusters 20% (A6)"],
                       topo_primary["clusters 20% (A3)"]]):
    for a, b, ma, mb, g, p, pc in row:
        _an = a.split("-")[-1].upper() if "-" in a else a
        _bn = b.split("-")[-1].upper() if "-" in b else b
        topo_tbl_b.append([label, f"{_an} vs {_bn}", f"{ma:.3f}", f"{mb:.3f}",
                           f"{g:+.1f}", fmt_p(p), fmt_p(pc)])
story.append(make_table(["Campaign", "A vs B", "med_A", "med_B", "gain%", "p",
                        "p_Holm"], topo_tbl_b[1:], [2.9, 1.8, 1.5, 1.5, 1.5,
                                                    1.5, 1.5], font=7.6))
tbl_cap("Cluster family, quality_auc @ T (higher better), n = 40 paired per "
        "campaign. A6 (T=1354): RA +3.0% vs FB (Holm-sig), RA ≥ CU "
        "confirmed, CU neutral. A3 pass-2 (T=2710): RA +2.6% vs FB "
        "(Holm-sig), RA vs CU +6.3% (strongest delta), CU neutral.")
img_keep(os.path.join(FIG, "fig_cluster_maps.png"), 15.2,
         "Sixteen cluster maps (topology=cluster, 100×100, 20% contiguous "
         "obstacle blocks, outer edge ring free). Real occlusion, "
         "preserved multi-path routing.")
img_keep(os.path.join(FIG, "fig_cluster_los.png"), 15.2,
         "LOS occlusion on a cluster map: same agent FOV (r = 5) with "
         "(right) and without (left) the wall-blocking rule. Contiguous "
         "blocks occlude the supercover line where i.i.d. single cells do "
         "not.")
body("<b>E. Clusters at 3 UAVs (pass-2): the finding is swarm-size "
     "robust.</b> The pass-2 rule ('A3 if the test is positive') triggers "
     "an A3 (3-UAV) replica on the same cluster-20% topology. RA's edge "
     "reproduces: RA vs FB <b>+2.6% (p = 0.0076, Holm-sig)</b>, RA vs CU "
     "<b>+6.3% (p = 0.0001, Holm-sig, the strongest delta of all cluster "
     "tests)</b>, no coverage regression in any cell. CU is again neutral "
     "(−1.2%, ns), consistent with the paper's dense regimes. The A6 "
     "cluster finding is not a 6-UAV artifact.")
body("<b>Cross-campaign verdict.</b> The five-campaign dossier maps the "
     "boundary of the paper's claims: perfect maze = negative for both "
     "candidates (single-path routing collapse, not precision loss); 10% "
     "loops = neutral (penalty gone, no gain); H=16 = inert (horizon is "
     "not the lever); clusters 20% A6 = RA advantage intact (+3.0% "
     "Holm-sig, RA ≥ CU confirmed); clusters 20% A3 = RA advantage "
     "reproduced (+2.6% Holm-sig). The mechanistic story is now coherent "
     "across the whole corpus: open space and clusters (multi-path) keep "
     "the advantage, a perfect maze (single path) flips it negative, and "
     "loops (partial routing) sit between. What matters is routing "
     "redundancy, not spatial fragmentation per se. This is a "
     "post-submission confirmatory extension and is labeled as such; the "
     "paper's headline claims (sparse open space, Sections 6.4–6.10) are "
     "unchanged.")


# ======================================================================
# 7. DISCUSSION
# ======================================================================
sec("7. Discussion")
body("<b>The falsifications delimit where the signal works.</b> Richness "
     "configurations fail as a direct target-selection rule (E1), as a "
     "mode-switching rule (E2/E3), and fail to move the saturated binary "
     "quality fraction (E4 primary). Each failure is informative: the "
     "bounded-horizon geometric frame already captures the coverage gains "
      "that these signals were expected to add, replicating the finding of "
      "our internal validation campaign. This is the “Occam’s razor” "
      "narrative made quantitative, and "
     "it is the honest baseline against which any positive claim must be "
     "read.")
body("<b>The positive effect is a budget effect, not an accuracy effect.</b> "
     "In unbounded episodes every method reaches quality_final = 1.0 and "
     "residual bounds at parity. Under a fixed mission time, Coverage-U "
     "reduces the residual oracle bound by ~21% median with no coverage "
      "cost. The mechanism is visible in the traces: standard coverage "
      "spends remaining time reaching cheap frontier cells, leaving angular "
      "gaps; Coverage-U spends the same steps re-observing under-determined "
      "cells. "
     "The result matches the classical GDOP intuition [Bishop2004, "
     "Bishop2009] — accuracy is bought with baseline geometry — and shows "
     "that a <i>local, cheap</i> proxy can capture part of that intuition "
     "without a centralized planner.")
body("<b>Why the primary metric failed and what it teaches.</b> quality_auc "
     "is a thresholded binary fraction that saturates; it cannot rank "
     "improvements below the well-localization threshold. The continuous "
     "residual bound is the metric aligned with the actual objective "
     "(accuracy). We report the preregistered primary as FAIL and the "
     "secondary as a confirmed discovery — a protocol lesson worth "
     "publishing: binary saturation metrics conceal accuracy effects that "
     "continuous bounds reveal.")
body("<b>Boundary conditions.</b> The Coverage-U advantage is not "
      "statistically significant at 20% obstacle density (A6_obs020: +5.4%, "
      "ns after Holm correction; undetermined slightly reversed), marking a "
      "boundary of the effect: "
      "dense obstacles fragment the known-free space and dilute the signal. "
      "The mechanism is an integral-image dilution: the Coverage-U bonus "
      "scores under-observed cells through a fixed-area FOV window, so "
      "under dense obstacles the window's footprint is dominated by "
      "blocked cells — the `under_count_FOV` integral image is starved, "
      "the FOV area stays constant, and the average bonus per free cell "
      "collapses even though genuinely under-determined cells exist. "
      "Richness-Angular avoids this dilution because it scores the raw "
      "per-cell richness without averaging over the FOV area, which is why "
      "the angular-selection component keeps a significant +24% gain in the "
      "same dense regime (Section 6.6) while the plain under-coverage "
      "component saturates. The conclusion is therefore conditional: "
      "config-count prioritization helps in open, low-obstacle environments "
      "under time pressure — precisely the mission profile where partial "
      "coverage is unavoidable and residual accuracy matters most — and its "
      "angular variant extends the benefit to denser obstacle fields.")
body("<b>Topology: routing redundancy, not occlusion, is the boundary "
     "variable.</b> The post-submission topology dossier (Section 6.11) "
     "isolates the mechanism. Under single-path routing (perfect maze), "
     "both candidates' open-space bonus flips sign — a significant "
     "coverage regression (not a precision one) driven by corridor "
     "over-lingering; re-opening 10% of the walls neutralizes the penalty "
     "without restoring any edge; a longer horizon is inert. When the same "
     "LOS occlusion is applied to contiguous blocks that preserve "
     "multi-path routing (cluster topology), Coverage-U returns to neutral "
     "and Richness-Angular's advantage is intact at 6 UAVs (+3.0%, "
     "Holm-sig) and reproduced at 3 UAVs (+2.6%, Holm-sig) with the "
     "directional RA ≥ CU comparison confirmed in both. The reading is "
     "that the under-observed/richness signals over-fit the open-space "
     "landscape and misjudge dead ends in single-path corridors, while "
     "their angular/richness content survives real occlusion when routing "
     "choice remains — refining the paper's boundary from 'spatial "
     "fragmentation' to 'routing redundancy'.")
body("<b>The dense-regime failure is a dilution, not a defect of the "
     "under-set signal.</b> Because the dilution is a denominator artifact "
     "of the score (Section 4.4.1), it can be attacked directly: "
     "normalizing by the number of traversable cells in the footprint "
     "instead of the constant FOV area re-scales the bonus by what the "
     "sensor can actually observe, restoring signal strength in fragmented "
     "windows. We deliberately do not report results for that variant here — "
     "changing the score is a new intervention and deserves its own "
     "preregistered evaluation rather than a post-hoc rerun — but it is the "
     "first and cheapest fix on the path to dense-terrain operation, and the "
     "hybrid that switches to the angular signal when the local window is "
     "mostly blocked (Section 9) is the second.")

# ======================================================================
# 8. SCOPE AND BOUNDARIES
# ======================================================================
sec("8. Scope and boundaries")
body("The paper answers a deliberately narrow, preregistered question: whether a "
     "config-count richness signal can reduce residual bearing-only localization "
     "error at equal coverage under a finite mission budget. Every boundary below "
     "is a scoping choice that keeps that question answerable with paired, "
     "preregistered evidence, and the robustness sweeps in Section 6.10 already "
     "relax the two idealizations that matter for the claim. What is not modeled "
     "is outside the claim, not a gap in it: within this framework the campaign "
     "is complete — falsifications, confirmation, robustness, and the "
     "local-vs-global ladder are all reported under the same locked protocol.")
story.append(BUL("Sensing is idealized where it is not the object of study. The "
                 "decision signal is deliberately tested under self-localization "
                 "noise (σ = 0.5, 1.0, Section 6.10), while the CRLB metric "
                 "intentionally uses the true geometry so it measures "
                 "localization work rather than sensor-model fidelity. "
                 "Bearing/measurement noise and dropout sit outside the paper's "
                 "question and would affect every method under comparison equally "
                 "in the paired design."))
story.append(BUL("The environment is a single-scale grid with randomly placed "
                "obstacles — the topology the effect is claimed for. A "
                "post-submission topology dossier (Section 6.11) extends this "
                "with maze and cluster layouts under the same LOS model: the "
                "advantage survives occlusion when routing is preserved "
                "(clusters) and flips negative under single-path routing "
                "(perfect maze); multi-floor layouts and real terrain remain "
                "outside the claim."))
story.append(BUL("The confirmed effect is scoped to sparse-obstacle regimes under "
                 "a finite mission budget — precisely the regime where coverage "
                 "and accuracy compete. No claim of a general accuracy gain is "
                 "made, and none is needed for the paper's conclusion."))
story.append(BUL("Communication is proximity-triggered fusion, the mechanism the "
                 "decentralized claim rests on. Message topologies, delays, and "
                 "bandwidth belong to the communication layer and are outside the "
                 "finite-budget accuracy question."))
story.append(BUL("The centralized oracle (E5) assumes idealized perfect "
                 "fusion; real centralized planners would face communication, "
                 "latency, and map-fidelity costs not modeled here. The "
                 "oracle's regression and the never-binding coverage guard are "
                 "results reported and discussed in Sections 6.9 and 7, not "
                 "limitations of this paper."))
story.append(BUL("Compute cost is measured per decision for the core comparison "
                 "(Section 6.6): Coverage-U is at FB cost (2.17 ms vs 2.16 ms) "
                 "and ~16% cheaper than the occupancy-entropy scorer (2.52 ms) on "
                 "one serial benchmark. GDOP/FIM-style matrix inversions are not "
                 "benchmarked directly; CU's advantage over them rests on its "
                 "O(1) integral-image design plus the measured gap to the entropy "
                 "scorer."))

# ======================================================================
# 9. FUTURE WORK
# ======================================================================
sec("9. Future Work")
body("E5 is complete: the centralized perfect oracle regresses both metrics, "
     "and the local-vs-global ladder (same config-count signal, +30% local "
     "vs regression global) locates the value in locality. E5-CORRECTED "
     "confirms the conclusion holds when the movement frame is held "
     "identical — the global under-set signal, not the oracle frame, is "
      "what collapses coverage. The coverage-guarded diagnostic did not "
      "bind in any real trajectory — a negative structural result that "
      "narrows the calibration-vs-structure question — so a definitive "
      "arbitration would need a guard that provably activates (e.g. "
      "inflated λ or a normed "
      "bonus). The immediate next step is therefore (i) a strict "
     "CPU-per-decision benchmark against GDOP/FIM "
     "planners; (ii) GPS-noise and communication-topology robustness "
     "(Phase 2 of the thesis plan) — Section 6.10 already extends "
     "self-localization noise and fusion range for the two confirmed "
      "regimes, and both robustness margins hold at moderate departures; "
      "(iii) the maze/cluster dossier (Section 6.11) shows the advantage "
      "survives occlusion when routing is preserved and fails under "
      "single-path routing — the immediate follow-up is multi-floor and "
      "room-and-door layouts to complete the routing-redundancy axis, "
      "plus the corridor-aware rescoring that addresses the dead-end "
      "misjudgment directly; (iv) adaptive λ (the Pareto plateau invites tuning, "
      "but we deliberately report the fixed preregistered value); (v) "
      "combining continuous prioritization with the deploy/orbit mechanics "
      "that failed as a pure mode switch — the positive result suggests the "
      "mechanics were sound and the gating was wrong; (vi) other richness "
      "estimators (ACE, Jackknife) and entropy baselines under the same "
      "budget protocol, to place the config-count proxy on the "
      "information-signal ladder.")
body("Two follow-ups target the dense-regime boundary directly. First, the "
     "dynamic-normalization variant of Section 4.4.1 — normalize the "
     "Coverage-U bonus by traversable area instead of the constant FOV "
     "window — is the cheapest fix for the integral-image dilution diagnosed "
     "in Section 7, and deserves its own preregistered evaluation before any "
     "dense-terrain claim. Second, a low-cost hybrid policy that runs "
     "Coverage-U in open terrain and switches to the angular signal when the "
     "local window is mostly blocked: both methods share the same "
     "Frontier-Bounded movement frame, so the switch is a threshold on the "
     "local blocked fraction (e.g. free_count_FOV / FOV_area < 0.5, exactly "
     "the free-count statistic introduced in Section 4.4.1) rather than a new "
     "planner. Because Richness-Angular is the only method whose advantage "
     "survives 20% obstacle density and Coverage-U the stable, cheaper "
     "continuous weighting in open terrain (Section 10), the hybrid is the "
     "natural composition of the two positive results — a fallback, not a "
     "third method.")
body("Finally, two steps move the evaluation off the synthetic grid. Real "
     "terrain: digital elevation models (MarsTrek, airborne LiDAR) would "
     "replace the random-obstacle maps with structured relief, testing the "
     "dense-regime claims under topography rather than uniform density. Real "
     "hardware: the measured per-decision cost of Coverage-U (2.17 ms, "
     "Section 6.6) fits the budget of an embedded flight controller, and a "
     "mini-drone fleet (e.g. Ryze Tello-class vehicles) or a Raspberry Pi "
     "4B-class onboard processor is the concrete platform for a "
     "communication-limited outdoor trial. These are deliberately framed as "
     "validation steps — the claims of this paper are sim-based, and the "
     "paper says so (Section 8).")

# ======================================================================
# 10. CONCLUSION
# ======================================================================
sec("10. Conclusion")
body("We asked whether a statistical-richness signal transposed to angular "
      "observation configurations can operate as a decision lever for "
      "multi-UAV bearing-only localization, evaluated honestly against a "
      "matched receding-horizon geometric control and a preregistered "
      "protocol. Three levers fail and one succeeds. Richness as target "
      "selection and as mode switching is falsified; continuous "
      "U-prioritized coverage (Coverage-U) reduces the residual oracle CRLB "
      "bound by a median 20.9% at equal coverage under finite mission "
      "budgets, robustly across λ, in sparse-obstacle environments. The "
       "centralized perfect oracle regresses both metrics, and the decisive "
       "comparison is the local-vs-global ladder: the same config-count "
       "signal gains ~+30% when scored locally and regresses when the same "
       "under-set is fused globally. E5-CORRECTED confirms the conclusion "
       "with the movement frame held identical: even in Coverage-U’s own "
        "frame the global under-set signal regresses both metrics, so the "
        "effect is the signal’s locality, not the oracle’s frame. "
        "<b>Locality</b> "
        "is therefore the property that "
      "makes the trade-off attainable, not a limitation of the "
      "decentralized setting. The "
      "practical reading is sharp: when mission time is the scarce resource, "
      "spend the remaining coverage on cells that are angularly "
      "under-determined — and a cheap local singleton/doubleton count is "
      "sufficient to decide where. The two positive signals should be read "
       "as a trade-off rather than competing winners: Richness-Angular "
       "delivers the highest peak accuracy and is the most robust accuracy "
       "driver under spatial fragmentation — the only method whose edge "
       "survives 20% obstacle density — but as a raw target-selection signal "
       "it fails the "
       "preregistered primary and is prone to oscillation; Coverage-U "
      "provides a stable, equally cheap continuous weighting that is "
      "optimal in sparse environments and robust across its plateau. "
       "The universal takeaway is not which signal wins, but the "
       "<b>locality</b> "
       "of the decision signal: the same under-set gains ~+30% scored "
       "locally and regresses fused globally. The post-submission "
       "topology dossier (Section 6.11) sharpens the boundary: the "
       "advantage survives line-of-sight occlusion when routing choice is "
       "preserved (cluster topologies, reproduced at 3 and 6 UAVs) and "
       "flips negative only under single-path routing (perfect maze) — "
       "routing redundancy, not spatial fragmentation per se, is the "
       "condition for the trade-off. The headline claims of the paper "
       "(sparse open space, Sections 6.4–6.10) are unchanged.")

# ======================================================================
# REFERENCES
# ======================================================================
sec("References")
refs = [
    ["Yamauchi1997", "B. Yamauchi, “A frontier-based approach for autonomous "
     "exploration,” in Proc. IEEE Int. Symp. on Computational Intelligence "
     "in Robotics and Automation (CIRA), 1997, pp. 146–151."],
    ["Yamauchi1998", "B. Yamauchi, “Frontier-based exploration using multiple "
     "robots,” in Proc. 2nd Int. Conf. on Autonomous Agents, 1998, pp. 47–53."],
    ["Burgard2005", "W. Burgard, M. Moors, C. Stachniss, and F. Schneider, "
     "“Coordinated multi-robot exploration,” IEEE Trans. Robotics, vol. 21, "
     "no. 3, pp. 376–386, 2005."],
    ["Gonzalez2002", "H. González-Baños and J.-C. Latombe, “Navigation "
     "strategies for exploring indoor environments,” Int. J. Robotics "
     "Research, vol. 21, no. 10–11, pp. 829–848, 2002."],
    ["Franchi2009", "A. Franchi, L. Freda, G. Oriolo, and M. Vendittelli, "
     "“The sensor-based random graph method for cooperative robot "
     "exploration,” IEEE/ASME Trans. Mechatronics, vol. 14, no. 2, "
     "pp. 163–175, 2009."],
    ["BasilicoAmigoni2011", "N. Basilico and F. Amigoni, “Exploration "
     "strategies based on multi-criteria decision making for searching "
     "environments in rescue operations,” Autonomous Robots, vol. 31, "
     "no. 4, pp. 401–417, 2011."],
    ["Bircher2016", "A. Bircher, M. Kamel, K. Alexis, H. Oleynikova, and "
     "R. Siegwart, “Receding horizon ‘next-best-view’ planner for 3D "
     "exploration,” in Proc. IEEE ICRA, 2016, pp. 1462–1468."],
    ["Bircher2018", "A. Bircher, M. Kamel, K. Alexis, H. Oleynikova, and "
      "R. Siegwart, “Receding horizon ‘next-best-view’ planner for 3D "
      "exploration,” IEEE Trans. Robotics, vol. 34, no. 3, pp. 625–634, 2018."],
    ["Zhou2023", "B. Zhou, H. Xu, and S. Shen, “RACER: Rapid collaborative "
      "exploration with a decentralized multi-UAV system,” IEEE Trans. "
      "Robotics, vol. 39, no. 3, pp. 1816–1835, 2023."],
    ["Bourgault2002", "F. Bourgault, A. A. Makarenko, S. B. Williams, "
     "B. Grocholsky, and H. F. Durrant-Whyte, “Information based adaptive "
     "robotic exploration,” in Proc. IEEE/RSJ IROS, 2002, pp. 540–545."],
    ["Stachniss2005", "C. Stachniss, G. Grisetti, and W. Burgard, "
     "“Information gain-based exploration using Rao-Blackwellized particle "
     "filters,” in Proc. Robotics: Science and Systems (RSS), 2005."],
    ["Julian2014", "B. J. Julian, S. Karaman, and D. Rus, “On mutual "
     "information-based control of range sensing robots for mapping "
     "applications,” in Proc. IEEE/RSJ IROS, 2014, pp. 5156–5163."],
    ["Charrow2015", "B. Charrow, G. Kahn, S. Patil, S. Liu, K. Goldberg, "
      "P. Abbeel, N. Michael, and V. Kumar, “Information-theoretic planning "
      "with trajectory optimization for dense 3D mapping,” in Proc. Robotics: "
      "Science and Systems (RSS), 2015."],
    ["Bai2014", "H. Bai, D. Hsu, and W. S. Lee, “Integrated perception and "
      "planning in the continuous space: A POMDP approach,” Int. J. Robotics "
      "Research, vol. 33, no. 9, pp. 1288–1302, 2014."],
    ["Grocholsky2002", "B. Grocholsky, “Information-Theoretic Control of "
     "Multiple Sensor Platforms,” Ph.D. dissertation, Univ. of Sydney, "
     "2002."],
    ["Ponda2012", "S. S. Ponda, L. B. Johnson, A. N. Kopeikin, H.-L. Choi, "
      "and J. P. How, “Distributed planning strategies to enable network-level "
      "cooperation for autonomous systems,” in Proc. ACC, 2012."],
    ["Pongsirijinda2025", "K. Pongsirijinda, Z. Cao, P. L. B. Lau, R. Liu, "
      "and U.-X. Tan, “MEF-Explore: Communication-constrained multi-robot "
      "entropy-field-based exploration,” IEEE Trans. Automation Science and "
      "Engineering, 2025."],
    ["Sagale2024", "A. Sagale, T. Kargar Tasooji, and R. Parasuraman, "
      "“DCL-Sparse: Distributed range-only cooperative localization of "
      "multi-robots in noisy and sparse sensing graphs,” arXiv:2412.14793, "
      "2024."],
    ["Liu2026", "H. Liu, W. Jiang, Q. Long, Q. Xia, and X. Chen, "
      "“A high-precision cooperative localization method for UAVs based on "
      "multi-condition constraints,” Sensors, vol. 26, no. 5, art. 1641, "
      "2026."],
    ["Li2026", "D. Li, Y. Wang, Z. Li, L. Zhang, J. Luo, Y. Yu, and J. Cheng, "
      "“Formation-constrained cooperative localization for UAV swarms in "
      "GNSS-denied environments,” Sensors, vol. 26, no. 6, art. 1984, "
      "2026."],
    ["Ruan2022", "J. Ruan, S. Li, Y. Dai, Y. Tian, Q. Fan, C. Wang, and "
      "W. Dai, “Cooperative relative localization for UAV swarm in "
      "GNSS-denied environment based on coalition formation game,” IEEE "
      "Internet of Things Journal, vol. 9, no. 13, pp. 11560–11577, 2022."],
    ["Woosley2021", "B. Woosley, C. Nieto-Granda, J. Rogers, N. Fung, and "
      "A. Schang, “Bid prediction for multi-robot exploration with disrupted "
      "communications,” Proc. IEEE Int. Symp. Safety, Security, and Rescue "
      "Robotics (SSRR), pp. 210–216, 2021."],
    ["Batinovic2020", "A. Batinović, J. Oršulić, T. Petrović, and S. Bogdan, "
      "“Decentralized strategy for cooperative multi-robot exploration and "
      "mapping,” IFAC-PapersOnLine, vol. 53, no. 2, pp. 9682–9687, 2020."],
    ["Chen2020", "M. Chen, Z. Xiong, J. Liu, R. Wang, and J. Xiong, "
      "“Cooperative navigation of unmanned aerial vehicle swarm based on "
      "cooperative dilution of precision,” Int. J. Advanced Robotic Systems, "
      "vol. 17, no. 3, 2020."],
    ["Lauri2023", "M. Lauri, P. Krusi, A. Farinelli, and W. Burgard, "
     "“Active perception and exploration in multi-robot systems: A survey,” "
     "IEEE/CAA J. Automatica Sinica, vol. 10, no. 2, pp. 307–332, 2023."],
    ["Yarlagadda2000", "R. Yarlagadda, I. Ali, N. Al-Dhahir, and J. Hershey, "
     "“GPS GDOP metric,” IEE Proc. Radar, Sonar and Navigation, vol. 147, "
     "no. 5, pp. 259–264, 2000."],
    ["Kaplan2017", "E. D. Kaplan and C. J. Hegarty, Understanding GPS/GNSS: "
     "Principles and Applications, 3rd ed. Artech House, 2017."],
    ["Ucinski2005", "D. Uciński, Optimal Measurement Methods for Distributed "
     "Parameter System Identification. CRC Press, 2005."],
    ["Martinez2006", "S. Martínez and F. Bullo, “Optimal sensor "
     "placement and motion coordination for target tracking,” Automatica, "
     "vol. 42, no. 4, pp. 661–668, 2006."],
    ["Krause2008", "A. Krause, A. Singh, and C. Guestrin, “Near-optimal "
     "sensor placements in Gaussian processes: Theory, efficient algorithms "
     "and empirical studies,” J. Machine Learning Research, vol. 9, "
     "pp. 235–284, 2008."],
    ["Cramér1946", "H. Cramér, Mathematical Methods of Statistics. "
     "Princeton Univ. Press, 1946."],
    ["Rao1945", "C. R. Rao, “Information and accuracy attainable in the "
     "estimation of statistical parameters,” Bull. Calcutta Math. Soc., "
     "vol. 37, pp. 81–91, 1945."],
    ["NardoneAidala1981", "S. C. Nardone and V. J. Aidala, “Observability "
     "criteria for bearings-only target motion analysis,” IEEE Trans. "
     "Aerospace and Electronic Systems, vol. 17, no. 2, pp. 162–166, 1981."],
    ["Passerieux1998", "J.-M. Passerieux and D. Van Cappel, “Optimal "
     "observer maneuver for bearings-only tracking,” IEEE Trans. Aerospace "
     "and Electronic Systems, vol. 34, no. 3, pp. 777–788, 1998."],
    ["Oshman1999", "Y. Oshman and P. Davidson, “Optimization of "
     "observer trajectories for bearings-only target localization,” IEEE "
     "Trans. Aerospace and Electronic Systems, vol. 35, no. 3, pp. 892–902, "
     "1999."],
    ["Dogancay2012", "K. Doğançay, “UAV path planning for passive emitter "
     "localization,” IEEE Trans. Aerospace and Electronic Systems, vol. 48, "
     "no. 2, pp. 1150–1166, 2012."],
    ["Bishop2004", "A. N. Bishop, B. Fidan, B. D. O. Anderson, "
     "K. Doğançay, and P. N. Pathirana, “Optimality analysis of "
     "sensor-target localization geometries,” Automatica, vol. 40, no. 4, "
     "pp. 677–687, 2004."],
    ["Bishop2009", "A. N. Bishop, B. D. O. Anderson, B. Fidan, P. N. "
     "Pathirana, and G. Mao, “Bearing-only localization using geometrically "
     "constrained optimization,” IEEE Trans. Aerospace and Electronic "
     "Systems, vol. 45, no. 1, pp. 308–320, 2009."],
    ["Patwari2005", "N. Patwari, J. N. Ash, S. Kyperountas, A. O. Hero, "
     "R. L. Moses, and N. S. Correal, “Locating the nodes: Cooperative "
     "localization in wireless sensor networks,” IEEE Signal Processing "
     "Magazine, vol. 22, no. 4, pp. 54–69, 2005."],
    ["Wymeersch2009", "H. Wymeersch, J. Lien, and M. Z. Win, “Cooperative "
     "localization in wireless networks,” Proc. IEEE, vol. 97, no. 2, "
     "pp. 427–450, 2009."],
    ["Ristic2004", "B. Ristic, S. Arulampalam, and N. Gordon, Beyond the "
     "Kalman Filter: Particle Filters for Tracking Applications. Artech "
     "House, 2004."],
    ["Chao1984", "A. Chao, “Nonparametric estimation of the number of "
     "classes in a population,” Scandinavian J. Statistics, vol. 11, no. 4, "
     "pp. 265–270, 1984."],
    ["ChaoLee1992", "A. Chao and S.-M. Lee, “Estimating the number of "
     "classes via sample coverage,” J. American Statistical Association, "
     "vol. 87, no. 417, pp. 210–217, 1992."],
    ["ChaoYang1993", "A. Chao and M. C. K. Yang, “Stopping rules and "
     "estimation for recapture debugging with unequal failure rates,” "
     "Biometrika, vol. 80, no. 1, pp. 193–201, 1993."],
    ["BurnhamOverton1978", "K. P. Burnham and W. S. Overton, “Estimation of "
     "the size of a closed population when capture probabilities vary among "
     "animals,” Biometrika, vol. 65, no. 3, pp. 625–633, 1978."],
]
for key, text in refs:
    story.append(P(f"[{key}] {text}", "ref"))

if __name__ == "__main__":
    build()
    print("WROTE", OUT_PDF)
