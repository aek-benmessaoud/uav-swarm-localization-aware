"""
analysis/fig_paper.py — quantitative figures for the paper (ver0).

Reads the raw campaign CSVs and renders publication-style PNGs into
results/figures/. Figures:

  fig_paper_phase1.png    E1: quality_auc medians (Random / FB / Richness-Angular)
  fig_paper_e3.png        E3: steps_dual medians (FB vs Deploy-U), 4 regimes
  fig_paper_e4confirm.png E4-CONFIRM: mean_bound_final @ T (FB vs Coverage-U)
  fig_paper_coverage.png  E4-CONFIRM: final_coverage @ T guard (FB vs CU)
  fig_paper_undetermined.png E4-CONFIRM: undetermined_final @ T
  fig_paper_lambda.png    E4-PARETO: rel-reduction vs lambda (A3, A6)

Run:  python analysis/fig_paper.py
"""

import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

RESULTS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results")
FIG = os.path.join(RESULTS, "figures")

FB_C = "#2c7bb6"
CU_C = "#d7191c"
DU_C = "#fdae61"
RA_C = "#2ca02c"
RD_C = "#999999"
GRAY = "#4d4d4d"

REGIMES = ["A2_obs005", "A3_obs005", "A6_obs005", "A6_obs020"]
REGIME_LABEL = {"A2_obs005": "A2 (2 UAVs)", "A3_obs005": "A3 (3 UAVs)",
                "A6_obs005": "A6 (6 UAVs)", "A6_obs020": "A6 20% obs"}


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


def style_ax(ax, xlabel, ylabel):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRAY)
    ax.tick_params(colors=GRAY, labelsize=9)
    ax.set_xlabel(xlabel, fontsize=10, color=GRAY)
    ax.set_ylabel(ylabel, fontsize=10, color=GRAY)
    ax.grid(axis="y", alpha=0.3, lw=0.6)


def group_bars(ax, labels, a, b, sig_idx, a_name, b_name, a_col, b_col,
               yscale=None):
    a_med = [m[0] for m in a]
    b_med = [m[0] for m in b]
    a_err = np.array([[m[0] - m[1], m[2] - m[0]] for m in a]).T
    b_err = np.array([[m[0] - m[1], m[2] - m[0]] for m in b]).T
    x = np.arange(len(labels))
    w = 0.36
    ax.bar(x - w / 2, a_med, w, yerr=a_err, capsize=3, color=a_col,
           alpha=0.9, label=a_name, edgecolor="white")
    ax.bar(x + w / 2, b_med, w, yerr=b_err, capsize=3, color=b_col,
           alpha=0.9, label=b_name, edgecolor="white")
    if yscale == "log":
        top = [max(a, b) for a, b in zip(a_med, b_med)]
        for i in sig_idx:
            ax.text(x[i], top[i] * 1.5, "*", ha="center", fontsize=13,
                    color=GRAY)
    else:
        top = [max(a, b) for a, b in zip(a_med, b_med)]
        for i in sig_idx:
            ax.text(x[i], top[i] * 1.05, "*", ha="center", fontsize=13,
                    color=GRAY)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    if yscale:
        ax.set_yscale(yscale)


def fig_phase1():
    d = os.path.join(RESULTS, "phase1_S1_fov5")
    names = ["Random", "Frontier-Bounded", "Richness-Angular"]
    cols = [RD_C, FB_C, RA_C]
    med, err = [], []
    for m in names:
        mm = med_iqr(vec(load(d, m), "quality_auc"))
        med.append(mm[0])
        err.append([mm[0] - mm[1], mm[2] - mm[0]])
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(range(3), med, 0.5, color=cols, alpha=0.9, edgecolor="white",
           yerr=np.array(err).T, capsize=3)
    fb = vec(load(d, "Frontier-Bounded"), "quality_auc")
    rd = vec(load(d, "Random"), "quality_auc")
    ra = vec(load(d, "Richness-Angular"), "quality_auc")
    p_fb_rd = wilcox(fb, rd)
    p_ra_fb = wilcox(ra, fb)
    ax.text(0, med[0] + 0.02, r"$\dagger$", ha="center", fontsize=12,
            color=GRAY)
    ax.text(1, med[1] + 0.02, r"$\dagger$", ha="center", fontsize=12,
            color=GRAY)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Random", "Frontier-\nBounded", "Richness-\nAngular"],
                       fontsize=9)
    ax.set_ylim(0.4, 1.0)
    style_ax(ax, "", "quality AUC (E1)")
    ax.set_title("Phase-1 (E1): localization-quality A/B, 6 UAVs, FOV 5",
                 fontsize=10, color=GRAY)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_paper_phase1.png"), dpi=200)
    plt.close(fig)
    print(f"[fig] phase1  (FB vs Random p={p_fb_rd:.4f}, "
          f"RA vs FB p={p_ra_fb:.4f})")


def _pair(dirname, method):
    d = os.path.join(RESULTS, dirname)
    a = load(d, method)
    return d, a


def fig_e3():
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    labels = [REGIME_LABEL[r] for r in REGIMES]
    fb = [med_iqr(vec(load(os.path.join(RESULTS, f"deploy_{r}"),
                           "Frontier-Bounded"), "steps_dual")) for r in REGIMES]
    du = [med_iqr(vec(load(os.path.join(RESULTS, f"deploy_{r}"),
                           "Deploy-U"), "steps_dual")) for r in REGIMES]
    group_bars(ax, labels, fb, du, [],
               "Frontier-Bounded", "Deploy-U", FB_C, DU_C, yscale="log")
    style_ax(ax, "", "steps to dual goal (cov≥90% AND quality≥0.9)")
    ax.set_title("E2/E3: localization-aware deployment (Deploy-U) vs FB control",
                 fontsize=10, color=GRAY)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_paper_e3.png"), dpi=200)
    plt.close(fig)
    print("[fig] e3")


def _budget_pair(rg, metric):
    d = os.path.join(RESULTS, f"budget_{rg}")
    return (med_iqr(vec(load(d, "Frontier-Bounded"), metric)),
            med_iqr(vec(load(d, "Coverage-U"), metric)),
            wilcox(vec(load(d, "Frontier-Bounded"), metric),
                   vec(load(d, "Coverage-U"), metric)))


def fig_e4confirm():
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    labels = [REGIME_LABEL[r] for r in REGIMES]
    fb = [_budget_pair(r, "mean_bound_final")[0] for r in REGIMES]
    cu = [_budget_pair(r, "mean_bound_final")[1] for r in REGIMES]
    sig = [i for i, r in enumerate(REGIMES)
           if _budget_pair(r, "mean_bound_final")[2] < 0.05]
    group_bars(ax, labels, fb, cu, sig,
               "Frontier-Bounded", "Coverage-U (λ=0.5)", FB_C, CU_C)
    style_ax(ax, "", "mean residual CRLB bound @ T")
    ax.set_title("E4-CONFIRM: residual localization error at fixed budget "
                 "(n=40 paired, *)", fontsize=10, color=GRAY)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_paper_e4confirm.png"), dpi=200)
    plt.close(fig)
    print("[fig] e4confirm")


def fig_coverage():
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    labels = [REGIME_LABEL[r] for r in REGIMES]
    fb = [_budget_pair(r, "final_coverage")[0] for r in REGIMES]
    cu = [_budget_pair(r, "final_coverage")[1] for r in REGIMES]
    sig = [i for i, r in enumerate(REGIMES)
           if _budget_pair(r, "final_coverage")[2] < 0.05]
    group_bars(ax, labels, fb, cu, sig,
               "Frontier-Bounded", "Coverage-U (λ=0.5)", FB_C, CU_C)
    style_ax(ax, "", "final coverage @ T (%)")
    ax.set_title("E4-CONFIRM: coverage guard — no significant regression",
                 fontsize=10, color=GRAY)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_paper_coverage.png"), dpi=200)
    plt.close(fig)
    print("[fig] coverage")


def fig_undetermined():
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    labels = [REGIME_LABEL[r] for r in REGIMES]
    fb = [_budget_pair(r, "undetermined_final")[0] for r in REGIMES]
    cu = [_budget_pair(r, "undetermined_final")[1] for r in REGIMES]
    sig = [i for i, r in enumerate(REGIMES)
           if _budget_pair(r, "undetermined_final")[2] < 0.05]
    group_bars(ax, labels, fb, cu, sig,
               "Frontier-Bounded", "Coverage-U (λ=0.5)", FB_C, CU_C,
               yscale="log")
    style_ax(ax, "", "unobserved-cell fraction @ T")
    ax.set_title("E4-CONFIRM: corroborating signal (cells never observed)",
                 fontsize=10, color=GRAY)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_paper_undetermined.png"), dpi=200)
    plt.close(fig)
    print("[fig] undetermined")


def fig_lambda():
    lams = [0.25, 0.5, 1.0, 2.0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for ax, rg in ((ax1, "A3_obs005"), (ax2, "A6_obs005")):
        d = os.path.join(RESULTS, f"budget_{rg}")
        p = os.path.join(RESULTS, f"pareto_{rg}")
        fb = vec(load(d, "Frontier-Bounded"), "mean_bound_final")
        red, cov = [], []
        for lam in lams:
            if lam == 0.5:
                cu_rows = load(d, "Coverage-U")
            else:
                cu_rows = load(p, "Coverage-U", tag=f"lam{lam:g}")
            cu = vec(cu_rows, "mean_bound_final")
            red.append(100.0 * (np.median(fb) - np.median(cu)) / np.median(fb))
            cov.append(med_iqr(vec(cu_rows, "final_coverage"))[0])
        ax.plot([0] + lams, [0.0] + red, "-o", color=CU_C, ms=5, lw=1.6)
        ax.axhline(10, ls="--", color=GRAY, lw=1, alpha=0.6)
        ax.text(0.03, 11.0, "prereg. +10% threshold", fontsize=7, color=GRAY)
        for x, y, c in zip(lams, red[1:], cov[1:]):
            ax.annotate(f"cov {c:.0f}%", (x, y), textcoords="offset points",
                        xytext=(8, 2), fontsize=7, color=GRAY)
        ax.set_xticks([0] + lams)
        ax.set_xticklabels([0.0, 0.25, 0.5, 1.0, 2.0], fontsize=8)
        style_ax(ax, "λ", "mean-bound reduction vs FB (%)")
        ax.set_title(REGIME_LABEL[rg], fontsize=10, color=GRAY)
        ax.set_ylim(0, 35)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_paper_lambda.png"), dpi=200)
    plt.close(fig)
    print("[fig] lambda")


# ----------------------------------------------------------------------
# Robustness sweeps (FUTURE_WORK_NOTES phases B/C/D), n = 40 paired each.
# ----------------------------------------------------------------------
def _sweep_pair(rg, tag, metric, method="Coverage-U"):
    """CU vs FB rows both under the same variant `tag` (None = confirmed base)."""
    d = os.path.join(RESULTS, f"budget_{rg}")
    base = os.path.join(RESULTS, f"budget_{rg}__{tag}" if tag else d)
    cu = load(base, method, tag=tag)
    fb = load(base, "Frontier-Bounded", tag=tag)
    return (med_iqr(vec(fb, metric)), med_iqr(vec(cu, metric)),
            wilcox(vec(fb, metric), vec(cu, metric)))


def fig_sweep_sigma():
    """CU vs FB mean_bound_final at sigma_loc in {0, 0.5, 1.0}."""
    tags = [None, "s05", "s10"]
    labels = ["σ = 0", "σ = 0.5", "σ = 1.0"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for ax, rg in ((ax1, "A3_obs005"), (ax2, "A6_obs005")):
        fb_med, cu_med, sig = [], [], []
        for t in tags:
            mf, mc, p = _sweep_pair(rg, t, "mean_bound_final")
            fb_med.append(mf[0]); cu_med.append(mc[0])
            sig.append(p < 0.05)
        x = np.arange(len(labels))
        w = 0.34
        ax.bar(x - w / 2, fb_med, w, color=FB_C, alpha=0.9, label="FB",
               edgecolor="white")
        ax.bar(x + w / 2, cu_med, w, color=CU_C, alpha=0.9,
               label="Coverage-U", edgecolor="white")
        for i, s in enumerate(sig):
            if s:
                ax.text(x[i], max(fb_med[i], cu_med[i]) * 1.05, "*",
                        ha="center", fontsize=13, color=GRAY)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.legend(fontsize=8, frameon=False)
        style_ax(ax, "self-loc. noise σ (cells)", "mean residual CRLB bound @ T")
        ax.set_title(REGIME_LABEL[rg], fontsize=10, color=GRAY)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_paper_sweep_sigma.png"), dpi=200)
    plt.close(fig)
    print("[fig] sweep_sigma")


def fig_sweep_budget():
    """CU rel-red% vs FB across budget fractions {0.3, 0.5, 0.7, 0.9}."""
    fracs = [0.3, 0.5, 0.7, 0.9]
    tags = ["b03", "b05", None, "b09"]
    cols = {"A3_obs005": FB_C, "A6_obs005": CU_C}
    xoff = {"A3_obs005": 6, "A6_obs005": -6}
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 3.9))
    for rg, c in cols.items():
        d = os.path.join(RESULTS, f"budget_{rg}")
        red, cov, sig = [], [], []
        for t in tags:
            base = os.path.join(RESULTS, f"budget_{rg}__{t}" if t else d)
            fb = vec(load(base, "Frontier-Bounded", tag=t), "mean_bound_final")
            cu = vec(load(base, "Coverage-U", tag=t), "mean_bound_final")
            red.append(100.0 * (np.median(fb) - np.median(cu)) / np.median(fb))
            cov.append(med_iqr(vec(load(base, "Coverage-U", tag=t),
                                   "final_coverage"))[0])
            sig.append(wilcox(cu, fb) < 0.05)
        ax.plot(fracs, red, "-o", color=c, ms=5, lw=1.7,
                label=REGIME_LABEL[rg])
        for i, s in enumerate(sig):
            if s:
                ax.annotate("*", (fracs[i], red[i]),
                            textcoords="offset points", xytext=(0, 7),
                            ha="center", fontsize=12, color=GRAY)
        for x, y, cc in zip(fracs, red, cov):
            ax.annotate(f"cov {cc:.0f}%", (x, y), textcoords="offset points",
                        xytext=(xoff[rg], -11), fontsize=7, color=GRAY)
    ax.axhline(10, ls="--", color=GRAY, lw=1, alpha=0.6)
    ax.text(0.31, 11.5, "prereg. +10% threshold", fontsize=7, color=GRAY)
    ax.set_xticks(fracs)
    ax.set_xticklabels([f"{f:.1f}" for f in fracs], fontsize=8)
    ax.set_ylim(0, 45)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    style_ax(ax, "budget T (× FB steps_90)", "mean-bound reduction vs FB (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_paper_sweep_budget.png"), dpi=200)
    plt.close(fig)
    print("[fig] sweep_budget")


def fig_sweep_comm():
    """CU vs FB mean_bound_final at comm_range in {1.25, 2.5, 5.0}."""
    tags = ["r125", "r25", None]
    labels = ["R = 1.25", "R = 2.5", "R = 5 (FOV)"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for ax, rg in ((ax1, "A3_obs005"), (ax2, "A6_obs005")):
        fb_med, cu_med, sig = [], [], []
        for t in tags:
            mf, mc, p = _sweep_pair(rg, t, "mean_bound_final")
            fb_med.append(mf[0]); cu_med.append(mc[0])
            sig.append(p < 0.05)
        x = np.arange(len(labels))
        w = 0.34
        ax.bar(x - w / 2, fb_med, w, color=FB_C, alpha=0.9, label="FB",
               edgecolor="white")
        ax.bar(x + w / 2, cu_med, w, color=CU_C, alpha=0.9,
               label="Coverage-U", edgecolor="white")
        for i, s in enumerate(sig):
            if s:
                ax.text(x[i], max(fb_med[i], cu_med[i]) * 1.05, "*",
                        ha="center", fontsize=13, color=GRAY)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.legend(fontsize=8, frameon=False)
        style_ax(ax, "fusion range R (cells)", "mean residual CRLB bound @ T")
        ax.set_title(REGIME_LABEL[rg], fontsize=10, color=GRAY)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_paper_sweep_comm.png"), dpi=200)
    plt.close(fig)
    print("[fig] sweep_comm")


def fig_sweep_bearing():
    """CU vs FB mean_bound_final at bearing noise in {0, 2} degrees."""
    tags = [None, "be2"]
    labels = ["σθ = 0", "σθ = 2°"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for ax, rg in ((ax1, "A3_obs005"), (ax2, "A6_obs005")):
        fb_med, cu_med, sig = [], [], []
        for t in tags:
            mf, mc, p = _sweep_pair(rg, t, "mean_bound_final")
            fb_med.append(mf[0]); cu_med.append(mc[0])
            sig.append(p < 0.05)
        x = np.arange(len(labels))
        w = 0.34
        ax.bar(x - w / 2, fb_med, w, color=FB_C, alpha=0.9, label="FB",
               edgecolor="white")
        ax.bar(x + w / 2, cu_med, w, color=CU_C, alpha=0.9,
               label="Coverage-U", edgecolor="white")
        for i, s in enumerate(sig):
            if s:
                ax.text(x[i], max(fb_med[i], cu_med[i]) * 1.05, "*",
                        ha="center", fontsize=13, color=GRAY)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.legend(fontsize=8, frameon=False)
        style_ax(ax, "bearing noise σθ (deg)", "mean residual CRLB bound @ T")
        ax.set_title(REGIME_LABEL[rg], fontsize=10, color=GRAY)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_paper_sweep_bearing.png"), dpi=200)
    plt.close(fig)
    print("[fig] sweep_bearing")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    fig_phase1()
    fig_e3()
    fig_e4confirm()
    fig_coverage()
    fig_undetermined()
    fig_lambda()
    fig_sweep_sigma()
    fig_sweep_budget()
    fig_sweep_comm()
    fig_sweep_bearing()
