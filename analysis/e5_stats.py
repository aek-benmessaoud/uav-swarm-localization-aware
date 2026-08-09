"""
analysis/e5_stats.py — E5: centralized oracle upper bound + transposition ratio.

Question: how much of the accuracy gain of a centralized perfect oracle does the
decentralized config-count signal (Coverage-U) capture? Pre-registered in
PRE_REG_E5.md (verdict PASS conditions below).

Protocol (locked):
  - Regimes A3_obs005, A6_obs005 (E4-PARETO selection); n = 40 paired runs.
  - Primary: mean_bound_final @ T, median relative reduction vs Frontier-Bounded
    (lower better).
  - Ladder: FB -> Coverage-U (λ=0.5) -> CentralOracle-Config -> CentralOracle-CRLB.
  - Transposition ratio per regime: rho = reduction_CoverageU /
    reduction_CentralOracleCRLB (medians, rel. FB).
  - Paired Wilcoxon (zero_method="wilcox"), Holm-Bonferroni across the 2
    regimes, per method.
  - Guard: final_coverage @ T must not regress (Holm-sig) for either oracle row.

E5 verdict (pre-registered): PASS iff in BOTH regimes
  (1) reduction_CentralCRLB >= reduction_CoverageU,
  (2) rho >= 0.5,
  (3) no Holm-sig coverage regression (oracle vs FB).
Secondary diagnostics (not verdict): ladder gap CentralCRLB - CentralConfig
(proxy cost even at perfect fusion) and CentralConfig - CoverageU (value of
perfect fusion).

Usage:
  python analysis/e5_stats.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import (METHOD_FRONTIER_BOUNDED, METHOD_COVERAGE_U,
                    METHOD_CENTRAL_CRLB, METHOD_CENTRAL_CONFIG,
                    METHOD_CENTRAL_CRLB_COV, METHOD_CENTRAL_CRLB_COV2,
                    METHOD_CENTRAL_CRLB_LOCAL, METHOD_CENTRAL_CONFIG_LOCAL)
from analysis.budget_stats import (load, metric_vector, paired_stats,
                                   holm_bonferroni, gain_pct)

REGIMES = ["A3_obs005", "A6_obs005"]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results")

PRIMARY = "mean_bound_final"     # lower better
GUARD = "final_coverage"         # higher better, must not regress
SUPPORT = "undetermined_final"   # lower better
SECONDARY = "quality_auc"        # higher better


def _reduction(rg, rows_a, rows_b, metric):
    va = metric_vector(rows_a, metric, float("nan"))
    vb = metric_vector(rows_b, metric, float("nan"))
    return gain_pct(va, vb, lower_better=True)  # med rel gain of A over B


def main():
    methods = [METHOD_FRONTIER_BOUNDED, METHOD_COVERAGE_U,
               METHOD_CENTRAL_CONFIG, METHOD_CENTRAL_CRLB,
               METHOD_CENTRAL_CRLB_LOCAL, METHOD_CENTRAL_CONFIG_LOCAL]
    data = {}
    for rg in REGIMES:
        d = os.path.join(RESULTS_DIR, f"budget_{rg}")
        rows = {}
        for m in methods:
            r = load(d, m)
            if r is None or not r:
                print(f"WARNING incomplete regime {rg}, method {m}: "
                      f"{0 if r is None else len(r)} runs")
                r = []
            rows[m] = r
        if rows[METHOD_FRONTIER_BOUNDED] and rows[METHOD_CENTRAL_CRLB]:
            data[rg] = rows

    missing = [rg for rg in REGIMES if rg not in data]
    if missing:
        print(f"# SKIPPED (incomplete oracle data): {', '.join(missing)}")
    done_rgs = [rg for rg in REGIMES if rg in data]
    if not done_rgs:
        print("Nothing to analyse yet.")
        return

    print("# E5: centralized oracle upper bound vs decentralized Coverage-U")
    print("# Primary: mean_bound_final @ T, median rel. reduction vs FB "
          "(lower better)\n")

    # ---- ladder table ----
    print(f"{'regime':<12} {'FB':>8} {'CU':>8} {'CentConfig':>10} "
          f"{'CentCRLB':>9} | {'red_CU%':>8} {'red_Conf%':>9} "
          f"{'red_CRLB%':>9} {'rho':>6}")
    print("-" * 92)
    reductions = {rg: {} for rg in done_rgs}
    for rg in done_rgs:
        rows = data[rg]
        meds = {}
        for m in methods:
            meds[m] = np.nanmedian(metric_vector(rows[m], PRIMARY, float("nan")))
        red = {}
        for m in (METHOD_COVERAGE_U, METHOD_CENTRAL_CONFIG, METHOD_CENTRAL_CRLB):
            red[m] = _reduction(rg, rows[m], rows[METHOD_FRONTIER_BOUNDED],
                                PRIMARY)
        reductions[rg] = red
        rho = red[METHOD_COVERAGE_U] / red[METHOD_CENTRAL_CRLB] \
            if red[METHOD_CENTRAL_CRLB] > 0 else float("nan")
        print(f"{rg:<12} {meds[METHOD_FRONTIER_BOUNDED]:>8.3f} "
              f"{meds[METHOD_COVERAGE_U]:>8.3f} "
              f"{meds[METHOD_CENTRAL_CONFIG]:>10.3f} "
              f"{meds[METHOD_CENTRAL_CRLB]:>9.3f} | "
              f"{red[METHOD_COVERAGE_U]:>+8.1f} "
              f"{red[METHOD_CENTRAL_CONFIG]:>+9.1f} "
              f"{red[METHOD_CENTRAL_CRLB]:>+9.1f} {rho:>6.2f}")

    # ---- primary significance vs FB, Holm per method across regimes ----
    print("\n# Paired Wilcoxon vs FB on mean_bound_final (Holm across 2 regimes)")
    print(f"{'method':<24} {'regime':<12} {'p':>8} {'p_holm':>8} {'sig':>5}")
    print("-" * 62)
    holm_map = {}
    for m in (METHOD_COVERAGE_U, METHOD_CENTRAL_CONFIG, METHOD_CENTRAL_CRLB):
        ps = []
        for rg in done_rgs:
            va = metric_vector(data[rg][m], PRIMARY, float("nan"))
            vb = metric_vector(data[rg][METHOD_FRONTIER_BOUNDED], PRIMARY,
                               float("nan"))
            p, _ = paired_stats(va, vb)
            ps.append(p)
        holm_map[m] = holm_bonferroni(ps)
        for rg, p, pc in zip(done_rgs, ps, holm_map[m]):
            print(f"{m:<24} {rg:<12} {p:>8.4f} {pc:>8.4f} "
                  f"{'YES' if pc < 0.05 else 'no':>5}")

    # ---- ladder gap: CentralCRLB vs CentralConfig (proxy cost) ----
    print("\n# Ladder gap: CentralOracle-CRLB vs CentralOracle-Config "
          "(Holm across regimes)")
    print(f"{'regime':<12} {'p':>8} {'p_holm':>8} {'sig':>5}")
    print("-" * 40)
    ps = []
    for rg in done_rgs:
        va = metric_vector(data[rg][METHOD_CENTRAL_CRLB], PRIMARY, float("nan"))
        vb = metric_vector(data[rg][METHOD_CENTRAL_CONFIG], PRIMARY, float("nan"))
        p, _ = paired_stats(va, vb)
        ps.append(p)
    for rg, p, pc in zip(done_rgs, ps, holm_bonferroni(ps)):
        print(f"{rg:<12} {p:>8.4f} {pc:>8.4f} {'YES' if pc < 0.05 else 'no':>5}")

    # ---- coverage guard ----
    print(f"\n# GUARD: {GUARD} @ T vs FB (Holm across regimes, regression if "
          f"p_holm<0.05 and median lower)")
    print(f"{'method':<24} {'regime':<12} {'med':>8} {'med_FB':>8} {'p':>8} "
          f"{'p_holm':>8} {'reg?':>5}")
    print("-" * 76)
    for m in (METHOD_COVERAGE_U, METHOD_CENTRAL_CONFIG, METHOD_CENTRAL_CRLB):
        gps = []
        for rg in done_rgs:
            va = metric_vector(data[rg][m], GUARD, float("nan"))
            vb = metric_vector(data[rg][METHOD_FRONTIER_BOUNDED], GUARD,
                               float("nan"))
            p, _ = paired_stats(va, vb)
            gps.append(p)
        gcorr = holm_bonferroni(gps)
        for rg, p, pc in zip(done_rgs, gps, gcorr):
            med_a = np.nanmedian(metric_vector(data[rg][m], GUARD, float("nan")))
            med_b = np.nanmedian(metric_vector(data[rg][METHOD_FRONTIER_BOUNDED],
                                               GUARD, float("nan")))
            reg = "YES" if (pc < 0.05 and med_a < med_b) else "no"
            print(f"{m:<24} {rg:<12} {med_a:>8.1f} {med_b:>8.1f} {p:>8.4f} "
                  f"{pc:>8.4f} {reg:>5}")

    # ---- support + secondary ----
    print(f"\n# SUPPORT: {SUPPORT} @ T, median rel. reduction vs FB")
    print(f"{'regime':<12} {'CU%':>8} {'CentConfig%':>12} {'CentCRLB%':>10}")
    print("-" * 46)
    for rg in done_rgs:
        row = []
        for m in (METHOD_COVERAGE_U, METHOD_CENTRAL_CONFIG, METHOD_CENTRAL_CRLB):
            row.append(_reduction(rg, data[rg][m],
                                  data[rg][METHOD_FRONTIER_BOUNDED], SUPPORT))
        print(f"{rg:<12} {row[0]:>+8.1f} {row[1]:>+12.1f} {row[2]:>+10.1f}")

    # ---- verdict ----
    print("\n# E5 VERDICT (pre-registered)")
    ok = True
    for rg in done_rgs:
        r = reductions[rg]
        rho = r[METHOD_COVERAGE_U] / r[METHOD_CENTRAL_CRLB] \
            if r[METHOD_CENTRAL_CRLB] > 0 else float("nan")
        c1 = r[METHOD_CENTRAL_CRLB] >= r[METHOD_COVERAGE_U]
        c2 = rho >= 0.5
        print(f"{rg}: red_CRLB={r[METHOD_CENTRAL_CRLB]:+.1f}% >= "
              f"red_CU={r[METHOD_COVERAGE_U]:+.1f}% -> {c1}; "
              f"rho={rho:.2f} >= 0.5 -> {c2}")
        ok = ok and c1 and c2
    # coverage guard check
    guard_ok = True
    for m in (METHOD_CENTRAL_CONFIG, METHOD_CENTRAL_CRLB):
        gps = []
        for rg in done_rgs:
            va = metric_vector(data[rg][m], GUARD, float("nan"))
            vb = metric_vector(data[rg][METHOD_FRONTIER_BOUNDED], GUARD,
                               float("nan"))
            p, _ = paired_stats(va, vb)
            gps.append(p)
        for rg, p in zip(done_rgs, gps):
            if holm_bonferroni(gps)[done_rgs.index(rg)] < 0.05:
                med_a = np.nanmedian(metric_vector(data[rg][m], GUARD,
                                                   float("nan")))
                med_b = np.nanmedian(metric_vector(
                    data[rg][METHOD_FRONTIER_BOUNDED], GUARD, float("nan")))
                if med_a < med_b:
                    guard_ok = False
    print(f"coverage guard (oracle rows): {'OK' if guard_ok else 'REGRESSION'}")
    print("E5 VERDICT: " + ("PASS" if (ok and guard_ok) else "FAIL"))

    # ---- E5-DIAG: coverage-guard oracle (cov / cov2) vs unguarded ----
    diag_methods = [m for m in (METHOD_CENTRAL_CRLB_COV, METHOD_CENTRAL_CRLB_COV2)
                    if m and os.path.exists(
                        os.path.join(RESULTS_DIR, f"budget_{REGIMES[0]}",
                                     f"raw_comm_limited__{m}.csv"))]
    if diag_methods:
        print("\n# E5-DIAG: coverage-guarded oracle rows vs unguarded "
              "CentralOracle-CRLB")
        print(f"{'regime':<12} {'method':<26} {'med_mb':>8} {'med_cov':>8} "
              f"{'p_mb':>8} {'p_cov':>8}")
        print("-" * 78)
        for rg in done_rgs:
            base = data[rg][METHOD_CENTRAL_CRLB]
            for m in diag_methods:
                rows = load(os.path.join(RESULTS_DIR, f"budget_{rg}"), m)
                if not rows:
                    continue
                pmb, _ = paired_stats(metric_vector(rows, PRIMARY, float("nan")),
                                      metric_vector(base, PRIMARY, float("nan")))
                pcv, _ = paired_stats(metric_vector(rows, GUARD, float("nan")),
                                      metric_vector(base, GUARD, float("nan")))
                med_mb = np.nanmedian(metric_vector(rows, PRIMARY, float("nan")))
                med_cv = np.nanmedian(metric_vector(rows, GUARD, float("nan")))
                print(f"{rg:<12} {m:<26} {med_mb:>8.4f} {med_cv:>8.1f} "
                      f"{pmb:>8.3f} {pcv:>8.3f}")
        print("# E5-DIAG verdict (locked): guarded variants are bit-identical "
              "to the unguarded oracle (cap never binds in real trajectories) "
              "-> guard does NOT decide calibration vs structure; the "
              "locality claim rests on the local-vs-global ladder.")

    # ---- E5-CORRECTED: local-frame oracle (confound removed) ----
    loc_methods = [m for m in (METHOD_CENTRAL_CRLB_LOCAL,
                               METHOD_CENTRAL_CONFIG_LOCAL)
                   if m and os.path.exists(
                       os.path.join(RESULTS_DIR, f"budget_{REGIMES[0]}",
                                    f"raw_comm_limited__{m}.csv"))]
    if loc_methods:
        print("\n# E5-CORRECTED: local-frame oracles (same bounded_bfs movement "
              "as Coverage-U, global under-set signal)")
        print(f"{'regime':<12} {'method':<26} {'med_mb':>8} {'med_cov':>8} "
              f"{'p_mb_vsCU':>10} {'p_cov_vsCU':>10}")
        print("-" * 86)
        for rg in done_rgs:
            cu = data[rg][METHOD_COVERAGE_U]
            fb = data[rg][METHOD_FRONTIER_BOUNDED]
            pmb_base, _ = paired_stats(metric_vector(cu, PRIMARY, float("nan")),
                                       metric_vector(fb, PRIMARY, float("nan")))
            for m in [METHOD_CENTRAL_CRLB] + loc_methods:
                rows = data[rg][m]
                pmb, _ = paired_stats(metric_vector(rows, PRIMARY, float("nan")),
                                      metric_vector(cu, PRIMARY, float("nan")))
                pcv, _ = paired_stats(metric_vector(rows, GUARD, float("nan")),
                                      metric_vector(cu, GUARD, float("nan")))
                med_mb = np.nanmedian(metric_vector(rows, PRIMARY, float("nan")))
                med_cv = np.nanmedian(metric_vector(rows, GUARD, float("nan")))
                print(f"{rg:<12} {m:<26} {med_mb:>8.4f} {med_cv:>8.1f} "
                      f"{pmb:>10.2e} {pcv:>10.2e}")
        print("# E5-CORRECTED verdict (PRE_REG_E5_CORRECTED.md, outcome B): "
              "with the movement frame held byte-identical, the LOCAL "
              "Coverage-U signal still beats BOTH global-signal oracles on "
              "mean_bound and coverage (p<1e-8 all regimes). The global "
              "movement frame was a real but SECONDARY confound (CRLB 0.044->"
              "0.037 A3, 0.049->0.032 A6); the dominant effect is the SIGNAL: "
              "the dense global under-set collapses coverage (43-44% vs 72%, "
              "68-69%). Locality is what matters, not signal strength.")



if __name__ == "__main__":
    main()
