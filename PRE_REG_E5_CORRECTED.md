# PRE-REGISTRATION — E5-CORRECTED : local-frame oracle vs Coverage-U

STATUS : PRE-REGISTERED 2026-08-06
CAMPAIGN : `CentralOracle-CRLB-local` / `CentralOracle-Config-local`
SEEDS : paired 0..39 (same env seeds as all previous campaigns)
RUNS : n = 40 per (method, regime)

## Why this correction exists

E5 originally compared Coverage-U (LOCAL bounded_bfs movement frame, LOCAL
under-set signal) against CentralOracle-CRLB/Config (GLOBAL movement frame,
GLOBAL under-set signal). Two confounds were discovered and instrumented:

1. **The coverage guard never bound.** In the global frame the oracle's
   reachable targets are always far (D >= 5, min D median = 6, max bonus 55
   vs the > 127 the cap needs at D = 6). 0 bindings across 4,800 decisions.
   The `-cov`/`-cov2` variants are bit-identical to the unguarded oracle
   (p = 1.0), so E5-DIAG cannot adjudicate calibration-vs-structure
   (verdict: INCONCLUSIVE).
2. **The movement frame itself is confounded.** The oracle's global
   `_global_bfs` excludes every cell visited by ANY agent from the path. The
   reachable set therefore has no D <= 2 target, and pure-coverage motion is
   slower than the local frame: oracle lambda=0 (global frame) covers 20.0%
   vs Frontier-Bounded 29.6% at 1200 steps on seed 0. So E5's FAIL conflates
   the frame artifact with the signal question.

E5-CORRECTED removes confound 2 (the guard is abandoned — it cannot bind in
any oracle frame and tuning it would test a different policy). Both new
oracle rows use the **byte-identical local `bounded_bfs` movement frame of
Coverage-U / Frontier-Bounded**; only the under-set SIGNAL is global
(perfect fusion). The remaining comparison is therefore exactly:

> local signal  vs  global signal, same movement frame, same scoring form.

## Design

- `frame="local"` added to `CentralOraclePolicy` (policies/central_oracle.py):
  reachable set from `bounded_bfs(env, agent_id)` (identical to Coverage-U),
  target mask = the agent's LOCAL unknown cells, movement actions from
  `curdir` exactly as Coverage-U. Fallback explore unchanged.
- `CentralOracle-CRLB-local` : under-set = global CRLB bound > 1.5 cells.
- `CentralOracle-Config-local` : under-set = global config count <= 1
  (perfect-fusion version of the Coverage-U signal).
- lambda = 0.5 (same CENTRAL_ORACLE_LAMBDA), horizon 8, fov 5 — identical to
  the original oracle rows.

## Regimes and budgets (unchanged from E4/E5)

| regime | agents | obstacles | budget |
|--------|--------|-----------|--------|
| A3_obs005 | 3 | 0.05 | 3200 |
| A6_obs005 | 6 | 0.05 | 1600 |

Paired same env seeds as all prior runs (runs 0..39). Raw files land in
results/budget_A3_obs005 and results/budget_A6_obs005 as
raw_comm_limited__CentralOracle-{CRLB,Config}-local.csv.

## RESULT (filled in after data)

Decision: **B — local wins anyway.** With the movement frame held byte-identical
(local bounded_bfs), the LOCAL Coverage-U signal still decisively beats BOTH
global-signal oracles on the primary accuracy metric AND on coverage.

Paired Wilcoxon (n=40, one row per regime):

| A3_obs005 | CU vs CRLB-local | CU vs Config-local | CU vs FB |
|---|---|---|---|
| mean_bound_final | p=1.8e-12 | p=1.8e-12 | p=9.6e-6 |
| final_coverage | p=1.8e-12 | — | p=0.75 (ns) |
| quality_auc | p=0.087 (ns) | — | p=0.98 (ns) |

| A6_obs005 | CU vs CRLB-local | CU vs Config-local | CU vs FB |
|---|---|---|---|
| mean_bound_final | p=9.1e-12 | — | p=6.4e-6 |
| final_coverage | p=1.8e-12 | — | p=0.94 (ns) |
| quality_auc | p=0.17 (ns) | — | p=0.11 (ns) |

Median mean_bound_final (lower better):

| method | A3 | A6 |
|---|---|---|
| Coverage-U | 0.018 | 0.018 |
| CRLB global frame | 0.044 | 0.049 |
| CRLB local frame | 0.037 | 0.032 |
| Config local frame | 0.035 | 0.031 |
| Frontier-Bounded | 0.022 | 0.024 |

Decomposition of the E5 gap:

1. **Frame effect** (CRLB global 0.044 -> CRLB local 0.037 in A3; 0.049 ->
   0.032 in A6): real but small, p<1e-11. The global movement frame was a
   genuine confound and is now removed.
2. **Signal effect (dominant)**: even with the frame identical, the local
   signal beats the perfect-fusion global under-set by ~2x on mean_bound
   (0.037 vs 0.018 A3; 0.032 vs 0.018 A6) and on coverage (43.7% vs 72.0% A3;
   41.9% vs 68.7% A6). The global under-set is dense across the whole map, so
   the bonus term dominates distance and agents over-chase far accuracy
   targets instead of discovering. The local proxy under-counts (each agent
   only sees its own observations) which keeps the coverage term in the
   tradeoff.

VERDICT: the qualitative conclusion of E5 SURVIVES the confound fix. What
matters is the LOCALITY of the signal, not its strength. A perfect global
under-set oracle loses to the decentralized local proxy even when movement is
matched.

## Primary comparison and decision rules

Primary accuracy metric : `mean_bound_final` (lower is better), exactly as in
E4/E5. Secondary : `quality_auc`, `final_coverage`, `undetermined_final`.

1. Local-frame oracle vs Coverage-U (Holm-corrected Wilcoxon, n = 40, one
   row per regime × signal mode). Report reduction in mean_bound vs FB.
2. CRLB-local vs Config-local (which global signal is stronger, if either).
3. CRLB-local vs the ORIGINAL global-frame CRLB oracle (frame effect, same
   signal).
4. Coverage-U vs FB (control replication, unchanged from E4).

Decision text — three possible outcomes (written before looking at data):

- **A (signal helps)**: a local-frame oracle beats Coverage-U on mean_bound
  without collapsing coverage. Then E5's original FAIL is fully explained by
  the frame artifact and the honest headline is "local proxies capture most
  of the global-signal value".
- **B (local wins anyway)**: even with an identical frame, the LOCAL signal
  (discovery + local config-count) yields equal-or-better accuracy/coverage
  than the GLOBAL under-set. Then E5's qualitative conclusion ("the locality
  of the proxy, not its strength, is what matters") SURVIVES, now on a
  non-confounded comparison.
- **C (tradeoff)**: global signal improves mean_bound at the cost of
  coverage, or vice versa. Report the Pareto honestly.

The E5 section in the paper and the PRE_REG_E5_DIAG.md verdict are updated to
cite THIS campaign as the fair comparison; the earlier ladder remains but is
explicitly labeled as frame-confounded.
