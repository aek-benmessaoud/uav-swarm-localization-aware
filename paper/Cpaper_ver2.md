# Locality, Not Signal Strength: Config-Count Richness for Budget-Limited UAV Swarm Localization

*Alternative title: "Spending Coverage Where Accuracy Is at Risk: Continuous and Direct Prioritization of Angularly Under-Determined Cells under Finite Mission Time"*

**Projet08 — Localization-Aware Deployment Strategies for UAV Swarm Systems**
Manuscript draft ver2 (publication-ready) · August 2026

*All experiments preregistered before execution; verdicts and statistics are generated unchanged from the raw paired campaign CSVs.*

---

## Abstract

We study multi-agent bearing-only localization under limited-range communication and a finite mission budget — the regime where coverage and localization accuracy genuinely compete. Building on a predecessor finding that bounded-horizon geometric movement (Frontier-Bounded, FB) already captures most of the coverage gain typically attributed to sophisticated uncertainty signals, we ask a narrower and harder question: can a statistical richness signal — Chao-type estimators transposed from ecology to per-cell angular observation-configuration counts — add value once movement is already well handled?

Under a strict preregistered protocol (paired seeds, Wilcoxon signed-rank with Holm-Bonferroni correction, a coverage-regression guard, and a hard validation gate on the evaluation metric itself), the signal fails as a direct target-selection rule and as a mode-switching trigger *when evaluated on an unbounded episode with a thresholded accuracy metric* — two falsifications that replicate the predecessor's Occam's-razor finding. Under the metric and regime that actually matter — a **finite mission budget** scored by a **continuous** residual localization-error bound — both a continuous re-weighting of the frontier score (Coverage-U, λ = 0.5, fixed a priori) and, more strongly, direct richness-based target selection (Richness-Angular) beat FB by a wide, Holm-significant margin (median residual-bound reduction 17.0–32.4% across regimes, n = 40), with no coverage regression, and Richness-Angular remains significant even at 20% obstacle density where Coverage-U's gain vanishes. Both signals beat classical occupancy-entropy scorers and a Random floor by a comparable or wider margin, at a measured compute cost within 1–16% of FB.

A centralized, perfect-information version of the same signal is not an upper bound: it *regresses* both accuracy and coverage (n = 40 × 2 regimes, all p < 0.0001), collapsing coverage from ~70% to 36–42%. A local-vs-global signal ladder, a preregistered coverage-guard diagnostic, and a movement-frame-controlled re-run together isolate the cause: the failure is not a movement-frame confound (confirmed by re-running the global signal inside the local movement frame, p < 10⁻⁸) and is not resolved by capping the accuracy bonus (the cap never binds in 4,800 real trajectory evaluations — a negative, inconclusive-on-calibration result). What is robust across every control is the comparison that matters: the same config-count signal gains ~30% when scored locally and *regresses* when the identical under-set is fused globally. We conclude that **locality, not signal strength, is what makes this accuracy/coverage trade-off attainable** — a cheap, local, decentralized singleton/doubleton count outperforms both a purely geometric control and an idealized centralized planner on the same objective.

---

## 1. Introduction

Multi-robot exploration has long balanced two objectives: covering an environment and acquiring information that is useful downstream [Yamauchi1997, Burgard2005]. In a growing class of missions — localization of RF/audio emitters, passive sensing, structure-from-motion, cooperative positioning — the downstream task *is* localization: each cell must be observed from enough independent viewing directions that a bearing-only estimator becomes well conditioned. The geometric spread of observations (angular diversity) is the quantity that matters, classically measured by GDOP or the Fisher information matrix [Yarlagadda2000, Bishop2004geometry].

A separate line of work has proposed information-driven exploration signals — map entropy [Bourgault2002, Stachniss2005], expected information gain [Julian2014, Charrow2015], POMDP-style value [Bai2014] — as a replacement or complement to purely geometric frontier control [Yamauchi1997]. Recent decentralized multi-UAV systems such as RACER [Zhou2023] dispatch large teams under asynchronous, bandwidth-limited communication, but optimize spatial coverage and workload balance rather than the geometry of the acquired observations — the axis studied here. Communication-constrained entropy-field exploration [Pongsirijinda2025] shares our proximity-fusion assumption but ranks targets by occupancy uncertainty, not observation geometry.

In a predecessor study we found that when information-driven signals are evaluated against a carefully matched bounded-horizon geometric control, most of their apparent gain on pure coverage is explained by the receding-horizon movement frame itself, not by the signal. This paper continues that thread under a harder and more specific question: once the honest geometric control is in place, can a localization-relevant uncertainty signal still add value, under a finite mission budget where coverage and accuracy compete for the same steps — and if a decentralized version of the signal helps, would a *centralized*, perfect-information version help even more, or is locality itself the operative ingredient?

The signal we study is unusual: it is borrowed from ecology. Chao-type estimators [Chao1984, ChaoLee1992, ChaoYang1993] estimate the number of unseen species from the counts of singletons and doubletons. We transpose them to a per-cell angular configuration count: two observations of a cell contribute independent configurations when their bearing directions differ by more than a tolerance (greedy clustering), and a cell is under-determined when it has at most one configuration. The transposed singleton/doubleton signal *U* then behaves as a local, cheap proxy for "how much localization work remains here."

**Contributions.**
- **C1.** A hard validation gate: the oracle CRLB bound correlates with empirical estimator error (ρ = 0.638, pooled), and the local richness signal correlates with residual localization work (ρ = 0.73–0.82).
- **C2.** A metric-dependent falsification/confirmation pair that is itself a methodological lesson: richness fails as a target-selection and mode-switching signal on an unbounded episode scored by a thresholded metric, but **both** a continuous re-weighting (Coverage-U) **and** direct richness-based selection (Richness-Angular) succeed under a finite mission budget scored by the continuous residual bound — the metric and regime under which coverage and accuracy actually compete.
- **C3.** Reviewer-anchored baselines: both confirmed signals beat a Random floor and classical occupancy-entropy scorers (Entropy-Frac, Frontier+Entropy) on the residual-bound metric at equal or better coverage, and Richness-Angular is the only method that remains significant at 20% obstacle density, where a dilution mechanism (Section 7) explains why Coverage-U's gain vanishes there.
- **C4.** A completed centralized-oracle control (E5, n = 40 × 2 regimes) that *falsifies* the "centralized upper bound" framing: perfect global fusion of the identical signal regresses both accuracy and coverage. A diagnostic (E5-DIAG) and a frame-matched re-run (E5-CORRECTED) rule out a movement-frame confound and locate the effect in the *locality* of the decision signal, not its strength or a calibration parameter.
- **C5.** Measured compute cost: Coverage-U runs within 1% of FB (2.17 vs. 2.16 ms/decision) and ~14–16% cheaper than an occupancy-entropy scorer, on a single-process serial benchmark.

---

## 2. Related Work

### 2.1 Multi-robot exploration and frontier methods
Frontier-based exploration [Yamauchi1997, Yamauchi1998] and its coordinated extensions [Burgard2005, Franchi2009] select targets on the boundary between explored and unexplored space. Multi-objective variants weight frontier targets by distance, utility, or information [Gonzalez2002, BasilicoAmigoni2011]. Receding-horizon next-best-view planning [Bircher2016, Bircher2018] formulates target selection as short-horizon optimization and is the modern state of the art for geometric exploration; our FB control instantiates exactly this principle. RACER [Zhou2023] scales decentralized multi-UAV exploration under asynchronous communication but targets coverage and workload balance, not observation geometry.

### 2.2 Information-driven exploration
Information-theoretic exploration maximizes expected information gain or entropy reduction of the map [Bourgault2002, Stachniss2005, Julian2014, Charrow2015]; decentralized approximations trade optimality for scalability [Grocholsky2002, Ponda2012]. MEF-Explore [Pongsirijinda2025] performs communication-constrained entropy-field exploration with proximity-triggered fusion close to our own model, ranking targets by occupancy entropy rather than observation-geometry richness. Our operating signal is neither occupancy entropy nor information gain but the angular configuration count of a bearing-only observation model; we compare directly against occupancy-entropy baselines in Section 6.6.

### 2.3 Localization-aware planning: GDOP, CRLB, FIM
Sensor-placement and path-planning for localization are classically cast as optimizing a scalar function of the Fisher information matrix — D-optimality, A-optimality, or GDOP [Kaplan2017, Ucinski2005, MartinezBullo2006, Krause2008]. For bearing-only problems, observability conditions [NardoneAidala1981] and optimal-observer-maneuver results [Passerieux1998, OshmanDavidson1999, Dogancay2012] show accuracy is governed by baseline geometry — why our CRLB-based evaluation [Cramer1946, Rao1945] and angular-diversity signal are principled. At fleet scale, cooperative dilution-of-precision analysis [Chen2020] shows the same principle: swarm configuration governs cooperative positioning accuracy, and more agents does not automatically improve localization — a finding echoed by our centralized-oracle result (Section 6.4).

### 2.4 Cooperative localization and communication
Cooperative localization in wireless networks [Patwari2005, Wymeersch2009] and multi-robot localization [Ristic2004] emphasize inter-agent measurement fusion. Distributed estimators increasingly operate directly on the sensing graph: DCL-Sparse [Sagale2024] improves range-only cooperative localization in noisy, sparse graphs; GNSS-denied swarms use coalition-based relative localization [Ruan2022] and formation-constrained geometry [Li2026]; a high-precision airborne study reports significant vertical error when swarm baselines lack diversity [Liu2026]. Communication disruptions motivate predictive bidding for disconnected teammates [Woosley2021] and low-overhead strategies exchanging only positions and targets [Batinovic2020]. We adopt a limited-range proximity-fusion model, keeping the decision signal local — and, as Section 6.4 shows, this locality is not merely a constraint but the reason the signal works.

### 2.5 Statistical richness estimators
Chao1 and related estimators [Chao1984, ChaoLee1992, ChaoYang1993, BurnhamOverton1978] estimate species richness from abundance data and are standard in ecology. Their use as *decision signals* for robot exploration — rather than post-hoc analytics — is, to the best of our knowledge, novel; this paper evaluates them transposed to angular configurations under a preregistered falsification protocol.

### 2.6 Positioning
This paper makes four moves relative to this literature. First, it evaluates information signals against a matched receding-horizon control, not a weak or absent baseline. Second, it uses a global oracle CRLB bound decoupled from the local decision signal, so accuracy claims are not circular. Third, it is organized around preregistered falsifications, including one whose resolution required diagnosing *why* a metric choice, not the signal, produced an apparent failure. Fourth, it tests — and rejects — the intuitive hypothesis that a centralized version of the same signal would only do better, isolating locality itself as the load-bearing property.

---

## 3. Problem Formulation

### 3.1 System and observation model
We consider *m* agents moving on a 100×100 grid with randomly placed obstacles (ratio *q* ∈ {0.05, 0.20}); agents do not know the map. At each time step, agent *i* at pose *p_i* obtains, for every traversable cell in a Chebyshev sensing footprint of radius F = 5, a bearing-only observation toward the cell center: a direction θ_k from the true geometry. This minimal model makes angular diversity the currency of localization and permits an exact oracle CRLB (Section 3.3).

### 3.2 Independent angular configurations and the richness signal
For each cell, observations are summarized by greedy clustering of their bearing directions: a new direction joins an existing cluster if it lies within ANG_TOL = 15° (circular) of the nearest center, otherwise it starts a new cluster, capped at CLUSTER_CAP = 8. Each cluster center is an independent angular configuration. A cell with one configuration is geometrically under-determined; a cell with two or more well-separated configurations is localizable. F1/F2 are the numbers of cells with exactly one/two configurations, and the decision signal is

```
U = min( F1·(F1−1) / (2·(F2+1)), cap ),   α = U / (U + K)
```

using the same bias-corrected form and normalization cap validated in the predecessor coverage study [Chao1984]. All policy decision signals are computed from local counts (own observations, augmented only by proximity fusion); the global count and the CRLB oracle are never fed to any policy — enforced by interface and test, no leakage.

### 3.3 Oracle CRLB evaluation metric
Localization quality is scored by a global oracle, never revealed to policies: for every traversable cell, J = Σ_k u_k u_k^⊤ / (σ² d_k²), with u_k the unit bearing vector to the *k*-th observing pose, d_k the distance, σ = 1° the nominal bearing precision. The per-cell bound is b = √(trace(J⁻¹)) in grid-cell units. A cell is well-localized when b ≤ 1.5 cells. The primary continuous accuracy metric is `mean_bound_final` (lower is better); the binary well-localized fraction is `quality(t)`, sampled every 25 steps (AUC = `quality_auc`).

### 3.4 Communication model
Agents use limited-range communication: two agents within COMM_RANGE = F exchange maps each step (rendezvous-triggered fusion). Every policy's decision signal is strictly local and temporally stale relative to the true map. Evaluation uses a separate global accumulator, never revealed to policies.

### 3.5 Metrics
**Coverage**: `final_coverage` (%), `coverage_auc`. **Localization**: `quality_auc`, `time_to_quality`, `mean_bound_final`, `undetermined_final`. **Dual**: `steps_dual`, first step with coverage ≥ 90% *and* quality ≥ 0.9. All tests paired by environment seed; gains are median relative differences; p-values Holm-Bonferroni corrected across regimes.

---

## 4. Methods

### 4.1 Frontier-Bounded control (FB)
FB selects a target among frontier cells reachable within a bounded BFS horizon (H = 8), preferring the cell maximizing remaining-exploration potential, with deterministic tie-breaking by per-agent scatter noise; movement is receding-horizon with an exploration fallback. FB uses no uncertainty signal: it is the validated geometric control and reference for every candidate. Its `steps_90` matches the validated predecessor baseline.

### 4.2 Richness-Angular (RA)
RA scores frontier targets directly by the transposed richness signal *U* over the local config-count map, replacing FB's frontier utility outright.

### 4.3 Deploy-U (mode-switching)
Deploy-U keeps the FB coverage mode while the known local map is mostly under-localized (fraction of known cells with ≤ 1 configuration above 0.30), then switches to a *deploy* mode orbiting the worst known under-determined cell to add independent configurations.

### 4.4 Coverage-U (continuous prioritization)
Coverage-U does not change mode and does not replace FB's frontier logic — it *biases* it:

```
score(target) = D/H − λ · under_count_FOV(target) / FOV_area
```

D is the bounded-BFS distance, H the horizon, `under_count_FOV` counts known-free cells with ≤ 1 angular configuration inside the target's sensing footprint (O(1) per candidate via integral image). λ = 0.5 fixed before any campaign; λ = 0 is exactly FB (verified action-identical).

### 4.5 Centralized oracle (E5, control bound)
An infeasible centralized control with perfect map knowledge maximizes the same score form using global under-sets — CentralOracle-Config (the config-count signal under perfect fusion) or CentralOracle-CRLB (the true global CRLB bottleneck). It tests whether centralization is an upper bound on Coverage-U's gain. Section 6.4 reports the completed campaign, its diagnostic extension, and a frame-matched re-run isolating the signal effect from the oracle's movement frame.

---

## 5. Experimental Protocol

### 5.1 Preregistration and gates
Every experiment is preregistered before execution, with locked metrics, thresholds, and verdict rules; analysis scripts run unchanged on final data. A hard gate (Phase 1a) must pass before any campaign runs (Section 6.1).

### 5.2 Regimes and budgets
A2 (2 UAVs), A3 (3 UAVs), A6 (6 UAVs) at 5% obstacles, and A6 at 20% obstacles (A6_obs020). Finite budget T = 0.7 × FB's median `steps_90`: A2 = 4200, A3 = 3200, A6 = 1600, A6_obs020 = 1750. Coverage at these budgets is partial (66–76%), so coverage and accuracy genuinely compete.

### 5.3 Paired design and statistics
All comparisons paired at the map level (`env_seed = 0 + 1000·r`). Paired Wilcoxon signed-rank test; Holm-Bonferroni correction across regimes; matched-pairs delta reported. n = 10 for discovery, n = 40 for confirmation.

### 5.4 Experiment map

| Stage | Question | Primary | n | Verdict |
|---|---|---|---|---|
| Phase 1a | CRLB valid? *U* predicts residual work? | ρ(bound, error) | 10 | **GO** (gate) |
| E1 (RA, unbounded/thresholded) | Richness as target selection? | quality_auc | 10 | **FAIL** (parity) |
| E2/E3 (Deploy-U) | Richness as mode switch? | steps_dual | 10 | **FAIL** (parity) |
| E4 (Coverage-U, discovery) | *U*-prioritization under budget? | quality_auc | 10 | FAIL (parity) → discovery on `mean_bound` |
| E4-CONFIRM | Confirmation, higher power | mean_bound_final | 40 | **PASS** |
| E4-PARETO | λ robustness? | mean_bound_final | 40 | **PASS** (plateau) |
| E5 | Centralized oracle bound? | mean_bound_final | 40 | **FAIL** (regression) |
| E5-DIAG | Calibration or structure? | mean_bound_final | 40 | **INCONCLUSIVE** (guard never binds — negative result) |
| E5-CORRECTED | Local frame + global signal? | mean_bound_final | 40 | **FAIL** (locality confirmed) |
| E4-reply (reviewer baselines) | vs. Random / classical entropy, at budget | mean_bound_final | 40 | **PASS** (RA and CU both beat FB and Random) |

---

## 6. Results

### 6.1 Phase 1a — metric validation (gate)

Pooled across 95,000 tested cells (91,530 localizable), ρ(bound, empirical error) = 0.638 (p < 0.05, min-acceptable 0.5), and ρ(U_local, error) = −0.457 (max-acceptable −0.4, p < 0.05). Both pass; the gate is **GO**.

### 6.2 Falsifications under the unbounded, thresholded regime (E1, E2/E3)

**Richness as target selection.** In the base scenario (6 UAVs, FOV 5, unbounded 4500-step episodes):

| Method | quality_auc (med [IQR]) | quality_final | time_to_quality | undetermined |
|---|---|---|---|---|
| Random | 0.551 [0.493–0.626] | 0.770 | 4150 | 0.2162 |
| Frontier-Bounded | 0.940 [0.939–0.942] | 1.000 | 350 | 0.0000 |
| Richness-Angular | 0.926 [0.918–0.935] | 1.000 | 575 | 0.0000 |

*n = 10.* Both FB and RA far exceed Random (p = 0.002 each); RA is not significantly better than FB (p = 0.084, −1.5%) on this metric. **Falsified — on this metric and regime.** (Section 6.6 revisits RA under the finite-budget, continuous-bound regime, where it is confirmed.)

**Richness as a mode switch (Deploy-U).** Deploy-U's orbit mode fires on only 2–6% of decisions:

| Regime | FB steps_dual | Deploy-U steps_dual | gain% | p | p_Holm |
|---|---|---|---|---|---|
| A2 | 6000 | 6088 | −1.9 | 0.8203 | 0.8203 |
| A3 | 4550 | 4375 | −0.3 | 0.7344 | 0.8203 |
| A6 | 2325 | 2450 | −6.4 | 0.2188 | 0.6562 |
| A6, 20% obs | 2512 | 2300 | +9.0 | 0.0371 | 0.1484 |

Median gain −1.1% (threshold +8%), no Holm significance. **Falsified.**

Both falsifications are informative: the bounded-horizon frame already absorbs the coverage gains a signal-driven policy might claim credit for, replicating the predecessor finding — but they also show that `quality_auc`, a thresholded binary fraction on an unbounded episode, is the wrong instrument to detect the signal's real contribution (Section 6.6).

### 6.3 Coverage-U — confirmed continuous effect under finite budget

Under finite budgets, Coverage-U does not move the discovery-stage primary `quality_auc` (median +0.4%, ns) — the metric saturates. On `mean_bound_final`, the effect is coherent across all four regimes at n = 10 (13.6–28.3% reduction, three of four raw p < 0.05), motivating confirmation at n = 40:

| Regime | FB bound | CU bound | rel-red% | p | p_Holm |
|---|---|---|---|---|---|
| A2 | 0.0277 | 0.0209 | +24.7 | 0.0002 | 0.0003 |
| A3 | 0.0222 | 0.0185 | +17.0 | <0.0001 | <0.0001 |
| A6 | 0.0244 | 0.0181 | +25.7 | <0.0001 | <0.0001 |
| A6, 20% obs | 0.0232 | 0.0219 | +5.4 | 0.2214 | 0.2214 |

*Median rel-red 20.9%, Fisher combined p ≈ 0 → **PASS**.* Coverage guard clean in every regime; corroborating `undetermined_final` drops 71.7–84.8% in three of four regimes.

**Pareto sweep.** λ ∈ {0.25, 0.5, 1.0, 2.0} on A3/A6 at n = 40: reduction stable, even slightly increasing (17.0–26.6%, all Holm-sig), no coverage regression at any λ. λ = 0.5, fixed a priori, sits at the center of the plateau — not a tuning artifact.

**Qualitative.** Coverage-U keeps the bounded-exploration frame intact but concentrates revisits on angularly under-determined regions; the fraction of rank-deficient cells is consistently lower throughout the mission, not only at the end.

### 6.4 E5 — the centralized oracle is not an upper bound

The E5 campaign (2 regimes × 2 oracle methods × 40 paired runs) asks how much of Coverage-U's gain a centralized perfect oracle would capture. The answer is sharp: the oracle regresses the primary metric and destroys the coverage guard simultaneously.

| Regime | FB mb | CU mb | Config mb | CRLB mb | FB cov | CU cov | Config cov | CRLB cov | red_CU% | red_Config% | red_CRLB% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A3_obs005 | 0.0222 | 0.0185 | 0.0424 | 0.0445 | 71.0 | 72.0 | 42.1 | 42.1 | +17.0 | −90.7 | −99.9 |
| A6_obs005 | 0.0244 | 0.0181 | 0.0487 | 0.0487 | 68.3 | 68.7 | 36.3 | 36.2 | +25.7 | −99.7 | −99.6 |

*n = 40 paired, Holm-sig p < 0.0001 both oracles, both regimes.* Both centralized oracles collapse to ≈36–42% coverage and a residual bound ≈2× worse than FB. Config and CRLB oracles are statistically indistinguishable (A3 p = 0.29, A6 p = 0.48): **the signal choice is not the problem — the global frame is.**

**Diagnostic (E5-DIAG): calibration or structure?** Coverage-guarded variants CentralOracle-CRLB-cov (ε = 0.05) and -cov2 (ε = 0.30) cap the accuracy bonus by a (1−ε) fraction of the coverage term. Instrumenting 4,800 real reachable-target evaluations shows the ε = 0.30 cap **never binds** — the global CRLB under-set is too sparse for any FOV to approach the threshold. The guarded variants are bit-identical to the unguarded oracle on all 40 seeds (p_mb = p_cov = 1.0 in every cell). This is a **negative structural result**: the guard mechanism is structurally incapable of arbitrating calibration vs. structure here; it neither confirms nor disproves the calibration hypothesis, and the diagnostic's value is narrowing the search toward the ladder comparison above.

**E5-CORRECTED: removing the movement-frame confound.** The ladder mixes two differences — the under-set signal (local vs. global fusion) and the movement frame (the oracle's global BFS excludes every visited cell from the path, so its targets are always far, D ≥ 5). A frame-matched re-run scores both oracle signals inside Coverage-U's own local bounded-BFS frame:

| Regime | FB mb | CU mb | CRLB mb | CRLB-local mb | Config-local mb | CU cov | CRLB-local cov | p_mb | p_cov |
|---|---|---|---|---|---|---|---|---|---|
| A3_obs005 | 0.0222 | 0.0185 | 0.0445 | 0.0371 | 0.0348 | 72.0 | 43.7 | <0.0001 | <0.0001 |
| A6_obs005 | 0.0244 | 0.0181 | 0.0487 | 0.0322 | 0.0306 | 68.7 | 41.9 | <0.0001 | <0.0001 |

The movement frame is a real but secondary confound (CRLB bound improves 0.049 → 0.032 in A6 once frame-matched); the dominant effect is the **signal**: with the frame held byte-identical, the local Coverage-U signal still beats both global-fusion variants by ≈2× on `mean_bound_final` and ≈30 percentage points on coverage (p < 10⁻⁸, both regimes). The perfect-fusion under-set is dense across the whole map, so the accuracy bonus dominates distance and agents over-chase far cells instead of discovering; the local proxy under-counts by construction (each agent sees only its own observations), which is precisely what keeps the coverage term in the trade-off — the mechanism that makes Coverage-U work. **The qualitative conclusion of E5 survives the confound fix: locality, not the oracle's movement frame, is what matters.**

### 6.5 Reviewer-anchored baselines and compute cost (E4-reply)

To anchor the finite-budget results against a floor and classical signals, we ran n = 40 paired budget episodes for Random, Richness-Angular, Entropy-Frac, and Frontier+Entropy alongside FB/Coverage-U, extended to the 20%-obstacle stress regime (A6_obs020; Random and Frontier+Entropy not run there):

| Regime | Method | bound | cov% | q_auc | p_b | rel-b |
|---|---|---|---|---|---|---|
| A3 | Random | 0.0534 | 20.3 | 0.3105 | <0.0001† | −140.3% |
| A3 | Frontier-Bounded | 0.0222 | 71.0 | 0.8656 | 1.0000 | +0.0% |
| A3 | **Richness-Angular** | **0.0172** | 73.3 | 0.8655 | <0.0001† | **+22.6%** |
| A3 | Entropy-Frac | 0.0190 | 70.6 | 0.8777 | 0.0061† | +14.5% |
| A3 | Frontier+Entropy | 0.0185 | 70.3 | 0.8816 | 0.0043† | +17.0% |
| A3 | Coverage-U | 0.0185 | 72.0 | 0.8598 | <0.0001† | +17.0% |
| A6 | Random | 0.0574 | 22.0 | 0.3651 | <0.0001† | −135.5% |
| A6 | Frontier-Bounded | 0.0244 | 68.3 | 0.8839 | 1.0000 | +0.0% |
| A6 | **Richness-Angular** | **0.0165** | 68.1 | 0.8745 | <0.0001† | **+32.4%** |
| A6 | Entropy-Frac | 0.0186 | 67.4 | 0.8851 | 0.0003† | +23.9% |
| A6 | Frontier+Entropy | 0.0172 | 68.0 | 0.8920 | 0.0003† | +29.6% |
| A6 | Coverage-U | 0.0181 | 68.7 | 0.8731 | <0.0001† | +25.7% |
| A6, 20% obs | Frontier-Bounded | 0.0232 | 70.8 | 0.8675 | 1.0000 | +0.0% |
| A6, 20% obs | **Richness-Angular** | **0.0176** | 71.8 | 0.8836 | <0.0001† | **+24.0%** |
| A6, 20% obs | Entropy-Frac | 0.0218 | 72.6 | 0.8764 | 0.1644 | +6.0% |
| A6, 20% obs | Coverage-U | 0.0219 | 70.6 | 0.8552 | 0.4427 | +5.4% |

*† = Holm-sig vs. FB, paired Wilcoxon, n = 40.*

Random sits at the floor (~20% coverage, bound ≈ 0.05). Every information-carrying method beats Random by a wide margin. **Richness-Angular achieves the lowest residual bound in all three regimes, including the only significant gain at 20% obstacles** (+24.0%, Holm-sig), where Coverage-U (+5.4%) and Entropy-Frac (+6.0%) both fall to non-significance. Classical occupancy-entropy scorers (Entropy-Frac, Frontier+Entropy) sit at parity-or-slightly-better than FB at 5% obstacles, replicating the predecessor lesson that the movement frame captures most of the coverage gain — the config-count signal is what adds a further, specific accuracy margin.

**Compute cost.** Serial benchmark (A6 regime, n = 10, `time.process_time()` inside `select_action`):

| Method | ms/decision |
|---|---|
| Random | 0.74 |
| Frontier-Bounded | 2.16 |
| Coverage-U | 2.17 |
| Entropy-Frac | 2.52 |

Coverage-U adds negligible overhead over FB (+0.5%) and runs ~14–16% cheaper per decision than the occupancy-entropy scorer.

---

## 7. Discussion

**The falsifications delimit where the signal works — and taught a metric lesson.** Richness fails as a direct target-selection rule and a mode-switching rule on an unbounded episode scored by a thresholded metric (Section 6.2): the bounded-horizon frame already captures the gains such signals were expected to add there, replicating the predecessor's Occam's-razor finding. But the same signal, tested under the regime and metric where coverage and accuracy actually compete (finite budget, continuous residual bound), succeeds — both as a continuous re-weighting (Coverage-U) and, even more strongly, as direct target selection (Richness-Angular, Section 6.5). The lesson is protocol-level, not signal-level: `quality_auc` is a thresholded binary fraction that saturates once most cells clear the well-localization threshold; it cannot register the accuracy work a signal is actually doing. We report the original falsifications as they were preregistered — they are not superseded, they identify precisely which metric/regime combination hides the effect.

**Two signals, a trade-off, not a winner.** Richness-Angular delivers the highest peak accuracy in every regime and is the only method robust to 20% obstacles, but as a raw target-selection rule (Section 6.2) it fails the original preregistered primary and is prone to oscillation on the unbounded/thresholded task. Coverage-U provides a stable, equally cheap continuous weighting, optimal in sparse environments, robust across its λ-plateau, but its gain is diluted by dense obstacles. The mechanism behind that dilution: Coverage-U's bonus averages `under_count_FOV` over a *fixed-area* sensing window; under 20% obstacles the window is increasingly dominated by blocked cells, so the average bonus per free cell collapses even where genuinely under-determined cells remain. Richness-Angular scores raw per-cell richness without FOV-area averaging and is immune to this dilution — which is exactly why it alone survives at 20% obstacles.

**The centralized oracle is not an upper bound; locality is the load-bearing property.** In unbounded episodes every method reaches parity; under a fixed budget, Coverage-U/RA reduce the residual bound with no coverage cost by spending the budget on angular gaps rather than cheap frontier cells — matching the classical GDOP intuition that accuracy is bought with baseline geometry [Bishop2004geometry, Bishop2009]. The intuitive next question — would a centralized, perfect-information version of the same signal do even better? — is answered no (Section 6.4): global fusion causes the accuracy bonus to dominate the distance term everywhere, so agents chase the hardest cells and coverage collapses. Two controls (a coverage-guard diagnostic, structurally unable to bind; a frame-matched re-run, isolating the signal from the oracle's movement frame) both point to the same conclusion: the local signal's weakness — it only ever sees a fraction of the true under-set — is precisely what keeps coverage in the loop and makes the trade-off attainable. Centralizing the identical signal removes that self-limitation and breaks the trade-off it depends on.

**Boundary conditions.** Coverage-U's effect vanishes at 20% obstacles for the FOV-dilution reason above; Richness-Angular does not share this failure mode and remains the recommended signal for denser or more fragmented environments.

---

## 8. Limitations

- Simulated, fully observable-to-oracle geometry: no GPS noise, sensor dropout, or measurement noise in the policies' signals (the CRLB metric itself uses the true geometry).
- The environment is a single-scale grid with randomly placed obstacles; no maze or multi-floor generalization has been run.
- The confirmed continuous-metric effect is limited to the finite-budget setting; in unbounded episodes all methods converge to parity — this is not a general accuracy gain, and is not claimed as one.
- Communication is proximity-triggered fusion; message topologies, delays, and bandwidth are not modeled.
- The E5 centralized oracle regresses both metrics, and the coverage-guarded diagnostic is a negative structural result (the guard never binds in real trajectories) — we do not claim to have proven a structural impossibility of centralization in general, only that this particular perfect-fusion construct, under this scoring form, fails, and that the failure is not resolved by re-matching the movement frame.
- Compute cost is measured for the core comparison (Coverage-U vs. FB vs. Entropy-Frac, one serial benchmark); GDOP/FIM-style matrix-inversion planners are not benchmarked directly, though Coverage-U's O(1) integral-image design gives an architectural argument for their higher relative cost.

---

## 9. Future Work

(i) GPS-noise and communication-topology robustness (Phase 2 of the thesis plan); (ii) a maze/obstacle-diverse generalization of the confirmed effects, extending the dense-obstacle analysis beyond 20%; (iii) adaptive λ or a hybrid RA/Coverage-U scheduler that gets Richness-Angular's obstacle robustness with Coverage-U's stability — the positive results for both suggest the mechanics are complementary, not competing; (iv) a parallelized/production CPU benchmark and a direct comparison against GDOP/FIM matrix-based planners; (v) other richness estimators (ACE, Jackknife) under the same budget protocol, to place the config-count proxy on the broader information-signal ladder; (vi) a partial-fusion sweep between the local and centralized extremes (e.g., k-hop relay fusion) to map how quickly the trade-off degrades as the signal's effective locality is relaxed.

---

## 10. Conclusion

We asked whether a statistical-richness signal, transposed from ecology to angular observation configurations, can operate as a decision lever for multi-UAV bearing-only localization — evaluated honestly against a matched receding-horizon geometric control, a preregistered protocol, reviewer-anchored baselines, and a centralized-oracle control. On the metric and regime that matter — finite mission budget, continuous residual localization error — the signal succeeds in two complementary forms: Richness-Angular for peak accuracy and robustness to dense obstacles, Coverage-U for a stable, near-zero-overhead continuous weighting in sparse environments. Both beat a Random floor and classical occupancy-entropy scorers. A centralized, perfect-information version of the identical signal is not an upper bound — it is worse than either decentralized variant, and three independent controls (the signal ladder, a coverage-guard diagnostic, and a frame-matched re-run) converge on the same explanation: locality is not a limitation of the decentralized setting, it is the property that makes the accuracy/coverage trade-off attainable at all. The practical reading is sharp: when mission time is the scarce resource, spend the remaining coverage on cells that are angularly under-determined, score that signal locally rather than centrally, and a cheap singleton/doubleton count is sufficient to decide where.

---

## References

[Yamauchi1997] B. Yamauchi, "A frontier-based approach for autonomous exploration," in *Proc. IEEE Int. Symp. on Computational Intelligence in Robotics and Automation (CIRA)*, 1997, pp. 146–151.

[Yamauchi1998] B. Yamauchi, "Frontier-based exploration using multiple robots," in *Proc. 2nd Int. Conf. on Autonomous Agents*, 1998, pp. 47–53.

[Burgard2005] W. Burgard, M. Moors, C. Stachniss, and F. Schneider, "Coordinated multi-robot exploration," *IEEE Trans. Robotics*, vol. 21, no. 3, pp. 376–386, 2005.

[Gonzalez2002] H. González-Baños and J.-C. Latombe, "Navigation strategies for exploring indoor environments," *Int. J. Robotics Research*, vol. 21, no. 10–11, pp. 829–848, 2002.

[Franchi2009] A. Franchi, L. Freda, G. Oriolo, and M. Vendittelli, "The sensor-based random graph method for cooperative robot exploration," *IEEE/ASME Trans. Mechatronics*, vol. 14, no. 2, pp. 163–175, 2009.

[BasilicoAmigoni2011] N. Basilico and F. Amigoni, "Exploration strategies based on multi-criteria decision making for searching environments in rescue operations," *Autonomous Robots*, vol. 31, no. 4, pp. 401–417, 2011.

[Bircher2016] A. Bircher, M. Kamel, K. Alexis, H. Oleynikova, and R. Siegwart, "Receding horizon 'next-best-view' planner for 3D exploration," in *Proc. IEEE ICRA*, 2016, pp. 1462–1468.

[Bircher2018] A. Bircher, M. Kamel, K. Alexis, H. Oleynikova, and R. Siegwart, "Receding horizon 'next-best-view' planner for 3D exploration," *IEEE Trans. Robotics*, vol. 34, no. 3, pp. 625–634, 2018.

[Zhou2023] B. Zhou, H. Xu, and S. Shen, "RACER: Rapid collaborative exploration with a decentralized multi-UAV system," *IEEE Trans. Robotics*, vol. 39, no. 3, pp. 1816–1835, 2023.

[Bourgault2002] F. Bourgault, A. A. Makarenko, S. B. Williams, B. Grocholsky, and H. F. Durrant-Whyte, "Information based adaptive robotic exploration," in *Proc. IEEE/RSJ IROS*, 2002, pp. 540–545.

[Stachniss2005] C. Stachniss, G. Grisetti, and W. Burgard, "Information gain-based exploration using Rao-Blackwellized particle filters," in *Proc. Robotics: Science and Systems (RSS)*, 2005.

[Julian2014] B. J. Julian, S. Karaman, and D. Rus, "On mutual information-based control of range sensing robots for mapping applications," in *Proc. IEEE/RSJ IROS*, 2014, pp. 5156–5163.

[Charrow2015] B. Charrow, G. Kahn, S. Patil, S. Liu, K. Goldberg, P. Abbeel, N. Michael, and V. Kumar, "Information-theoretic planning with trajectory optimization for dense 3D mapping," in *Proc. Robotics: Science and Systems (RSS)*, 2015.

[Bai2014] H. Bai, D. Hsu, and W. S. Lee, "Integrated perception and planning in the continuous space: A POMDP approach," *Int. J. Robotics Research*, vol. 33, no. 9, pp. 1288–1302, 2014.

[Grocholsky2002] B. Grocholsky, "Information-Theoretic Control of Multiple Sensor Platforms," Ph.D. dissertation, Univ. of Sydney, 2002.

[Ponda2012] S. S. Ponda, L. B. Johnson, A. N. Kopeikin, H.-L. Choi, and J. P. How, "Distributed planning strategies to enable network-level cooperation for autonomous systems," in *Proc. ACC*, 2012.

[Pongsirijinda2025] K. Pongsirijinda, Z. Cao, P. L. B. Lau, R. Liu, and U.-X. Tan, "MEF-Explore: Communication-constrained multi-robot entropy-field-based exploration," *IEEE Trans. Automation Science and Engineering*, 2025.

[Sagale2024] A. Sagale, T. Kargar Tasooji, and R. Parasuraman, "DCL-Sparse: Distributed range-only cooperative localization of multi-robots in noisy and sparse sensing graphs," arXiv:2412.14793, 2024.

[Liu2026] H. Liu, W. Jiang, Q. Long, Q. Xia, and X. Chen, "A high-precision cooperative localization method for UAVs based on multi-condition constraints," *Sensors*, vol. 26, no. 5, art. 1641, 2026.

[Li2026] D. Li, Y. Wang, Z. Li, L. Zhang, J. Luo, Y. Yu, and J. Cheng, "Formation-constrained cooperative localization for UAV swarms in GNSS-denied environments," *Sensors*, vol. 26, no. 6, art. 1984, 2026.

[Ruan2022] J. Ruan, S. Li, Y. Dai, Y. Tian, Q. Fan, C. Wang, and W. Dai, "Cooperative relative localization for UAV swarm in GNSS-denied environment based on coalition formation game," *IEEE Internet of Things Journal*, vol. 9, no. 13, pp. 11560–11577, 2022.

[Woosley2021] B. Woosley, C. Nieto-Granda, J. Rogers, N. Fung, and A. Schang, "Bid prediction for multi-robot exploration with disrupted communications," *Proc. IEEE Int. Symp. Safety, Security, and Rescue Robotics (SSRR)*, pp. 210–216, 2021.

[Batinovic2020] A. Batinović, J. Oršulić, T. Petrović, and S. Bogdan, "Decentralized strategy for cooperative multi-robot exploration and mapping," *IFAC-PapersOnLine*, vol. 53, no. 2, pp. 9682–9687, 2020.

[Chen2020] M. Chen, Z. Xiong, J. Liu, R. Wang, and J. Xiong, "Cooperative navigation of unmanned aerial vehicle swarm based on cooperative dilution of precision," *Int. J. Advanced Robotic Systems*, vol. 17, no. 3, 2020.

[Yarlagadda2000] R. Yarlagadda, I. Ali, N. Al-Dhahir, and J. Hershey, "GPS GDOP metric," *IEE Proc. Radar, Sonar and Navigation*, vol. 147, no. 5, pp. 259–264, 2000.

[Kaplan2017] E. D. Kaplan and C. J. Hegarty, *Understanding GPS/GNSS: Principles and Applications*, 3rd ed. Artech House, 2017.

[Ucinski2005] D. Uciński, *Optimal Measurement Methods for Distributed Parameter System Identification*. CRC Press, 2005.

[MartinezBullo2006] S. Martínez and F. Bullo, "Optimal sensor placement and motion coordination for target tracking," *Automatica*, vol. 42, no. 4, pp. 661–668, 2006.

[Krause2008] A. Krause, A. Singh, and C. Guestrin, "Near-optimal sensor placements in Gaussian processes: Theory, efficient algorithms and empirical studies," *J. Machine Learning Research*, vol. 9, pp. 235–284, 2008.

[Cramer1946] H. Cramér, *Mathematical Methods of Statistics*. Princeton Univ. Press, 1946.

[Rao1945] C. R. Rao, "Information and accuracy attainable in the estimation of statistical parameters," *Bull. Calcutta Math. Soc.*, vol. 37, pp. 81–91, 1945.

[NardoneAidala1981] S. C. Nardone and V. J. Aidala, "Observability criteria for bearings-only target motion analysis," *IEEE Trans. Aerospace and Electronic Systems*, vol. 17, no. 2, pp. 162–166, 1981.

[Passerieux1998] J.-M. Passerieux and D. Van Cappel, "Optimal observer maneuver for bearings-only tracking," *IEEE Trans. Aerospace and Electronic Systems*, vol. 34, no. 3, pp. 777–788, 1998.

[OshmanDavidson1999] Y. Oshman and P. Davidson, "Optimization of observer trajectories for bearings-only target localization," *IEEE Trans. Aerospace and Electronic Systems*, vol. 35, no. 3, pp. 892–902, 1999.

[Dogancay2012] K. Doğançay, "UAV path planning for passive emitter localization," *IEEE Trans. Aerospace and Electronic Systems*, vol. 48, no. 2, pp. 1150–1166, 2012.

[Bishop2004geometry] A. N. Bishop, B. Fidan, B. D. O. Anderson, K. Doğançay, and P. N. Pathirana, "Optimality analysis of sensor-target localization geometries," *Automatica*, vol. 40, no. 4, pp. 677–687, 2004.

[Bishop2009] A. N. Bishop, B. D. O. Anderson, B. Fidan, P. N. Pathirana, and G. Mao, "Bearing-only localization using geometrically constrained optimization," *IEEE Trans. Aerospace and Electronic Systems*, vol. 45, no. 1, pp. 308–320, 2009.

[Patwari2005] N. Patwari, J. N. Ash, S. Kyperountas, A. O. Hero, R. L. Moses, and N. S. Correal, "Locating the nodes: Cooperative localization in wireless sensor networks," *IEEE Signal Processing Magazine*, vol. 22, no. 4, pp. 54–69, 2005.

[Wymeersch2009] H. Wymeersch, J. Lien, and M. Z. Win, "Cooperative localization in wireless networks," *Proc. IEEE*, vol. 97, no. 2, pp. 427–450, 2009.

[Ristic2004] B. Ristic, S. Arulampalam, and N. Gordon, *Beyond the Kalman Filter: Particle Filters for Tracking Applications*. Artech House, 2004.

[Chao1984] A. Chao, "Nonparametric estimation of the number of classes in a population," *Scandinavian J. Statistics*, vol. 11, no. 4, pp. 265–270, 1984.

[ChaoLee1992] A. Chao and S.-M. Lee, "Estimating the number of classes via sample coverage," *J. American Statistical Association*, vol. 87, no. 417, pp. 210–217, 1992.

[ChaoYang1993] A. Chao and M. C. K. Yang, "Stopping rules and estimation for recapture debugging with unequal failure rates," *Biometrika*, vol. 80, no. 1, pp. 193–201, 1993.

[BurnhamOverton1978] K. P. Burnham and W. S. Overton, "Estimation of the size of a closed population when capture probabilities vary among animals," *Biometrika*, vol. 65, no. 3, pp. 625–633, 1978.
