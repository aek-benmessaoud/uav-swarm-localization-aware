# Spending Coverage Where Accuracy Is at Risk: Continuous Prioritization of Angularly Under-Determined Cells under Finite Mission Budgets

**Projet08 — Localization-Aware Deployment Strategies for UAV Swarm Systems**
Manuscript draft ver1 · August 2026

*All experiments preregistered before execution; verdicts and statistics are generated unchanged from the raw paired campaign CSVs.*

---

## Abstract

We study multi-agent bearing-only localization under limited-range communication and a finite mission budget — the regime where coverage and localization accuracy genuinely compete. Building on a predecessor finding that bounded-horizon geometric movement (Frontier-Bounded, FB) already captures most of the coverage gain typically attributed to sophisticated uncertainty signals, we ask a narrower and harder question: can a statistical richness signal — Chao-type estimators transposed from ecology to per-cell angular observation-configuration counts — still add value once *how to move* is already well handled, when the decision axis is instead *which frontier target to spend remaining coverage on*?

Under a strict preregistered protocol (paired seeds, Wilcoxon signed-rank with Holm-Bonferroni correction, a coverage-regression guard, and a hard validation gate on the evaluation metric itself), we find that richness does **not** help as a direct target-selection rule or as a mode-switching trigger — two falsifications that replicate the predecessor's Occam's-razor finding. It **does** help in a third, more specific role: continuous, always-on prioritization. Coverage-U — an FB frontier score linearly biased toward frontier cells surrounded by angularly under-determined neighbors (λ = 0.5, fixed before any data was collected) — reduces the residual oracle CRLB localization-error bound at mission end by a median of **20.9%** relative to FB, confirmed at n = 40 paired runs in three of four regimes (Fisher combined p ≈ 0), with no coverage regression and a corroborating reduction in never-observed cells. The effect is a plateau across λ ∈ [0.25, 2.0], not a knife-edge, and is specific to the budget-limited, sparse-obstacle regime: in unbounded episodes, all methods converge to parity. We interpret config-count richness as a reliable *state witness* — it correctly flags where localization work remains — that becomes an *operating lever* precisely when mission time, not distance, is the scarce resource.

---

## 1. Introduction

Multi-robot exploration has long balanced two objectives: covering an environment and acquiring information that is useful downstream [Yamauchi1997, Burgard2005]. In a growing class of missions — localization of RF/audio emitters, passive sensing, structure-from-motion, cooperative positioning — the downstream task *is* localization: each cell must be observed from enough independent viewing directions that a bearing-only estimator becomes well conditioned. The geometric spread of observations (angular diversity) is the quantity that matters, classically measured by GDOP or the Fisher information matrix [Yarlagadda2000, Bishop2004geometry].

A separate line of work has proposed information-driven exploration signals — map entropy [Bourgault2002, Stachniss2005], expected information gain [Julian2014, Charrow2015], POMDP-style value [Bai2014] — as a replacement or complement to purely geometric frontier control [Yamauchi1997]. In a predecessor study we found that when such signals are evaluated against a carefully matched bounded-horizon geometric control, most of their apparent gain on pure coverage is explained by the receding-horizon movement frame itself, not by the signal. This paper continues that thread under a harder and more specific question: once the honest geometric control is in place, can a localization-relevant uncertainty signal still add value, under a finite mission budget where coverage and accuracy compete for the same steps?

The signal we study is unusual: it is borrowed from ecology. Chao-type estimators [Chao1984, ChaoLee1992, ChaoYang1993] estimate the number of unseen species from the counts of singletons and doubletons. We transpose them to a per-cell angular configuration count: two observations of a cell contribute independent configurations when their bearing directions differ by more than a tolerance (greedy clustering), and a cell is under-determined when it has at most one configuration. The transposed singleton/doubleton signal *U* then behaves as a local, cheap proxy for "how much localization work remains here."

Rather than presenting a single positive claim, we report a **preregistered, negative-first evaluation**: three tested operating levers for this signal fail, and isolating exactly why they fail is what leads to the one role in which the signal succeeds.

**Contributions.**
- **C1.** A hard validation gate: the oracle CRLB bound correlates with empirical estimator error (ρ = 0.638, pooled), and the local richness signal correlates with residual localization work (ρ = 0.73–0.82) — establishing that both the evaluation metric and the decision signal are meaningful before any campaign is run.
- **C2.** Two documented falsifications: richness as a direct target-selection signal and as a coverage/deploy mode-switching trigger do not beat the FB control on their preregistered primaries.
- **C3.** A confirmed positive effect: continuous U-prioritized coverage (Coverage-U, λ = 0.5, fixed a priori) reduces the residual oracle CRLB bound at mission end by a median 20.9% at equal coverage (n = 40), robust across λ ∈ [0.25, 2.0], specific to budget-limited missions in sparse-obstacle environments.
- **C4.** A protocol lesson: the preregistered primary metric (a thresholded, binary well-localized fraction) saturates and conceals the effect that a continuous residual-error bound reveals — a general caution for accuracy claims built on thresholded metrics.

---

## 2. Related Work

### 2.1 Multi-robot exploration and frontier methods
Frontier-based exploration [Yamauchi1997, Yamauchi1998] and its coordinated extensions [Burgard2005, Franchi2009] select targets on the boundary between explored and unexplored space. Multi-objective variants weight frontier targets by distance, utility, or information [Gonzalez2002, BasilicoAmigoni2011]. Receding-horizon next-best-view planning [Bircher2016, Bircher2018] formulates target selection as short-horizon optimization and is the modern state of the art for geometric exploration; our FB control instantiates exactly this principle and serves as the movement baseline throughout.

### 2.2 Information-driven exploration
Information-theoretic exploration maximizes expected information gain or entropy reduction of the map [Bourgault2002, Stachniss2005, Julian2014, Charrow2015]; decentralized approximations trade optimality for scalability in cooperative settings [Grocholsky2002, Ponda2012]. These methods are computationally heavier than frontier selection and typically assume richer sensing. Our operating signal is neither occupancy entropy nor information gain but the angular configuration count of a bearing-only observation model.

### 2.3 Localization-aware planning: GDOP, CRLB, FIM
Sensor-placement and path-planning for localization are classically cast as optimizing a scalar function of the Fisher information matrix — D-optimality, A-optimality, or GDOP [Kaplan2017, Ucinski2005, MartinezBullo2006, Krause2008]. For bearing-only problems, observability conditions [NardoneAidala1981] and optimal-observer-maneuver results [Passerieux1998, OshmanDavidson1999, Dogancay2012] show that accuracy is governed by baseline geometry — precisely why our CRLB-based evaluation [Cramer1946, Rao1945] and angular-diversity signal are principled.

### 2.4 Cooperative localization and communication
Cooperative localization in wireless networks [Patwari2005, Wymeersch2009] and multi-robot localization [Ristic2004] emphasize inter-agent measurement fusion. We adopt a limited-range proximity-fusion model, which keeps the decision signal local and honest — no policy ever observes the global map or the evaluation oracle.

### 2.5 Statistical richness estimators
Chao1 and related estimators [Chao1984, ChaoLee1992, ChaoYang1993, BurnhamOverton1978] estimate species richness from abundance data and are standard in ecology. Their use as *decision signals* for robot exploration — rather than post-hoc analytics — is, to the best of our knowledge, novel; this paper evaluates them transposed to angular configurations under a preregistered falsification protocol.

### 2.6 Positioning
This paper makes three moves relative to this literature. First, it evaluates information signals against a matched receding-horizon control, not a weak or absent baseline — the dominant failure mode of information-driven exploration claims. Second, it uses a global oracle CRLB bound decoupled from the local decision signal, so accuracy claims are not circular. Third, it is organized around preregistered falsifications: we report which operating levers failed before claiming the one that succeeded, and the failures are load-bearing evidence, not a footnote.

---

## 3. Problem Formulation

### 3.1 System and observation model
We consider *m* agents moving on a 100×100 grid with randomly placed obstacles (ratio *q* ∈ {0.05, 0.20}); agents do not know the map. At each time step, agent *i* at pose *p_i* obtains, for every traversable cell in a Chebyshev sensing footprint of radius F = 5, a bearing-only observation toward the cell center: a direction θ_k from the true geometry. This minimal model makes angular diversity the currency of localization and permits an exact oracle CRLB (Section 3.3).

### 3.2 Independent angular configurations and the richness signal
For each cell, observations are summarized by greedy clustering of their bearing directions: a new direction joins an existing cluster if it lies within ANG_TOL = 15° (circular) of the nearest center, otherwise it starts a new cluster, capped at CLUSTER_CAP = 8. Each cluster center is an independent angular configuration. A cell with one configuration is geometrically under-determined (a single bearing gives a rank-deficient Fisher matrix); a cell with two or more well-separated configurations is localizable. We transpose the Chao richness vocabulary to these counts: F1/F2 are the numbers of cells with exactly one/two configurations, and the decision signal is

```
U = min( F1·(F1−1) / (2·(F2+1)), cap ),   α = U / (U + K)
```

using the same bias-corrected form and normalization cap validated in the predecessor coverage study [Chao1984]. All decision signals used by policies are computed from local counts (own observations, augmented only by proximity fusion); the global count and the CRLB oracle are never fed to any policy — enforced by interface and test, no leakage.

### 3.3 Oracle CRLB evaluation metric
Localization quality is scored by a global oracle, never revealed to policies: for every traversable cell, the Fisher information matrix from the true observation geometry is J = Σ_k u_k u_k^⊤ / (σ² d_k²), with u_k the unit bearing vector to the *k*-th observing pose, d_k the distance, and σ the nominal bearing precision (1°). The per-cell bound is b = √(trace(J⁻¹)) in grid-cell units. A cell is well-localized when b ≤ QUALITY_THRESHOLD = 1.5 cells. The primary continuous accuracy metric is the mean residual bound across traversable cells at mission end, `mean_bound_final` (lower is better); the binary fraction of well-localized cells is `quality(t)`, sampled every 25 steps (normalized AUC = `quality_auc`).

### 3.4 Communication model
Agents use limited-range communication: two agents within COMM_RANGE = F exchange maps each step (rendezvous-triggered fusion). Every policy's decision signal is therefore strictly local and temporally stale relative to the true map — the honest distributed setting. Evaluation uses a separate global accumulator, never revealed to policies.

### 3.5 Metrics
**Coverage**: `final_coverage` (%), `coverage_auc`. **Localization**: `quality_auc`, `time_to_quality`, `mean_bound_final`, `undetermined_final` (fraction of traversable cells never observed). **Dual** (for mode-switching tests): `steps_dual`, first step with coverage ≥ 90% *and* quality ≥ 0.9. All tests are paired by environment seed; gains are median relative differences; p-values are Holm-Bonferroni corrected across regimes.

---

## 4. Methods

### 4.1 Frontier-Bounded control (FB)
FB selects a target among frontier cells reachable within a bounded BFS horizon (H = 8), preferring the frontier cell that maximizes remaining-exploration potential, with deterministic tie-breaking by per-agent scatter noise; movement is receding-horizon with an exploration fallback. FB uses no uncertainty signal: it is the validated geometric control and the reference against which every candidate is tested. Its `steps_90` matches the validated baseline of the predecessor study.

### 4.2 Two falsified alternatives (condensed)
Before arriving at the confirmed method, we tested and preregistered two more direct uses of the richness signal, both of which failed on their primary metric (full results in Section 6.2):

- **Richness-Angular (RA)** scores frontier targets directly by the transposed richness signal *U* over the local config-count map, replacing FB's frontier utility outright. It tests whether richness as a *direct target-selection signal* beats FB.
- **Deploy-U** keeps the FB coverage mode while the known local map is mostly under-localized, then switches to a *deploy* mode that orbits the worst known under-determined cell to add independent configurations. It tests richness as a *mode-switching signal*.

### 4.3 Coverage-U (confirmed method)
Coverage-U does not change mode and does not replace FB's frontier logic — it *biases* it. Inside the same bounded frame, it replaces the frontier utility by a continuous target score

```
score(target) = D/H − λ · under_count_FOV(target) / FOV_area
```

where D is the bounded-BFS distance to the target, H the horizon, and `under_count_FOV` counts the known-free cells with ≤ 1 angular configuration inside the target's sensing footprint, computed in O(1) per candidate via an integral image. λ = 0.5 was fixed before any campaign (no tuning); λ = 0 is exactly FB (verified by an action-identical test). The hypothesis: with finite mission time, standard coverage spends the remaining budget on frontier cells that are cheap to reach but leave angular gaps; biasing target choice toward under-observed regions buys residual accuracy at the same coverage cost.

### 4.4 Centralized oracle (E5 — status: incomplete, not a result)
An infeasible centralized control with perfect map knowledge, maximizing the same score form using global under-sets, was designed to bound how much of the accuracy gain a decentralized config-count proxy captures (transposition ratio ρ = reduction(Coverage-U) / reduction(CentralCRLB)). At the time of writing, only a 4-run validation of the CRLB oracle exists (Section 6.5); it is **not reported as a result** and does not affect any preregistered verdict in this paper. It is the immediate next experiment (Section 9).

---

## 5. Experimental Protocol

### 5.1 Preregistration and gates
Every experiment is preregistered in the repository before execution, with locked metrics, thresholds, and verdict rules; analysis scripts implement the verdicts exactly and are run unchanged on the final data. A hard gate (Phase 1a) must pass before any campaign runs: the oracle CRLB bound must correlate with empirical estimator error, and the local richness signal must correlate with residual work (Section 6.1).

### 5.2 Regimes and budgets
Regimes vary team size and obstacle ratio: A2 (2 UAVs), A3 (3 UAVs), A6 (6 UAVs) at 5% obstacles, and A6 at 20% obstacles (A6_obs020). The finite budget T per regime is 0.7 × the FB median `steps_90` measured in the predecessor study: A2 = 4200, A3 = 3200, A6 = 1600, A6_obs020 = 1750. At these budgets, coverage is partial (66–76%), so coverage and accuracy genuinely compete.

### 5.3 Paired design and statistics
All comparisons are paired at the map level: run index *r* uses `env_seed = 0 + 1000·r` for every method. Significance is assessed by the paired Wilcoxon signed-rank test; p-values are Holm-Bonferroni corrected across regimes; matched-pairs delta (rank-biserial) is reported. Primary/guard/secondary metrics and verdict rules are fixed per stage. n = 10 pairs for discovery (RA, Deploy-U, initial Coverage-U), n = 40 for confirmation (Coverage-U-CONFIRM, Coverage-U-PARETO).

### 5.4 Experiment map

| Stage | Question | Primary | n | Verdict |
|---|---|---|---|---|
| Phase 1a | CRLB valid? *U* predicts residual work? | ρ(bound, error) | 10 | **GO** (gate) |
| RA | Richness as target selection? | quality_auc | 10 | **FAIL** (parity) |
| Deploy-U | Richness as mode switch? | steps_dual | 10 | **FAIL** (parity) |
| Coverage-U (discovery) | *U*-prioritization under budget? | quality_auc | 10 | FAIL (parity) → discovery on `mean_bound` |
| Coverage-U-CONFIRM | Confirmation, higher power | mean_bound_final | 40 | **PASS** |
| Coverage-U-PARETO | λ robustness? | mean_bound_final | 40 | **PASS** (plateau) |
| E5 (oracle) | Centralized upper bound? | mean_bound_final | 40 | **INCOMPLETE** |

*All stages preregistered; Coverage-U-CONFIRM and -PARETO fixed λ = 0.5 and budgets before any data were seen.*

---

## 6. Results

### 6.1 Phase 1a — metric validation (gate)

The oracle CRLB bound is a valid accuracy proxy: pooled across 95,000 tested cells (91,530 localizable), ρ(bound, empirical error) = 0.638 on localizable cells (p < 0.05, min-acceptable 0.5), and the local richness signal correlates negatively with empirical error (ρ(U_local, error) = −0.457, max-acceptable −0.4, p < 0.05). Both conditions pass; the gate is **GO**. This confirms both the evaluation metric and the decision signal are meaningful before any campaign is interpreted.

### 6.2 Two falsifications (condensed)

**Richness as target selection (RA).** In the base scenario (6 UAVs, FOV 5, unbounded 4500-step episodes):

| Method | quality_auc (med [IQR]) | quality_final | time_to_quality | undetermined |
|---|---|---|---|---|
| Random | 0.551 [0.493–0.626] | 0.770 | 4150 | 0.2162 |
| Frontier-Bounded | 0.940 [0.939–0.942] | 1.000 | 350 | 0.0000 |
| Richness-Angular | 0.926 [0.918–0.935] | 1.000 | 575 | 0.0000 |

*n = 10.* Both FB (p = 0.002 vs. Random) and RA (p = 0.002 vs. Random) far exceed the Random floor; RA is not significantly better than FB (p = 0.084, −1.5%). **Richness as a direct target-selection signal is falsified** — the bounded-horizon frame already captures the coverage gains this signal was expected to add, replicating the predecessor finding.

**Richness as a mode switch (Deploy-U).** Deploy-U's orbit mode fires on only 2–6% of decisions and never changes the dual-objective outcome:

| Regime | FB steps_dual | Deploy-U steps_dual | gain% | p | p_Holm |
|---|---|---|---|---|---|
| A2 | 6000 | 6088 | −1.9 | 0.8203 | 0.8203 |
| A3 | 4550 | 4375 | −0.3 | 0.7344 | 0.8203 |
| A6 | 2325 | 2450 | −6.4 | 0.2188 | 0.6562 |
| A6, 20% obs | 2512 | 2300 | +9.0 | 0.0371 | 0.1484 |

Median gain −1.1% (preregistered threshold +8%), no Holm significance. **Falsified.** Final localization quality is at parity everywhere (`quality_final` = 1.0).

Both failures are informative, not incidental: they show precisely where a well-tuned geometric control already absorbs the gains a signal-driven policy would otherwise claim credit for. This sets up the discriminating question for Section 6.3: is there *any* role left for the signal once movement and mode are already well handled?

### 6.3 Coverage-U — confirmed effect under finite budget

Under finite budgets, Coverage-U (λ = 0.5) does not move the preregistered discovery primary `quality_auc` (median gain +0.4%, no Holm significance) — the binary well-localized fraction saturates and cannot register improvements below threshold. However, on the continuous accuracy metric `mean_bound_final`, the effect is coherent across all four regimes at n = 10 (relative reduction vs. FB 13.6–28.3%, three of four raw p < 0.05). This directional discovery motivated a higher-power preregistered confirmation.

**Confirmation (n = 40 paired runs per regime).** The residual oracle CRLB bound at mission end is significantly lower for Coverage-U in three of four regimes, median relative reduction **20.9%**, Fisher combined p ≈ 0:

| Regime | FB bound | Coverage-U bound | rel-red% | p | p_Holm |
|---|---|---|---|---|---|
| A2 (2 UAVs) | 0.0277 | 0.0209 | +24.7 | 0.0002 | 0.0003 |
| A3 (3 UAVs) | 0.0222 | 0.0185 | +17.0 | <0.0001 | <0.0001 |
| A6 (6 UAVs) | 0.0244 | 0.0181 | +25.7 | <0.0001 | <0.0001 |
| A6, 20% obs | 0.0232 | 0.0219 | +5.4 | 0.2214 | 0.2214 |

*Median rel-red 20.9% (preregistered threshold 10%); any Holm-significant = true; coverage regression = none; Fisher p ≈ 0 → **PASS**.*

**Coverage guard.** Final coverage at mission end is not significantly reduced in any regime — Coverage-U achieves its accuracy gain without spending extra steps or sacrificing coverage.

**Corroborating signal.** The fraction of traversable cells never observed drops sharply alongside the bound:

| Regime | FB undetermined | Coverage-U undetermined | gain% | p | p_Holm |
|---|---|---|---|---|---|
| A2 | 0.0109 | 0.0032 | +71.7 | 0.0170 | 0.0341 |
| A3 | 0.0085 | 0.0009 | +84.8 | 0.0002 | 0.0009 |
| A6 | 0.0110 | 0.0031 | +83.2 | 0.0094 | 0.0281 |
| A6, 20% obs | 0.0104 | 0.0125 | −37.2 | 0.2591 | 0.2591 |

Holm-significant in three of four regimes; A6_obs020 is the single non-significant (mildly reversed) regime — see Section 7, boundary conditions.

### 6.4 Pareto sweep — plateau, not a knife-edge

Sweeping λ ∈ {0.25, 0.5, 1.0, 2.0} on the two confirmed regimes (A3, A6) at n = 40 shows the reduction is stable — even slightly increasing — with no coverage regression at any λ:

| Regime | λ=0.25 | λ=0.5 | λ=1.0 | λ=2.0 |
|---|---|---|---|---|
| A3 (3 UAVs) | +17.0%* (72% cov) | +17.0%* (72% cov) | +16.4%* (70% cov) | +23.9%* (71% cov) |
| A6 (6 UAVs) | +23.9%* (68% cov) | +25.7%* (69% cov) | +23.4%* (66% cov) | +26.6%* (67% cov) |

*All Holm-significant (p ≤ 0.0002). λ = 0.5, fixed before any campaign, sits at the center of the plateau — the effect is not a tuning artifact.*

### 6.5 Qualitative illustration

Representative trajectories (median and 75th-percentile residual-bound reduction) show that Coverage-U keeps the bounded-exploration frame intact but concentrates revisits around angularly under-determined regions, rather than diverging into a different movement pattern. The fraction of traversable cells left in the rank-deficient, single-configuration state over time is consistently lower for Coverage-U than for FB throughout the mission, not only at the end — the ambiguous residue is resolved progressively, not in a late correction.

### 6.6 E5 — centralized oracle (status: incomplete)

The full E5 campaign (2 regimes × 2 oracle methods × 40 runs) is designed to bound how much of the achievable accuracy gain the decentralized config-count proxy captures. At the time of writing, only a 4-run validation of the CRLB oracle in A3 exists; all four validation runs confirm the oracle is well-formed (residual bounds ≈0.04 vs. ≈0.022 for FB at the same budget, consistent with an oracle that trades coverage for accuracy more aggressively than the decentralized proxy). **This is not reported as a result.** It is listed in Section 9 as the immediate next experiment; its outcome does not affect any preregistered verdict above.

---

## 7. Discussion

**The falsifications delimit where the signal works.** Richness configurations fail as a direct target-selection rule and as a mode-switching rule; each failure is informative, not a dead end. The bounded-horizon geometric frame already captures the coverage gains these signals were expected to add — the same Occam's-razor pattern found in the predecessor coverage study, now made quantitative in a localization setting. This is the honest baseline against which the positive result in Section 6.3 must be read: it is not the first thing we tried, it is what survived after two more direct uses of the same signal did not.

**The positive effect is a budget effect, not a general accuracy effect.** In unbounded episodes every method reaches `quality_final` = 1.0 and residual bounds converge to parity. Under a fixed mission time, Coverage-U reduces the residual oracle bound by ≈21% median with no coverage cost. The mechanism is visible in the traces: standard coverage spends remaining time reaching cheap frontier cells, leaving angular gaps; Coverage-U spends the same steps re-observing under-determined cells instead. This matches the classical GDOP intuition [Bishop2004geometry, Bishop2009] — accuracy is bought with baseline geometry — and shows that a *local, cheap* proxy can capture part of that intuition without a centralized planner.

**Why the preregistered primary failed, and what it teaches.** `quality_auc` is a thresholded binary fraction; it saturates and cannot rank improvements once most cells already clear the well-localization threshold. The continuous residual bound is the metric actually aligned with the objective. We report the preregistered primary as FAIL and the continuous secondary as a confirmed discovery — a protocol lesson we consider worth publishing in its own right: binary saturation metrics can conceal real accuracy effects that continuous bounds reveal.

**Boundary conditions.** The effect is absent at 20% obstacles (A6_obs020: +5.4%, not significant; `undetermined` mildly reversed). Dense obstacles fragment the known-free cells, leaving the sensing footprint few under-observed cells to target, and constrain mobility generally. The conclusion is therefore conditional: config-count prioritization helps in open, low-obstacle environments under time pressure — precisely the mission profile where partial coverage is unavoidable and residual accuracy matters most.

---

## 8. Limitations

- Simulated, fully observable-to-oracle geometry: no GPS noise, sensor dropout, or measurement noise in the policies' signals (the CRLB metric itself uses the true geometry).
- The environment is a single-scale grid with randomly placed obstacles; no maze or multi-floor generalization has been run.
- The confirmed effect is limited to sparse-obstacle regimes and the finite-budget setting; it is not a general accuracy gain, and is explicitly not claimed as one.
- Communication is proximity-triggered fusion; message topologies, delays, and bandwidth are not modeled.
- The E5 centralized-oracle upper bound is incomplete; the transposition ratio ρ is therefore not yet available.
- Compute cost per decision (CPU/decision) was not measured for all methods in this campaign; the design-level claim that Coverage-U is cheaper than GDOP/FIM-style planners (O(1) integral-image scoring vs. matrix optimization) is architecturally supported but not yet benchmarked rigorously.

---

## 9. Future Work

The immediate next step is completing the E5 oracle campaign (160 remaining episodes, ≈3–4h) to report the transposition ratio ρ = reduction(Coverage-U) / reduction(CentralCRLB), and the full ladder FB → Coverage-U → CentralOracle-Config → CentralOracle-CRLB. Beyond it: (i) a strict CPU-per-decision benchmark against GDOP/FIM planners; (ii) GPS-noise and communication-topology robustness; (iii) a maze/obstacle-diverse generalization of the confirmed effect; (iv) adaptive λ — the Pareto plateau invites tuning, but we deliberately report only the fixed preregistered value in this paper; (v) combining continuous prioritization with the deploy/orbit mechanics that failed as a pure mode switch (Section 6.2) — the positive result suggests the underlying mechanics were sound and the *triggering rule*, not the mechanism, was wrong; (vi) other richness estimators (ACE, Jackknife) and entropy baselines under the same budget protocol, to place the config-count proxy on the broader information-signal ladder.

---

## 10. Conclusion

We asked whether a statistical-richness signal, transposed from ecology to angular observation configurations, can operate as a decision lever for multi-UAV bearing-only localization — evaluated honestly against a matched receding-horizon geometric control and under a preregistered protocol. Two levers fail; a third succeeds. Richness as direct target selection and as a mode-switching trigger is falsified. Continuous U-prioritized coverage reduces the residual oracle CRLB bound by a median 20.9% at equal coverage under finite mission budgets, robustly across λ, in sparse-obstacle environments. The practical reading is sharp: when mission time is the scarce resource, spend the remaining coverage on cells that are angularly under-determined — and a cheap, local, decentralized singleton/doubleton count is sufficient to decide where.

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

[Bourgault2002] F. Bourgault, A. A. Makarenko, S. B. Williams, B. Grocholsky, and H. F. Durrant-Whyte, "Information based adaptive robotic exploration," in *Proc. IEEE/RSJ IROS*, 2002, pp. 540–545.

[Stachniss2005] C. Stachniss, G. Grisetti, and W. Burgard, "Information gain-based exploration using Rao-Blackwellized particle filters," in *Proc. Robotics: Science and Systems (RSS)*, 2005.

[Julian2014] B. J. Julian, S. Karaman, and D. Rus, "On mutual information-based control of range sensing robots for mapping applications," in *Proc. IEEE/RSJ IROS*, 2014, pp. 5156–5163.

[Charrow2015] B. Charrow, G. Kahn, S. Patil, S. Liu, K. Goldberg, P. Abbeel, N. Michael, and V. Kumar, "Information-theoretic planning with trajectory optimization for dense 3D mapping," in *Proc. Robotics: Science and Systems (RSS)*, 2015.

[Grocholsky2002] B. Grocholsky, "Information-Theoretic Control of Multiple Sensor Platforms," Ph.D. dissertation, Univ. of Sydney, 2002.

[Ponda2012] S. S. Ponda, L. B. Johnson, A. N. Kopeikin, H.-L. Choi, and J. P. How, "Distributed planning strategies to enable network-level cooperation for autonomous systems," in *Proc. ACC*, 2012.

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
