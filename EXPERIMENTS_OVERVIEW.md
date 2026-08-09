# Projet08 — Vue d'ensemble : but, description et "why" de toutes les expériences

> Document compagnon de `code_concatene_complet.py` (concaténation de TOUT le
> code source, avec un en-tête par fichier : AIM / DESCRIPTION / IMPORTANT DETAILS).
> Ce fichier répond au « what and why » : que fait chaque expérience, pourquoi,
> quel verdict, et où le trouver dans le code.

---

## 1. Le but général (AIM of the project)

Simulation multi-agents (essaim d'UAV) de **localisation à repère de cap
uniquement (bearing-only localization)**, sous communication limitée et budget
de mission fini. Question scientifique centrale :

> Un signal de « richesse statistique » transposé au comptage de configurations
> angulaires indépendantes (héritage des estimateurs écologiques Chao-U, F1/F2)
> peut-il servir de **levier de décision** pour améliorer la **précision de
> localisation collective** — pas seulement la couverture — par rapport à un
> contrôle purement géométrique à horizon borné (Frontier-Bounded) ?

Le contrôle honnête **Frontier-Bounded** est le point de référence obligatoire
(leçon du projet précédent : l'essentiel des gains attribués aux signaux
sophistiqués était en fait le cadre de mouvement à horizon borné, pas le signal).

Trois leçons méthodologiques structurantes appliquées partout :
1. **Appariement par graine** : toutes les méthodes partagent la même graine
   d'environnement par index de run (cartes et positions initiales identiques).
2. **Pré-enregistrement** : métrique primaire, seuils, garde-fous et verdicts
   fixés AVANT de voir les données (`PRE_REG_*.md`, verdicts implémentés
   littéralement dans les scripts d'analyse).
3. **Métrique découplée** : l'évaluation est un oracle CRLB global par cellule
   (`sqrt(trace(J⁻¹))`, seuil de bonne localisation 1.5 cellules), jamais fourni
   aux politiques — seul un comptage local de configurations angulaires est un
   signal de décision.

---

## 2. Le modèle (quoi est simulé)

- Grille 100×100, obstacles 5% (un régime à 20%), agents = 2/3/6.
- Chaque cellule accumule des **configurations angulaires indépendantes** : une
  nouvelle observation ajoute une configuration si son cap est à plus de
  `ANG_TOL_DEG = 15°` (circulaire) de tout centre de cluster déjà présent,
  clustering glouton, plafond `CLUSTER_CAP = 8`.
- FOV carré de rayon 5, communication par portée limitée (`COMM_RANGE = FOV`),
  **fusion par rendez-vous** : deux agents à portée échangent leurs cartes
  (visites = max élément par élément, vus/obstacles = union, angles = union
  brute re-clustérisée de façon déterministe).
- **Oracle CRLB** : FIM par cellule `J = Σ_k u_k u_kᵀ / (σ² d_k²)` (σ = 1°),
  bound = `sqrt(trace(J⁻¹))` ; `+inf` si rang déficient. `quality(t)` = fraction
  de cellules franchissables bien localisées (bound ≤ 1.5), échantillonné tous
  les 25 pas ; `quality_auc` = AUC normalisée ; `mean_bound_final` = bound moyen
  résiduel en fin de mission (métrique continue, primaire d'E5, plus bas = mieux).
- Cellule « under-determined » = observée libre avec ≤ 1 configuration angulaire
  (FIM rang-déficient → le vrai goulot de précision). C'est le cible du signal.

---

## 3. Les méthodes comparées (policies/)

| Méthode | Fichier | Rôle | Signal |
|---|---|---|---|
| Random + répulsion | `random_walk.py` | plancher | aucun |
| Frontier | `frontier.py` | baseline géométrique (V4) | aucun |
| Frontier-Bounded | `frontier_bounded.py` | **contrôle validé obligatoire** | aucun (cadre borné) |
| Richness-Angular | `frontier_richness_angular.py` | E1 : richesse-configs comme sélection de cible | U config-count |
| Deploy-U | `frontier_richness_deploy.py` | E2/E3 : richesse-configs comme changement de mode | U config-count |
| Coverage-U | `coverage_u.py` | **E4/E5 : la méthode positive** — priorisation continue | under-count local |
| CentralOracle-* | `central_oracle.py` | E5 : oracle centralisé parfait (contrôle infaisable) | under-set global (config ou CRLB) |

Cadre de mouvement commun : `bounded_bfs` (BFS plafonné à `horizon=8`, chemins
à travers les cellules inconnues + connues-libres-non-visitées, jamais à
travers une cellule déjà visitée) — défini dans `policies/_common.py`.

---

## 4. Le protocole (régimes, budgets, gate)

- **Régimes** : `A2_obs005` (2 agents, budget 4200), `A3_obs005` (3 agents,
  3200), `A6_obs005` (6 agents, 1600), `A6_obs020` (6 agents, 20% obstacles,
  1750). Budget = 0.7 × `steps_90` médian FB mesuré en E3 (pré-enregistré).
- **Grain** : n = 10 (découverte) → n = 40 (confirmation, seeds appariées
  `BASE_SEED + run × SEED_STRIDE(1000)`).
- **Gate dur** : `experiments/run_phase1a.py` valide que (a) le bound CRLB
  corrèle avec l'erreur empirique de triangulation (ρ ≥ 0.5, localisable) et
  (b) le comptage local de configs corrèle négativement avec l'erreur
  (ρ ≤ −0.4). S'il passe, écrit `gates/phase1_GO.txt`, requis par toutes les
  campagnes.
- **Stats** : Wilcoxon apparié (`zero_method="wilcox"`), Holm-Bonferroni,
  gain relatif médian, garde-fou couverture (pas de régression Holm-significative).

---

## 5. La carte des expériences (what, why, verdict)

| Exp. | Question | Quoi | Why / hypothèse | Résultat |
|---|---|---|---|---|
| Phase 1a (gate) | La métrique et le signal sont-ils valides ? | Triangulation Gauss-Newton sous bruit 1°, corrélation de Spearman bound→erreur et U→erreur | Sans validité de la métrique et du signal, rien ne suit | **GO** (ρ(bound,err)=0.638 loc., ρ(U,err)=−0.457) |
| **E1** (Phase 1) | La richesse-configs comme **sélection directe de cible** bat-elle FB ? | Richness-Angular vs Random vs FB sur `quality_auc` | U prédit le travail restant → le prioriser aide | **FAIL** (parité) ; le signal est un témoin fiable (ρ(U,gap)=0.73–0.82) mais pas un pilote en sélection de cible |
| **E2/E3** | La richesse-configs comme **changement de mode** (déployer/orbiter) bat-elle FB ? | Deploy-U : FB tant que la carte est surtout sous-localisée, sinon orbite la pire cellule ≤ 1 config | Forcer la diversité angulaire par station-keeping | **FAIL** (`steps_dual` médian −1.1%, seuil +8%) ; `deploy_frac` 2–6% → mode sous-alimenté |
| **E4** | **Priorisation continue** sous budget fini améliore-t-elle la précision ? | Coverage-U : `score = D/horizon − λ·(under_count_FOV/FOV_area)`, λ=0.5 fixé d'avance | Sous temps limité, la couverture standard dépense le budget sur des cibles bon marché à erreur résiduelle ; les prioriser achète de l'accuracy | Primaire `quality_auc` **FAIL** (sature) mais secondaire **cohérent** : `mean_bound_final` réduit 13.6–28.3% sans régression de couverture |
| **E4-CONFIRM** | Confirmation haute puissance | n=40 paires/régime, primaire `mean_bound_final` | Verdict pré-écrit avant les données | **PASS** : réduction médiane **+20.9%**, 3/4 régimes Holm-sig, Fisher p≈0, aucune régression de couverture |
| **E4-PARETO** | L'effet est-il un artefact de λ ? | λ ∈ {0.25, 0.5, 1.0, 2.0}, A3 + A6, n=40 | | **PASS** : plateau 16.4–26.6% sur [0.25, 2.0], aucune régression → λ=0.5 (d'office) au cœur du plateau |
| **E5** | L'oracle centralisé parfait capture-t-il plus que le proxy local ? | CentralOracle-Config (fusion parfaite du signal CU) et CentralOracle-CRLB (vrai goulot CRLB global), cadre global | Bornes la perte de la transposition décentralisée | **FAIL** : les deux oracles RÉGRESSENT (bound ≈2× FB, couverture 42/36% vs 72%) |
| **E5-DIAG** | L'échec est-il calibration ou structure ? | Garde-fou de couverture `-cov` (ε=0.05) / `-cov2` (ε=0.30) plafonnant le bonus accuracy | Un oracle gardé qui limite l'accuracy-chase isolerait le mécanisme | **INCONCLUSIF** : le garde ne se déclenche **jamais** (0 bindings sur 4 800 évaluations ; cibles toujours D≥5, bonus max 55 < seuil >127) → variants bit-identiques (p=1.0) |
| **E5-CORRECTED** | L'échec vient-il du cadre de mouvement global ou du signal ? | Mêmes signaux globaux mais dans le **cadre local `bounded_bfs` byte-identique** à Coverage-U | Séparer l'artefact de cadre (les chemins globaux excluent toute cellule visitée → cibles toujours loin) de l'effet signal | **Scénario B** : le cadre global était un confond réel mais **secondaire** (CRLB 0.049→0.032 en A6) ; l'effet dominant est le **signal** — le sous-ensemble global dense noie le terme de distance → même cadre identique, CU local bat les oracles globaux ≈2× sur mean_bound et +30 pp de couverture (p<1e-8). **La localité du signal, pas sa force, est ce qui compte.** |
| **E4-RÉPLIQUE** (baselines reviewer #2/#3) | Le floor Random et les signaux classiques (entropie) tiennent-ils sous budget fini ? | n=40 paires A3+A6 (5% obs) pour Random, Richness-Angular, Entropy-Frac, Frontier+Entropy (complétés dans `budget_*`) + même batterie étendue au régime dense **A6_obs020** (sans Random ni F+E) | Ancrer le tableau budget contre le plancher et les signaux d'entropie, pas seulement contre le cadre FB ; tester la robustesse du signal à la densité d'obstacles | **Floor Random ≈20% couverture, bound ≈0.05** (les 5 méthodes le battent largement). **Entropie à parité ou mieux que FB sur bound à couverture égale** (leçon V4 : le cadre capture déjà les gains de couverture). **Richness-Angular = bound le plus bas des 3 régimes (+23–32% vs FB, Holm-sig partout)** — seule méthode significative à 20% d'obstacles (+24%, p<1e-6) alors que Coverage-U (+5.4%, ns) et Entropy-Frac (+6.0%, ns) s'éteignent en régime dense. Couverture sans régression Holm-sig dans tous les régimes. Médailles en section 6.9 du PDF |
| **CPU/decision** (reviewer #4) | Coverage-U est-il réellement bon marché à décider ? | Benchmark sériel `benchmark_cpu.py` (A6, n=10, `time.process_time()`) → `ms_per_decision` | Le claim « CU ≈ FB, moins cher que l'entropie » était par design (O(1) par image intégrale), pas mesuré | **Mesuré** : FB 2.16 ms, CU 2.17 ms, Entropy-Frac 2.52 ms, Random 0.74 ms → CU au coût de FB, ~2.9× moins cher que le scoreur d'entropie |

---

## 6. Où trouver quoi dans le code concaténé

Ordre du fichier `code_concatene_complet.py` :
1. `config.py` — toutes les constantes (single source of truth).
2. `utils/` — seeds appariées, chemins CSV, logger passif.
3. `estimators/` — modèle angulaire (clustering), estimateurs de richesse (Chao-U, ACE, Jackknife), validation Gauss-Newton/Spearman.
4. `env.py` — l'environnement (mouvement, FOV, fusion, oracle CRLB).
5. `metrics.py` — métriques de couverture (mesures globales, jamais des entrées de politique).
6. `policies/` — toutes les politiques E1→E5 (+ `_common.py` : `bounded_bfs`, `box_sum`, explore/exploit ; `factory.py` : construction canonique).
7. `experiments/` — le runner partagé et toutes les campagnes (phase1a, phase1, budget/E4, deploy/E3, pareto, traces, figures qualitatives, benchmark_cpu).
8. `analysis/` — tous les scripts de stats et verdicts (budget_stats, e5_stats, pareto_stats, phase1_stats, deploy_stats, pairwise_stats, validate_u_gap, headroom_check, compute_entropy, fig_paper, paper_build → PDF).
9. `run_all_tests.py` + `tests/` — suite de non-régression (74 tests).

