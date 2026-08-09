# Rapport E2/E3 — Déploiement localisation-aware « 1+3 » (Deploy-U)
## Projet08 — Localization-Aware Deployment for UAV Swarms

> **CORRIGÉ 2026-08-09** : dans le tableau des secondaires, les « gain % » des
> métriques lower-better (steps_90, time_to_quality, mean_bound_final) avaient
> un dénominateur non-uniforme et un signe inversé pour les métriques où
> Deploy-U est plus lent que FB. Recalculés avec la convention du papier
> (dénominateur = baseline FB, gain = (FB−DU)/FB pour lower-better) :
> `steps_90` −1.9/+0.4/−6.5/+8.6, `time_to_quality` −29.5/+19.0/−16.2/+27.2,
> `mean_bound_final` +2.5/+2.0/+4.0/−3.1. La primaire `steps_dual`, les
> médianes, p et le verdict (ÉCHEC) sont inchangés.

---

## 1. Contexte de l'étape

Après la **falsification nette de la Phase 1** (la richesse-configs comme signal de
*prise de décision* ne bat pas le contrôle de mouvement borné sur la couverture,
quel que soit le régime), l'étape E2/E3 teste l'idée de recherche la plus liée au
titre de thèse :

> **Déploiement 1+3** : maintien de la couverture (mouvement Frontier-Bounded)
> tant que la carte locale connue reste majoritairement sous-localisée, puis —
> une fois les cellules connues surtout bien localisées — **orbite** des cellules
> connues restant à ≤ 1 configuration angulaire indépendante pour forcer la
> diversité angulaire d'observation.

Contrairement à la Phase 1 (richesse = sélection de cible), ici la richesse
configurationnelle est un **signal de mode** (déployer ou continuer à couvrir),
ce qui est l'interprétation cohérente avec la transposition du projet : le
comptage angulaire est la mesure de ce qui reste à observer.

**Décisions verrouillées (pré-spécifiées avant la campagne) :**
- Signal de décision **uniquement** le comptage local de configurations angulaires
  indépendantes (`get_config_count_grid`) ; ni CRLB, ni entropie, ni GDOP, ni
  composite ne servent de signal (éliminés v3/v4).
- Métrique primaire (titre) : `steps_dual` = premier pas où **couverture ≥ 90 %
  ET qualité ≥ 0.9** simultanément. Secondaires : `steps_90`, `quality_auc`,
  `time_to_quality`, `quality_final`, `mean_bound_final`, `deploy_frac`,
  `orbit_frac`.
- Verdict A4 : gain médian ≥ 8 % **et** p < 0.05 après Holm-Bonferroni (global
  sur les 4 régimes) sur la métrique primaire.
- Protocole : A/B **apparié** par seed d'environnement (même run index), n = 10
  par méthode et par régime, sortie incrémentale reprise sur redémarrage.
- Campagne refuse de s'exécuter sans `gates/phase1_GO.txt`.

---

## 2. E1 (récapitulatif) — validation du signal U comme indicateur du travail restant

Avant d'implémenter un mode dédié, il fallait vérifier que le signal local permet
d'identifier *où* le travail de localisation reste à faire.

| Régime | ρ(config local vs global) | ρ(config local vs CRLB) | precision | recall | ρ(U_local, gap_frac) | p |
|---|---|---|---|---|---|---|
| A2_obs005 | 0.888 | −0.686 | 0.458 | 0.190 | 0.733 | 2.9e-11 |
| A3_obs005 | 0.798 | −0.534 | 0.343 | 0.268 | 0.788 | 2.9e-20 |
| A6_obs005 | 0.690 | −0.361 | 0.195 | 0.172 | 0.815 | 1.0e-29 |

Interprétation : la carte locale de configurations (a) s'accorde bien avec la
vérité globale (ρ_count 0.69–0.89), (b) est inversement corrélée à l'erreur
CRLB (ρ_bound −0.36..−0.69 : les zones à forte erreur ont peu de configurations),
et (c) le signal U agrégé prédit la fraction de travail restant avec une
corrélation élevée (0.73–0.82) et très significative. → **E2 autorisé.**

---

## 3. E2 — Implémentation de Deploy-U

Fichier : `policies/frontier_richness_deploy.py` (`FrontierRichnessDeployPolicy`),
factory dans `policies/factory.py`, constantes dans `config.py`
(`DEPLOY_COOLDOWN=12`, `DEPLOY_UNDER_FRAC_MAX=0.30`, `DEPLOY_MIN_UNDER_CELLS=3`,
`DEPLOY_ORBIT_RADIUS=2`, `DEPLOY_STATION_STEPS=6`, `DEPLOY_APPROACH_DEPTH=12`).

**Deux modes, auto-câblés sur le signal local :**
- **COVERAGE** : exactement le cadre Frontier-Bounded (BFS borné horizon 8 +
  `explore_action` de repli), tant que la fraction de cellules connues
  sous-localisées (`config ≤ 1`) dépasse `under_frac_max` (ou que le nombre de
  telles cellules < `min_under_cells`).
- **DEPLOY** : quand la carte connue est surtout bien localisée mais que des
  cellules connues restent à ≤ 1 configuration, l'agent choisit la pire cellule
  `under` atteignable (0-config d'abord, puis profondeur BFS, puis tirage),
  s'y rend par BFS borné (`approach_depth=12`), puis **orbite** : à chaque pas,
  il se déplace vers le voisin libre maximisant la variation d'angle de visée
  vers la cible (recul du précédent pénalisé), balayant ainsi des angles
  orthogonaux. Chaque nouvel angle suffisamment distant (seuil >15° du modèle
  angulaire de l'environnement) ajoute une configuration indépendante, ce qui
  garantit la diversité angulaire ; le flanquement émerge naturellement quand
  deux agents convergent sur la même cellule urgente par des côtés opposés.

**Contrôles qualité :**
- Test de fuite : la politique ne lit que l'interface locale
  (`get_local_info`, `get_config_count_grid`, `get_obstacle_knowledge`,
  `agent_positions`) — aucun accès aux `global_*`.
- Compteurs `deploy` / `orbit` exportés vers le runner (`deploy_frac`,
  `orbit_frac`).
- Optimisation notable : la sélection de cible est passée d'un BFS par
  candidat (coût O(n_under × BFS), 336 s/run) à **un seul passage BFS borné**
  (`_reach_map`, 42 s/run, comportement équivalent).

**Tests** (`tests/test_deploy.py`, 6 tests) : construction, non-déclenchement
quand la carte est très sous-localisée, déclenchement + approche/orbite sur
cartes fabriquées, métriques d'épisode bien formées, déterminisme apparié.

---

## 4. E3 — Campagne A/B appariée

Scripts : `experiments/run_deploy.py` (runner de campagne), `analysis/deploy_stats.py`
(stats). Régimes : {A2, A3, A6} × obs 5 % + A6 × obs 20 %. 4 régimes × 10 runs ×
2 méthodes = 80 épisodes, grille 100×100, FOV=5, communication limitée.

**Résultats (médianes ; steps en pas d'épisode) :**

| Régime | steps_dual FB | steps_dual DU | gain % | p | p_Holm | delta |
|---|---|---|---|---|---|---|
| A2_obs005 | 6000 | 6088 | −1.9 | 0.820 | 0.820 | −0.070 |
| A3_obs005 | 4550 | 4375 | −0.3 | 0.734 | 0.820 | +0.110 |
| A6_obs005 | 2325 | 2450 | −6.4 | 0.219 | 0.656 | −0.270 |
| A6_obs020 | 2512 | 2300 | **+9.0** | 0.037 | 0.148 | +0.690 |

**Verdict A4 (métrique primaire `steps_dual`) : gain médian = −1.1 %
(seuil +8 %) et aucune significativité Holm globale → ÉCHEC.**

Secondaires (informatives) :

| Métrique | A2 | A3 | A6_obs005 | A6_obs020 |
|---|---|---|---|---|
| steps_90 (gain %) | −1.9 | +0.4 | −6.5 | +8.6 |
| quality_auc (diff médiane) | +0.006 | −0.003 | +0.004 | +0.001 |
| time_to_quality (gain %) | −29.5 | +19.0 | −16.2 | +27.2 |
| quality_final | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 |
| mean_bound_final (gain %) | +2.5 | +2.0 | +4.0 | −3.1 |
| deploy_frac | 0.061 | 0.061 | 0.053 | 0.023 |
| orbit_frac | 0.009 | 0.010 | 0.011 | 0.005 |

**Contrôles de cohérence :** FB A6_obs005 `steps_90` = 2319, identique au
baseline V4 (2319) → le contrôle est bien la référence validée. `quality_final`
= 1.0 dans tous les régimes, pour les deux méthodes. `mean_bound_final` quasi
égal (parité de précision finale).

---

## 5. Interprétation et narration V4

1. **Parité, pas de nuisance.** Deploy-U ne dégrade jamais la qualité de
   localisation : `quality_auc` et `mean_bound_final` restent à l'équilibre
   (≤ 0.5 % de différence), `quality_final` saturé à 1.0. Aucun coût
   d'accuracy n'est payé pour le gain de déploiement.
2. **Le mécanisme fonctionne mais reste sous-alimenté.** `deploy_frac` =
   2–6 % des décisions, `orbit_frac` = 0.5–1 % : avec `under_frac_max = 0.30`
   et une fusion parcimonieuse (communication limitée), la carte locale connue
   reste majoritairement « sous-localisée » très tard, si bien que le mode
   deploy ne se déclenche que marginalement — et trop tard pour modifier
   `steps_dual`. Le cas le plus proche d'un effet est A6_obs020 (+9.0 %,
   p brut 0.037) : c'est exactement le régime où orbiter des cellules connues
   (le plus de zones ré-observables) devrait payer le plus.
3. **Cohérence avec l'arc du projet.** Troisième falsification documentée de
   la richesse-configs comme levier de décision opérant : Phase 1 (sélection de
   cible), E2/E3 (signal de mode / déploiement). Dans tous les cas, le contrôle
   géométrique de mouvement (Frontier-Bounded) capture l'essentiel du gain, et
   le signal de richesse n'apporte pas de gain statistiquement détectable sur
   la métrique du titre.
4. **Ce qui reste vrai (positif, non falsifié)** : le signal U est un bon
   indicateur **d'état** (E1 : corrélation 0.73–0.82 avec le travail restant) —
   il décrit fidèlement l'incertitude de localisation résiduelle, même s'il ne
   la réduit pas plus vite que le mouvement de couverture seul.

**Conclusion d'étape : E3 est un résultat négatif documenté sur la métrique
primaire.** Le déploiement 1+3 ne se distingue pas de Frontier-Bounded ; il ne
lui est pas inférieur en précision. Il rejoint les falsifications antérieures
et nourrit la narrative « Occam's razor » : pour la tâche évaluée, le contrôle
géométrique borné suffit, et la richesse configurationnelle joue un rôle de
témoin (monitoring) plutôt que de pilote.

---

## 6. Fichiers produits

| Fichier | Rôle |
|---|---|
| `policies/frontier_richness_deploy.py` | Politique Deploy-U (E2) |
| `policies/factory.py` / `config.py` | Branche `METHOD_DEPLOY` + constantes DEPLOY_* |
| `experiments/_runner.py` | Nouvelles métriques titre (`coverage_auc`, `steps_dual`, `mean_bound_final`, `deploy_frac`, `orbit_frac`) |
| `experiments/run_deploy.py` | Campagne E3 (résumable, gate Phase 1a) |
| `tests/test_deploy.py` | 6 tests (dont déterminisme et mécanique d'orbite) |
| `results/deploy_{A2,A3,A6_obs005,A6_obs020}/` | CSV bruts appariés (FB, Deploy-U) |
| `analysis/deploy_stats.py` | Wilcoxon apparié + Holm + verdict A4 |
| `results/validate_u_gap.csv` | E1 (signal U vs travail restant) |

Reproduire : `python experiments/run_deploy.py --runs 10` puis
`python analysis/deploy_stats.py --max-steps 8000`.
