# Rapport E4 — Couverture priorisée par le signal U sous budget fini (Coverage-U)
## Projet08 — Localization-Aware Deployment for UAV Swarms

> **CORRIGÉ 2026-08-09** : les « gain % » des secondaires lower-better de ce
> rapport (mean_bound_final, undetermined_final) avaient été calculés avec le
> mauvais dénominateur — `(FB−CU)/CU` au lieu de `(FB−CU)/FB` (convention du
> papier, voir `analysis/paper_build.py`). Tous les gains de ce rapport sont
> recalculés avec le dénominateur = baseline FB. Les valeurs p, médianes et
> verdicts (PASS/ÉCHEC) sont inchangés (les tests sont appariés, la formule ne
> change que l'échelle du % relatif).

---

## 1. Contexte de l'étape

Après trois falsifications documentées de la richesse-configurations comme
**levier de décision** (Phase 1 = sélection de cible ; E2/E3 = signal de
mode/déploiement), et après le verdict utilisateur « Document E3 as negative,
write V4 narrative », la direction suivante retenue (option 1) est :

> **U-prioritized coverage under budget** : ne pas changer de mode, mais
> **prioriser en continu** — dans le cadre de mouvement Frontier-Bounded — les
> cibles dont le voisinage FOV contient le plus de cellules connues encore
> sous-localisées (≤ 1 configuration angulaire). L'hypothèse : à budget de
> mission fini (temps d'arrêt fixé), la couverture *déployée* par la stratégie
> standard couvre vite mais laisse une erreur de localisation résiduelle ; en
> biaisant la sélection de cible vers les zones qui manquent de configurations,
> on réduit cette erreur résiduelle **au même coût de couverture**.

**Décisions verrouillées (pré-spécifiées avant la campagne) :**
- Score continu de cible dans le cadre FB :
  `score(target) = D/horizon − λ · under_count_FOV(target)/FOV_area`,
  où `under_count_FOV` compte les cellules connues-libres avec ≤ 1 configuration
  dans le carré FOV (footprint `get_fov_mask`) de la cible. **λ = 0.5** (paramètre
  unique, aucun tuning). λ = 0 == Frontier-Bounded exact (vérifié par test).
- Signal de décision **uniquement** le comptage local de configurations
  angulaires indépendantes ; ni CRLB, ni entropie, ni GDOP.
- Budget T par régime = **0.7 × médiane FB de `steps_90` mesurée en E3**
  (pré-spécifié) : A2 = 4200, A3 = 3200, A6_obs005 = 1600, A6_obs020 = 1750.
- Métrique primaire E4 : **`quality_auc` @ T** (précision intégrée sous budget ;
  plus haut = mieux). Garde Pareto : **`final_coverage` @ T** ne doit pas
  régresser significativement. Secondaires : `coverage_auc`, `mean_bound_final`
  @ T (plus bas = mieux), `undetermined_final` @ T (plus bas = mieux),
  `steps_dual`, `time_to_quality`.
- Verdict A4 : gain médian ≥ 8 % **et** Holm-sig sur la primaire **et** pas de
  régression de couverture. Deploy-U = ligne de contexte.
- Protocole : A/B **apparié** par seed (même run index), n = 10 × méthode ×
  régime, 3 méthodes (FB, Deploy-U, Coverage-U), sortie incrémentale.

---

## 2. Implémentation

Fichier : `policies/coverage_u.py` (`CoverageUPolicy`), branche
`METHOD_COVERAGE_U = "Coverage-U"` dans `policies/factory.py`,
`COVERAGE_U_LAMBDA = 0.5` dans `config.py`.

**Mécanisme :** même cadre que Frontier-Bounded (BFS borné horizon 8,
`explore_action` de repli, mêmes tirages) ; seule la fonction de cible change.
`under(target)` = nombre de cellules `connue & ~obstacle & config ≤ 1` dans le
footprint FOV Chebyshev de rayon `fov_radius`. Le calcul est vectorisé par
**image intégrale** (`_box_sum`) sur le carré du footprint → O(1) par candidat.
Compteurs `fallback` / `random_walk` exportés au runner.

**Contrôle sans fuite :** la politique ne lit que l'interface locale
(`get_local_info`, `get_config_count_grid`, `get_obstacle_knowledge`,
`agent_positions`) — aucun accès aux `global_*` ni à l'oracle CRLB.

**Tests** (`tests/test_coverage_u.py`, 6 tests, tous passés) : construction ;
`_box_sum` vs référence naïve ; **λ = 0 action-identique à Frontier-Bounded sur
2 environnements seed-identiques** (le test le plus fort : à λ = 0 la politique
est indistinguable du contrôle) ; λ = 0.5 diverge ; métriques d'épisode bien
formées ; déterminisme apparié.

---

## 3. Campagne A/B appariée

Scripts : `experiments/run_budget.py` (runner de campagne, résumable),
`analysis/budget_stats.py` (stats). 4 régimes × 10 runs × 3 méthodes = 120
épisodes, grille 100×100, FOV=5, communication limitée, budgets T = 0.7 × FB
`steps_90` (E3). Campagne terminée en 4912 s (~82 min).

### Primaire : `quality_auc` @ T (plus haut = mieux)

| Régime | med_FB | med_CU | gain % | p | p_Holm | delta |
|---|---|---|---|---|---|---|
| A2_obs005 | 0.843 | 0.837 | −0.8 | 1.000 | 1.000 | −0.060 |
| A3_obs005 | 0.873 | 0.855 | −1.0 | 0.432 | 1.000 | −0.200 |
| A6_obs005 | 0.890 | 0.890 | −0.8 | 0.625 | 1.000 | +0.040 |
| A6_obs020 | 0.876 | 0.839 | −3.1 | 0.275 | 1.000 | −0.240 |

### Garde : `final_coverage` @ T (pas de régression autorisée)

| Régime | med_FB | med_CU | p | p_Holm | régression ? |
|---|---|---|---|---|---|
| A2_obs005 | 64.5 | **67.7** | 1.000 | 1.000 | non |
| A3_obs005 | 67.7 | **71.5** | 0.846 | 1.000 | non |
| A6_obs005 | 68.2 | **69.5** | 1.000 | 1.000 | non |
| A6_obs020 | 67.3 | **68.2** | 1.000 | 1.000 | non |

**Verdict A4 (primaire `quality_auc`) : gain médian = −0.9 % (seuil +8 %),
aucune Holm-sig, pas de régression de couverture → ÉCHEC.**

### Secondaire précision continue : `mean_bound_final` @ T (plus bas = mieux)

| Régime | med_FB | med_CU | gain %* | p | p_Holm | delta |
|---|---|---|---|---|---|---|
| A2_obs005 | 0.0290 | **0.0208** | +29.7 | 0.049 | 0.098 | −0.660 |
| A3_obs005 | 0.0232 | **0.0193** | +19.3 | 0.106 | 0.106 | −0.400 |
| A6_obs005 | 0.0229 | **0.0174** | +19.5 | 0.027 | 0.098 | −0.400 |
| A6_obs020 | 0.0265 | **0.0229** | +13.0 | 0.049 | 0.098 | −0.460 |

\* gain = (FB−CU)/FB, dénominateur = baseline (convention du papier,
paper_build.py ; corr. 2026-08-09 — la version précédente divisait par CU) ;
la **réduction du bound CRLB résiduel rel. au contrôle** vaut 13.0–29.7 %
selon le régime (médiane ≈ 20 %).
3/4 p bruts < 0.05, toutes les différences dans le même sens ; aucune ne passe
le Holm global (p_Holm max 0.098).

### Autres secondaires (informatives)

| Métrique | A2 | A3 | A6_obs005 | A6_obs020 |
|---|---|---|---|---|
| coverage_auc (gain %) | −1.6 | −2.7 | −1.7 | −5.5 |
| undetermined_final (gain %) | +71.7 | +45.3 | +86.4 | −52.2 |
| time_to_quality (gain %) | −17 | +20 | −15 | −22 |
| Deploy-U vs FB, quality_auc (gain %) | +1.2 | +0.4 | +0.5 | +1.4 |

`quality_final` = 1.0 partout (les deux méthodes finissent avec toute la carte
localisée au-delà du seuil quand on les laisse courir ; sous budget, la
différence de précision résiduelle est portée par `mean_bound_final`, pas par la
fraction binaire).

---

## 4. Interprétation et narration

1. **La primaire pré-enregistrée échoue — parité sur la fraction « bien
   localisée ».** `quality_auc` mesure une fraction binaire saturée (cellules
   ≥ 2 configurations). Sous budget, les deux méthodes produisent la même
   fraction au-dessus du seuil : le signal U priorise *à l'intérieur* de la zone
   sous-seuil, ce que la fraction binaire ne voit pas.
2. **Le signal déplace la métrique continue de précision, pas la couverture.**
   `mean_bound_final` @ T (bound CRLB résiduel moyen) est **plus bas dans les 4
   régimes** (réduction 13.0–29.7 % vs FB, 3/4 p bruts < 0.05, toutes même sens)
   **sans aucune régression de couverture** — au contraire, `final_coverage`
   médiane est même légèrement supérieure partout (67.7 vs 64.5 … 68.2 vs 67.3).
   C'est le premier signal directionnel positif de la richesse-configurations
   comme levier de décision : à budget fixe, prioriser la dépense de couverture
   vers les zones sous-observées achète de l'erreur de localisation résiduelle
   **à couverture égale ou meilleure** — exactement le compromis du titre
   (« Maximizing Accuracy and Coverage »).
3. **Caveat statistique honnête.** Aucun p ne passe le Holm global (max 0.098) :
   avec n = 10 × 4 régimes, l'effet est cohérent mais non concluant après
   contrôle des comparaisons multiples. `undetermined_final` est mitigé (3/4
   régimes très améliorés, A6_obs020 légèrement dégradé) ; `coverage_auc` est
   légèrement inférieur (−1.6 à −5.5 %, FB couvre un peu plus vite au début) ;
   `time_to_quality` mitigé. Le pattern net et cohérent est porté par
   `mean_bound_final`.
4. **Pourquoi la primaire était mal choisie (leçon de protocole).** `quality_auc`
   est une mesure de saturation binaire à seuil 2-configurations : elle plafonne
   et ne résout pas les améliorations de précision *sous* le seuil. La métrique
   alignée avec l'objectif réel (« accuracy ») est la borne continue
   (`mean_bound_final`), secondaire pré-enregistrée depuis E2/E3. Le verdict A4
   est rendu sur la primaire telle que verrouillée (ÉCHEC), et l'effet de
   précision est rapporté explicitement comme **découverte cohérente, non
   concluante au seuil global**.
5. **Cohérence avec l'arc du projet.** Quatrième étape, premier résultat
   directionnellement positif du signal U comme *pilote* (et non plus seulement
   témoin) : là où la sélection de cible directe (Phase 1) et le changement de
   mode (E2/E3) ne battaient pas le contrôle géométrique, la **priorisation
   continue dans le même cadre de mouvement** réduit l'erreur résiduelle à
   budget fixe. Frontière claire : le signal ne gagne pas de la *vitesse* de
   couverture, il gagne de l'*accuracy terminale sous contrainte de temps*.

**Conclusion d'étape : ÉCHEC sur la primaire pré-enregistrée, mais premier
résultat positif cohérent (non concluant après Holm) sur la précision continue
sous budget fini.** L'effet est suffisamment net et aligné avec le titre de
thèse pour justifier une **campagne de confirmation à plus grande puissance**
(n > 10) sur `mean_bound_final` @ T comme primaire — lancée immédiatement
(pré-enregistrement : `PRE_REG_E4_CONFIRM.md`), voir section 5.

---

## 5. Confirmation haute puissance (E4-CONFIRM, pré-enregistrée)

> Pré-enregistrement avant tout résultat : `PRE_REG_E4_CONFIRM.md`
> (protocole, seuils, filet anti-dérive). Analyse codée dans
> `analysis/budget_stats.py` (section `E4-CONFIRM`), exécutée sans modification
> sur les données finales.

### Protocole (verrouillé avant lancement)
- n total = **40 paires par régime** (runs 0–9 conservés de E4, 30 nouvelles
  paires ajoutées par reprise — mêmes seeds, appariement intact).
- Méthodes : Frontier-Bounded + Coverage-U (Deploy-U = contexte, non reproduit).
- Budgets T inchangés (A2=4200, A3=3200, A6_obs005=1600, A6_obs020=1750).
- **Métrique primaire : `mean_bound_final` @ T** (plus bas = mieux), Wilcoxon
  apparié + Holm (4 régimes), Fisher combiné en énoncé global secondaire.
- **Succès pré-spécifié** : (i) réduction médiane relative du bound rel. FB
  ≥ 10 % ; (ii) au moins un régime Holm-sig ; (iii) aucune régression de
  couverture Holm-sig.
- Puissance estimée par bootstrap des différences appariées de E4 : Holm power
  0.86–1.00 à n = 40 (régime le plus faible A3 : 0.86).
- Campagne : 240 runs (2 méthodes × 4 régimes × 30 nouveaux), 10852 s (~3 h).

### Résultats (n = 40 paires/régime)

`mean_bound_final` @ T (plus bas = mieux) :

| Régime | med_FB | med_CU | rel-red % (vs FB) | p | p_Holm | delta |
|---|---|---|---|---|---|---|
| A2_obs005 | 0.0277 | **0.0209** | +24.7 | 0.0002 | 0.0003 | −0.576 |
| A3_obs005 | 0.0222 | **0.0185** | +17.0 | <0.0001 | <0.0001 | −0.554 |
| A6_obs005 | 0.0244 | **0.0181** | +25.7 | <0.0001 | <0.0001 | −0.526 |
| A6_obs020 | 0.0232 | 0.0219 | +5.4 | 0.221 | 0.221 | −0.139 |

**Verdict E4-CONFIRM : réduction médiane +20.9 % (seuil +10 %), 3/4 régimes
Holm-sig, aucune régression de couverture, Fisher combiné p ≈ 0
→ **PASS**.**

`undetermined_final` @ T (plus bas = mieux), corroboration indépendante :

| Régime | med_FB | med_CU | gain % | p | p_Holm |
|---|---|---|---|---|---|
| A2_obs005 | 0.0109 | **0.0032** | +71.7 | 0.017 | 0.034 |
| A3_obs005 | 0.0085 | **0.0009** | +84.8 | 0.0002 | 0.0009 |
| A6_obs005 | 0.0110 | **0.0031** | +83.2 | 0.009 | 0.028 |
| A6_obs020 | 0.0104 | 0.0125 | −37.2 | 0.259 | 0.259 |

Contrôles : `final_coverage` @ T sans régression (médianes quasi identiques :
67.4/67.2, 71.0/72.0, 68.3/68.7, 70.8/70.6) ; `quality_auc` @ T reste à parité
(gain médian +0.4 %, aucune Holm-sig) — la fraction binaire saturée ne bouge
pas, la précision continue oui.

### Interprétation de la confirmation
1. **L'effet est réel, pas du bruit à n = 10.** À n = 40, la réduction du bound
   CRLB résiduel sous budget est très significative dans les 3 régimes à
   obstacles rares (p ≤ 0.0002, Holm-sig) et corroborée par la fraction de
   cellules indéterminées (3/4 régimes Holm-sig, −37 % à +85 %). Fisher combiné
   p ≈ 0. Le pattern de E4 (4/4 régimes même sens, 3/4 bruts significatifs)
   se confirme et gagne en puissance.
2. **Limite systématique : le régime à 20 % d'obstacles.** A6_obs020 reste non
   significatif (+5.4 %, p = 0.221) et `undetermined_final` y est même légèrement
   dégradé (−37.2 %, ns). Interprétation : avec des obstacles denses, les
   cellules connues-libres sont rares et très fragmentées ; le bonus FOV a peu de
   sous-observé à viser et la mobilité est contrainte — l'effet de priorisation
   s'évanouit. L'effet est donc **conditionnel à un environnement peu obstrué**.
3. **Pas de coût de couverture.** Aucune régression de `final_coverage` (ni de
   `quality_auc`) ; le prix est une très légère baisse de `coverage_auc`
   (−0.5 à −4.0 %, FB couvre un peu plus vite en début d'épisode) et une
   `time_to_quality` mitigée. Le gain d'accuracy est acheté à couverture égale,
   pas contre elle.
4. **Signification pour la thèse.** Premier résultat **statistiquement
   significatif** de la richesse-configurations comme *pilote* de décision : à
   budget fini, prioriser la dépense de couverture vers les zones sous-observées
   réduit l'erreur de localisation résiduelle (bound CRLB et indéterminées) à
   couverture égale — le compromis « Accuracy + Coverage » du titre. L'effet est
   **propre à la contrainte de budget** : en épisodes non bornés (E2/E3),
   `quality_final` saturé et `mean_bound_final` à parité ; c'est sous T que la
   priorisation paie.

---

## 5bis. Sensibilité à λ / frontière Pareto (E4-PARETO, pré-enregistrée)

> Pré-enregistrement : `PRE_REG_E4_PARETO.md`. Analyse :
> `analysis/pareto_stats.py` (section `E4-PARETO`), exécutée telle quelle.

### Protocole (verrouillé avant lancement)
- λ ∈ {0.25, 0.5, 1.0, 2.0} sur **A3_obs005 + A6_obs005** (les deux régimes à
  effet confirmé) ; budgets T inchangés ; seeds 0–39.
- λ = 0 (FB) et λ = 0.5 déjà acquis à n = 40 → seuls λ ∈ {0.25, 1.0, 2.0}
  rejoués (240 nouveaux épisodes). Le runner a été parallélisé pour cette
  campagne (le filtre de workers — 4 processus — n'était auparavant jamais
  branché ; voir `num_workers` dans `_runner.py`).
- Wilcoxon apparié λ vs FB, **Holm intra-régime sur les 4 λ** ; garde
  `final_coverage` sans régression Holm-sig.
- **Succès pré-spécifié** : ≥ 2 des 4 λ satisfont (rel-red ≥ 10 % **et**
  Holm-sig) dans ≥ 1 régime, sans régression de couverture.

### Résultats (n = 40 paires/régime ; rel-red % vs FB)

| Régime | λ=0 (FB) | λ=0.25 | λ=0.5 | λ=1.0 | λ=2.0 |
|---|---|---|---|---|---|
| A3_obs005 | 0.0222 / 71.0 | **−17.0 %*** / 72.0 | −17.0 %* / 72.0 | −16.4 %* / 69.8 | **−23.9 %*** / 71.0 |
| A6_obs005 | 0.0244 / 68.3 | −23.9 %* / 68.0 | −25.7 %* / 68.7 | −23.4 %* / 65.9 | **−26.6 %*** / 67.3 |

(valeur = réduction rel. FB de `mean_bound_final` @ T (* = Holm-sig, tous
p ≤ 0.0002) / médiane `final_coverage` % ; aucune régression de couverture
Holm-sig, `med_cov%` jamais significativement inférieur à FB)

**Verdict E4-PARETO : les 4 λ passent dans les 2 régimes, garde couverture OK
→ **PASS**.**

### Interprétation
1. **Plateau, pas couteau.** Le gain de précision est stable sur tout le
   domaine λ ∈ [0.25, 2.0] (16.4–26.6 %), même légèrement croissant avec λ,
   **sans coût de couverture** (médianes 65.9–72.0 %, aucune Holm-sig). λ = 0.5
   (fixé d'office avant E4) est au cœur du plateau : le résultat confirmé
   **n'est pas un artefact de tuning**.
2. **La couverture est remarquablement inélastique.** Même à λ = 2.0, la
   priorisation n'abaisse pas significativement `final_coverage` — les agents
   continuent de couvrir (le bonus FOV n'éloigne pas des frontières, il choisit
   *laquelle*). La frontière Pareto du texte
   (`max f_cov − λ·Tr(P_loc)`) est ici presque horizontale en couverture pour
   cette gamme de λ : l'accuracy s'améliore « gratuitement » jusqu'au point où
   λ n'apporte plus rien.
3. **Recommandation pratique** : λ ≈ 1.0–2.0 est légèrement supérieur à 0.5 ;
   la différence est faible (≤ 7 points) et non contrôlée par un test
   λ-vs-λ. On conserve λ = 0.5 (valeur figée, pré-spécifiée) comme point de
   référence pour E5.

---

## 6. Fichiers produits

| Fichier | Rôle |
|---|---|
| `policies/coverage_u.py` | Politique Coverage-U (score continu λ=0.5 ; `_box_sum` image intégrale) |
| `policies/factory.py` / `config.py` | Branche `METHOD_COVERAGE_U` + `COVERAGE_U_LAMBDA = 0.5` |
| `experiments/run_budget.py` | Campagne E4 (budgets 0.7 × FB steps_90, résumable) |
| `tests/test_coverage_u.py` | 6 tests (dont λ=0 action-identique FB) |
| `analysis/budget_stats.py` | Wilcoxon apparié + Holm + verdict A4 + section secondaires précision |
| `results/budget_{A2,A3,A6_obs005,A6_obs020}/` | CSV bruts appariés FB / Deploy-U / Coverage-U (E4 : runs 0–9 ; E4-CONFIRM : runs 10–39 ajoutés, n=40) |
| `results/pareto_{A3,A6_obs005}/` | CSV Coverage-U par λ (n=40 chacun) |
| `PRE_REG_E4_CONFIRM.md` / `PRE_REG_E4_PARETO.md` | Pré-enregistrements (avant résultats) |
| `results/budget_campaign.log` / `budget_confirm.log` / `pareto_campaign*.log` | Logs de campagne |
| `experiments/_runner.py` | Parallélisation réelle (`num_workers`, pool de processus, écriture CSV unique parent) |

Reproduire : `python experiments/run_budget.py --runs 10` (E4) puis
`python experiments/run_budget.py --methods Frontier-Bounded Coverage-U --runs 40`
(E4-CONFIRM, reprend les runs 0–9) puis
`python experiments/run_pareto.py --lambdas 0.25 1.0 2.0 --workers 4`
(E4-PARETO ; λ=0.5 réutilisé) puis
`python analysis/budget_stats.py` et `python analysis/pareto_stats.py`.
