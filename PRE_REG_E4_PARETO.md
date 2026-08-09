# PRE-REGISTRATION — E4-PARETO (λ sensitivity / accuracy-coverage Pareto frontier)
## Projet08 — Coverage-U under finite budget

Date : 2026-08-05. Décidé après le verdict E4-CONFIRM (PASS, n=40, réduction
médiane +20.9 % sur `mean_bound_final` @ T, 3/4 régimes Holm-sig). Tout est
verrouillé avant le lancement ; aucune analyse rétrospective.

## Pourquoi
λ = 0.5 a été fixé d'office avant E4 (paramètre unique, aucun tuning). La
question honnête restante : l'effet confirmé est-il un couteau à λ = 0.5, et où
se trouve le point de bascule couverture ? On cartographie donc la frontière
précision-vs-couverture de Coverage-U.

## Protocole verrouillé
- **Régimes** : A3_obs005 et A6_obs005 (les deux où l'effet est confirmé et le
  plus fort) ; budgets T inchangés (3200 / 1600).
- **Grille de λ** : {0.25, 0.5, 1.0, 2.0}. λ = 0 est le contrôle Frontier-Bounded
  (déjà acquis). **λ = 0.5 est réutilisé tel quel** (déjà acquis à n = 40) —
  seuls λ ∈ {0.25, 1.0, 2.0} sont rejoués : 2 régimes × 3 λ × 40 seeds = **240
  nouveaux épisodes** (~2 h estimées).
- **Appariement** : même échelle de seeds que budget_* (0–39), même budget T,
  même grille/FOV/communication. Les nouvelles runs sont écrites dans
  `results/pareto_{régime}/raw_comm_limited__Coverage-U__lam{λ}.csv`.
- **Métriques** : précision = `mean_bound_final` @ T (plus bas = mieux),
  réduction rel. FB ; garde = `final_coverage` @ T ; support =
  `undetermined_final` @ T.
- **Tests** : Wilcoxon apparié λ vs FB, **Holm intra-régime sur les 4 λ**
  (λ = 0 exclu des tests), par régime.
- **Succès (pré-spécifié)** : PASS ssi, dans **au moins un** des deux régimes,
  **≥ 2 des 4 λ** satisfont (réduction rel. FB ≥ 10 % **et** Holm-sig sur
  `mean_bound_final`) **et** aucun de ces λ n'a de régression de couverture
  Holm-sig.
- **Lecture secondaire** : forme de la frontière — plateau vs couteau —
  (combien de λ passent, où la couverture commence à chuter, où la précision
  cesse de croître).

## Filet anti-dérive
- Aucun tuning rétrospectif de λ ni de seuils ; aucun filtrage de runs ; aucun
  ajout/suppression de régime. Analyse codée dans `analysis/pareto_stats.py`
  (section `E4-PARETO`), exécutée telle quelle sur les données finales.
- Les données λ = 0.5 et FB proviennent de `results/budget_*` (E4 +
  E4-CONFIRM, n = 40) ; les trois nouveaux λ de `results/pareto_*`.
