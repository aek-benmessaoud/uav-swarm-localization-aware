# PRE-REGISTRATION — E4-CONFIRM (higher-power confirmation campaign)
## Projet08 — Coverage-U under finite budget

Date : 2026-08-05. Décidé après le verdict E4 (primaire `quality_auc` = ÉCHEC,
mais effet cohérent sur la métrique continue `mean_bound_final` @ T). Aucune
analyse rétrospective : tout est verrouillé avant le lancement.

## Pourquoi
Avec n = 10 paires par régime, aucun p ne passe le Holm global (max 0.098) sur
l'effet de précision, pourtant présent dans les 4 régimes (réduction du bound
CRLB résiduel 13.6–28.3 %, 3/4 p bruts < 0.05). Estimation de puissance par
bootstrap des différences appariées observées :
- n = 30 : puissance Holm ≈ 0.72–0.98 selon le régime (régime le plus faible :
  A3, 0.72).
- n = 40 : puissance Holm ≈ 0.86–1.00 dans les 4 régimes (A3 : 0.86).

## Protocole verrouillé
- **Méthodes** : Frontier-Bounded (contrôle) + Coverage-U uniquement (Deploy-U
  = ligne de contexte, déjà acquise, non reproduite).
- **n total = 40 paires par régime** : les runs 0–9 existants sont conservés
  (mêmes seeds, appariement intact), 30 nouvelles paires (runs 10–39) ajoutées
  via le mécanisme de reprise du runner.
- **Budgets T inchangés** : A2=4200, A3=3200, A6_obs005=1600, A6_obs020=1750
  (0.7 × FB steps_90 E3).
- **Métrique primaire** : `mean_bound_final` @ T (plus bas = mieux).
  **Garde** : `final_coverage` @ T sans régression Holm-sig.
- **Test** : Wilcoxon apparié par régime + Holm-Bonferroni global (4 régimes).
  Fisher combiné (méta-analytique) rapporté comme énoncé global secondaire.
- **Succès (pré-spécifié)** :
  1. réduction médiane relative du bound (rel. FB) ≥ 10 % ;
  2. au moins un régime Holm-sig sur `mean_bound_final` @ T ;
  3. aucune régression de couverture Holm-sig.
- **Coût estimé** : 240 runs (2 méthodes × 4 régimes × 30 nouveaux) ≈ 41 s/run
  → ~2,7–3 h.

## Filet anti-dérive
- Aucun tuning de λ (0.5, figé), aucun filtrage de runs, aucune suppression de
  régimes. L'analyse s'exécute sur les CSV complets de `results/budget_*`.
- L'analyse de confirmation est codée dans `analysis/budget_stats.py` (section
  `E4-CONFIRM`) et s'exécute sans modification sur les données finales.
