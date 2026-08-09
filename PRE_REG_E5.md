# PRE-REGISTRATION — E5 (plafond centralisé / oracle CRLB)
## Projet08 — Transposition du signal config-count

Date : 2026-08-05. Décidé après E4-PARETO (PASS : plateau sur λ, tous Holm-sig
dans les 2 régimes). Tout est verrouillé avant le lancement ; aucune analyse
rétrospective.

## Pourquoi
E4 a validé l'utilisation du signal config-count local (Coverage-U, λ = 0.5).
Question honnête restante : **quelle part du gain d'un orateur centralisé
parfait** (connaissance globale + vraie géométrie CRLB) le transposé décentralisé
capture-t-il ? E5 mesure ce plafond et le ratio de transposition :

    ρ(régime) = réduction_CoverageU / réduction_CentralOracleCRLB   (médianes, rel. FB)

Un orateur centralisé maximise `score(cible) = D/horizon − λ·Tr(P_loc)/FOV_area`
(forme exacte du texte GDOP multi-objectif) avec connaissance globale. C'est un
**contrôle, jamais une méthode proposée** (infaisable déployé : carte vraie +
CRLB oracle). Il borne toute politique décentralisée.

## Protocole verrouillé
- **Régimes** : A3_obs005 et A6_obs005 (ceux d'E4-PARETO) ; budgets T inchangés
  (3200 / 1600), seeds 0–39, même grille/FOV/communication.
- **Nouvelles méthodes** (λ = 0.5, même forme de score que Coverage-U, cadre de
  mouvement BFS global = carte vraie) :
  - `CentralOracle-Config` : sous-ensemble = cellules libres observées avec
    config-count global ≤ 1 (fusion parfaite du signal Coverage-U) ;
  - `CentralOracle-CRLB` : sous-ensemble = cellules libres observées dont le
    CRLB oracle `sqrt(trace(J⁻¹)) > QUALITY_THRESHOLD` (1.5 cellules) — le vrai
    goulot de précision, inaccessible localement.
  - Échelle : CU(config) → CentralConfig → CentralCRLB isole (i) la valeur de
    la fusion parfaite et (ii) l'écart de proxy config-count vs CRLB.
- **Données existantes réutilisées** : FB (contrôle) et Coverage-U λ = 0.5 déjà
  à n = 40 dans `results/budget_*` (E4 + E4-CONFIRM). Seules les deux méthodes
  oracle sont rejouées : 2 régimes × 2 méthodes × 40 = **160 nouveaux épisodes**
  (~3–4 h, 4 workers).
- **Métriques** : précision = `mean_bound_final` @ T (plus bas = mieux, réduction
  rel. FB) ; garde = `final_coverage` @ T ; support = `undetermined_final` @ T ;
  secondaire = `quality_auc` @ T.
- **Tests** : Wilcoxon apparié vs FB, Holm sur les 2 régimes par méthode ;
  comparaison appariée CentralConfig vs CentralCRLB sur la réduction.
- **Succès (pré-spécifié)** : **PASS** ssi, dans **les deux** régimes :
  1. réduction_CentralCRLB ≥ réduction_CoverageU (l'oracle centralisé ne fait
     jamais moins bien que la méthode décentralisée — échelle monotone) ;
  2. ratio de transposition ρ ≥ 0.5 (le transposé config-count garde ≥ la moitié
     du plafond centralisé) ;
  3. aucune régression de couverture Holm-sig (oracle vs FB).
- **Lecture secondaire (diagnostic, pas un verdict)** : écart de ladder
  réduction_CentralCRLB − réduction_CentralConfig ≥ 0 (le proxy config-count
  perd-il visiblement face au CRLB même à fusion parfaite ?) et valeur de la
  fusion parfaite réduction_CentralConfig − réduction_CoverageU.

## Filet anti-dérive
- Aucun tuning rétrospectif (ni de λ, ni de seuils, ni de régimes) ; aucun
  filtrage de runs. Analyse codée dans `analysis/e5_stats.py` (section `E5`),
  exécutée telle quelle sur les données finales.
- L'oracle reste hors des tableaux « méthodes proposées » ; il est rapporté
  comme borne/contrôle.
