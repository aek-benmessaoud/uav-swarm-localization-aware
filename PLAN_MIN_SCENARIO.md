# PLAN — Scénario minimum « réponses aux baselines du reviewer » (MIS EN PAUSE)

> Statut : PAUSÉ le 2026-08-06 à la demande de l'utilisateur.
> Reprise : relancer les commandes ci-dessous telles quelles (runner résumable).

## Objectif
Compléter le tableau budget (A3+A6, n=40) avec Random, Richness-Angular,
Entropy-Frac, Frontier+Entropy ; ajouter un benchmark CPU/décision ; mettre à
jour les stats et le papier. Répond aux points #2/#3/#4 du retour reviewer
(l'entropie classique, le floor random, la richesse pure, le claim CPU).

## État actuel des données
- `results/budget_A3_obs005/raw_comm_limited__Random.csv` : **16/40** (arrêté)
- Autres : 0/40 (à lancer). Régime A6 : rien de nouveau (0/40).
- Rien d'autre à changer ; les campagnes E5/E5-CORRECTED existantes sont intactes.

## Commandes (reprises à reprendre)

### Étape 1 — Campagnes budget (320 épisodes, ~2.7 h wall, 4 workers)
```
python experiments/run_budget.py --methods Random Richness-Angular Entropy-Frac Frontier+Entropy --runs 40 --regimes A3_obs005 --workers 4
python experiments/run_budget.py --methods Random Richness-Angular Entropy-Frac Frontier+Entropy --runs 40 --regimes A6_obs005 --workers 4
```
Temps estimés : A3 ≈ 72 min, A6 ≈ 91 min (108 s / 137 s par batch×4).
Resume automatique : Random A3 reprendra aux runs 16–39.

### Étape 2 — Benchmark CPU/décision (~60 min, sériel)
Petit script inline réutilisant `run_episode(..., timing=True)` (déjà dans
`experiments/_runner.py` + `TIMING_FIELDS`), sur FB / Coverage-U / Entropy-Frac /
Random, 10 runs par méthode, longueur budget (3200 / 1600), `max_steps`=budget.
Sortie : `ms_per_decision` par méthode → tableau dans le papier.

### Étape 3 — Stats + papier (~15 min)
- Nouveau bloc dans `analysis/budget_stats.py` (ou script dédié
  `analysis/baseline_budget_stats.py`) : table Random/FB/RA/Entropy-Frac/
  Frontier+Entropy/Coverage-U sur mean_bound_final, final_coverage, quality_auc
  @ T ; Wilcoxon apparié + Holm.
- Mettre à jour `analysis/paper_build.py` : étendre la table E4/E5 avec les
  nouvelles lignes budget + la ligne CPU, régénérer `paper_ver02.pdf`.
- Ajuster EXPERIMENTS_OVERVIEW.md et PRE_REG si nécessaire (ces runs utilisent
  le même protocole pré-enregistré E4 ; l'entropie est une falsification
  documentée attendue, pas une nouvelle découverte).

## Verdicts attendus (hypothèses, à confirmer)
- Random : floor sur couverture ET mean_bound.
- Entropy-Frac / Frontier+Entropy : à parité ou pire que FB (leçon V4) →
  falsification documentée de plus.
- Richness-Angular : couverture plus lente, mean_bound intermédiaire.
- Coverage-U : reste le meilleur sur mean_bound à couverture égale.
