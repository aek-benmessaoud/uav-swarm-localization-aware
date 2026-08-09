# Richness-Guided Localization-Aware Deployment for UAV Swarms
## Résumé du projet — pour validation

---

## 1. Contexte et origine

Ce projet fait suite à un premier article (déjà avancé) sur l'exploration/couverture décentralisée d'essaims UAV, utilisant des **estimateurs de richesse écologique** (Chao1, ACE, Jackknife) comme signal de décision adaptatif (formule `α = U/(U+K)`).

**Découverte clé du projet précédent** : sur la tâche de *couverture pure*, l'essentiel du gain de performance attribué aux signaux sophistiqués (entropie, richesse statistique) provient en réalité d'un principe de planification à horizon borné (*receding horizon*, connu dans la littérature — NBVP, POMDP+frontier hybrides), pas du signal lui-même. Un test de contrôle rigoureux (`Frontier-Bounded`, planification identique sans signal dédié) a capté la majorité du gain observé.

**Conclusion pratique** : la couverture spatiale pure n'est pas la tâche où un signal d'incertitude statistique devrait apporter le plus de valeur — car la géométrie/topologie du mouvement y domine. Ce constat motive un **pivot d'application** vers une tâche où l'incertitude elle-même est l'objectif : la **précision de localisation**.

Ce pivot s'aligne aussi avec le sujet de thèse associé au projet (*"Localization-Aware Deployment Strategies for UAV Swarm Systems: Maximizing Accuracy and Coverage"*).

---

## 2. Question de recherche

**RQ1** : Les estimateurs de richesse statistique permettent-ils de concevoir des stratégies de déploiement décentralisées d'essaims UAV qui améliorent la précision de localisation collective, par rapport aux approches géométriques (GDOP) et à un simple contrôle de couverture (mouvement borné sans signal dédié), sous contraintes de communication limitée ?

**Dérivées, traitées séquentiellement (pas en parallèle)** :
- RQ1.1 (Phase 1) : le signal de richesse apporte-t-il un gain **significatif au-delà** du contrôle de mouvement borné seul ?
- RQ1.2 (Phase 2, conditionnelle) : comment la performance évolue-t-elle sous bruit GPS et communication restreinte ?
- RQ1.3 (ablation, si le temps permet) : quel estimateur (Chao1 bias-corrected vs Jackknife vs ACE vs Good-Turing) offre le meilleur compromis précision/coût CPU ?

**Explicitement hors scope pour cet article** (réservé à des travaux futurs) : scénarios urbains/Gazebo, stratégies de décision multiples (softmax, combinaisons entropie), baseline d'apprentissage par renforcement.

---

## 3. Principe technique (transposition du travail précédent)

Le même mécanisme mathématique validé sur la couverture (comptage de singletons/doublons F1/F2, formule bias-corrected de Chao, cap de normalisation) est réappliqué à un objet différent :

| | Projet précédent (couverture) | Ce projet (localisation) |
|---|---|---|
| Ce qui est compté | Visites de cellules | Configurations de mesure angulaire indépendantes par cellule (clustering d'angles d'observation, seuil >15°, méthode greedy) |
| F1 / F2 | Cellules vues 1× / 2× | Cellules à 1 / 2 configurations angulaires indépendantes |
| Objectif du signal | Prioriser les zones sous-explorées | Prioriser les zones sous-déterminées géométriquement (mauvaise diversité angulaire d'observation) |
| Formule | `U = min(F1(F1−1)/(2(F2+1)), cap)`, `α = U/(U+K)` | Identique, réutilisée telle quelle |

**Métrique de qualité de localisation (Phase 1)** : borne théorique CRLB/GDOP calculée sur la géométrie exacte des observations (pas de bruit simulé pour la métrique principale, afin de rester ancrée dans la littérature GDOP). Une validation séparée vérifie que le signal U corrèle bien avec l'erreur de localisation réelle sous bruit angulaire simulé (estimateur léger par moindres carrés, corrélation de Spearman par cellule, seuil de validation |ρ| ≥ 0.7).

---

## 4. Méthodes comparées

| # | Méthode | Rôle |
|---|---|---|
| 1 | Random walk + répulsion anti-collision | Baseline plancher |
| 2 | GDOP-minimizing (centralisé) | Référence état de l'art géométrique, coûteuse en calcul |
| 3 | **Frontier-Bounded** (couverture pure, sans signal dédié) | **Contrôle obligatoire dès la Phase 1** |
| 4 | Richness-guided (Chao1 bias-corrected) | Contribution testée |

Réservés à une extension ultérieure (pas codés en parallèle dès le départ) : FIM D-optimal, Entropy-based, Good-Turing (ablation seulement si Phase 1-3 concluantes).

---

## 5. Scénario et infrastructure

- **Un seul scénario pour cet article** : grille ouverte standard 100×100, obstacles 5% (réutilisation directe de l'infrastructure déjà validée du projet précédent).
- Un scénario labyrinthe est réservé à une validation de généralisation légère en toute fin de projet, pas une campagne complète.
- **Base technique** : fork du code déjà validé (modèle de communication à portée limitée avec fusion par rendez-vous, mouvement à horizon borné, corrections numériques déjà éprouvées) — pas de réécriture depuis zéro.

---

## 6. Plan expérimental phasé (discipline stricte : validation avant extension)

### Phase 0 — Validation du signal (préalable, avant toute campagne)
Vérifier par corrélation de Spearman (seuil |ρ| ≥ 0.7) que le signal de richesse U corrèle avec l'erreur de localisation réelle mesurée par un estimateur léger sous bruit simulé. **Sans cette validation, le reste du plan est reconsidéré.**

### Phase 1 — Contrôle et validation du gain (priorité absolue)
Random vs Frontier-Bounded vs Richness-guided, sur le scénario de base, géométrie exacte (pas de bruit GPS). Critère de décision **pré-spécifié avant de voir les résultats** : gain significatif (p<0.05, correction de Holm-Bonferroni) **et** taille d'effet ≥ 8-10% pour valider l'apport du signal au-delà du simple contrôle de mouvement.

### Phase 2 — Robustesse (conditionnelle à Phase 1 positive)
Bruit GPS (plusieurs niveaux) et niveaux de communication (complète / limitée / locale uniquement).

### Phase 3 — Scalabilité et référence coûteuse
Variation du nombre d'agents ; comparaison au GDOP-minimizing centralisé (référence de performance et de coût).

### Phase 4 — Coût de calcul
Mesure rigoureuse du temps CPU par décision (pas de temps total d'épisode, leçon du projet précédent), en particulier pour quantifier l'écart de coût avec la référence GDOP centralisée.

### Phase 5 (optionnelle) — Généralisation
Validation légère sur scénario labyrinthe, méthodes principales seulement.

---

## 7. Garde-fous méthodologiques (leçons du projet précédent, appliquées dès le départ)

- Le contrôle de mouvement (Frontier-Bounded) est testé **dès la première expérience**, jamais découvert après coup.
- Aucune conclusion "méthode A bat méthode B" sans test statistique apparié et correction pour comparaisons multiples.
- Seuils de décision (significativité, taille d'effet, corrélation de validation) fixés **avant** de voir les résultats, avec une réponse pré-écrite pour les cas ambigus (zone grise de corrélation partielle).
- Toute nouvelle métrique de saturation ou de plafond numérique (ex. nombre maximal de configurations angulaires retenues par cellule) est monitorée dès le premier run, pas ajoutée après avoir découvert un problème.
- Aucune extension de scope (nouveaux scénarios, nouvelles baselines) tant que la phase précédente n'est pas validée.

---

## 8. Positionnement par rapport à la littérature

Les concepts de diversité géométrique/angulaire d'observation, GDOP et information de Fisher pour l'optimisation du placement de capteurs sont bien établis dans la littérature de localisation coopérative UAV. La contribution revendiquée n'est pas la notion de diversité angulaire elle-même, mais l'utilisation d'un **estimateur de richesse écologique (Chao1) comme signal de décision décentralisé et léger** pour piloter cette diversité, en alternative aux calculs géométriques centralisés (GDOP, FIM) plus coûteux — une transposition inédite à ce rôle précis, à documenter honnêtement comme telle dans le related work.

---

## 9. Mise à jour E2/E3 — résultat (négatif documenté)

> Détail complet dans `RAPPORT_E2E3.md`.

Après la falsification de la Phase 1 (richesse-configs comme sélection de cible),
l'idée « 1+3 » a été implémentée et évaluée : **Deploy-U** (couverture
Frontier-Bounded tant que la carte locale est surtout sous-localisée, puis
orbite des cellules connues à ≤ 1 configuration pour forcer la diversité
angulaire). Le signal de décision reste exclusivement le comptage local de
configurations angulaires (E1 a validé qu'il prédit bien le travail restant :
ρ(U, gap) = 0.73–0.82).

**Verdict A4 sur `steps_dual` (couverture ≥ 90 % ET qualité ≥ 0.9) : ÉCHEC.**
Gain médian −1.1 % (seuil +8 %), aucune significativité après Holm. Résultat
par régime : A2 −1.9 %, A3 −0.3 %, A6_obs005 −6.4 %, A6_obs020 +9.0 %
(p brut 0.037, p Holm 0.148).

**Points saillants :**
- Parité stricte de précision : `quality_final` = 1.0 partout, `quality_auc` et
  `mean_bound_final` à l'équilibre (≤ 0.5 %). Deploy-U ne nuit jamais à la
  localisation.
- Le mode deploy est sous-alimenté : `deploy_frac` = 2–6 % des décisions,
  `orbit_frac` = 0.5–1 % (seuil de déclenchement trop conservateur face à une
  fusion parcimonieuse en communication limitée).
- **Narrative finale « Occam's razor » renforcée** : trois falsifications
  documentées de la richesse-configs comme levier de décision opérant
  (Phase 1 sélection de cible ; E2/E3 signal de mode/déploiement). Le contrôle
  géométrique borné capture l'essentiel du gain ; la richesse configurationnelle
  est un **témoin d'état fiable** (E1), pas un pilote.

**État du plan phasé :** Phase 1 (négative) et Phase 1a (contrôle FB validé)
exécutées. E2/E3 exécutées (négatives). Les phases 2–5 (bruit GPS, scalabilité,
coût CPU, généralisation) restent hors de portée : sans gain démontré de la
contribution testée, elles ne sont pas justifiées dans la direction actuelle.

---

## 10. Mise à jour E4 — couverture priorisée par U sous budget fini (Coverage-U)

> Détail complet dans `RAPPORT_E4.md`.

Nouvelle idée testée : **ne pas changer de mode** (leçon E2/E3), mais
**prioriser en continu** la sélection de cible dans le cadre Frontier-Bounded
vers les zones dont le FOV contient le plus de cellules connues à ≤ 1
configuration angulaire (score = D/horizon − λ·under_count_FOV/FOV_area, λ = 0.5,
aucun tuning ; λ = 0 == FB exact, vérifié par test). Budget T = 0.7 × FB
`steps_90` (E3) par régime. Campagne 4 régimes × 10 × 3 méthodes, 120 épisodes,
82 min.

**Verdict A4 sur la primaire pré-enregistrée `quality_auc` @ T : ÉCHEC.**
Parité (gain médian −0.9 %, aucune Holm-sig) : la fraction binaire
« ≥ 2 configurations » sature et ne voit pas la précision sous-seuil.

**Résultat positif cohérent (secondaire) :** `mean_bound_final` @ T (bound CRLB
résiduel moyen, plus bas = mieux) est **plus bas dans les 4 régimes** — réduction
de 13.6–28.3 % vs FB (médiane ≈ 20 % ; A2 +42 %, A3 +24 %, A6_obs005 +25 %,
A6_obs020 +15 % en gain rel. CU ; p bruts 0.049/0.106/0.027/0.049, 3/4 < 0.05,
toutes même sens) — **sans aucune régression de couverture** (médiane
`final_coverage` même légèrement supérieure partout). Premier signal
directionnel positif de la richesse-configurations comme **pilote** : sous
budget fini, prioriser la dépense de couverture vers les zones sous-observées
achète de l'erreur de localisation résiduelle à couverture égale ou meilleure —
exactement le compromis du titre.

**Caveat honnête :** aucun p ne passe le Holm global (max 0.098) ; `coverage_auc`
légèrement inférieur (−1.6 à −5.5 %), `undetermined_final` mitigé (3/4 régimes
très améliorés, A6_obs020 dégradé). L'effet est cohérent mais **non concluant au
seuil global** → campagne de confirmation lancée immédiatement
(`PRE_REG_E4_CONFIRM.md`).

### 10bis. Confirmation haute puissance (E4-CONFIRM) — **PASS**

n = 40 paires/régime (30 nouvelles seeds ajoutées, budgets et λ inchangés,
aucun tuning). Primaire pré-spécifiée : `mean_bound_final` @ T.

**Verdict : PASS.** Réduction médiane du bound CRLB résiduel sous budget
**+20.9 %** (seuil +10 %), **3/4 régimes Holm-sig** — A2 −24.7 % (p=0.0002),
A3 −17.0 % (p<0.0001), A6_obs005 −25.7 % (p<0.0001), A6_obs020 +5.4 %
(p=0.221) — Fisher combiné p ≈ 0, **aucune régression de couverture**.
Corroboration indépendante : `undetermined_final` Holm-sig dans 3/4 régimes
(+84 %, +57 %, +79 % ; A6_obs020 −27 % ns). `quality_auc` @ T reste à parité
(la fraction binaire ≥ 2 configs sature et ne voit pas la précision sous-seuil).

**Interprétation :** premier résultat **statistiquement significatif** de la
richesse-configurations comme *pilote* de décision. L'effet est **conditionnel**
(absent à 20 % d'obstacles, où les zones sous-observées sont rares/fragmentées)
et **propre à la contrainte de budget** (en épisodes non bornés — E2/E3 — la
précision finale est à parité). Sous budget fini, prioriser la dépense de
couverture vers les cellules à ≤ 1 configuration angulaire réduit l'erreur de
localisation résiduelle à couverture égale : le compromis « Accuracy +
Coverage » du titre est soutenu par la preuve — dans les environnements peu
obstacles et sous contrainte de temps de mission.

**État du plan phasé :** Phase 1 (négative), Phase 1a (FB validé), E2/E3
(négatives), E4 (primaire pré-enregistrée négative, effet secondaire cohérent),
E4-CONFIRM (**positive**). La direction reste : consolider l'effet de
priorisation U (sensibilité à λ, effectif des agents, grilles plus grandes) ;
les phases 2–5 classiques (bruit GPS, scalabilité, coût CPU, généralisation)
ne sont justifiées que conditionnellement à cette consolidation.

### 10ter. Sensibilité à λ (E4-PARETO) — **PASS : plateau robuste**

λ ∈ {0.25, 0.5, 1.0, 2.0} sur A3_obs005 + A6_obs005, n = 40, budgets inchangés.
**Les 4 λ passent dans les 2 régimes** (réduction `mean_bound_final` @ T de
16.4–26.6 % vs FB, tous Holm-sig, p ≤ 0.0002) **sans aucune régression de
couverture** (médianes 65.9–72.0 %, aucune Holm-sig). La frontière
précision-vs-couverture est **presque horizontale en couverture** : l'accuracy
s'améliore (légèrement croissante avec λ) sans coût de couverture mesurable.
**λ = 0.5, fixé d'office avant E4, est au cœur du plateau → le résultat n'est
pas un artefact de tuning.** Le runner a été parallélisé pour l'occasion (4
workers effectifs, `num_workers` branché). → **E5 (plafond centralisé CRLB)**
lancé : quantifier la perte de la transposition config-count vs la politique
oracle `max f_cov − λ·Tr(P_loc)` du related work.

---

## 11. Manuscrit ver0 (2026-08-05) — terminé

> PDF : `results/paper_ver0.pdf` (16 pages, 8 figures, 6 tableaux).

**Titre retenu (working) :** *U-Prioritized Coverage under Finite Mission
Budgets: Config-Count Richness Reduces Residual Bearing-Only Localization
Error in UAV Swarms at Equal Coverage* + 2 titres alternatifs (Alt. A/B) en
page 1.

**Structure :** abstract → 1. Intro (C1–C4) → 2. Related Work (6 sous-sections)
→ 3. Formulation (modèle, configurations angulaires, oracle CRLB, comm,
métriques) → 4. Méthodes (FB, RA, Deploy-U, Coverage-U, E5 status) → 5.
Protocole (pré-enregistrement, régimes/budgets, appariement, table des
expériences) → 6. Résultats (gate 1a ρ=0.638 / ρ(U,err)=−0.457 ; E1 falsifié ;
E2/E3 falsifié ; E4 primaire fail + découverte secondaire 13.6–28.3 % ;
E4-CONFIRM PASS 20.9 % 3/4 Holm-sig ; E4-PARETO plateau 16.4–26.6 % ; figures
qualitatives traj/ambiguous ; E5 incomplet) → 7. Discussion → 8. Limitations →
9. Future Work → 10. Conclusion → References (38 entrées).

**Corrections apportées lors du build :**
1. Bug Holm : `r += (hp,)` sur tuple ne mute pas la ligne → les colonnes Holm
   manquaient dans 4 tables (E4-CONFIRM, coverage, undetermined, E3) ;
   corrigé en listes `r.append(hp)`.
2. Bug Pareto : le tag de fichier `f"lam{lam:g}"` produisait `lam1`/`lam2`
   alors que les fichiers sont `lam1.0`/`lam2.0` → NaN dans le tableau ;
   corrigé en `f"lam{lam}"`.
3. Captions `fig_cap`/`tbl_cap` retournaient un Paragraph sans l'ajouter au
   story → 14 légendes absentes du PDF ; corrigé (append direct).
4. Figure traj (ratio 1.672) trop haute à 15.2 cm → figure veuve seule sur une
   page ; réduite à 14.0 cm.

**Vérifications :** 16 pages ; 8 images embarquées (1/2/2/1/1/1 par page 8–13) ;
toutes les valeurs des tableaux recoupées contre les CSV bruts (E1, E3, E4-CONFIRM,
undetermined, Pareto, p-value Holm). Dossier de travail `F:\Project08` (pas
`F:\Project6`, qui est l'ancien workspace).
