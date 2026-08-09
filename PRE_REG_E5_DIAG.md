# PRE-REGISTRATION — E5-DIAG (diagnostic : oracle à garde de couverture)
## Projet08 — Origine du FAIL E5 : calibration vs structure

Date : 2026-08-05. Rédigé AVANT tout run. Suite au verdict E5 = FAIL
(les deux oracles centralisés CentralOracle-Config/CRLB régressent la couverture
de 68–71 % → 36–42 % et dégradent mean_bound_final vs FB, Holm-sig, n = 40).

## Question diagnostique (une seule)

Le FAIL d'E5 vient-il
(a) de la **calibration** : λ = 0.5, calibré pour un sous-ensemble LOCAL,
    appliqué tel quel à un sous-ensemble GLOBAL (bien plus dense) rend le bonus
    d'accuracy dominant → les agents convergent tous vers la même zone à fort
    bonus et la dispersion spatiale (donc la couverture) s'effondre ; ou
(b) de la **structure** : prioriser l'accuracy à connaissance parfaite est
    intrinsèquement perdant sous budget fini, quelle que soit la force du signal ?

## Variante unique (une seule ligne ajoutée, aucun autre paramètre)

`CentralOracle-CRLB-cov` — identique à CentralOracle-CRLB (même sous-ensemble
CRLB global, même λ = 0.5, même score, même cadre de mouvement BFS global,
même horizon) SAUF que le bonus d'accuracy est **plafonné par le terme de
couverture** dans le score de cible :

    score(t) = D(t)/H − min( λ·U(t)/FOV_area , (1−ε)·D(t)/H ),   ε = 0.05 fixé

Le terme de distance (couverture) garde donc toujours ≥ 5 % de poids : le bonus
d'accuracy peut réordonner les cibles de couverture équivalentes (le mécanisme
du papier) mais ne peut plus faire gagner une cible lointaine sur une proche.
Pas de tuning : ε = 0.05 et λ = 0.5 sont fixés d'office, avant le run.

## Protocole verrouillé

- **Régimes** : A3_obs005 et A6_obs005 (ceux d'E5), budgets T inchangés
  (3200 / 1600), seeds 0–39, mêmes grilles/FOV/communication.
- **n = 40** runs appariés, une seule nouvelle méthode rejouée.
- **Métriques** : primaire = `mean_bound_final` @ T (médiane, réduction rel. FB) ;
  garde = `final_coverage` @ T ; support = `undetermined_final` ; secondaire =
  `quality_auc` @ T ; mécanisme = `overlap` (re-observation) et `coverage_auc`.
- **Tests** : Wilcoxon apparié vs FB, Holm sur les 2 régimes.
- **Lecture (diagnostic, PAS un verdict de méthode)** :
  - Si `red_Cov-CRLB ≥ 0` (amélioration) **et** garde propre → l'hypothèse (a)
    calibration est soutenue : la centralisation PEUT aider quand le bonus ne
    domine pas la couverture ; E5-FAIL s'explique par une force de signal
    mal calibrée à l'échelle globale.
  - Si `red_Cov-CRLB ≤ 0` **ou** garde en régression → l'hypothèse (b) structure
    est soutenue : la priorisation centralisée de l'accuracy est
    intrinsèquement perdante sous budget fini ; la localité de Coverage-U est
    la propriété qui compte, pas la force du signal.
  - La comparaison `red_Cov-CRLB` vs `red_CoverageU` quantifie : la garde de
    couverture suffit-elle à dépasser le proxy local (si oui, de combien) ?

## Filet anti-dérive

- Ceci est un DIAGNOSTIC de la discussion, pas une correction de l'E5 original :
  le verdict E5 = FAIL reste inchangé et est rapporté tel quel.
- Aucun tuning rétrospectif ; analyse codée dans `analysis/e5_stats.py`
  (section E5-DIAG), exécutée telle quelle sur les données finales.
- Si le diagnostic (a) est soutenu, l'exploration d'une variante « λ oracle
  réduit » constitue une NOUVELLE pré-enregistrement, pas un ajustement ici.

---

# RÉSULTAT (saisi APRÈS les runs, à la lecture du verdict)

## Extension verrouillée : CentralOracle-CRLB-cov2 (ε = 0.30)

Décision prise avant tout run cov2, pour lever l'ambiguïté laissée par le
premier diagnostic : le cap ε = 0.05 ne s'enclenche jamais pour D ≥ 5
(condition `bonus > 28.7·D` vs max FOV = 121), donc il ne teste pas la zone de
dégradation. `cov2` réutilise EXACTEMENT la même méthode
(CentralOracle-CRLB, même score, même λ = 0.5, même cadre BFS) avec le seul
paramètre prévu changé :

    ε = 0.05 → ε = 0.30   (cap : bonus_term ≤ 0.70 · cov_term)

Condition de déclenchement vérifiée par construction AVANT le run :
`bonus > 21.2·D` → le cap lie pour D = 1..5 (max 121), donc touche bien la
chasse aux cibles lointaines. Vérifié sur trajectoire : 30/900 décisions
changées entre cov et cov2 (3.3 %), preuve que le cap n'est pas inerte.
Fichiers : `raw_comm_limited__CentralOracle-CRLB-cov2.csv`, n = 40 × 2 régimes.

## Verdict DIAGNOSTIQUE (lecture verrouillée, pas un verdict de méthode)

| régime | méthode | mb (médiane) | cov (médiane) | p apparié vs Oracle |
|---|---|---|---|---|
| A3_obs005 | Oracle-CRLB | 0.0445 | 42.1 % | — |
| A3_obs005 | Oracle-cov (ε .05) | 0.0445 | 42.1 % | p(mb)=0.85, p(cov)=0.82 |
| A3_obs005 | Oracle-cov2 (ε .30) | 0.0445 | 42.1 % | p(mb)=1.00, p(cov)=0.96 |
| A6_obs005 | Oracle-CRLB | 0.0487 | 36.2 % | — |
| A6_obs005 | Oracle-cov (ε .05) | 0.0487 | 36.2 % | p(mb)=0.84, p(cov)=0.92 |
| A6_obs005 | Oracle-cov2 (ε .30) | 0.0487 | 36.2 % | p(mb)=0.90, p(cov)=0.82 |

Mécanique : cov2 change de vraies décisions (A/B 3.3 %) et déplace les runs
individuels dans les deux sens (A3 : mb mieux 15 / pire 13 / égal 12 ;
A6 : 16 / 15 / 9), mais l'effet net est nul — médianes identiques à la 4e
décimale, aucun test significatif. Autant de runs améliorés que dégradés.

## CONCLUSION DU DIAGNOSTIC (corrigée après instrumentation)

Le garde-fou ne s'est **jamais déclenché** : instrumenté sur de vraies
trajectoires (4 800 cibles atteignables évaluées), le cap `bonus > 21.2·D`
présente **0 binding** même en ε = 0.30 — le sous-ensemble CRLB global est
trop dispersé pour qu'aucun FOV n'atteigne le seuil. Les variantes
`-cov` / `-cov2` sont donc **bit-identiques** à l'oracle non gardé sur les
40 seeds des deux régimes (mb et cov identiques à la 4e décimale, p = 1.0).
Un test A/B naïf (appels séquentiels de `select_action` sur le même env)
avait suggéré 3.3 % de décisions changées ; ce test était invalide car il
mutait l'état partagé de l'env. Le test correct (envs séparés) confirme des
trajectoires identiques.

Le DIAG est donc **inconclusif comme test mécaniste** : le garde-fou ne
peut pas trancher calibration-vs-structure dans ce régime.

**L'argument « localité » ne dépend PAS de ce diagnostic.** Il repose sur
l'échelle local-vs-global, qui est solide et inchangée :
- A3 : Coverage-U (config-count LOCAL) red = **+32.5 %** vs FB, couv. 72 % ;
  CentralOracle-Config (même signal, fusion GLOBALE) red = **−47.7 %**,
  couv. 42 % ;
- A6 : Coverage-U red = **+29.1 %** (couv. 69 %) ; Oracle-Config red =
  **−50.7 %** (couv. 36 %).

Le même signal, scoré localement, gagne ; scoré globalement, il régresse.
C'est la preuve que la contrainte locale (cadre de couverture borné + score
local) est la propriété qui rend le compromis atteignable.
