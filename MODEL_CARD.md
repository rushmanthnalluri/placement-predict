# Model Card — Placement Predict (Gradient Boosting, v2 — calibrated)

## Model details

- **Model:** `HistGradientBoostingClassifier` (scikit-learn 1.9), default depth, lr 0.1
- **Calibration (new in v2):** Platt sigmoid — a logistic map fit on 3-fold
  out-of-fold predictions within the training split
  (`CalibratedClassifierCV(method="sigmoid", ensemble=False)`); the base model
  is refit on the full training split and the sealed test set never
  participates. ROC-AUC is unchanged by construction (monotone remap);
  Brier/log-loss improve.
- **Selected:** by 3-fold cross-validated ROC-AUC on a 12,000-row stratified
  subsample of the training split, against a logistic-regression baseline and
  a 150-tree random forest
- **Serving:** all three candidates stay fitted and selectable — the UI model
  dropdown and the `model` field of `/api/predict` accept any of them, with
  the champion as the default/"best". `/api/benchmark` reports the shared
  sealed-split evaluation for any subset. Per-model artifact files
  (`model_artifact_<key>.joblib`, sha256- and recipe-version-validated) load
  on demand; the main artifact carries the champion for a fast cold start.
- **Owner/repo:** [rushmanthnalluri/placement-predict](https://github.com/rushmanthnalluri/placement-predict) · audit: `docs/audit/FINAL_AUDIT.md`

## Intended use

Predict the probability that an engineering student is placed, from academic
and skill-profile features, for **education/demo purposes** — portfolio project
for a 12-week classical-ML track. **Not for admissions, hiring, or any
consequential decision about a real person.**

## Training data

- Synthetic 50,000-record dataset (31 fields) modelled on Indian
  engineering-college placement data; bundled at
  `flask_project/data/placement_predict_50k.csv`.
- Cleaning: the corrupt sentinel row (StudentID 0 — its values are the
  per-column missing counts) is dropped → 50,000 usable records.
- The 1,750 records flagged `IsAnomaly` are **retained deliberately**: their
  placement rate matches the population (65.71%) and they act as
  label-consistent noise. Disclosed, not silent.
- Missing values (19,976 cells across Workshops, AptitudeTestScore,
  SoftSkillsRating, CodingTestScore, MockInterviewScore) are imputed with
  **training-split means**; the linear model's features are z-score
  standardised (train-only statistics); trees read raw imputed values.

## Features (12)

CGPA, AttendancePercent, Internships, Projects, Workshops, Certifications,
Publications, AptitudeTestScore, SoftSkillsRating, CodingTestScore,
MockInterviewScore, ExtraCurricular. Target: `PlacementStatus` (1 = placed).

**Excluded deliberately:** StudentID (identifier), IsAnomaly (quality flag),
CGPA_Tier (noisy CGPA proxy — no added signal), SGPA_Sem1–8 (subsumed by CGPA),
and all demographics (Gender, CollegeTier, Stream, …) from the headline model.

## Evaluation

Protocol: stratified 80/20 split (seed 42) sealed before any transform was
fit; model selection by 3-fold CV on a 12,000-row stratified subsample of the
training split; calibration fit on out-of-fold training predictions; the test
set was assessed **once**. The v1 (pre-calibration) numbers were reproduced
byte-identically by the forensic audit (`docs/audit/`); the table below is the
current v2 recipe.

| Model | CV ROC-AUC (train) | Accuracy | Precision | Recall | F1 | ROC-AUC (test) | Brier ↓ | Log-loss ↓ |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.9638 ± 0.0022 | 0.8923 | 0.9021 | 0.9379 | 0.9196 | 0.9595 | 0.0744 | 0.2385 |
| Random Forest | 0.9725 ± 0.0023 | 0.9082 | 0.9146 | 0.9489 | 0.9314 | 0.9716 | 0.0662 | 0.2175 |
| **Gradient Boosting (champion)** | **0.9726 ± 0.0025** | **0.9086** | **0.9155** | **0.9484** | **0.9317** | **0.9733** | **0.0619** | **0.1927** |

Confusion matrix (champion, sealed test, threshold 0.5): TP 6,232 · FP 575 ·
FN 339 · TN 2,854.

Top drivers (RF importance / target correlation agree at the top): CGPA,
MockInterviewScore, then the skill-score cluster.

## Limitations

- **Synthetic data.** Patterns are real-looking but generated; do not
  generalise conclusions to real students.
- **Feature timing.** Mock-interview and test scores may be concurrent with
  the placement process; a truly pre-placement model would need earlier
  features only.
- **Calibration is fitted, not perfect.** v2 Platt-calibrates every served
  model (sigmoid, 3-fold out-of-fold); the evaluation page shows reliability
  curves against the sealed test set. Treat a predicted 80% as ≈80% *on this
  synthetic dataset* — it remains an estimate, not a frequency guarantee for
  any individual.
- **No fairness audit on outcomes.** Demographics are excluded from the
  feature set, but proxy bias via CGPA/skills is possible; no group metrics
  are computed in v1.
- **Fixed threshold (0.5).** No cost-sensitive tuning is applied.

## Reproducibility

Seed 42 everywhere; two fresh-process training runs produce byte-identical
metric bundles (audit-verified for the v1 recipe; v2 adds calibration on top
of the same split and transforms). Retrain: start the app and hit `/train`
(~40 s cold with calibration, cached after). Full environment:
`requirements.txt`; container: `Dockerfile`.
