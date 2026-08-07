# Model Card — Placement Predict (Gradient Boosting, v1)

## Model details

- **Model:** `HistGradientBoostingClassifier` (scikit-learn 1.9), default depth, lr 0.1
- **Selected:** by 3-fold cross-validated ROC-AUC on a 12,000-row stratified
  subsample of the training split, against a logistic-regression baseline and
  a 150-tree random forest
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
training split; the test set was assessed **once**. Verified by the forensic
audit (`docs/audit/`), which reproduced every number below by re-running the
pipeline in fresh processes.

| Model | CV ROC-AUC (train) | Accuracy | Precision | Recall | F1 | ROC-AUC (test) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9638 ± 0.0022 | 0.8925 | 0.9023 | 0.9379 | 0.9198 | 0.9595 |
| Random Forest | 0.9725 ± 0.0023 | 0.9090 | 0.9119 | 0.9536 | 0.9323 | 0.9716 |
| **Gradient Boosting (champion)** | **0.9726 ± 0.0025** | **0.9087** | **0.9160** | **0.9480** | **0.9317** | **0.9733** |

Confusion matrix (champion, sealed test, threshold 0.5): TP 6,229 · FP 571 ·
FN 342 · TN 2,858.

Top drivers (RF importance / target correlation agree at the top): CGPA,
MockInterviewScore, then the skill-score cluster.

## Limitations

- **Synthetic data.** Patterns are real-looking but generated; do not
  generalise conclusions to real students.
- **Feature timing.** Mock-interview and test scores may be concurrent with
  the placement process; a truly pre-placement model would need earlier
  features only.
- **Calibration.** Probabilities are ranking-accurate (AUC 0.973) but not
  isotonic/Platt-calibrated; treat the percentage as a score, not a frequency
  guarantee. Calibration is the named next iteration.
- **No fairness audit on outcomes.** Demographics are excluded from the
  feature set, but proxy bias via CGPA/skills is possible; no group metrics
  are computed in v1.
- **Fixed threshold (0.5).** No cost-sensitive tuning is applied.

## Reproducibility

Seed 42 everywhere; two fresh-process training runs produce byte-identical
metric bundles (audit-verified). Retrain: start the app and hit `/train`
(~9 s cold, cached after). Full environment: `requirements.txt`; container:
`Dockerfile`.
