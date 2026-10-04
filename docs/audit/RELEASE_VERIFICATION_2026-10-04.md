# Release Verification — 2026-10-04

## Purpose

This record supersedes stale presentation and environment assumptions from the August forensic audit. The historical audit remains preserved; this document records the current release-consistency contract used before publishing the project externally.

## Reproducibility lock

| Item | Current contract |
|---|---|
| Python | 3.12 in CI and deployment baseline |
| scikit-learn | **1.9.0 exact pin** |
| Model artifact version | **5** |
| Random seed | 42 |
| Split | stratified 80/20 |
| Champion selection | 3-fold ROC-AUC on a 12,000-row stratified training subsample |
| Dataset | 50,000 rows × 32 fields |

Every model artifact now stores the exact Python, NumPy, pandas, and scikit-learn versions used to create it. A stale artifact from another environment is rejected and the application retrains instead of silently serving an incompatible model.

## Authoritative model result

The repository's audited 1.9.0 evaluation remains the reference result for the pinned release:

| Model | CV ROC-AUC | Test ROC-AUC |
|---|---:|---:|
| Logistic Regression | 0.9638 ± 0.0022 | 0.9595 |
| Random Forest | 0.9725 ± 0.0023 | 0.9716 |
| **Gradient Boosting** | **0.9726 ± 0.0025** | **0.9733** |

Salary regression remains **MAE 0.67 LPA** and **R² 0.9593** on the sealed placed-student test rows.

## Visual coverage

The capture tooling now covers all nine pipeline stages plus the overview:

1. Overview
2. Upload Dataset
3. Analyse Features
4. Descriptive Statistics
5. Missing Value Analysis
6. Data Visualization
7. Preprocessing
8. Model Training
9. Model Evaluation
10. Predict Placement

The demo GIF was rebuilt at 1200×760 with deliberate per-screen durations instead of sub-second transitions.

## Test baseline

The current repository contains **84 pytest tests**. This consistency pass did not add a new test function; it added an artifact-environment assertion to the existing artifact round-trip test. The historical 34-test timing statement in `FINAL_AUDIT.md` is no longer treated as the current suite timing and should not be used as a release metric.

## CI release gate

`.github/workflows/ci.yml` now has a dedicated visual-regeneration job. It installs the pinned dependencies on Python 3.12, trains fresh artifacts, renders the GitHub Pages snapshot, captures all ten screens, rebuilds the GIF, records runtime versions and model metrics, and uploads the resulting visual bundle.

The current ChatGPT runtime did not have the pinned scikit-learn 1.9.0 and Flask application dependencies installed locally, so a second independent 1.9.0 training execution was not falsely represented as a release result. The repository now enforces that reproduction through its pinned CI workflow and environment-bound artifacts.
