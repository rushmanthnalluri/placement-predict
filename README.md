---
title: Placement Predict
sdk: docker
---

<div align="center">

# Placement Predict System

**An end-to-end ML pipeline that predicts engineering-student placements —
from raw CSV to a deployed prediction service, in one nine-stage web app.**

[![CI](https://github.com/rushmanthnalluri/placement-predict/actions/workflows/ci.yml/badge.svg)](https://github.com/rushmanthnalluri/placement-predict/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org)

[🚀 **Live app**](https://placement-predict-p2z1.onrender.com) ·
[📊 **Static showcase**](https://rushmanthnalluri.github.io/placement-predict/) ·
[📋 **Model card**](MODEL_CARD.md) ·
[🔍 **Forensic audit**](docs/audit/FINAL_AUDIT.md)

![Demo: explore the data, then get a placement call](screenshots/demo.gif)

</div>

## Quick start

```bash
pip install -r requirements.txt
python flask_project/app.py        # → http://127.0.0.1:5000
```

or one container, any host:

```bash
docker build -t placement-predict . && docker run -p 7860:7860 placement-predict
```

## What it does

Every stage of the ML lifecycle is a live page, computed from the real
dataset on every load — nothing is hardcoded:

| # | Stage | What it shows |
|---|-------|---------------|
| 01 | Upload Dataset | Drag-and-drop CSV/Excel intake with instant profiling |
| 02 | Analyse Features | Full 31-field registry: types, roles, coverage, samples |
| 03 | Descriptive Statistics | Centre/spread/range for 20 numeric fields, split by outcome |
| 04 | Missing Value Analysis | 19,976 missing cells across 5 columns, mean-imputed |
| 05 | Data Visualization | Distributions, z-scores, 21×21 correlation heatmap, boxplots |
| 06 | Preprocessing | Stratified 80/20 split (seed 42), frozen train-only transforms |
| 07 | Model Training | Logistic regression, random forest, gradient boosting |
| 08 | Model Evaluation | Sealed-test metrics, ROC curves, confusion matrix, importances |
| 09 | Predict Placement | Validated profile form → champion's call + probability |

## How it fits together

```
CSV upload ──► EDA bundle (cached) ──► split 80/20 (sealed test)
                                              │
              build time: train 3 models ──► champion by 3-fold CV
                                              │
                       validated 0.3 MB artifact (sha256 + recipe version)
                                              │
              ┌───────────────────────────────┼────────────────────┐
              ▼                               ▼                    ▼
        Flask web app                   JSON API              Static showcase
        (9 live stages)           /api/predict · /api/health   (GitHub Pages)
```

Runtime never trains on request for the bundled dataset — the artifact loads
in ~50 ms; uploads retrain once (~9 s) and are cached.

## Results (sealed test set, assessed once)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | 0.893 | 0.902 | 0.938 | 0.920 | 0.960 |
| Random Forest | 0.909 | 0.912 | 0.954 | 0.932 | 0.972 |
| **Gradient Boosting (champion)** | **0.909** | **0.916** | **0.948** | **0.932** | **0.973** |

Top drivers: CGPA (0.65), Mock Interview Score (0.63), Soft Skills Rating (0.60).
Champion selection: 3-fold CV ROC-AUC on a 12k stratified training subsample.
Every number reproduced by the independent [forensic audit](docs/audit/FINAL_AUDIT.md).

![Model evaluation](screenshots/evaluate.png)

## JSON API

```bash
curl https://placement-predict-p2z1.onrender.com/api/health
# → {"status":"ok","model":"Gradient Boosting","roc_auc":0.9733, ...}

curl -X POST https://placement-predict-p2z1.onrender.com/api/predict \
  -H "Content-Type: application/json" \
  -d '{"CGPA": 8.6, "MockInterviewScore": 88, "CodingTestScore": 85}'
# → {"placed": true, "probability": 99.9, "threshold": 0.5,
#    "model": "Gradient Boosting", "roc_auc": 0.9733, ...}
```

All 12 fields optional (absent = dataset median). Errors are JSON: `400` with
per-field details, `415` for non-JSON, `503` when no model can train.

## Engineering practices worth pointing at

- **Leakage found in the wild** — the dataset ships a corrupt sentinel row
  (StudentID 0, holding per-column missing counts as values): detected,
  dropped, disclosed in the UI.
- **Honest evaluation** — the test set is sealed before any transform is fit
  and touched exactly once; all preprocessing statistics are train-only.
- **Graceful failure** — off-schema uploads, single-class datasets, and tiny
  files each get a clear explanation, never a crash or a traceback.
- **Secure by construction** — ephemeral session keys, path containment,
  schema-validated uploads, security headers, non-root container, strict
  pip-audit in CI.
- **Accessible & responsive** — keyboard-navigable, AA contrast,
  reduced-motion support, phone-to-desktop layouts.

![Data visualization](screenshots/visualize.png)
![Prediction form](screenshots/predict.png)

## Testing & CI

41 pytest tests cover every route × dataset state, the API contract, model
artifacts, and degenerate-input guards:

```bash
pytest -q                 # full suite (~30s)
pytest -m "not slow" -q   # fast subset (~3s)
```

Every push runs **pytest + Docker build + pip-audit** on GitHub Actions.

## Deploy it

**This app lives on Render:** https://placement-predict-p2z1.onrender.com
(free tier sleeps when idle — first hit after a lull takes ~30–60 s to wake).

- **Render**: New → Blueprint → this repo (`render.yaml` included).
- **Railway / Fly.io / any container host**: portable Dockerfile; expose `$PORT`.
- **Hugging Face Spaces**: Docker SDK works, but HF now requires PRO for Docker runtimes.
- **GitHub Pages**: `python flask_project/export_pages.py` re-renders `docs/`.

## Project structure

```
flask_project/
├── app.py              # routes, JSON API, error handlers, security
├── eda.py              # cached dataset → EDA artifact computation
├── model.py            # split, CV selection, train, evaluate, infer, artifacts
├── train_artifact.py   # build-time pretraining for zero-cost cold starts
├── export_pages.py     # renders the static GitHub Pages snapshot
├── data/               # bundled 50k dataset
├── static/             # design system CSS, Chart.js builders
└── templates/          # Jinja templates, one per stage
tests/                  # 41-test pytest suite
docs/                   # Pages site + audit trail (docs/audit/)
MODEL_CARD.md           # intended use, methodology, limitations
Dockerfile · render.yaml · requirements.txt
eda.ipynb               # original exploratory notebook
```

## Data

Synthetic 50,000-record dataset modelled on Indian engineering-college
placement data: 8 semester SGPAs, CGPA, attendance, experience counts,
four skill scores, and the placement outcome. The corrupt sentinel row
(StudentID 0) is dropped; the 1,750 `IsAnomaly` records are *retained* —
their placement rate matches the population (65.7%), so they act as
label-consistent noise. A deliberate, disclosed choice.

## Roadmap

- Probability calibration (isotonic/Platt) + cost-based threshold control
- Per-prediction explanations (SHAP-style "why this call")
- Fairness slices: performance by Gender/CollegeTier with group metrics

## License

MIT — see [LICENSE](LICENSE).

---

Built as the capstone of a 12-week classical-ML self-learning track (25SC2107E,
KL Deemed to be University): EDA → preprocessing → linear baseline → tree
ensembles → honest evaluation → deployment.
