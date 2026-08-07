# File Coverage

Line-by-line unless noted. "Executed" = ran and probed, not just read.

| File | Lines | Reviewed | Status |
|------|-------|----------|--------|
| flask_project/app.py | 504 | YES (full) | Reviewed + 25+ live probes + Docker probes |
| flask_project/eda.py | 442 | YES (full) | Every output independently recomputed — all matched |
| flask_project/model.py | 237 | YES (full) | Reproduced in 3 fresh processes; determinism verified |
| flask_project/export_pages.py | 84 | YES (full) | Static + output-verified against fresh training |
| flask_project/static/js/script.js | 103 | YES (full) | Executed in Flask + static builds (Chrome) |
| flask_project/static/js/charts.js | 311 | YES (full) | All 8 builders executed; payload shapes field-verified |
| flask_project/static/css/style.css | 1327 | Skimmed (dead-class grep) | 2 dead classes found → P3 |
| templates/base.html | 68 | YES | Executed |
| templates/index.html | 166 | YES | Executed (multi-state) |
| templates/upload.html | 113 | YES | Executed + contract trace |
| templates/features.html | 89 | YES | Executed |
| templates/descriptive.html | 97 | YES | Executed |
| templates/missing.html | 112 | YES | Executed (nan-render finding) |
| templates/visualize.html | 223 | YES | Executed (incl. None-box guards) |
| templates/preprocess.html | 110 | YES | Executed (all branches) |
| templates/train.html | 81 | YES | Executed (settings strings cross-checked) |
| templates/evaluate.html | 137 | YES | Executed (page JSON == bundle verified) |
| templates/predict.html | 110 | YES | Executed (GET/POST/error states) |
| templates/stage.html | 22 | YES | N.A. — unreachable (all 9 stages live); dead template |
| templates/error.html | 14 | YES | Executed |
| templates/_pager.html | 16 | YES | Executed (first/last edges) |
| Dockerfile | 14 | YES (full) | Built + run + probed (9 routes + predict + 404 + 413) |
| render.yaml | 11 | YES (full) | Static + live deploy verified |
| requirements.txt | 5 | YES (full) | pip-audit cross-check |
| .gitignore | — | YES | Verified vs git ls-files |
| README.md | 122 | YES (full) | Every factual claim checked by execution |
| eda.ipynb | — | Partial (grep-level) | Reference artifact; consistency with app confirmed |
| docs/* (generated) | — | Output-verified | Generated artifact; compared to fresh bundle |
| data/placement_predict_50k.csv | 50,002 rows | YES (full parse) | Forensicated |
| placement_predict_50k Dataset.csv (root) | 50,001 rows | YES (full parse) | Same cohort, sentinel-free, kept (user's original) |
| flask_project/placement_predict_50k Dataset.xlsx | — | YES (full parse) | Value-identical to CSV twin |

**Unreviewed source files: none.** No file marked BLOCKED except README's external HF-PRO claim (account-specific, unverifiable; claim removed).
