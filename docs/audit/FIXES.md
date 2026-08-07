# Fixes Applied

Each fix: what changed, how it was verified, and the regression test that guards it (tests/ suite, 34 tests, `pytest -q` green).

## Security pack (app.py, Dockerfile, requirements.txt, .dockerignore)

| Fix | Verification evidence | Regression test |
|---|---|---|
| F-01 P0: ephemeral `secrets.token_hex(32)` key when `SECRET_KEY` unset (+ startup warning); session stores only basename; `_dataset_path()` re-derives under UPLOAD_FOLDER with `commonpath` containment; legacy/abs cookie values ignored | Cookie signed with `dev-only-change-me` pointing at outside sentinel CSV → sentinel content absent from every response; default dataset shown | `test_upload.py::test_forged_session_cookie_cannot_read_outside_file` |
| F-07 containment on delete | clear removes only namespaced upload; 302 → default restored | `test_upload.py::test_clear_restores_default` |
| F-08 schema validation at upload (missing columns listed, file deleted, not made active) | bad-schema CSV rejected; home still shows default | `test_upload.py::test_bad_schema_rejected` |
| F-03(extension)/junk: `.xls` dropped from whitelist; generic parse-error message + server-side log + failed file unlinked | .txt and .xls rejected; garbage CSV shows no parser internals; no orphans | `test_upload.py::test_wrong_extension`, `test_garbage_rejected` |
| F-05(uuid namespacing) | `uuid8_name.csv` on disk; display name unchanged | `test_upload.py::test_valid_slice_accepted` |
| Security headers + cookie flags | `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy` observed live; SameSite=Lax | `test_routes.py::test_security_headers` |
| Branded catch-all (405 etc.) | PUT /predict → branded "Method Not Allowed" | `test_routes.py::test_405_branded` |
| FLASK_DEBUG opt-in, default off | server log "Debug mode: off" | static |
| Dockerfile non-root `USER appuser`; requirements floors flask≥3.1.3, jinja2≥3.1.6, werkzeug≥3.1.8; `.dockerignore` (git, uploads, docs, screenshots, xlsx, caches) | image rebuilt clean by CI job; xlsx exclusion verified safe (no code path reads it) | CI: docker build job |

## ML-correctness pack (model.py, eda.py, templates, script.js, README)

| Fix | Verification evidence | Regression test |
|---|---|---|
| F-02 split-first + train-only imputation | `impute_means` == train-only means (e.g. AptitudeTestScore 68.85 vs full 68.86) | `test_model_bundle.py` bundle sanity |
| F-06 champion by 5-fold stratified CV ROC-AUC on train; final fit on train; test reported once; CV mean±std shown on /train | fresh build: GB cv 0.9744±0.0009 > RF 0.9731 > LR 0.9626; test assessed once | `test_model_bundle.py::test_champion_and_cv` |
| F-03 single-flight `threading.Lock`/`RLock` | 10-thread cold burst → exactly one training run | (timing-verified; suite covers correctness) |
| F-04 2-entry LRU both caches | interleave two datasets → 2 trains, no thrash; third evicts LRU | bundle-identity tests |
| F-05 robust bundle (conditional gender split; 0-row/all-object → schema_ok False) + upload-time validation | no-Gender upload: all stages 200 under its own name, no swap, no gender card | `test_upload.py::test_bad_schema_rejected` + degenerate fixture tests |
| Graceful degenerate failures (0-row, NaN target, string dtype, single-class with real label) | each renders a human-readable alert, never 500 | `test_model_bundle.py::test_degenerate_*` (×3) |
| Cache key includes file size | same-mtime content change invalidates | `test_eda_bundle.py` sanity |
| nan→"n/a", empty-histogram cards skipped, dynamic copy (drivers/fields/counts/plurals), heatmap frame note | rendered pages inspected in Chrome | template-render assertions in route tests |
| `window.CHAMPION` dynamic; clamp comment corrected; form_meta step fix | static docs/predict.html: note shows live champion + AUC | static E2E re-run |
| README truth pass: real retrain cost, CV selection note, pytest claim, updated metrics table, IsAnomaly disclosure | every number below re-measured | docs truth (this audit) |

## Orchestrator fixes

- gunicorn `--threads 4` in Dockerfile + render.yaml (was 1 sync thread).
- README: IsAnomaly retention disclosed with rationale.
- upload.html: `.xls` removed from hint/accept (follow-through from security pack).

## Final measured metrics (post-fix, fresh processes, seed 42)

| Model | CV ROC-AUC (5-fold train) | Accuracy | Precision | Recall | F1 | ROC-AUC (test) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9626 ± 0.0013 | 0.8925 | 0.9023 | 0.9379 | 0.9198 | 0.9595 |
| Random Forest | 0.9731 ± 0.0007 | 0.9079 | 0.9119 | 0.9518 | 0.9314 | 0.9716 |
| **Gradient Boosting (champion)** | **0.9744 ± 0.0009** | **0.9087** | **0.9160** | **0.9480** | **0.9317** | **0.9733** |

Split 40,000/10,000 stratified (placed rate 65.7/65.7). Confusion (GB, test): tn 2858, fp 571, fn 342, tp 6229. Test-set numbers moved ≤0.001 vs pre-fix — confirming the leakage had no practical impact, while the code now matches every claim.
