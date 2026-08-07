# Function Coverage Matrix

Every function in application source. Tested = covered by the committed pytest suite (tests/). Executed = run during audit.

## app.py

| Function | Lines | Reviewed | Executed | Tested | Status |
|---|---|---|---|---|---|
| `_find_step` | — | YES | YES | YES | PASS |
| `_step_pager` | — | YES | YES | YES | PASS |
| `inject_pipeline` | — | YES | YES | YES | PASS |
| `_dataset_path` (added in fix) | — | YES | YES (forged/abs/relative paths collapse safely) | YES | PASS |
| `_active_dataset` | — | YES | YES (incl. forged cookie attempt) | YES | PASS (post-fix) |
| `_active_bundle` | — | YES | YES | YES | PASS |
| `_allowed_file` | — | YES | YES (.txt/.xls rejected) | YES | PASS |
| `_read_dataset` | — | YES | YES | YES | PASS |
| `_build_preview` | — | YES | YES | YES | PASS |
| `home` | — | YES | YES (default + schema-broken) | YES | PASS |
| `upload_dataset` | — | YES | YES (no-file/empty-name/bad-ext/xls/>10MB/garbage/0-byte/traversal/valid/schema-reject/clear-restore) | YES | PASS (post-fix) |
| `clear_dataset` | — | YES | YES (idempotent, containment-checked) | YES | PASS (post-fix) |
| `_eda_stage_view` + 4 EDA views | — | YES | YES (×4 routes) | YES | PASS |
| `_model_stage_view` + 3 model views | — | YES | YES (never-500 catch verified) | YES | PASS |
| `predict_placement` | — | YES | YES (missing/non-numeric/inf/nan/-inf/out-of-range/200KB-string/XSS/extra-fields/blank/valid) | YES | PASS |
| `_make_stage_view` + stub loop | — | YES | N.A. (all stages live; dead path) | N.A. | N.A. |
| `not_found` / `too_large` / `server_error` / HTTPException fallback | — | YES | YES (404/413/405 executed; 500 via instrumented client) | YES | PASS |
| `set_security_headers` (added in fix) | — | YES | YES (headers observed live) | YES | PASS |
| `__main__` | — | YES | YES (debug off by default; FLASK_DEBUG opt-in) | static | PASS (post-fix) |

## eda.py

| Function | Reviewed | Executed | Tested | Status |
|---|---|---|---|---|
| `_f` / `_i` | YES | YES (direct calls) | YES | PASS |
| `_histogram` | YES | YES (bins bit-identical; empty-series stub) | YES | PASS |
| `_box_stats` | YES | YES (1.5·IQR hand-verified; None on empty) | YES | PASS |
| `_heat_color` | YES | YES (hand-computed) | YES | PASS |
| `_clean_categories` | YES | YES (groupby match; garbage filtered) | YES | PASS |
| `load_dataframe` | YES | YES (identity, invalidation, size-in-key) | YES | PASS (post-fix) |
| `schema_ok` | YES | YES | YES | PASS |
| `get_bundle` | YES | YES (LRU + RLock post-fix) | YES | PASS (post-fix) |
| `_build_bundle` | YES | YES (every artifact area recomputed) | YES | PASS (post-fix: Gender-conditional, 0-row guard) |

## model.py

| Function | Reviewed | Executed | Tested | Status |
|---|---|---|---|---|
| `_cache_key` | YES | YES | YES | PASS (post-fix: size in key) |
| `get_model_bundle` | YES | YES (single-flight + LRU verified under 10-thread burst) | YES | PASS (post-fix) |
| `get_fitted` | YES | YES | YES | PASS |
| `_subsample_curve` | YES | YES (AUC Δ ≤ 0.0007 full→80pts) | YES | PASS |
| `_train_all` | YES | YES (default + 5 degenerate fixtures) | YES | PASS (post-fix: all failures graceful) |
| `_fit_and_evaluate` | YES | YES | YES | PASS (post-fix: train-only imputation) |
| `_fit_and_evaluate_inner` | YES | YES (metrics reproduced) | YES | PASS (post-fix: CV-based selection) |
| `predict` | YES | YES (3 profiles; feature order enforced) | YES | PASS |

## export_pages.py / static JS

| Function | Reviewed | Executed | Tested | Status |
|---|---|---|---|---|
| `_rewrite` / `main` | YES | YES (re-export run post-fix) | smoke via E2E | PASS |
| script.js handlers (reveal/stepper/dropzone/static-note/static-predict) | YES | YES (Chrome; LR math vs Python ≤2.5e-07 diff) | YES (static E2E) | PASS (post-fix: dynamic champion) |
| charts.js 8 builders + dispatcher | YES | YES (25/25 canvases, 0 console errors) | YES (browser sweep) | PASS |

## Verdicts key
PASS — verified by execution unless marked static. "Post-fix" = failed or concerned at audit start, verified after the fix wave.
