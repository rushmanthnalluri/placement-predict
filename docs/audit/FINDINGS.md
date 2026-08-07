# Consolidated Findings

Deduplicated across all auditors. Severity: P0 critical · P1 high · P2 medium · P3 low. Each finding was reproduced by at least one auditor; P0/P1 were independently confirmed by two or more.

## P0 — Critical

**F-01 · Session forgery → arbitrary file read + delete.** `SECRET_KEY` hardcoded fallback (`"dev-only-change-me"`) + session-stored absolute filesystem path trusted by `_active_dataset` and `os.remove`'d by `clear_dataset`. Flask client-side cookies signed with the known key are forgeable. Exploit executed by three independent auditors: read a sentinel CSV outside the app tree via `/upload` preview; deleted an arbitrary file via `/upload/clear`. Render deploy was protected (`render.yaml` generateValue); Dockerfile/local dev were not. **FIXED** — see FIXES.md F-01.

## P1 — High

**F-02 · Imputation fit on full frame before the split.** `X.mean()` over all 50,000 rows (incl. test) filled NaNs pre-split, contradicting "training means"/"sealed before anything is fit" claims in app notes, preprocess page, and README. Measured impact: ΔAUC ≤ 1e-6 (nil) — but definitional leakage + false claims. **FIXED** (split first; train means only).

**F-03 · Training thundering herd.** No single-flight lock: 10 concurrent cold `POST /predict` each trained a full 3-model suite (wall 33.3s, 10× duplicate work). **FIXED** (single-flight lock; burst now trains exactly once).

**F-04 · Single-slot global caches thrash across sessions.** `.clear()` on any key miss; two interleaved datasets forced retraining into nearly every request (alternating 0.08s/0.9s/3.5s). **FIXED** (2-entry LRU in both eda and model caches).

**F-05 · Silent dataset swap.** Uploads passing the schema gate but missing `Gender` (or 0-row/all-object frames) raised inside `_build_bundle`; `_active_bundle` swallowed it and served the *default* dataset under the upload's name. **FIXED** (bundle is robust: conditional gender split, degenerate frames → schema alert under the upload's own name; upload-time schema validation added).

## P2 — Medium

**F-06 · Champion selected on the "sealed" test set** (max test AUC), contradicting "assessed once" copy. **FIXED** — champion now picked by 5-fold stratified CV ROC-AUC on the training split; test set reported once. Champion unchanged (Gradient Boosting).

**F-07 · `clear_dataset` had no path-containment check** — the delete primitive behind F-01. **FIXED** (commonpath containment).

**F-08 · No schema validation at upload time** — garbage-schema CSVs became the active dataset. **FIXED** (rejected at upload with the missing columns listed; file deleted).

**F-09 · Vulnerable dependency floors** (flask 3.0.3 / jinja2 3.1.4 / werkzeug 3.0.3 have PYSEC advisories; local env affected; image resolved clean by luck). **FIXED** (floors bumped flask≥3.1.3, jinja2≥3.1.6, werkzeug≥3.1.8; strict pip-audit job in CI).

**F-10 · Container ran as root.** **FIXED** (non-root `USER appuser`).

**F-11 · No `.dockerignore`** — local upload residue (incl. an 8MB xlsx) baked into the image. **FIXED** (`.dockerignore` added).

**F-12 · CI missing; README claimed tests ran with Playwright — no tests existed anywhere. FIXED** (34-test pytest suite + 3-job GitHub Actions workflow).

**F-13 · 1,750 `IsAnomaly` rows silently trained/evaluated on** (systematically shifted profiles; label-consistent, identical placement rate). **FIXED by disclosure** (README now documents the deliberate retention and why); metrics unchanged.

## P3 — Low (all fixed unless noted)

- `app.run(debug=True)` the documented local entrypoint → **fixed** (FLASK_DEBUG opt-in, default off).
- Raw parser exceptions echoed to clients → **fixed** (generic message, real error logged).
- `.xls` whitelisted but `xlrd` not installed (always failed) → **fixed** (whitelist + hint now csv/xlsx).
- Orphaned upload files on parse failure → **fixed** (failed files deleted; uploads dir remains gitignored; no janitor for dead-session files — documented limitation).
- Cross-session same-filename clobber → **fixed** (uuid-prefixed storage names).
- Absolute server path disclosed in client-readable cookie → **fixed** (basename only).
- Unbranded 405/other HTTP errors → **fixed** (branded HTTPException fallback).
- No security headers; cookie flags → **fixed** (nosniff/DENY/Referrer-Policy; SameSite=Lax; Secure via env).
- README "retrain in ~2 s" false (7.25 s measured at the time) → **fixed** (states true ~22 s cold incl. CV, cached after).
- script.js hardcoded "ROC-AUC 0.9733"/"gradient-boosting champion" → **fixed** (dynamic `window.CHAMPION` from the bundle).
- script.js clamp comment claimed server parity that doesn't exist → **fixed** (comment states the real divergence).
- Stale-able copy: "CGPA leads by a wide margin" (margin 0.015), "All 31 fields", "Counts below 50,000", "corrupt record" singular → **fixed** (dynamic/generic/pluralized).
- All-NaN column rendered `nan` as imputed mean → **fixed** ("n/a (no observed values)").
- 0-row upload produced cryptic sklearn/indexer text on model stages → **fixed** (human-readable guard messages).
- `_train_all` head outside failure wrapper (string-dtype / NaN-target uncaught) → **fixed**.
- Single-class guard message mislabeled non-0/1 labels → **fixed** (prints actual value).
- `predict()` docstring claimed a tuple return → **fixed**.
- Cache keyed by path+mtime only (stale on preserved mtime) → **fixed** (size added to key).
- Heatmap (raw pairwise corr) vs influence (imputed corr) show different values for the same pair → **fixed by disclosure** (clarifying note on the page).
- Champion argmax ran on 4dp-rounded AUC → **fixed** (argmax on raw CV means).
- Dead CSS (`.btn-sm`, `.muted`), unstyled `.result-idle` → **not fixed** (cosmetic; documented here).
- form_meta `step` for whole-number float columns → **fixed** (step=1 when all non-null values are integral).

## Non-findings (explicitly verified clean)

No secrets in tree or full git history (HF device token never committed). Path traversal neutralized by `secure_filename`. No stack traces ever reached a client. No feature-target leakage (max |corr| 0.65; CGPA_Tier correctly excluded). Zero StudentID overlap across the split. Determinism bit-identical across fresh processes. Static demo honestly labeled (LR in-browser vs GB champion). Confusion matrix, importances, ROC curves reproduced bit-for-bit. Sentinel row content = per-column missing counts (corrupt), dropped everywhere.
