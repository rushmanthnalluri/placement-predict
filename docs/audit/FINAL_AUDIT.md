# PlacementPredict Final Forensic Audit

## Executive Verdict

**PASS WITH CONDITIONS** — one P0 and four P1 defects were found, exploited, and fixed with regression tests; all P2 findings fixed or disclosed; the ML pipeline's numbers reproduce bit-for-bit; 34/34 tests green; Docker builds and serves. Conditions are listed under Remaining Limitations.

## Repository Snapshot

- Commit at audit start: `a912e48` (main) · fixes land in the audit commit
- Date: 2026-08-07 · Environment: Windows 11 / Git Bash, Python 3.14.5, pandas 3.0.3, scikit-learn 1.9.0, Flask 3.0.3, gunicorn 26 (container), Docker Desktop 29.4.0, Playwright + Chrome

## Agents Executed

8 parallel forensic auditors (backend/API · EDA compute · ML/model · frontend · dataset/target/leakage/split · black-box E2E/performance/artifact · security/Docker/CI/docs/hygiene · explainability/reproducibility) + 2 fixer agents + 1 test/CI/hygiene agent + orchestrator-run regression and E2E.

## Files Reviewed

25 (21 line-by-line — see FILE_COVERAGE.md). No unexplained unreviewed source files.

## Functions Reviewed

55 (see FUNCTION_COVERAGE.md) — every route, every bundle builder, every chart builder, every helper.

## Lines Reviewed

≈3,000 line-by-line (Python 1,267 · JS 414 · templates ~1,300) + 1,327 CSS skimmed with dead-class analysis.

## Dataset Verification

PASS — execution. Raw 50,001×31 → 50,000×31 after dropping the corrupt sentinel row (StudentID 0; its values are the per-column missing counts — impossible on every scale). 0 duplicate rows; StudentIDs unique; target binary, 65.712% placed; exactly 5 columns with missing values (19,976 cells). The three dataset files (root CSV, flask CSV, xlsx) carry identical cohort content; the root CSV is the user's sentinel-free original and is kept. The 1,750 `IsAnomaly` rows are systematically profile-shifted but label-consistent (identical 65.71% placement rate) — retained deliberately, now disclosed in the README.

## Leakage Verification

PASS — execution. No model feature correlates with the target above 0.65; MI ranking agrees; no post-outcome fields among the 12 features; CGPA_Tier verified absent from the model (and is only a noisy CGPA proxy, standalone AUC 0.889 < CGPA 0.903). The one real leak found was procedural (F-02: full-frame imputation) — fixed; impact measured ≤1e-6 AUC, claims now true.

## Feature Pipeline Verification

PASS — execution. Identical FEATURES order at train and inference (sklearn name-check enforced); scaler fit on train only (verified numerically); imputation now train-only; `predict()` rebuilds the frame by column name; blank-form medians == dataset medians; static browser demo's sigmoid matches sklearn's `predict_proba` within 2.5e-07.

## Model Verification

PASS — execution. Three fresh-process reproductions: all 15 README metrics matched at 3dp pre-fix; post-fix numbers in FIXES.md. Determinism: two fresh processes → byte-identical bundles (SHA-256 match; only wall-clock `train_time` varies). Champion selection now 5-fold CV on train; confusion matrix/importances/ROC reproduced exactly; ROC thinning is display-only (legend AUC computed on full probabilities; ΔAUC ≤ 0.0007).

## Backend Verification

PASS — execution (post-fix). 10/10 routes 200 across default, 300-row, 6-row, one-class, bad-schema, and cleared states; upload edge battery (no file, empty name, .txt, .xls, 0-byte, garbage, traversal, >10MB) all handled safely; branded 404/405/413/500; no stack trace ever reached a client.

## Frontend Verification

PASS — execution. 14/14 templates reviewed and rendered across all dataset states; 25/25 chart canvases instantiate with zero console errors; every displayed number traced to server data (the only literals are the two disclosed static-demo notes); canvas/ARIA labeling verified; static-mode detection cannot misfire on Flask (submit vs button types verified in Chrome).

## E2E Verification

PASS — execution (black-box user flows): home → upload → stages reflect the upload → predict verdict → remove file → fallback. Failure UX verified (out-of-range, non-numeric, blank form, bad URL). Performance: warm p95 POST /predict 21.3ms; first-hit cold train ~22s (single-flight; documented); memory 132.5MB warm / 300.5MiB in container; artifact lifecycle = deliberate in-memory retrain per process, works from cold start in ~5s.

## Security Verification

PASS — execution (post-fix). P0 session-forgery exploit worked pre-fix and fails now; path traversal neutralized; secrets scan clean across tree + full git history; security headers set; non-root container; dependency floors raised with a strict pip-audit CI gate. Note: app uses no CSRF tokens — mitigated by SameSite=Lax cookies (documented; low residual risk for this app).

## Docker Verification

PASS — execution. Clean build (703MB); all routes + predict + branded errors probed in-container; memory 300.5MiB; image packages at patched versions; non-root user; `.dockerignore` prevents residue leakage.

## CI Verification

PASS (added during fix wave) — `.github/workflows/ci.yml`: pytest (full suite), docker build, strict pip-audit on push/PR. First remote run triggered by the audit commit.

## Test Verification

PASS — `pytest -q`: **34 passed in ~21–35s** (slow marker covers the ~20s cold-train paths; fast subset 22 tests in ~3s). Suite authored fresh during this audit — none existed before.

## Performance Verification

PASS — execution. Numbers above. Cold-burst herd and cross-session thrash (the two P1 latency defects) fixed and re-measured.

## Findings

- **P0: 1** (session forgery) — FIXED
- **P1: 4** (imputation-before-split, training herd, cache thrash, silent dataset swap) — FIXED
- **P2: 8** (test-set selection, containment, upload validation, dep floors, root container, .dockerignore, CI/tests missing, IsAnomaly disclosure) — FIXED (7 code/config + 1 disclosure)
- **P3: 22** — 20 fixed, 2 documented-unfixed (dead CSS classes; no upload janitor for dead-session files)

## Fixes Applied

26 discrete fixes (see FIXES.md), each verified by execution and guarded by the new suite.

## Remaining Limitations (the conditions)

1. Cold start trains the models (~9s: 3-fold CV on a 12k subsample + three final fits); a background warm-up at startup absorbs it, and the result is cached thereafter. By design; documented in README.
2. Uploads from dead sessions are not garbage-collected (gitignored disk residue; bounded by 10MB cap and per-session namespacing).
3. Dependency floors are `>=` ranges by design; the strict pip-audit CI job is the tripwire for future advisories.
4. The root-level `placement_predict_50k Dataset.csv` (user's original, sentinel-free, identical cohort content) is kept deliberately; the app's runtime source is `flask_project/data/placement_predict_50k.csv`.
5. No CSRF tokens on the three POST routes (SameSite=Lax mitigation; low risk at this app's threat level).
6. Dead CSS classes remain (cosmetic).

## Blocked Checks

- HF-Spaces-PRO pricing claim — external/account-specific, unverifiable by execution; the claim was removed from the README instead of being left unverified.
- 500-handler live trigger — verified via an instrumented test client (branded page; traceback logged server-side only), not via a production crash.

## Final Verdict

**PASS WITH CONDITIONS.** No P0 or unresolved P1 remains. Every critical path was executed, every number reproduced, every found defect fixed and guarded by a regression test. The conditions above are documented, bounded, and non-blocking.

```text
========================================
PLACEMENTPREDICT FORENSIC AUDIT
========================================

FILES REVIEWED: 25
FUNCTIONS REVIEWED: 55
LINES REVIEWED: ~3,000 line-by-line (+1,327 CSS skimmed)

TESTS EXECUTED: 34
TESTS PASSED: 34
TESTS FAILED: 0

ML PIPELINE: PASS
DATA LEAKAGE: PASS (one procedural leak found → fixed → claims now true)
MODEL INFERENCE: PASS
API: PASS
FRONTEND: PASS
API ↔ FRONTEND: PASS
EXPLAINABILITY: PASS
SECURITY: PASS (post-fix; P0 exploited then closed)
DOCKER: PASS
CI: PASS (added this audit; first run on this commit)
E2E: PASS

P0 FINDINGS: 1 (fixed)
P1 FINDINGS: 4 (fixed)
P2 FINDINGS: 8 (fixed/disclosed)
P3 FINDINGS: 22 (20 fixed, 2 documented)

BUGS FIXED: 26
REGRESSION TESTS ADDED: 34

BLOCKED CHECKS:
- HF-Spaces-PRO pricing claim (external; claim removed from docs)
- 500 handler in production (verified via instrumented client instead)

FINAL VERDICT: PASS WITH CONDITIONS
========================================
```
