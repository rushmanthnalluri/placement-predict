# PlacementPredict — Audit Plan

**Mission:** independently verify that the implemented PlacementPredict system is correct, functional, reproducible, secure, and production-quality. Trust nothing pre-existing: every claim required evidence.

**Repository snapshot at audit start:** branch `main`, commit `a912e48` (pre-fix). Environment: Windows 11 / Git Bash, Python 3.14.5, pandas 3.0.3, scikit-learn 1.9.0, Flask 3.0.3, Docker Desktop 29.4.0, Playwright + Chrome.

**Excluded from review:** `.git/`, `__pycache__/`, `docs/` (generated snapshot — verified as output, not source), `screenshots/` (binary), `data/uploads/` (gitignored runtime residue), `.vs/`.

## Method

Two waves of independent agents + orchestrator verification:

| Wave | Agent | Scope (maps to mandated agents) |
|------|-------|----------------------------------|
| 1 | Backend + API | app.py line-by-line; every route probed live (agents 14, 20 partial) |
| 1 | EDA compute | eda.py line-by-line; every number independently recomputed (agents 2, 6, 12 partial) |
| 1 | ML / model | model.py line-by-line; metrics reproduced fresh-process (agents 8, 9, 11, 15 partial, 23) |
| 1 | Frontend | all 14 templates + 2 JS files line-by-line; hardcoded-value hunt (agents 15, 16, 18 partial) |
| 1 | Dataset / target / leakage / split | dataset forensics, leakage hunt, split integrity (agents 2, 3, 4, 5, 10) |
| 1 | Black-box E2E + performance | user flows, latency, concurrency, artifact lifecycle (agents 20, 22, 13) |
| 1 | Security + Docker + CI + docs-truth + hygiene | secrets scan incl. git history, container build/run, README claims (agents 18, 19, 21) |
| 1 | Explainability + reproducibility | importances, ROC fidelity, determinism, static-predict parity (agents 12, 23) |
| 2 | Security fixer | app.py, Dockerfile, requirements.txt, .dockerignore |
| 2 | ML-correctness fixer | model.py, eda.py, templates, script.js, README |
| 3 | Tests + CI + hygiene | tests/, .github/workflows/ci.yml, .gitignore, junk deletions |

**Fix protocol:** discover → reproduce → document → fix → regression test → targeted tests → full suite → E2E re-run. No speculative refactors.

**Severity scale:** P0 critical (leakage/forgery/fabrication) · P1 high (skew, broken integration, silent wrong data) · P2 medium (missing validation, deployment weaknesses) · P3 low (cosmetic, docs, hygiene).
