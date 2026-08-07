# Audit Status Tracker

| Area | Agent verdict (evidence) | Findings | Fixed | Final status |
|------|--------------------------|----------|-------|--------------|
| Routing / all GET pages | PASS — execution (10/10 routes 200, 0 tracebacks) | — | — | PASS |
| Upload handling | PASS WITH CONCERN — execution | F: no schema validation, orphans, clobber, exception echo | ✔ all | PASS |
| Predict input handling | PASS — execution (median fallback, range/type guards, XSS escaped) | — | — | PASS |
| Error pages | PASS — execution (404/413/500 branded) | unbranded 405 | ✔ | PASS |
| Session management | FAIL — execution (forgery exploit worked) | **P0** | ✔ | PASS |
| Frontend↔backend contract | PASS — execution + static | — | — | PASS |
| EDA computations | PASS — execution (zero numeric mismatches anywhere) | P3 notes only | ✔ (P3s) | PASS |
| Caching | PASS WITH CONCERN — execution | mtime-only key, single-slot thrash, no lock | ✔ | PASS |
| ML training/metrics | PASS — execution (15/15 README numbers reproduced) | impute-before-split (**P1**), test-set champion selection (P2) | ✔ | PASS |
| Determinism | PASS — execution (two fresh processes bit-identical) | — | — | PASS |
| Degenerate-dataset guards | PASS WITH CONCERN — execution | head outside wrapper, cryptic 0-row msg | ✔ | PASS |
| Dataset / target / split | PASS — execution (shape, dupes, sentinel, stratification, 0 ID overlap) | IsAnomaly silent inclusion (P2) | ✔ disclosed | PASS |
| Feature leakage | PASS — execution (max feature-target corr 0.65, no proxies) | CGPA_Tier doc note wrong (P3) | ✔ | PASS |
| Frontend templates/JS | PASS WITH CONCERN — execution (14/14, all branches) | stale copy, hardcoded 0.9733, nan render (P3s) | ✔ | PASS |
| E2E user flows | PASS — execution (upload→analyse→train→predict→clear) | cold-burst herd (**P1**), latency findings | ✔ | PASS |
| Performance | PASS — execution (p95 POST /predict 21ms warm; cold train 22s documented) | first-hit sync cost (P3) | ✔ documented | PASS |
| Secrets | PASS — execution (tree + full git history clean) | — | — | PASS |
| Dependency security | PASS WITH CONCERN — pip-audit | vulnerable floors locally (P2) | ✔ floors + CI gate | PASS |
| Docker | PASS WITH CONCERN — build/run/probe executed | root user, no .dockerignore (P2) | ✔ | PASS |
| CI | MISSING | no workflows (P2) | ✔ added | PASS (first run pending at audit time) |
| Tests | FAIL/ABSENT → 34 pytest tests added | none existed (P2) | ✔ | PASS (34/34 green) |
| Docs truth | 17/19 README claims verified by execution | "retrain ~2s" false; Playwright-test claim unverifiable | ✔ | PASS |
| Repo hygiene | PASS WITH CONCERN | junk file, duplicate-ish root CSV (P3) | ✔ / kept+documented | PASS |
| Explainability | PASS — execution (importances match bit-for-bit, honest labels) | "wide margin" copy (P3) | ✔ | PASS |
| HF-Spaces-PRO README claim | BLOCKED — external account-specific | — | claim removed | N.A. |
