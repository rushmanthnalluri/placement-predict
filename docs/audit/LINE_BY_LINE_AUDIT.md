# Line-by-Line Audit Ledger

| File | Lines | Mode | Status |
|------|-------|------|--------|
| flask_project/app.py | 1–504 | line-by-line | PASS (post-fix) |
| flask_project/eda.py | 1–442 | line-by-line | PASS (post-fix) |
| flask_project/model.py | 1–237 | line-by-line | PASS (post-fix) |
| flask_project/export_pages.py | 1–84 | line-by-line | PASS |
| flask_project/static/js/script.js | 1–103 | line-by-line | PASS (post-fix) |
| flask_project/static/js/charts.js | 1–311 | line-by-line | PASS |
| flask_project/static/css/style.css | 1–1327 | skimmed + dead-class analysis | PASS WITH CONCERN (P3: `.btn-sm`, `.muted` dead; `.result-idle` unstyled) |
| templates/*.html (14 files) | 1–end each | line-by-line | PASS (post-fix) |
| Dockerfile / render.yaml / requirements.txt / .gitignore / .dockerignore | full | line-by-line | PASS (post-fix) |
| README.md | 1–122 | line-by-line, claims executed | PASS (post-fix) |
| eda.ipynb | — | grep-level (reference) | N.A. — notebook is a source artifact, not shipped code |
| docs/ (generated snapshot) | — | output-verified vs fresh training | PASS (regenerated post-fix) |

**Totals:** 21 source files line-by-line; ≈3,000 lines of Python/JS/HTML templating + 1,327 lines CSS skimmed; 3 dataset files fully parsed (50k+ rows each). Zero unexplained unreviewed application source files. Zero BLOCKED files (one external claim on a third-party platform marked unverifiable and removed from the README instead).

## Notable line-level evidence (spot index)

- `app.py:16-26` — SECRET_KEY: env or `secrets.token_hex(32)` ephemeral + startup warning (P0 fix).
- `app.py:185` — `_dataset_path()` containment: basename-only + `commonpath` check.
- `model.py:117-121` — split precedes imputation; means fit on train only (P1 fix).
- `model.py` CV block — champion = `max(cv_auc_mean)` over 5-fold stratified train CV (P2 fix); test assessed once.
- `model.py` / `eda.py` — `threading.Lock`/`RLock` single-flight + 2-entry LRU (P1 fixes); cache key `(abspath, mtime, size)`.
- `eda.py` — `gender_split` conditional; 0-row/all-object frames → `schema_ok: False` (P1 silent-swap fix).
