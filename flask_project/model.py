"""Model training, evaluation, benchmarking, and inference for the pipeline.

Trains three classifiers on the active dataset — a logistic-regression
baseline, a random forest, and a gradient-boosted ensemble — behind a sealed
80/20 stratified split (seed 42), picks the champion by cross-validated
ROC-AUC on the training split, and caches the JSON-safe evaluation bundle
plus every fitted candidate, so the training page, the benchmark API, and
the prediction form can each work with any selected model — never a
hardcoded one. Served probabilities are Platt-calibrated (sigmoid, 3-fold
out-of-fold within the training split), so a predicted 80% means ~80%.

Mirrors the workbook discipline: impute with training means, standardise for
the linear model, trees on raw features, assess once on the sealed test set.
"""

import os
import threading
import time
from collections import OrderedDict

import hashlib

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix, f1_score, log_loss,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import eda

FEATURES = [
    "CGPA", "AttendancePercent", "Internships", "Projects", "Workshops",
    "Certifications", "Publications", "AptitudeTestScore", "SoftSkillsRating",
    "CodingTestScore", "MockInterviewScore", "ExtraCurricular",
]
TARGET = "PlacementStatus"
SEED = 42
# Champion selection runs 3-fold CV on a stratified subsample of the training
# split — statistically identical ranking, a fraction of the cost, so a cold
# start on a weak host stays fast and inside free-tier memory.
CV_FOLDS = 3
CV_ROWS = 12_000

# The candidate roster, in canonical order. Each entry carries every fact the
# UI and API need about the model, so templates never hardcode per-model
# settings — one source of truth for training, benchmarking, and inference.
MODEL_REGISTRY = OrderedDict([
    ("logistic_regression", {
        "name": "Logistic Regression",
        "factory": lambda: LogisticRegression(max_iter=2000),
        "needs_scaling": True,
        "settings": "max_iter 2000 · lbfgs",
        "note": "interpretable linear baseline — the bar to beat",
    }),
    ("random_forest", {
        "name": "Random Forest",
        "factory": lambda: RandomForestClassifier(
            # bounded worker count: each loky worker copies the training
            # frame, and free-tier hosts are memory-capped
            n_estimators=150, n_jobs=2, random_state=SEED),
        "needs_scaling": False,
        "settings": "150 trees · n_jobs 2",
        "note": "deep decorrelated trees — averages away variance",
    }),
    ("gradient_boosting", {
        "name": "Gradient Boosting",
        "factory": lambda: HistGradientBoostingClassifier(random_state=SEED),
        "needs_scaling": False,
        "settings": "default depth · lr 0.1",
        "note": "shallow sequential trees — corrects residual bias",
    }),
])
MODEL_KEYS = list(MODEL_REGISTRY)
MODEL_NAMES = [spec["name"] for spec in MODEL_REGISTRY.values()]

# Selectors meaning "the champion of the latest full training run".
BEST_ALIASES = {"best", "best_model", "recommended", "auto", "champion"}

_NAME_TO_KEY = {spec["name"].lower(): key for key, spec in MODEL_REGISTRY.items()}


def resolve_model_key(raw):
    """Normalise a user-supplied model selector to a registry key, or None.

    Accepts canonical keys ("random_forest") and display names
    ("Random Forest"), case-insensitive. "best"-style aliases are the
    caller's business — they resolve against a specific bundle.
    """
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if text in MODEL_REGISTRY:
        return text
    return _NAME_TO_KEY.get(text)


def is_best_alias(raw):
    """True for selectors like "best" / "recommended" — the champion."""
    return raw is not None and str(raw).strip().lower() in BEST_ALIASES


_CACHE_MAX = 2  # the bundled dataset plus one upload, without retraining on every switch
_bundle_cache = OrderedDict()
# cache_key -> {"champion": name, "models": {name: (clf, scaler_or_None)},
#               "impute_means": pd.Series} — every candidate stays available
# for model-selected prediction, not just the champion
_fitted_cache = OrderedDict()
_cache_lock = threading.Lock()

# Bump whenever the training recipe or artifact layout changes — stale
# artifacts are ignored. v2: per-model artifact files sit next to the main
# bundle+champion artifact, so non-champion selections load in ~1 s instead
# of retraining. v3: served models are Platt-calibrated.
ARTIFACT_VERSION = 3


def _dataset_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifact_path(path):
    return os.path.join(os.path.dirname(path), "model_artifact.joblib")


def _model_artifact_path(path, key):
    return os.path.join(os.path.dirname(path), f"model_artifact_{key}.joblib")


def save_artifact(path):
    """Train on the dataset at `path` and persist bundle + every fitted
    candidate, so production never trains at request time. Used by
    train_artifact.py at image/deploy build time.

    Layout (ARTIFACT_VERSION 2): the main artifact carries the bundle, the
    impute means, and the fitted champion — small, so boot stays fast. Each
    candidate additionally gets its own compressed file; a non-champion
    selection then loads on demand in about a second (the 150-tree forest
    shrinks ~5x under zlib) instead of retraining."""
    bundle = get_model_bundle(path)  # trains if needed; fills the fitted cache
    if not bundle.get("ok"):
        raise RuntimeError(f"cannot build artifact: {bundle.get('error')}")
    sha = _dataset_sha(path)
    with _cache_lock:
        fitted = _fitted_cache.get(_cache_key(path))
    if fitted is None or len(fitted["models"]) != len(MODEL_REGISTRY):
        raise RuntimeError("fitted cache incomplete after training")
    champion_name = fitted["champion"]
    champ_clf, champ_scaler = fitted["models"][champion_name]
    joblib.dump(
        {
            "version": ARTIFACT_VERSION,
            "dataset_sha": sha,
            "bundle": bundle,
            "champion_name": champion_name,
            "champion": champ_clf,
            "scaler": champ_scaler,
            "impute_means": fitted["impute_means"],
        },
        _artifact_path(path),
    )
    for key, spec in MODEL_REGISTRY.items():
        clf, scaler = fitted["models"][spec["name"]]
        joblib.dump(
            {
                "version": ARTIFACT_VERSION,
                "dataset_sha": sha,
                "key": key,
                "clf": clf,
                "scaler": scaler,
            },
            _model_artifact_path(path, key),
            compress=3,
        )


def _load_validated(ap, path):
    """joblib-load the artifact file `ap` for dataset `path`, or None.
    Validates the recipe version and the dataset's content hash, so a stale
    artifact can never silently serve the wrong model."""
    if not os.path.exists(ap):
        return None
    try:
        payload = joblib.load(ap)
    except Exception:  # noqa: BLE001 - a corrupt artifact just retrains
        return None
    if payload.get("version") != ARTIFACT_VERSION:
        return None
    if payload.get("dataset_sha") != _dataset_sha(path):
        return None
    return payload


def _load_artifact(path):
    """A validated precomputed bundle + fitted champion for `path`, or None."""
    return _load_validated(_artifact_path(path), path)


def _load_model_artifact(path, key):
    """A validated precomputed fitted candidate for `path`, or None — same
    version + content-hash discipline as the main artifact."""
    return _load_validated(_model_artifact_path(path, key), path)


def _cache_key(path):
    st = os.stat(path)
    return (os.path.abspath(path), st.st_mtime, st.st_size)


def get_model_bundle(path):
    key = _cache_key(path)
    with _cache_lock:
        if key not in _bundle_cache:
            # evict the least-recently-used dataset, and its fitted model
            while len(_bundle_cache) >= _CACHE_MAX:
                old_key, _ = _bundle_cache.popitem(last=False)
                _fitted_cache.pop(old_key, None)
            # a precomputed artifact (built at deploy time) loads in ~ms;
            # otherwise train, under the lock, so a concurrent cold burst
            # waits for the first run instead of launching one per thread
            payload = _load_artifact(path)
            if payload is not None:
                _bundle_cache[key] = payload["bundle"]
                _fitted_cache[key] = {
                    "champion": payload["champion_name"],
                    "models": {
                        payload["champion_name"]: (
                            payload["champion"], payload["scaler"],
                        ),
                    },
                    "impute_means": payload["impute_means"],
                }
            else:
                _bundle_cache[key] = _train_all(path)
        else:
            _bundle_cache.move_to_end(key)
        return _bundle_cache[key]


def get_fitted(path, model_name=None):
    """(model, scaler_or_None, impute_means) for one candidate — the champion
    when `model_name` is None; None when no model could be trained.

    A candidate that isn't in memory yet loads from its per-model artifact
    (~1 s) or, failing that, re-trains solo on the identical sealed split.
    The result is cached either way, so model selection never costs a
    retrain per request."""
    bundle = get_model_bundle(path)  # ensures caches are warm
    if not bundle.get("ok"):
        return None
    name = model_name or bundle["best"]
    key = _NAME_TO_KEY.get(str(name).lower())
    if key is None:
        raise ValueError(f"unknown model: {name!r}")
    name = MODEL_REGISTRY[key]["name"]  # canonical casing
    cache_key = _cache_key(path)
    with _cache_lock:
        entry = _fitted_cache.get(cache_key)
        if entry is None:
            # only reachable if the fitted cache was cleared independently of
            # the bundle cache (tests) — rebuild a shell from the bundle
            entry = {
                "champion": bundle["best"],
                "models": {},
                "impute_means": pd.Series(bundle["impute_means"]),
            }
            _fitted_cache[cache_key] = entry
        hit = entry["models"].get(name)
        if hit is not None:
            return hit[0], hit[1], entry["impute_means"]
        # single-flight: fill under the lock so a concurrent burst waits
        payload = _load_model_artifact(path, key)
        if payload is not None:
            fitted = (payload["clf"], payload["scaler"])
        else:
            clf, scaler, means = _train_single(path, name)
            entry["impute_means"] = means
            fitted = (clf, scaler)
        entry["models"][name] = fitted
        return fitted[0], fitted[1], entry["impute_means"]


def warm_status(path):
    """Model readiness without triggering training — used by /api/health so
    health checks stay cheap. Reports the in-memory state plus whether a
    precomputed artifact is on disk (loads in ~ms on first use)."""
    with _cache_lock:
        bundle = _bundle_cache.get(_cache_key(path))
    artifact_ready = False
    if not bundle or not bundle.get("ok"):
        ap = _artifact_path(path)
        artifact_ready = os.path.exists(ap)
    if not bundle or not bundle.get("ok"):
        return {"trained": False, "artifact_available": artifact_ready}
    return {
        "trained": True,
        "artifact_available": True,
        "model": bundle["best"],
        "roc_auc": next(
            m["metrics"]["roc_auc"] for m in bundle["models"]
            if m["name"] == bundle["best"]
        ),
    }


def _subsample_curve(fpr, tpr, points=80):
    """Thin an ROC curve to ~80 points so the JSON payload stays small."""
    if len(fpr) <= points:
        return fpr, tpr
    idx = np.linspace(0, len(fpr) - 1, points).astype(int)
    return fpr[idx], tpr[idx]


def _train_all(path):
    try:
        return _train_all_inner(path)
    except Exception as exc:  # noqa: BLE001 - surface training failures as a page, not a 500
        return {
            "schema_ok": True, "ok": False,
            "error": f"Training failed on this dataset: {exc}",
        }


def _train_all_inner(path):
    df = eda.load_dataframe(path)
    if not eda.schema_ok(df):
        return {"schema_ok": False, "ok": False}

    # same cleaning as the EDA bundle: drop the corrupt sentinel row
    if "StudentID" in df.columns:
        df = df[df["StudentID"] != 0]

    # degenerate uploads get a clear reason instead of a traceback
    if len(df) == 0:
        return {
            "schema_ok": True, "ok": False,
            "error": "The file matches the placement schema but has no usable "
                     "rows — there is nothing to train on.",
        }
    if df[TARGET].isna().any():
        return {
            "schema_ok": True, "ok": False,
            "error": "The target column (PlacementStatus) has missing values — "
                     "every row needs a known 0/1 outcome before a model can "
                     "be trained.",
        }
    text_cols = [c for c in FEATURES if not pd.api.types.is_numeric_dtype(df[c])]
    if text_cols:
        return {
            "schema_ok": True, "ok": False,
            "error": "Feature columns must be numeric — "
                     f"{', '.join(text_cols)} arrived as text. Check the file "
                     "for stray headers or formatting and try again.",
        }

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    if y.nunique() < 2:
        only = int(y.iloc[0])
        friendly = {0: "Not placed", 1: "Placed"}.get(only)
        shown = f"{only} · {friendly}" if friendly else str(only)
        return {
            "schema_ok": True, "ok": False,
            "error": f"Every row in this dataset has the same outcome "
                     f"(PlacementStatus = {shown}). A classifier needs both "
                     "placed and not-placed examples to learn the difference.",
        }
    if len(df) < 50:
        return {
            "schema_ok": True, "ok": False,
            "error": f"Only {len(df)} usable rows — too few to train and evaluate "
                     "honestly. Upload at least a few hundred rows.",
        }
    return _fit_and_evaluate(path, df, X, y)


def _prepare_split(X, y):
    """The sealed stratified split plus train-only transforms (impute means,
    z-score scaler). Shared by the full evaluation and by single-model
    retraining, so both always see byte-identical data."""
    # split first — every transform below is fit on the training rows only,
    # so the sealed test set can never leak into them
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED,
    )
    impute_means = X_train.mean()
    X_train = X_train.fillna(impute_means)
    X_test = X_test.fillna(impute_means)
    scaler = StandardScaler().fit(X_train)
    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "X_train_s": scaler.transform(X_train),
        "X_test_s": scaler.transform(X_test),
        "impute_means": impute_means, "scaler": scaler,
    }


def _xy(path):
    """Feature matrix + target for a dataset already known to be usable —
    the shared input for training, single-model retraining, and fresh
    benchmark runs."""
    df = eda.load_dataframe(path)
    if "StudentID" in df.columns:  # same sentinel-row cleaning as the bundle
        df = df[df["StudentID"] != 0]
    return df[FEATURES].copy(), df[TARGET].astype(int)


def _cv_view(X_train, y_train):
    """The champion-selection view: 3-fold CV on a stratified subsample of
    the training split — identical ranking, a fraction of the cost."""
    if len(X_train) > CV_ROWS:
        X_cv, _, y_cv, _ = train_test_split(
            X_train, y_train, train_size=CV_ROWS, stratify=y_train, random_state=SEED,
        )
    else:
        X_cv, y_cv = X_train, y_train
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    return X_cv, y_cv, len(X_cv), cv


def _evaluate_model(key, sp, X_cv, y_cv, cv):
    """Fit and evaluate one candidate on the sealed split. Returns
    (bundle_entry, (fitted_pipeline, scaler_or_None)).

    The served pipeline is Platt-calibrated: a sigmoid calibrator fit on
    3-fold out-of-fold predictions within the training split, base model
    refit on all of it — the sealed test set never touches calibration.
    Champion selection still scores the raw model: calibration is monotone,
    so the ROC-AUC ranking is identical and the raw pass is cheaper."""
    spec = MODEL_REGISTRY[key]
    name, needs_scaling = spec["name"], spec["needs_scaling"]

    raw = spec["factory"]()
    cv_est = make_pipeline(StandardScaler(), clone(raw)) if needs_scaling else raw
    cv_scores = cross_val_score(cv_est, X_cv, y_cv, cv=cv, scoring="roc_auc")

    served = CalibratedClassifierCV(
        spec["factory"](), method="sigmoid", cv=CV_FOLDS, ensemble=False,
    )
    tr = sp["X_train_s"] if needs_scaling else sp["X_train"]
    te = sp["X_test_s"] if needs_scaling else sp["X_test"]
    t0 = time.time()
    served.fit(tr, sp["y_train"])
    train_time = time.time() - t0

    proba = served.predict_proba(te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(sp["y_test"], proba)
    fpr, tpr = _subsample_curve(np.asarray(fpr), np.asarray(tpr))
    # per-model confusion at the default 0.5 threshold, so every
    # candidate — not just the champion — can show its error profile
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(sp["y_test"], pred).ravel())
    # reliability curve: does a predicted 80% actually place ~80%?
    frac_pos, bin_mid = calibration_curve(
        sp["y_test"], proba, n_bins=10, strategy="uniform",
    )
    entry = {
        "key": key,
        "name": name,
        "needs_scaling": needs_scaling,
        "settings": spec["settings"],
        "calibration": "Platt sigmoid · 3-fold out-of-fold",
        "train_time": round(train_time, 2),
        "note": spec["note"],
        "cv_auc_mean": round(float(cv_scores.mean()), 4),
        "cv_auc_std": round(float(cv_scores.std()), 4),
        "metrics": {
            "accuracy": round(float(accuracy_score(sp["y_test"], pred)), 4),
            "precision": round(float(precision_score(sp["y_test"], pred)), 4),
            "recall": round(float(recall_score(sp["y_test"], pred)), 4),
            "f1": round(float(f1_score(sp["y_test"], pred)), 4),
            "roc_auc": round(float(roc_auc_score(sp["y_test"], proba)), 4),
            "brier": round(float(brier_score_loss(sp["y_test"], proba)), 4),
            "log_loss": round(float(log_loss(sp["y_test"], proba, labels=[0, 1])), 4),
        },
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "roc": {"fpr": [round(float(v), 4) for v in fpr],
                "tpr": [round(float(v), 4) for v in tpr]},
        "reliability": {
            "bin_mid": [round(float(v), 4) for v in bin_mid],
            "frac_pos": [round(float(v), 4) for v in frac_pos],
        },
    }
    return entry, (served, sp["scaler"] if needs_scaling else None)


def _train_single(path, name):
    """Re-fit one candidate on the sealed split — the on-demand path when no
    per-model artifact exists (an upload, or a checkout without build-time
    artifacts). Deterministic: same seed, split, and transforms as the full
    evaluation, so the solo fit matches the benchmarked one. The returned
    pipeline is the same Platt-calibrated one the evaluation serves."""
    key = _NAME_TO_KEY.get(str(name).lower())
    if key is None:
        raise ValueError(f"unknown model: {name!r}")
    spec = MODEL_REGISTRY[key]
    X, y = _xy(path)
    sp = _prepare_split(X, y)
    served = CalibratedClassifierCV(
        spec["factory"](), method="sigmoid", cv=CV_FOLDS, ensemble=False,
    )
    served.fit(
        sp["X_train_s"] if spec["needs_scaling"] else sp["X_train"], sp["y_train"]
    )
    return served, sp["scaler"] if spec["needs_scaling"] else None, sp["impute_means"]


def _fit_and_evaluate(path, df, X, y):
    sp = _prepare_split(X, y)
    X_train, X_test = sp["X_train"], sp["X_test"]
    y_train, y_test = sp["y_train"], sp["y_test"]
    scaler = sp["scaler"]
    impute_means = sp["impute_means"]
    X_cv, y_cv, cv_rows, cv = _cv_view(X_train, y_train)

    models = []
    fitted = {}
    for key in MODEL_REGISTRY:
        entry, fitted_entry = _evaluate_model(key, sp, X_cv, y_cv, cv)
        models.append(entry)
        fitted[entry["name"]] = fitted_entry

    # champion = best cross-validated ROC-AUC on the training split; the test
    # metrics above are reported, never used for selection
    best = max(models, key=lambda m: m["cv_auc_mean"])
    best_name = best["name"]

    # feature importance from the forest's inner estimator (stable,
    # model-agnostic enough to read)
    rf_clf = fitted["Random Forest"][0].calibrated_classifiers_[0].estimator
    imp = pd.Series(rf_clf.feature_importances_, index=FEATURES).sort_values(ascending=False)

    # the logistic baseline + its calibrator, exported so the static Pages
    # build can predict in the browser (z-scored dot product, then the
    # fitted sigmoid map — no server needed)
    lr_cc = fitted["Logistic Regression"][0].calibrated_classifiers_[0]
    lr_clf = lr_cc.estimator
    lr_cal = lr_cc.calibrators[0]  # p = expit(-(a·z + b)) on decision scores
    lr_auc = next(m["metrics"]["roc_auc"] for m in models if m["name"] == "Logistic Regression")
    lr_export = {
        "features": FEATURES,
        "coef": [round(float(c), 6) for c in lr_clf.coef_.ravel()],
        "intercept": round(float(lr_clf.intercept_[0]), 6),
        "mean": {c: round(float(v), 6) for c, v in zip(FEATURES, scaler.mean_)},
        "std": {c: round(float(v), 6) for c, v in zip(FEATURES, scaler.scale_)},
        "cal": {"a": round(float(lr_cal.a_), 6), "b": round(float(lr_cal.b_), 6)},
        "auc": lr_auc,
    }

    # input constraints + sensible defaults for the prediction form
    form_meta = []
    for col in FEATURES:
        s = df[col]
        non_null = s.dropna()
        form_meta.append({
            "name": col,
            "min": round(float(s.min()), 2),
            "max": round(float(s.max()), 2),
            # whole-number columns step in units even if NaNs made them float
            "step": "1" if len(non_null) and bool((non_null % 1 == 0).all()) else "0.1",
            "default": round(float(s.median()), 2),
        })

    bundle = {
        "schema_ok": True,
        "ok": True,
        "features": FEATURES,
        "seed": SEED,
        "cv_folds": CV_FOLDS,
        "cv_rows": int(cv_rows),
        "split": {
            "train": int(len(X_train)),
            "test": int(len(X_test)),
            "train_rate": round(float(y_train.mean() * 100), 1),
            "test_rate": round(float(y_test.mean() * 100), 1),
        },
        "impute_means": {c: round(float(impute_means[c]), 2) for c in FEATURES},
        "scaler": {
            "mean": {c: round(float(v), 3) for c, v in zip(FEATURES, scaler.mean_)},
            "std": {c: round(float(v), 3) for c, v in zip(FEATURES, scaler.scale_)},
        },
        "models": models,
        "best": best_name,
        "best_key": best["key"],
        # the champion's error profile at the 0.5 threshold (each candidate
        # also carries its own under "confusion")
        "confusion": best["confusion"],
        "importance": {
            "labels": [str(k) for k in imp.index],
            "values": [round(float(v), 4) for v in imp.values],
        },
        "lr_export": lr_export,
        "form_meta": form_meta,
    }

    # keep every fitted candidate + its transform for the prediction route —
    # the user picks the model, so all three must stay available
    _fitted_cache[_cache_key(path)] = {
        "champion": best_name,
        "models": fitted,
        "impute_means": impute_means,
    }
    return bundle


def benchmark(path, keys=None, fresh=False):
    """Comparative evaluation for the requested candidates (default: all).

    Every comparison is measured under identical conditions — same sealed
    split, same train-only transforms, one assessment on the test set.

    Default ("cached_evaluation"): read the shared pipeline run — the first
    call trains, later calls are instant, and a subset filters the shared
    evaluation without retraining.

    fresh=True ("fresh_run"): genuinely re-execute the training/evaluation
    pipeline for exactly the requested models — re-fit, re-calibrate,
    re-measure. The recipe is deterministic, so the numbers match the cached
    run by construction; the point of a fresh run is real execution, and it
    costs the full fit time (tens of seconds on a small host)."""
    bundle = get_model_bundle(path)
    if not bundle.get("ok"):
        return {
            "ok": False,
            "schema_ok": bundle.get("schema_ok", True),
            "error": bundle.get("error", "Training failed on this dataset."),
        }
    wanted = [k for k in (dict.fromkeys(keys) if keys else MODEL_KEYS)
              if k in MODEL_REGISTRY]
    if not wanted:
        return {"ok": False, "error": "No valid models were requested."}

    if fresh:
        X, y = _xy(path)
        sp = _prepare_split(X, y)
        X_cv, y_cv, _, cv = _cv_view(sp["X_train"], sp["y_train"])
        selected = [_evaluate_model(key, sp, X_cv, y_cv, cv)[0] for key in wanted]
        source = "fresh_run"
    else:
        selected = [m for m in bundle["models"] if m["key"] in wanted]
        selected.sort(key=lambda m: wanted.index(m["key"]))  # caller's order
        source = "cached_evaluation"

    best = max(selected, key=lambda m: m["cv_auc_mean"])
    return {
        "ok": True,
        "source": source,
        "seed": bundle["seed"],
        "cv_folds": bundle["cv_folds"],
        "cv_rows": bundle["cv_rows"],
        "split": bundle["split"],
        "selection_rule": "highest cross-validated ROC-AUC on the training "
                          "split — the sealed test set plays no part in selection",
        "models": selected,
        "best": {"key": best["key"], "name": best["name"]},
        # champion across all three candidates — the "Best model" option in
        # prediction always resolves to this, regardless of benchmark subsets
        "overall_best": {"key": bundle["best_key"], "name": bundle["best"]},
    }


def predict(path, values, model_name=None):
    """values: dict feature -> float. model_name: a display name from
    MODEL_REGISTRY (None = the champion). Returns the placement probability
    from the selected fitted model."""
    fitted = get_fitted(path, model_name)
    if fitted is None:
        raise RuntimeError("no trained model available for this dataset")
    model_obj, scaler, impute_means = fitted
    row = {c: values.get(c, float(impute_means[c])) for c in FEATURES}
    vec = pd.DataFrame([row], columns=FEATURES).fillna(impute_means)
    if scaler is not None:
        vec = scaler.transform(vec)
    proba = float(model_obj.predict_proba(vec)[0, 1])
    return proba
