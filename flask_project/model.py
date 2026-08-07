"""Model training, evaluation, and inference for the Placement Predict pipeline.

Trains three classifiers on the active dataset — a logistic-regression
baseline, a random forest, and a gradient-boosted ensemble — behind a sealed
80/20 stratified split (seed 42), picks the champion by cross-validated
ROC-AUC on the training split, and caches both the JSON-safe evaluation
bundle and the fitted champion for the prediction form.

Mirrors the workbook discipline: impute with training means, standardise for
the linear model, trees on raw features, assess once on the sealed test set.
"""

import os
import threading
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
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
CV_FOLDS = 5

# short human note per model, for the training page
MODEL_NOTES = {
    "Logistic Regression": "interpretable linear baseline — the bar to beat",
    "Random Forest": "deep decorrelated trees — averages away variance",
    "Gradient Boosting": "shallow sequential trees — corrects residual bias",
}

_CACHE_MAX = 2  # the bundled dataset plus one upload, without retraining on every switch
_bundle_cache = OrderedDict()
_fitted_cache = OrderedDict()
_cache_lock = threading.Lock()


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
            # built under the lock: a concurrent cold burst waits for the
            # first training run instead of launching one per thread
            _bundle_cache[key] = _train_all(path)
        else:
            _bundle_cache.move_to_end(key)
        return _bundle_cache[key]


def get_fitted(path):
    """(model, scaler_or_None, impute_means) for the champion — for inference."""
    get_model_bundle(path)  # ensures caches are warm
    with _cache_lock:
        return _fitted_cache.get(_cache_key(path))


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


def _fit_and_evaluate(path, df, X, y):
    # split first — every transform below is fit on the training rows only,
    # so the sealed test set can never leak into them
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED,
    )
    impute_means = X_train.mean()
    X_train = X_train.fillna(impute_means)
    X_test = X_test.fillna(impute_means)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    candidates = [
        ("Logistic Regression", LogisticRegression(max_iter=2000), True),
        ("Random Forest", RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=SEED), False),
        ("Gradient Boosting", HistGradientBoostingClassifier(random_state=SEED), False),
    ]
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    models = []
    fitted = {}
    for name, clf, needs_scaling in candidates:
        # champion selection runs on cross-validated ROC-AUC within the
        # training split — the test set stays sealed until the final read
        cv_est = make_pipeline(StandardScaler(), clone(clf)) if needs_scaling else clf
        cv_scores = cross_val_score(cv_est, X_train, y_train, cv=cv, scoring="roc_auc")

        tr, te = (X_train_s, X_test_s) if needs_scaling else (X_train, X_test)
        t0 = time.time()
        clf.fit(tr, y_train)
        train_time = time.time() - t0
        proba = clf.predict_proba(te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        fpr, tpr, _ = roc_curve(y_test, proba)
        fpr, tpr = _subsample_curve(np.asarray(fpr), np.asarray(tpr))
        models.append({
            "name": name,
            "needs_scaling": needs_scaling,
            "train_time": round(train_time, 2),
            "note": MODEL_NOTES[name],
            "cv_auc_mean": round(float(cv_scores.mean()), 4),
            "cv_auc_std": round(float(cv_scores.std()), 4),
            "metrics": {
                "accuracy": round(float(accuracy_score(y_test, pred)), 4),
                "precision": round(float(precision_score(y_test, pred)), 4),
                "recall": round(float(recall_score(y_test, pred)), 4),
                "f1": round(float(f1_score(y_test, pred)), 4),
                "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
            },
            "roc": {"fpr": [round(float(v), 4) for v in fpr],
                    "tpr": [round(float(v), 4) for v in tpr]},
        })
        fitted[name] = (clf, needs_scaling)

    # champion = best cross-validated ROC-AUC on the training split; the test
    # metrics above are reported, never used for selection
    best = max(models, key=lambda m: m["cv_auc_mean"])
    best_name = best["name"]
    best_clf, best_scaled = fitted[best_name]

    # confusion matrix for the champion at the default 0.5 threshold
    te = X_test_s if best_scaled else X_test
    cm = confusion_matrix(y_test, best_clf.predict(te))
    tn, fp, fn, tp = (int(v) for v in cm.ravel())

    # feature importance from the forest (stable, model-agnostic enough to read)
    rf_clf = fitted["Random Forest"][0]
    imp = pd.Series(rf_clf.feature_importances_, index=FEATURES).sort_values(ascending=False)

    # the logistic baseline, exported so the static Pages build can predict
    # in the browser (z-scored dot product + sigmoid — no server needed)
    lr_clf = fitted["Logistic Regression"][0]
    lr_auc = next(m["metrics"]["roc_auc"] for m in models if m["name"] == "Logistic Regression")
    lr_export = {
        "features": FEATURES,
        "coef": [round(float(c), 6) for c in lr_clf.coef_.ravel()],
        "intercept": round(float(lr_clf.intercept_[0]), 6),
        "mean": {c: round(float(v), 6) for c, v in zip(FEATURES, scaler.mean_)},
        "std": {c: round(float(v), 6) for c, v in zip(FEATURES, scaler.scale_)},
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
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "importance": {
            "labels": [str(k) for k in imp.index],
            "values": [round(float(v), 4) for v in imp.values],
        },
        "lr_export": lr_export,
        "form_meta": form_meta,
    }

    # keep the fitted champion + its transform for the prediction route
    _fitted_cache[_cache_key(path)] = (best_clf, scaler if best_scaled else None, impute_means)
    return bundle


def predict(path, values):
    """values: dict feature -> float. Returns (probability, model_name)."""
    model, scaler, impute_means = get_fitted(path)
    row = {c: values.get(c, float(impute_means[c])) for c in FEATURES}
    vec = pd.DataFrame([row], columns=FEATURES).fillna(impute_means)
    if scaler is not None:
        vec = scaler.transform(vec)
    proba = float(model.predict_proba(vec)[0, 1])
    return proba
