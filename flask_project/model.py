"""Model training, evaluation, and inference for the Placement Predict pipeline.

Trains three classifiers on the active dataset — a logistic-regression
baseline, a random forest, and a gradient-boosted ensemble — behind a sealed
80/20 stratified split (seed 42), and caches both the JSON-safe evaluation
bundle and the fitted champion for the prediction form.

Mirrors the workbook discipline: impute with training means, standardise for
the linear model, trees on raw features, assess once on the sealed test set.
"""

import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import eda

FEATURES = [
    "CGPA", "AttendancePercent", "Internships", "Projects", "Workshops",
    "Certifications", "Publications", "AptitudeTestScore", "SoftSkillsRating",
    "CodingTestScore", "MockInterviewScore", "ExtraCurricular",
]
TARGET = "PlacementStatus"
SEED = 42

# short human note per model, for the training page
MODEL_NOTES = {
    "Logistic Regression": "interpretable linear baseline — the bar to beat",
    "Random Forest": "deep decorrelated trees — averages away variance",
    "Gradient Boosting": "shallow sequential trees — corrects residual bias",
}

_bundle_cache = {}
_fitted_cache = {}


def _cache_key(path):
    return (os.path.abspath(path), os.path.getmtime(path))


def get_model_bundle(path):
    key = _cache_key(path)
    if key not in _bundle_cache:
        _bundle_cache.clear()
        _fitted_cache.clear()
        _bundle_cache[key] = _train_all(path)
    return _bundle_cache[key]


def get_fitted(path):
    """(model, scaler_or_None, impute_means) for the champion — for inference."""
    get_model_bundle(path)  # ensures caches are warm
    return _fitted_cache.get(_cache_key(path))


def _subsample_curve(fpr, tpr, points=80):
    """Thin an ROC curve to ~80 points so the JSON payload stays small."""
    if len(fpr) <= points:
        return fpr, tpr
    idx = np.linspace(0, len(fpr) - 1, points).astype(int)
    return fpr[idx], tpr[idx]


def _train_all(path):
    df = eda.load_dataframe(path)
    if not eda.schema_ok(df):
        return {"schema_ok": False, "ok": False}

    # same cleaning as the EDA bundle: drop the corrupt sentinel row
    if "StudentID" in df.columns:
        df = df[df["StudentID"] != 0]

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)
    impute_means = X.mean()
    X = X.fillna(impute_means)

    # degenerate uploads get a clear reason instead of a traceback
    if y.nunique() < 2:
        only = "Placed" if int(y.iloc[0]) == 1 else "Not placed"
        return {
            "schema_ok": True, "ok": False,
            "error": f"Every row in this dataset has the same outcome ({only}). "
                     "A classifier needs both placed and not-placed examples to "
                     "learn the difference.",
        }
    if len(df) < 50:
        return {
            "schema_ok": True, "ok": False,
            "error": f"Only {len(df)} usable rows — too few to train and evaluate "
                     "honestly. Upload at least a few hundred rows.",
        }
    return _fit_and_evaluate(path, df, X, y, impute_means)


def _fit_and_evaluate(path, df, X, y, impute_means):
    try:
        return _fit_and_evaluate_inner(path, df, X, y, impute_means)
    except Exception as exc:  # noqa: BLE001 - surface training failures as a page, not a 500
        return {
            "schema_ok": True, "ok": False,
            "error": f"Training failed on this dataset: {exc}",
        }


def _fit_and_evaluate_inner(path, df, X, y, impute_means):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED,
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    candidates = [
        ("Logistic Regression", LogisticRegression(max_iter=2000), True),
        ("Random Forest", RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=SEED), False),
        ("Gradient Boosting", HistGradientBoostingClassifier(random_state=SEED), False),
    ]

    models = []
    fitted = {}
    for name, clf, needs_scaling in candidates:
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

    best = max(models, key=lambda m: m["metrics"]["roc_auc"])
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
        form_meta.append({
            "name": col,
            "min": round(float(s.min()), 2),
            "max": round(float(s.max()), 2),
            "step": "1" if pd.api.types.is_integer_dtype(s) else "0.1",
            "default": round(float(s.median()), 2),
        })

    bundle = {
        "schema_ok": True,
        "ok": True,
        "features": FEATURES,
        "seed": SEED,
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
