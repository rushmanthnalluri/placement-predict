import os
import secrets
import threading
from uuid import uuid4

import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

import eda
import model

app = Flask(__name__)

# Needed for session (Flask signs the session cookie with this). Set the
# SECRET_KEY environment variable in production; with no key configured an
# ephemeral random one is generated at startup, so a signed cookie can never
# be forged from a hardcoded value (sessions just don't survive restarts).
_secret_key = os.environ.get("SECRET_KEY")
if _secret_key:
    app.secret_key = _secret_key
else:
    app.secret_key = secrets.token_hex(32)
    app.logger.warning(
        "SECRET_KEY is not set - using an ephemeral random key; "
        "sessions will not survive restarts."
    )

# ---------------------------------------------------------------------------
# Upload configuration
# ---------------------------------------------------------------------------
UPLOAD_FOLDER = os.path.join(app.root_path, "data", "uploads")
ALLOWED_EXTENSIONS = {"csv", "xlsx"}
MAX_PREVIEW_ROWS = 8

# Bundled dataset (CSV twin of "placement_predict_50k Dataset.xlsx", kept for
# fast loading) powers every page until the user uploads their own file.
DEFAULT_DATASET = os.path.join(app.root_path, "data", "placement_predict_50k.csv")
DEFAULT_DATASET_NAME = "placement_predict_50k.csv"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def _clean_uploads():
    """Remove files left by dead sessions. Sessions are signed with an
    ephemeral key by default, so nothing from a previous run is reachable
    anyway; a missing file just falls back to the bundled dataset."""
    for name in os.listdir(UPLOAD_FOLDER):
        try:
            os.remove(os.path.join(UPLOAD_FOLDER, name))
        except OSError:
            pass


_clean_uploads()

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
# Off by default so local http keeps working; set SESSION_COOKIE_SECURE=1
# when serving over https.
app.config["SESSION_COOKIE_SECURE"] = os.environ.get(
    "SESSION_COOKIE_SECURE", ""
).lower() in {"1", "true", "yes"}


@app.after_request
def set_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    if request.path.startswith("/api/"):
        # The JSON API answers without credentials (cross-origin fetches
        # don't carry the session cookie, so they always see the bundled
        # dataset) and is meant to be callable — the static showcase on
        # GitHub Pages falls back to it. Never combine "*" with credentials.
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


def _warm_model_cache():
    """Train the bundled dataset's models in the background at startup so
    the first user request never pays the full training cost. The
    single-flight lock in model.get_model_bundle makes this safe to race
    with an early request."""
    try:
        model.get_model_bundle(DEFAULT_DATASET)
        # absorb the one-time first-predict initialisation as well
        model.predict(DEFAULT_DATASET, {})
        app.logger.info("model warm-up complete")
    except Exception:  # noqa: BLE001 - warm-up failure must never kill the app
        app.logger.exception("model warm-up failed (will retry on first request)")


# Boot-time warm-up is opt-in: on memory-capped hosts, training at boot can
# OOM the worker before it ever serves. Set WARM_MODEL=1 on hosts with room.
if os.environ.get("WARM_MODEL", "").lower() in {"1", "true", "yes"}:
    threading.Thread(target=_warm_model_cache, daemon=True).start()

# ---------------------------------------------------------------------------
# The ML pipeline, left to right in build order. This single list drives
# the left sidebar stepper, the home page roadmap, and the stub page
# rendered for each stage that isn't wired up yet. Stages marked live=True
# render real pages driven by the active dataset.
# ---------------------------------------------------------------------------
PIPELINE_STEPS = [
    {
        "id": "upload",
        "step": "01",
        "label": "Upload Dataset",
        "endpoint": "upload_dataset",
        "eyebrow": "Stage 01 · Intake",
        "live": True,
        "note": "Bring in the raw placement records as a CSV or Excel "
        "file — or explore the bundled 50k-student cohort.",
    },
    {
        "id": "features",
        "step": "02",
        "label": "Analyse Features",
        "endpoint": "analyse_features",
        "eyebrow": "Stage 02 · Records",
        "live": True,
        "note": "Every column in the record — its type, role, coverage, "
        "and sample values — before any analysis begins.",
    },
    {
        "id": "descriptive",
        "step": "03",
        "label": "Descriptive Statistics",
        "endpoint": "descriptive_statistics",
        "eyebrow": "Stage 03 · Records",
        "live": True,
        "note": "Mean, median, spread, and range across twenty numeric "
        "fields, then split by placement outcome.",
    },
    {
        "id": "missing",
        "step": "04",
        "label": "Missing Value Analysis",
        "endpoint": "missing_values",
        "eyebrow": "Stage 04 · Records",
        "live": True,
        "note": "Five skill and activity columns have gaps — measured here, "
        "then repaired with mean imputation.",
    },
    {
        "id": "visualize",
        "step": "05",
        "label": "Data Visualization",
        "endpoint": "visualize_data",
        "eyebrow": "Stage 05 · Records",
        "live": True,
        "note": "Distributions, correlations, and placement splits — the "
        "shape of the data, made visible.",
    },
    {
        "id": "preprocess",
        "step": "06",
        "label": "Preprocessing",
        "endpoint": "preprocess_data",
        "eyebrow": "Stage 06 · Preparation",
        "live": True,
        "note": "Impute with training means, standardise for the linear "
        "model, and seal a stratified 80/20 split — before any model sees data.",
    },
    {
        "id": "train",
        "step": "07",
        "label": "Model Training",
        "endpoint": "train_model",
        "eyebrow": "Stage 07 · Modelling",
        "live": True,
        "note": "Three classifiers trained on one sealed split — inspect any "
        "of them, or benchmark any subset and see the best emerge from the data.",
    },
    {
        "id": "evaluate",
        "step": "08",
        "label": "Model Evaluation",
        "endpoint": "evaluate_model",
        "eyebrow": "Stage 08 · Modelling",
        "live": True,
        "note": "One honest look at the sealed test set — metrics, ROC "
        "curves, confusion matrix, and feature importance.",
    },
    {
        "id": "predict",
        "step": "09",
        "label": "Predict Placement",
        "endpoint": "predict_placement",
        "eyebrow": "Stage 09 · Assessment",
        "live": True,
        "note": "Enter a student's profile, pick a model — or the recommended "
        "best — and get a placement call with its calibrated probability.",
    },
]


def _find_step(step_id):
    return next(step for step in PIPELINE_STEPS if step["id"] == step_id)


def _step_pager(step_id):
    """Previous/next stages for the pager at the foot of each stage page."""
    idx = next(i for i, s in enumerate(PIPELINE_STEPS) if s["id"] == step_id)
    prev_step = PIPELINE_STEPS[idx - 1] if idx > 0 else None
    next_step = PIPELINE_STEPS[idx + 1] if idx + 1 < len(PIPELINE_STEPS) else None
    return prev_step, next_step


@app.context_processor
def inject_pipeline():
    # Makes pipeline_steps and the active dataset's name available in every
    # template without passing them explicitly from each view.
    return {
        "pipeline_steps": PIPELINE_STEPS,
        "active_dataset_name": session.get("dataset_name", DEFAULT_DATASET_NAME),
    }


# ---------------------------------------------------------------------------
# Active dataset helpers
# ---------------------------------------------------------------------------

def _dataset_path(stored_name):
    """Resolve a stored upload basename to a path inside UPLOAD_FOLDER.

    The session cookie only ever carries a bare filename — never a path —
    so a tampered or legacy value resolves inside the upload dir or not
    at all.
    """
    if not stored_name:
        return None
    upload_dir = os.path.abspath(app.config["UPLOAD_FOLDER"])
    path = os.path.abspath(os.path.join(upload_dir, os.path.basename(stored_name)))
    if os.path.commonpath((upload_dir, path)) != upload_dir:
        return None
    return path


def _active_dataset():
    """(path, display name, is_default) for the dataset currently in play."""
    path = _dataset_path(session.get("dataset_file"))
    if path and os.path.exists(path):
        return path, session.get("dataset_name", "uploaded dataset"), False
    return DEFAULT_DATASET, DEFAULT_DATASET_NAME, True


def _active_bundle():
    path, name, is_default = _active_dataset()
    try:
        bundle = eda.get_bundle(path)
    except Exception:  # noqa: BLE001 - unreadable upload falls back to default
        bundle = eda.get_bundle(DEFAULT_DATASET)
        path, name, is_default = DEFAULT_DATASET, DEFAULT_DATASET_NAME, True
    return bundle, name, is_default


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------

def _allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _read_dataset(filepath):
    if filepath.lower().endswith(".csv"):
        return pd.read_csv(filepath)
    return pd.read_excel(filepath)


def _build_preview(df, max_rows=MAX_PREVIEW_ROWS):
    preview_df = df.head(max_rows).copy()
    for col in preview_df.select_dtypes(include="number").columns:
        preview_df[col] = preview_df[col].round(2)

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isna().sum().sum()),
        "column_names": list(df.columns),
        "head": preview_df.to_dict(orient="records"),
        "dtypes": [{"name": col, "dtype": str(df[col].dtype)} for col in df.columns],
    }


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def _model_bundle_if_warm(path):
    """The model bundle only when it costs nothing — already trained, or a
    validated artifact on disk. The home page must never trigger a cold
    train on a fresh checkout; the model insight is simply omitted then."""
    try:
        status = model.warm_status(path)
        if status.get("trained") or status.get("artifact_available"):
            return model.get_model_bundle(path)
    except Exception:  # noqa: BLE001 - a model problem must never break home
        return None
    return None


def _dataset_insights(bundle, mb):
    """Auto-generated plain-language findings — every number computed from
    the active dataset's bundle, never hand-written."""
    ov = bundle["overview"]
    insights = [
        f"The dataset contains {ov['rows']:,} student records across "
        f"{ov['columns']} fields — {ov['placed']:,} placed "
        f"({ov['placement_rate']}%), {ov['not_placed']:,} not placed.",
    ]
    desc = bundle.get("descriptive", {}).get("table", {})
    cgpa = desc.get("CGPA")
    if cgpa:
        insights.append(
            f"Half of all students have a CGPA between {cgpa['25%']} and "
            f"{cgpa['75%']} (median {cgpa['50%']})."
        )
    drivers = bundle.get("top_drivers") or []
    if drivers:
        insights.append(
            f"{drivers[0]['name']} is the strongest single placement signal "
            f"(correlation {drivers[0]['value']})."
        )
    infl = dict(zip(
        bundle.get("influence", {}).get("labels", []),
        bundle.get("influence", {}).get("values", []),
    ))
    for candidate in ("Internships", "Projects", "Certifications"):
        if infl.get(candidate, 0) > 0.05:
            insights.append(
                f"More {candidate.lower()} come with a higher placement rate "
                f"(r = {infl[candidate]})."
            )
            break
    missing = bundle.get("missing")
    if missing and missing.get("total"):
        insights.append(
            f"{missing['total']:,} values are missing across "
            f"{len(missing['affected'])} columns — repaired with "
            "training-split mean imputation before modelling."
        )
    if mb and mb.get("ok"):
        auc = next(
            m["metrics"]["roc_auc"] for m in mb["models"] if m["name"] == mb["best"]
        )
        insights.append(
            f"{mb['best']} currently achieves the strongest benchmark "
            f"performance (sealed-test ROC-AUC {auc})."
        )
    return insights


@app.route("/")
def home():
    bundle, dataset_name, is_default = _active_bundle()
    schema_ok = bundle.get("schema_ok", False)
    mb = None
    if schema_ok:
        path, _, _ = _active_dataset()
        mb = _model_bundle_if_warm(path)
    return render_template(
        "index.html",
        schema_ok=schema_ok,
        bundle=bundle,
        overview=bundle.get("overview") if schema_ok else None,
        top_drivers=bundle.get("top_drivers", []) if schema_ok else [],
        features=bundle.get("features") if schema_ok else None,
        dropped_rows=bundle.get("dropped_rows", 0) if schema_ok else 0,
        insights=_dataset_insights(bundle, mb) if schema_ok else [],
        dataset_name=dataset_name,
        is_default=is_default,
        active_step=None,
    )


@app.route("/upload", methods=["GET", "POST"])
def upload_dataset():
    step = _find_step("upload")
    error = None
    preview = None

    if request.method == "POST":
        uploaded_file = request.files.get("dataset")

        if uploaded_file is None or uploaded_file.filename == "":
            error = "Choose a CSV or Excel file before uploading."
        elif not _allowed_file(uploaded_file.filename):
            error = "Only .csv or .xlsx files are accepted."
        else:
            display_name = uploaded_file.filename
            # Namespace the stored file per upload so concurrent sessions
            # can't clobber each other's dataset; the original name is
            # kept only for display.
            stored_name = f"{uuid4().hex[:8]}_{secure_filename(display_name)}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], stored_name)
            uploaded_file.save(filepath)

            try:
                df = _read_dataset(filepath)
            except Exception:  # noqa: BLE001 - any parse failure rejects the upload
                app.logger.exception("Could not parse uploaded file %s", stored_name)
                os.remove(filepath)
                error = ("That file couldn't be read as a CSV or Excel "
                         "dataset. Check the file and try again.")
            else:
                missing = sorted(eda.REQUIRED_COLS - set(df.columns))
                if missing:
                    os.remove(filepath)
                    error = ("That file is missing required columns: "
                             + ", ".join(missing) + ".")
                else:
                    session["dataset_file"] = stored_name
                    session["dataset_name"] = display_name
                    preview = _build_preview(df)

    # No fresh upload this request - if one is already on file, show it
    # (even alongside an error, so the active dataset stays visible).
    if preview is None and session.get("dataset_file"):
        try:
            df = _read_dataset(_dataset_path(session["dataset_file"]))
            preview = _build_preview(df)
        except Exception:  # noqa: BLE001 - stored file is missing or unreadable
            session.pop("dataset_file", None)
            session.pop("dataset_name", None)

    # With no upload on file, preview the bundled dataset instead.
    dataset_name = session.get("dataset_name")
    if preview is None and error is None:
        df = eda.load_dataframe(DEFAULT_DATASET)
        preview = _build_preview(df)
        dataset_name = DEFAULT_DATASET_NAME

    prev_step, next_step = _step_pager("upload")
    return render_template(
        "upload.html",
        step=step,
        active_step="upload",
        error=error,
        preview=preview,
        dataset_name=dataset_name,
        is_default=not session.get("dataset_file"),
        prev_step=prev_step,
        next_step=next_step,
    )


@app.route("/upload/clear", methods=["POST"])
def clear_dataset():
    stored_name = session.pop("dataset_file", None)
    session.pop("dataset_name", None)
    filepath = _dataset_path(stored_name)
    if filepath and os.path.exists(filepath):
        os.remove(filepath)
    return redirect(url_for("upload_dataset"))


def _eda_stage_view(step_id, template):
    """Real view for an EDA stage: active dataset's bundle + pager."""
    step = _find_step(step_id)
    bundle, dataset_name, is_default = _active_bundle()
    prev_step, next_step = _step_pager(step_id)
    return render_template(
        template,
        step=step,
        active_step=step_id,
        bundle=bundle,
        dataset_name=dataset_name,
        is_default=is_default,
        prev_step=prev_step,
        next_step=next_step,
    )


@app.route("/features")
def analyse_features():
    return _eda_stage_view("features", "features.html")


@app.route("/descriptive")
def descriptive_statistics():
    return _eda_stage_view("descriptive", "descriptive.html")


@app.route("/missing")
def missing_values():
    return _eda_stage_view("missing", "missing.html")


@app.route("/visualize")
def visualize_data():
    return _eda_stage_view("visualize", "visualize.html")


def _model_stage_view(step_id, template):
    step = _find_step(step_id)
    bundle, dataset_name, is_default = _active_bundle()
    prev_step, next_step = _step_pager(step_id)
    model_bundle = None
    if bundle["schema_ok"]:
        path, _, _ = _active_dataset()
        try:
            model_bundle = model.get_model_bundle(path)
        except Exception as exc:  # noqa: BLE001 - never 500 a stage page
            model_bundle = {"schema_ok": True, "ok": False, "error": str(exc)}
    return render_template(
        template,
        step=step,
        active_step=step_id,
        bundle=bundle,
        mb=model_bundle,
        dataset_name=dataset_name,
        is_default=is_default,
        prev_step=prev_step,
        next_step=next_step,
    )


@app.route("/preprocess")
def preprocess_data():
    return _model_stage_view("preprocess", "preprocess.html")


@app.route("/train")
def train_model():
    step = _find_step("train")
    bundle, dataset_name, is_default = _active_bundle()
    prev_step, next_step = _step_pager("train")
    mb = None
    if bundle["schema_ok"]:
        path, _, _ = _active_dataset()
        try:
            mb = model.get_model_bundle(path)
        except Exception as exc:  # noqa: BLE001 - never 500 a stage page
            mb = {"schema_ok": True, "ok": False, "error": str(exc)}

    # single-model drill-down: /train?model=random_forest — an unknown key
    # renders the page with a notice instead of a 404 (the URL is user-facing)
    requested_model = request.args.get("model", "").strip()
    selected = None
    if mb and mb.get("ok") and requested_model:
        selected_key = model.resolve_model_key(requested_model)
        if selected_key:
            selected = next(
                (m for m in mb["models"] if m["key"] == selected_key), None
            )

    return render_template(
        "train.html",
        step=step,
        active_step="train",
        bundle=bundle,
        mb=mb,
        selected=selected,
        requested_model=requested_model,
        dataset_name=dataset_name,
        is_default=is_default,
        prev_step=prev_step,
        next_step=next_step,
    )


@app.route("/evaluate")
def evaluate_model():
    return _model_stage_view("evaluate", "evaluate.html")


def _resolve_requested_model(raw, mb):
    """(display_name, key, requested, error) for a model selector from the
    predict form or the JSON API. `requested` is what the UI re-selects
    ("best" or a registry key); `key` is always the concrete model resolved
    from the latest training run."""
    if not raw or model.is_best_alias(raw):
        return mb["best"], mb["best_key"], "best", None
    key = model.resolve_model_key(raw)
    if key is None:
        return None, None, "best", (
            f"Unknown model “{raw}” — pick one of the listed models."
        )
    return model.MODEL_REGISTRY[key]["name"], key, key, None


def _prediction_note(placed, probability_pct, model_name, roc_auc):
    """Responsible one-liner under the verdict — banded by distance from the
    threshold, never a guarantee."""
    p = probability_pct / 100
    if placed:
        tone = ("predicts a high likelihood of placement"
                if p >= 0.8 else
                "leans toward placement, though the margin is modest")
    else:
        tone = ("predicts a low likelihood of placement"
                if p <= 0.2 else
                "leans against placement, though the margin is modest")
    return (f"The model {tone} based on the provided features. This is a "
            f"calibrated statistical estimate from {model_name} (sealed-test "
            f"ROC-AUC {roc_auc}), not a guarantee — probabilities near the "
            "50% threshold are genuinely uncertain calls.")


@app.route("/predict", methods=["GET", "POST"])
def predict_placement():
    step = _find_step("predict")
    bundle, dataset_name, is_default = _active_bundle()
    prev_step, next_step = _step_pager("predict")
    path, _, _ = _active_dataset()

    mb = None
    if bundle["schema_ok"]:
        try:
            mb = model.get_model_bundle(path)
        except Exception as exc:  # noqa: BLE001 - never 500 the page
            mb = {"schema_ok": True, "ok": False, "error": str(exc)}
    model_ready = bool(mb and mb.get("ok"))

    result = None
    errors = []
    invalid_fields = []
    values = {}
    selected_model = "best"

    if request.method == "POST" and model_ready:
        raw_model = request.form.get("model", "best")
        chosen, _, selected_model, model_error = _resolve_requested_model(raw_model, mb)
        if model_error:
            errors.append(model_error)
        for meta in mb["form_meta"]:
            name = meta["name"]
            raw = request.form.get(name, "").strip()
            if raw == "":
                raw = str(meta["default"])  # blank input -> dataset median
            try:
                val = float(raw)
            except ValueError:
                errors.append(f"{name}: “{raw}” is not a number.")
                invalid_fields.append(name)
                continue
            if not (meta["min"] <= val <= meta["max"]):
                errors.append(
                    f"{name}: {val:g} is outside the observed range "
                    f"{meta['min']:g}–{meta['max']:g}."
                )
                invalid_fields.append(name)
            values[name] = val
        if not errors:
            proba = model.predict(path, values, chosen)
            roc_auc = next(
                m["metrics"]["roc_auc"] for m in mb["models"] if m["name"] == chosen
            )
            placed = proba >= 0.5
            probability = round(proba * 100, 1)
            result = {
                "placed": placed,
                "probability": probability,
                "model": chosen,
                "is_champion": chosen == mb["best"],
                "roc_auc": roc_auc,
                "explanation": _prediction_note(placed, probability, chosen, roc_auc),
                "values": values,
            }

    return render_template(
        "predict.html",
        step=step,
        active_step="predict",
        bundle=bundle,
        mb=mb,
        model_ready=model_ready,
        dataset_name=dataset_name,
        is_default=is_default,
        prev_step=prev_step,
        next_step=next_step,
        result=result,
        errors=errors,
        invalid_fields=invalid_fields,
        values=values,
        selected_model=selected_model,
    )


LIVE_STAGES = {
    "upload", "features", "descriptive", "missing", "visualize",
    "preprocess", "train", "evaluate", "predict",
}


# ---------------------------------------------------------------------------
# JSON API — the service contract. Same validation and model selection as the
# UI form; health never triggers training.
# ---------------------------------------------------------------------------

@app.route("/api/health")
def api_health():
    path, name, is_default = _active_dataset()
    status = {"status": "ok", "dataset": name, "is_default_dataset": is_default}
    status.update(model.warm_status(path))
    return jsonify(status)


@app.route("/api/dataset")
def api_dataset():
    """Dataset overview for programmatic consumers: the same cached bundle
    numbers and auto-generated insights the home page renders."""
    bundle, dataset_name, is_default = _active_bundle()
    if not bundle["schema_ok"]:
        return jsonify({
            "error": "The active dataset does not match the placement schema.",
            "dataset": dataset_name,
        }), 503
    ov = bundle["overview"]
    feats = bundle["features"]
    path, _, _ = _active_dataset()
    insights = _dataset_insights(bundle, _model_bundle_if_warm(path))
    return jsonify({
        "dataset": dataset_name,
        "is_default_dataset": is_default,
        "summary": {
            "total_records": ov["rows"],
            "total_features": ov["columns"],
            "numerical_features": feats["n_numeric"],
            "categorical_features": feats["n_categorical"],
            "placed": ov["placed"],
            "not_placed": ov["not_placed"],
            "placement_rate": ov["placement_rate"],
            "missing_values": ov["missing_total"],
            "completeness": ov["completeness"],
        },
        "insights": insights,
        "distributions": bundle["histograms"],
        "rate_by_feature": bundle["rate_by_feature"],
        "correlation": bundle["heatmap_core"],
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not request.is_json:
        return jsonify({
            "error": "Content-Type must be application/json",
            "expected_fields": model.FEATURES,
        }), 415

    bundle, dataset_name, _ = _active_bundle()
    if not bundle["schema_ok"]:
        return jsonify({
            "error": "The active dataset does not match the placement schema.",
            "dataset": dataset_name,
        }), 503

    path, _, _ = _active_dataset()
    try:
        mb = model.get_model_bundle(path)
    except Exception as exc:  # noqa: BLE001
        mb = {"ok": False, "error": str(exc)}
    if not mb.get("ok"):
        return jsonify({
            "error": "No trained model available for the active dataset.",
            "detail": mb.get("error"),
        }), 503

    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "JSON body must be an object of feature values."}), 400

    # model selection: "model" may be a registry key ("random_forest"), a
    # display name ("Random Forest"), or "best" (the default) for the
    # champion of the latest training run
    chosen, chosen_key, _, model_error = _resolve_requested_model(
        payload.get("model", "best"), mb
    )
    if model_error:
        return jsonify({
            "error": model_error,
            "valid_models": model.MODEL_KEYS + ["best"],
        }), 400

    errors = []
    values = {}
    for meta in mb["form_meta"]:
        name = meta["name"]
        raw = payload.get(name)
        if raw is None or raw == "":
            values[name] = float(meta["default"])  # absent -> dataset median
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{name}: expected a number, got {raw!r}.")
            continue
        if not (meta["min"] <= val <= meta["max"]):
            errors.append(
                f"{name}: {val:g} is outside the observed range "
                f"{meta['min']:g}–{meta['max']:g}."
            )
        values[name] = val

    if errors:
        return jsonify({"error": "Validation failed", "details": errors}), 400

    proba = model.predict(path, values, chosen)
    return jsonify({
        "placed": proba >= 0.5,
        "probability": round(proba * 100, 1),
        "threshold": 0.5,
        "model": chosen,
        "model_key": chosen_key,
        "roc_auc": next(m["metrics"]["roc_auc"] for m in mb["models"] if m["name"] == chosen),
        "dataset": dataset_name,
    })


@app.route("/api/benchmark", methods=["POST"])
def api_benchmark():
    """Comparative evaluation of the requested models on the active dataset.

    Body (optional): {"models": ["logistic_regression", ...], "fresh": false}
    — an absent/empty body benchmarks all three candidates; "fresh": true
    genuinely re-runs the pipeline for the selection instead of reading the
    cached evaluation. Every number comes from the real training run for the
    active dataset (first call trains, later calls read the cached
    evaluation)."""
    payload = {}
    if request.is_json:
        payload = request.get_json(silent=True)
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            return jsonify({
                "error": 'JSON body must be an object like {"models": [...]}.'
            }), 400
    elif request.form or (request.data and request.data.strip()):
        return jsonify({
            "error": "Content-Type must be application/json",
            "valid_models": model.MODEL_KEYS,
        }), 415

    bundle, dataset_name, _ = _active_bundle()
    if not bundle["schema_ok"]:
        return jsonify({
            "error": "The active dataset does not match the placement schema.",
            "dataset": dataset_name,
        }), 503

    keys = None
    raw_models = payload.get("models")
    if raw_models is not None:
        if not isinstance(raw_models, list) or not all(
            isinstance(item, str) for item in raw_models
        ):
            return jsonify({
                "error": '"models" must be a list of model keys.',
                "valid_models": model.MODEL_KEYS,
            }), 400
        keys = []
        unknown = []
        for item in raw_models:
            key = model.resolve_model_key(item)
            (keys if key else unknown).append(key or item)
        if unknown:
            return jsonify({
                "error": "Unknown model(s): " + ", ".join(unknown) + ".",
                "valid_models": model.MODEL_KEYS,
            }), 400
        keys = list(dict.fromkeys(keys))
        if not keys:
            return jsonify({
                "error": "Select at least one model to benchmark.",
                "valid_models": model.MODEL_KEYS,
            }), 400

    path, _, _ = _active_dataset()
    # strict boolean: only an actual true (or an explicit truthy string)
    # triggers a fresh run — {"fresh": "false"} must not retrain
    raw_fresh = payload.get("fresh", False)
    fresh = raw_fresh is True or (
        isinstance(raw_fresh, str)
        and raw_fresh.strip().lower() in {"1", "true", "yes"}
    )
    result = model.benchmark(path, keys, fresh=fresh)
    if not result.get("ok"):
        return jsonify({
            "error": "No trained models available for the active dataset.",
            "detail": result.get("error"),
            "dataset": dataset_name,
        }), 503
    result["dataset"] = dataset_name
    return jsonify(result)


def _make_stage_view(step):
    """Builds a stub view for a pipeline stage. Replace the body of the
    returned function with real logic (pandas, sklearn, etc.) stage by
    stage - the routing and sidebar stay exactly as they are."""

    def view():
        prev_step, next_step = _step_pager(step["id"])
        return render_template(
            "stage.html",
            step=step,
            active_step=step["id"],
            prev_step=prev_step,
            next_step=next_step,
        )

    view.__name__ = step["endpoint"]
    return view


for _step in PIPELINE_STEPS:
    if _step["id"] in LIVE_STAGES:
        continue  # live stages have real routes above
    app.add_url_rule(
        f"/{_step['id']}",
        endpoint=_step["endpoint"],
        view_func=_make_stage_view(_step),
    )


@app.errorhandler(404)
def not_found(err):
    return render_template(
        "error.html",
        code=404,
        title="Page not found",
        message="That URL doesn't exist in this app. The nine pipeline stages "
        "are listed on the left — pick one to continue.",
        active_step=None,
    ), 404


@app.errorhandler(413)
def too_large(err):
    return render_template(
        "error.html",
        code=413,
        title="File too large",
        message="Uploads are capped at 10 MB. Trim the file or convert it to "
        "CSV (which compresses far better than Excel) and try again.",
        active_step="upload",
    ), 413


@app.errorhandler(500)
def server_error(err):
    return render_template(
        "error.html",
        code=500,
        title="Something broke on our side",
        message="An unexpected error occurred. Your dataset is unaffected — "
        "head back to the overview and pick up where you left off.",
        active_step=None,
    ), 500


@app.errorhandler(HTTPException)
def http_error(err):
    """Branded page for any HTTP error without its own handler (405, ...)."""
    code = err.code or 500
    return render_template(
        "error.html",
        code=code,
        title=err.name,
        message=err.description,
        active_step=None,
    ), code


if __name__ == "__main__":
    # Debug is opt-in: set FLASK_DEBUG=1 for local development.
    app.run(debug=os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"})
