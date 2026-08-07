import os

import pandas as pd
from flask import Flask, redirect, render_template, request, session, url_for
from werkzeug.utils import secure_filename

import eda
import model

app = Flask(__name__)

# Needed for session (Flask signs the session cookie with this). Set the
# SECRET_KEY environment variable in production; the fallback is dev-only.
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-change-me")

# ---------------------------------------------------------------------------
# Upload configuration
# ---------------------------------------------------------------------------
UPLOAD_FOLDER = os.path.join(app.root_path, "data", "uploads")
ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}
MAX_PREVIEW_ROWS = 8

# Bundled dataset (CSV twin of "placement_predict_50k Dataset.xlsx", kept for
# fast loading) powers every page until the user uploads their own file.
DEFAULT_DATASET = os.path.join(app.root_path, "data", "placement_predict_50k.csv")
DEFAULT_DATASET_NAME = "placement_predict_50k.csv"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

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
        "note": "Three classifiers trained and timed: an interpretable "
        "logistic baseline, a random forest, and gradient boosting.",
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
        "note": "Enter a student's profile and get a placement call, with "
        "a calibrated probability, from the champion model.",
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

def _active_dataset():
    """(path, display name, is_default) for the dataset currently in play."""
    path = session.get("dataset_path")
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

@app.route("/")
def home():
    bundle, dataset_name, is_default = _active_bundle()
    schema_ok = bundle.get("schema_ok", False)
    return render_template(
        "index.html",
        schema_ok=schema_ok,
        overview=bundle.get("overview") if schema_ok else None,
        top_drivers=bundle.get("top_drivers", []) if schema_ok else [],
        features=bundle.get("features") if schema_ok else None,
        dropped_rows=bundle.get("dropped_rows", 0) if schema_ok else 0,
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
            error = "Only .csv, .xlsx, or .xls files are accepted."
        else:
            filename = secure_filename(uploaded_file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            uploaded_file.save(filepath)

            try:
                df = _read_dataset(filepath)
            except Exception as exc:  # noqa: BLE001 - surface any parse error to the user
                error = f"Could not read that file: {exc}"
            else:
                session["dataset_path"] = filepath
                session["dataset_name"] = filename
                preview = _build_preview(df)

    # No fresh upload this request - if one is already on file, show it
    # (even alongside an error, so the active dataset stays visible).
    if preview is None and session.get("dataset_path"):
        try:
            df = _read_dataset(session["dataset_path"])
            preview = _build_preview(df)
        except Exception:  # noqa: BLE001 - stored file is missing or unreadable
            session.pop("dataset_path", None)
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
        is_default=not session.get("dataset_path"),
        prev_step=prev_step,
        next_step=next_step,
    )


@app.route("/upload/clear", methods=["POST"])
def clear_dataset():
    filepath = session.pop("dataset_path", None)
    session.pop("dataset_name", None)
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
    return _model_stage_view("train", "train.html")


@app.route("/evaluate")
def evaluate_model():
    return _model_stage_view("evaluate", "evaluate.html")


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
    values = {}

    if request.method == "POST" and model_ready:
        for meta in mb["form_meta"]:
            name = meta["name"]
            raw = request.form.get(name, "").strip()
            if raw == "":
                raw = str(meta["default"])  # blank input -> dataset median
            try:
                val = float(raw)
            except ValueError:
                errors.append(f"{name}: “{raw}” is not a number.")
                continue
            if not (meta["min"] <= val <= meta["max"]):
                errors.append(
                    f"{name}: {val:g} is outside the observed range "
                    f"{meta['min']:g}–{meta['max']:g}."
                )
            values[name] = val
        if not errors:
            proba = model.predict(path, values)
            result = {
                "placed": proba >= 0.5,
                "probability": round(proba * 100, 1),
                "model": mb["best"],
                "roc_auc": next(m["metrics"]["roc_auc"] for m in mb["models"] if m["name"] == mb["best"]),
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
        values=values,
    )


LIVE_STAGES = {
    "upload", "features", "descriptive", "missing", "visualize",
    "preprocess", "train", "evaluate", "predict",
}


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


if __name__ == "__main__":
    app.run(debug=True)
