"""Champion-model sanity on the bundled dataset + degenerate-upload handling.

The four bundle tests share one module-scoped training run (~20s, marked
slow). The degenerate-upload tests fail before any fitting, so they stay
in the fast subset.
"""

import pandas as pd
import pytest

import app as app_module
import model
from conftest import make_csv


@pytest.fixture(scope="module")
def model_bundle():
    return model.get_model_bundle(app_module.DEFAULT_DATASET)


@pytest.mark.slow
def test_champion_is_gradient_boosting(model_bundle):
    assert model_bundle["ok"] is True
    assert model_bundle["best"] == "Gradient Boosting"
    assert model_bundle["best_key"] == "gradient_boosting"


@pytest.mark.slow
def test_cv_auc_reported_for_all_three_models(model_bundle):
    assert len(model_bundle["models"]) == 3
    for entry in model_bundle["models"]:
        assert 0.5 < entry["cv_auc_mean"] <= 1.0


@pytest.mark.slow
def test_model_keys_and_settings_from_registry(model_bundle):
    keys = [m["key"] for m in model_bundle["models"]]
    assert keys == model.MODEL_KEYS
    for entry in model_bundle["models"]:
        spec = model.MODEL_REGISTRY[entry["key"]]
        assert entry["name"] == spec["name"]
        assert entry["settings"] == spec["settings"]
        assert entry["note"] == spec["note"]


@pytest.mark.slow
def test_champion_test_roc_auc(model_bundle):
    champ = next(
        m for m in model_bundle["models"] if m["name"] == model_bundle["best"]
    )
    assert champ["metrics"]["roc_auc"] > 0.95


@pytest.mark.slow
def test_confusion_covers_sealed_test_set(model_bundle):
    cm = model_bundle["confusion"]
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == 10_000
    assert model_bundle["split"]["test"] == 10_000


@pytest.mark.slow
def test_every_model_carries_a_full_confusion(model_bundle):
    for entry in model_bundle["models"]:
        cm = entry["confusion"]
        assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == 10_000


@pytest.mark.slow
def test_champion_matches_best_of_benchmark(model_bundle):
    result = model.benchmark(app_module.DEFAULT_DATASET)
    assert result["ok"] is True
    assert result["source"] == "cached_evaluation"
    assert len(result["models"]) == 3
    assert result["best"]["name"] == model_bundle["best"]
    assert result["overall_best"]["name"] == model_bundle["best"]


@pytest.mark.slow
def test_fresh_benchmark_reproduces_cached_numbers(model_bundle):
    """fresh=True re-runs the pipeline for real; the deterministic recipe
    must land on the cached evaluation exactly."""
    fresh = model.benchmark(
        app_module.DEFAULT_DATASET, ["gradient_boosting"], fresh=True
    )
    assert fresh["ok"] is True
    assert fresh["source"] == "fresh_run"
    cached = next(
        m for m in model_bundle["models"] if m["key"] == "gradient_boosting"
    )
    assert fresh["models"][0]["metrics"] == cached["metrics"]
    assert fresh["models"][0]["confusion"] == cached["confusion"]


@pytest.mark.slow
def test_models_are_calibrated(model_bundle):
    for entry in model_bundle["models"]:
        assert entry["calibration"].startswith("Platt")
        assert 0.0 < entry["metrics"]["brier"] < 1.0
        assert 0.0 < entry["metrics"]["log_loss"]
        rel = entry["reliability"]
        assert len(rel["bin_mid"]) == len(rel["frac_pos"]) > 0
    # on this dataset the champion is also the best-calibrated
    champ = next(
        m for m in model_bundle["models"] if m["name"] == model_bundle["best"]
    )
    others = [
        m for m in model_bundle["models"] if m["name"] != model_bundle["best"]
    ]
    assert all(champ["metrics"]["brier"] < m["metrics"]["brier"] for m in others)


@pytest.mark.slow
def test_benchmark_subset_is_a_fair_filter(model_bundle):
    result = model.benchmark(
        app_module.DEFAULT_DATASET, ["logistic_regression", "random_forest"]
    )
    assert result["ok"] is True
    assert [m["key"] for m in result["models"]] == [
        "logistic_regression", "random_forest",
    ]
    # best of the subset by the same champion rule; overall winner unchanged
    assert result["best"]["name"] == "Random Forest"
    assert result["overall_best"]["name"] == "Gradient Boosting"
    # subset numbers are the same measured numbers, not a re-run
    full = {m["key"]: m for m in model_bundle["models"]}
    for m in result["models"]:
        assert m["metrics"] == full[m["key"]]["metrics"]


@pytest.mark.slow
def test_predict_with_each_model(model_bundle):
    values = {m["name"]: m["default"] for m in model_bundle["form_meta"]}
    for name in model.MODEL_NAMES:
        proba = model.predict(app_module.DEFAULT_DATASET, values, name)
        assert 0.0 <= proba <= 1.0
    # default is the champion
    assert model.predict(app_module.DEFAULT_DATASET, values) == model.predict(
        app_module.DEFAULT_DATASET, values, model_bundle["best"]
    )


# -- degenerate uploads: rejected with a reason, before any training ---------


def test_single_class_target_rejected(tmp_path, default_df):
    df = default_df[default_df["StudentID"] != 0]
    df = df[df["PlacementStatus"] == 1].head(100)
    bundle = model.get_model_bundle(str(make_csv(tmp_path, df, "one_class.csv")))
    assert bundle["ok"] is False
    assert bundle["error"]


def test_too_few_rows_rejected(tmp_path, default_df):
    df = default_df[default_df["StudentID"] != 0]
    sample = pd.concat(
        [
            df[df["PlacementStatus"] == 1].head(15),
            df[df["PlacementStatus"] == 0].head(15),
        ]
    )
    bundle = model.get_model_bundle(str(make_csv(tmp_path, sample, "small.csv")))
    assert bundle["ok"] is False
    assert "30" in bundle["error"]


def test_string_typed_feature_rejected(tmp_path, default_df):
    df = default_df[default_df["StudentID"] != 0].head(100).copy()
    df["CGPA"] = df["CGPA"].astype(str) + "!"  # stays text after CSV round-trip
    bundle = model.get_model_bundle(str(make_csv(tmp_path, df, "text_cgpa.csv")))
    assert bundle["ok"] is False
    assert bundle["error"]
