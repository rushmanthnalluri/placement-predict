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


@pytest.mark.slow
def test_cv_auc_reported_for_all_three_models(model_bundle):
    assert len(model_bundle["models"]) == 3
    for entry in model_bundle["models"]:
        assert 0.5 < entry["cv_auc_mean"] <= 1.0


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
