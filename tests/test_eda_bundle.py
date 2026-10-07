"""EDA bundle facts for the bundled 50k cohort."""

import pandas as pd
import pytest

import app as app_module
import eda


@pytest.fixture(scope="module")
def bundle():
    return eda.get_bundle(app_module.DEFAULT_DATASET)


def test_schema_and_row_counts(bundle):
    assert bundle["schema_ok"] is True
    assert bundle["n_rows"] == 50_000
    assert bundle["dropped_rows"] == 0  # no StudentID-0 sentinel row in this revision


def test_missing_value_total(bundle):
    assert bundle["missing"]["total"] == 19_976
    assert len(bundle["missing"]["affected"]) == 5


def test_top_driver_is_cgpa(bundle):
    assert bundle["top_drivers"][0]["name"] == "CGPA"


def test_heatmap_supports_negative_correlations():
    assert eda._heat_color(-1.0) != eda._heat_color(1.0)
    assert eda._heat_color(-0.7).startswith("#")
    assert eda._heat_color(0.7).startswith("#")
    corr = pd.DataFrame([[1.0, -0.8], [-0.8, 1.0]], columns=["a", "b"], index=["a", "b"])
    matrix = eda._heat_matrix(corr, ["a", "b"])
    assert matrix[0][1]["v"] == -0.8
    assert matrix[0][1]["show"] is True


def test_validate_dataset_contract(default_df):
    ok, error = eda.validate_dataset(default_df.head(100))
    assert ok is True
    assert error is None
    bad = default_df.head(100).copy()
    bad["CGPA"] = float("inf")
    ok, error = eda.validate_dataset(bad)
    assert ok is False
    assert "non-finite" in error