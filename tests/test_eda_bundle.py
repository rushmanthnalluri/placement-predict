"""EDA bundle facts for the bundled 50k cohort."""

import pytest

import app as app_module
import eda


@pytest.fixture(scope="module")
def bundle():
    return eda.get_bundle(app_module.DEFAULT_DATASET)


def test_schema_and_row_counts(bundle):
    assert bundle["schema_ok"] is True
    assert bundle["n_rows"] == 50_000
    assert bundle["dropped_rows"] == 1  # the StudentID-0 sentinel row


def test_missing_value_total(bundle):
    assert bundle["missing"]["total"] == 19_976
    assert len(bundle["missing"]["affected"]) == 5


def test_top_driver_is_cgpa(bundle):
    assert bundle["top_drivers"][0]["name"] == "CGPA"
