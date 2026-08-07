"""Prediction route: input validation, median fallback, verdict rendering.

Every test here needs the trained champion model, so all are marked slow.
"""

import pytest

import app as app_module
import model


@pytest.fixture(scope="module")
def model_bundle():
    return model.get_model_bundle(app_module.DEFAULT_DATASET)


def _profile(bundle, key):
    """A complete form submission built from the bundle's form metadata."""
    return {m["name"]: str(m[key]) for m in bundle["form_meta"]}


def _has_verdict(body):
    return ('class="result-verdict">Placed<' in body) or (
        'class="result-verdict">Not placed<' in body
    )


@pytest.mark.slow
def test_predict_valid_profile_renders_verdict(client, model_bundle):
    resp = client.post("/predict", data=_profile(model_bundle, "default"))
    body = resp.get_data(as_text=True)
    assert resp.status_code == 200
    assert _has_verdict(body)
    assert "% probability" in body


@pytest.mark.slow
def test_predict_out_of_range_flagged(client, model_bundle):
    data = _profile(model_bundle, "default")
    data["CGPA"] = "999"  # observed CGPA range is 0-10
    resp = client.post("/predict", data=data)
    body = resp.get_data(as_text=True)
    assert resp.status_code == 200
    assert "outside the observed range" in body
    assert not _has_verdict(body)  # invalid input suppresses the verdict


@pytest.mark.slow
def test_predict_non_numeric_flagged(client, model_bundle):
    data = _profile(model_bundle, "default")
    data["CGPA"] = "abc"
    resp = client.post("/predict", data=data)
    body = resp.get_data(as_text=True)
    assert resp.status_code == 200
    assert "is not a number" in body


@pytest.mark.slow
def test_predict_all_blank_falls_back_to_medians(client):
    resp = client.post("/predict", data={})
    body = resp.get_data(as_text=True)
    assert resp.status_code == 200
    assert _has_verdict(body)
    assert "% probability" in body
