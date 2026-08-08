"""Page smoke tests on the bundled dataset, security headers, branded errors."""

import pytest

# EDA-only stages render from the cached EDA bundle — fast.
FAST_ROUTES = ["/", "/upload", "/features", "/descriptive", "/missing", "/visualize"]
# Model stages train the three candidates on first hit (~20s cold, once).
MODEL_ROUTES = ["/preprocess", "/train", "/evaluate", "/predict"]


@pytest.mark.parametrize("route", FAST_ROUTES)
def test_get_route_200(client, route):
    assert client.get(route).status_code == 200


@pytest.mark.slow
@pytest.mark.parametrize("route", MODEL_ROUTES)
def test_get_model_route_200(client, route):
    assert client.get(route).status_code == 200


@pytest.mark.slow
def test_train_model_drilldown(client):
    body = client.get("/train?model=random_forest").get_data(as_text=True)
    assert 'panel-title">Random Forest' in body
    assert "Confusion matrix" in body
    assert "ROC curve" in body
    # the dropdown reflects the selection
    assert 'value="random_forest" selected' in body


@pytest.mark.slow
def test_train_unknown_model_notice(client):
    resp = client.get("/train?model=svm")
    assert resp.status_code == 200
    assert "Unknown model" in resp.get_data(as_text=True)


@pytest.mark.slow
def test_train_benchmark_console_rendered(client):
    body = client.get("/train").get_data(as_text=True)
    assert "Best performing model" in body
    assert "Gradient Boosting" in body
    assert body.count('class="check-input bench-check"') == 3
    assert 'id="benchRun"' in body
    assert 'id="benchFresh"' in body
    assert "Brier" in body
    assert 'data-chart="benchmark"' in body


@pytest.mark.slow
def test_evaluate_shows_calibration(client):
    body = client.get("/evaluate").get_data(as_text=True)
    assert 'data-chart="calibration"' in body
    assert "Brier" in body
    assert "Log-loss" in body


def test_home_shows_default_overview(client):
    body = client.get("/").get_data(as_text=True)
    assert "50,000" in body          # usable records after the sentinel drop
    assert "65.7" in body            # placement rate, percent
    assert "placement_predict_50k.csv" in body


def test_home_dataset_overview_section(client):
    body = client.get("/").get_data(as_text=True)
    # the section, its four charts, and the stat cards
    assert "Dataset overview" in body
    assert 'data-chart="donut"' in body
    assert 'data-chart="ratefeat"' in body
    assert 'id="distSelect"' in body
    assert 'id="rateSelect"' in body
    assert "heat-grid" in body
    for label in ("Student records", "Total features", "Numerical features",
                  "Categorical features", "Placed", "Not placed",
                  "Placement rate", "Missing values"):
        assert label in body
    # auto-generated insights — computed values, not hand-written prose
    assert "Data insights" in body
    assert "50,000 student records" in body
    assert "CGPA between" in body          # IQR from the descriptive stats
    assert "strongest single placement signal" in body
    # chart payloads are embedded for the client-side renderers
    assert "window.EDA" in body
    assert "rateByFeature" in body


@pytest.mark.slow
def test_home_insight_reports_best_model(client):
    client.get("/train")  # warm the model cache so the insight can read it
    body = client.get("/").get_data(as_text=True)
    assert "Gradient Boosting currently achieves the strongest benchmark" in body
    assert "0.9733" in body


def test_security_headers(client):
    resp = client.get("/")
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"
    assert resp.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"


def test_404_branded(client):
    resp = client.get("/definitely-not-a-page")
    assert resp.status_code == 404
    assert "Page not found" in resp.get_data(as_text=True)


def test_405_branded(client):
    resp = client.post("/features")  # GET-only stage
    assert resp.status_code == 405
    assert "Method Not Allowed" in resp.get_data(as_text=True)
