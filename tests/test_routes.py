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


def test_home_shows_default_overview(client):
    body = client.get("/").get_data(as_text=True)
    assert "50,000" in body          # usable records after the sentinel drop
    assert "65.7" in body            # placement rate, percent
    assert "placement_predict_50k.csv" in body


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
