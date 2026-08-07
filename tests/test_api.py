"""JSON API contract tests — /api/health and /api/predict."""

import pytest


def test_health_shape(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert body["is_default_dataset"] is True
    assert "placement_predict_50k" in body["dataset"]
    # health never triggers training on a cold app
    assert body["trained"] is False or body["trained"] is True  # warm order-agnostic


@pytest.mark.slow
def test_api_predict_valid(client):
    client.get("/train")  # warm the model cache
    resp = client.post("/api/predict", json={
        "CGPA": 9.2, "AttendancePercent": 95, "Internships": 3, "Projects": 4,
        "Workshops": 3, "Certifications": 4, "Publications": 1,
        "AptitudeTestScore": 92, "SoftSkillsRating": 4.5,
        "CodingTestScore": 95, "MockInterviewScore": 90,
    })
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["placed"] is True
    assert body["probability"] > 90
    assert body["model"] == "Gradient Boosting"
    assert body["roc_auc"] > 0.95
    assert body["threshold"] == 0.5

    health = client.get("/api/health").get_json()
    assert health["trained"] is True
    assert health["model"] == "Gradient Boosting"


@pytest.mark.slow
def test_api_predict_empty_body_uses_medians(client):
    resp = client.post("/api/predict", json={})
    assert resp.status_code == 200
    body = resp.get_json()
    assert "placed" in body and "probability" in body


def test_api_predict_requires_json(client):
    resp = client.post("/api/predict", data="CGPA=8")
    assert resp.status_code == 415
    assert "expected_fields" in resp.get_json()


@pytest.mark.slow
def test_api_predict_validation(client):
    resp = client.post("/api/predict", json={"CGPA": 999})
    assert resp.status_code == 400
    assert "outside the observed range" in resp.get_json()["details"][0]

    resp = client.post("/api/predict", json={"CGPA": "abc"})
    assert resp.status_code == 400
    assert "expected a number" in resp.get_json()["details"][0]
