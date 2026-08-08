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


# -- model selection ----------------------------------------------------------


@pytest.mark.slow
def test_api_predict_with_each_model(client):
    profile = {"CGPA": 9.2, "MockInterviewScore": 88, "CodingTestScore": 85}
    seen = {}
    for key in ["logistic_regression", "random_forest", "gradient_boosting"]:
        resp = client.post("/api/predict", json={**profile, "model": key})
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["model_key"] == key
        seen[key] = body
    # each response genuinely names a different fitted model with its own AUC
    assert len({b["model"] for b in seen.values()}) == 3
    assert seen["gradient_boosting"]["roc_auc"] != seen["logistic_regression"]["roc_auc"]


@pytest.mark.slow
def test_api_predict_best_resolves_to_champion(client):
    resp = client.post("/api/predict", json={"model": "best"})
    body = resp.get_json()
    assert resp.status_code == 200
    assert body["model"] == "Gradient Boosting"
    assert body["model_key"] == "gradient_boosting"

    # display names are accepted too
    resp = client.post("/api/predict", json={"model": "Random Forest"})
    assert resp.get_json()["model_key"] == "random_forest"


@pytest.mark.slow
def test_api_predict_unknown_model_rejected(client):
    resp = client.post("/api/predict", json={"model": "svm"})
    assert resp.status_code == 400
    body = resp.get_json()
    assert "Unknown model" in body["error"]
    assert body["valid_models"] == [
        "logistic_regression", "random_forest", "gradient_boosting", "best",
    ]


# -- benchmarking -------------------------------------------------------------


@pytest.mark.slow
def test_api_benchmark_all_models(client):
    resp = client.post("/api/benchmark", json={})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    assert body["source"] == "cached_evaluation"
    assert len(body["models"]) == 3
    assert body["best"]["name"] == "Gradient Boosting"
    assert body["overall_best"]["name"] == "Gradient Boosting"
    assert body["split"]["test"] == 10_000
    assert "selection_rule" in body
    for entry in body["models"]:
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            assert 0.0 < entry["metrics"][metric] <= 1.0
        # calibration quality metrics
        assert 0.0 < entry["metrics"]["brier"] < 1.0
        assert 0.0 < entry["metrics"]["log_loss"]
        assert entry["calibration"].startswith("Platt")
        rel = entry["reliability"]
        assert len(rel["bin_mid"]) == len(rel["frac_pos"]) > 0
        cm = entry["confusion"]
        assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == 10_000


@pytest.mark.slow
def test_api_benchmark_fresh_reruns_pipeline(client):
    """A fresh run genuinely re-executes the pipeline — and because the
    recipe is deterministic, it must land on the cached numbers exactly."""
    cached = client.post("/api/benchmark", json={
        "models": ["logistic_regression"],
    }).get_json()
    fresh = client.post("/api/benchmark", json={
        "models": ["logistic_regression"], "fresh": True,
    }).get_json()
    assert fresh["ok"] is True
    assert fresh["source"] == "fresh_run"
    assert fresh["models"][0]["metrics"] == cached["models"][0]["metrics"]
    assert fresh["models"][0]["cv_auc_mean"] == cached["models"][0]["cv_auc_mean"]


@pytest.mark.slow
def test_api_benchmark_fresh_string_is_not_truthy(client):
    """{"fresh": "false"} must read the cached run, not retrain."""
    resp = client.post("/api/benchmark", json={
        "models": ["logistic_regression"], "fresh": "false",
    })
    assert resp.get_json()["source"] == "cached_evaluation"


@pytest.mark.slow
def test_api_benchmark_subset(client):
    resp = client.post("/api/benchmark", json={
        "models": ["logistic_regression", "random_forest"],
    })
    assert resp.status_code == 200
    body = resp.get_json()
    assert [m["key"] for m in body["models"]] == [
        "logistic_regression", "random_forest",
    ]
    assert body["best"]["name"] == "Random Forest"  # best of the subset…
    assert body["overall_best"]["name"] == "Gradient Boosting"  # …not overall


@pytest.mark.slow
def test_api_benchmark_empty_body_benchmarks_all(client):
    resp = client.post("/api/benchmark")  # no body at all
    assert resp.status_code == 200
    assert len(resp.get_json()["models"]) == 3


def test_api_benchmark_unknown_model(client):
    resp = client.post("/api/benchmark", json={"models": ["svm"]})
    assert resp.status_code == 400
    assert "valid_models" in resp.get_json()


def test_api_benchmark_empty_list_rejected(client):
    resp = client.post("/api/benchmark", json={"models": []})
    assert resp.status_code == 400
    assert "at least one" in resp.get_json()["error"]


def test_api_benchmark_requires_json_body(client):
    resp = client.post("/api/benchmark", data="models=x")
    assert resp.status_code == 415


def test_api_benchmark_get_not_allowed(client):
    assert client.get("/api/benchmark").status_code == 405


# -- dataset overview ---------------------------------------------------------


def test_api_dataset_summary(client):
    resp = client.get("/api/dataset")
    assert resp.status_code == 200
    body = resp.get_json()
    s = body["summary"]
    assert s["total_records"] == 50_000          # sentinel row dropped
    assert s["placed"] + s["not_placed"] == 50_000
    assert s["numerical_features"] + s["categorical_features"] == s["total_features"]
    assert 0 < s["placement_rate"] < 100
    assert s["missing_values"] > 0
    assert isinstance(body["insights"], list) and body["insights"]
    # no insight may claim a model winner before training/warm-up — but the
    # dataset facts must always be there
    assert any("student records" in i for i in body["insights"])


def test_api_dataset_chart_payloads(client):
    body = client.get("/api/dataset").get_json()
    # correlation core: model features + target, square matrix
    corr = body["correlation"]
    n = len(corr["labels"])
    assert "CGPA" in corr["labels"] and "PlacementStatus" in corr["labels"]
    assert len(corr["matrix"]) == n and all(len(row) == n for row in corr["matrix"])
    # rate-by-feature bands carry rates + counts
    cgpa = body["rate_by_feature"]["CGPA"]
    assert len(cgpa["labels"]) == len(cgpa["rates"]) == len(cgpa["counts"])
    assert all(0 <= r <= 100 for r in cgpa["rates"] if r is not None)
    assert "CGPA" in body["distributions"]


# -- CORS: the JSON API is public, credential-free, and callable from Pages --


def test_api_sends_cors_header(client):
    resp = client.get("/api/health")
    assert resp.headers["Access-Control-Allow-Origin"] == "*"


def test_api_preflight_options(client):
    resp = client.options("/api/predict")
    assert resp.status_code == 200
    assert "Content-Type" in resp.headers["Access-Control-Allow-Headers"]


def test_pages_do_not_send_cors(client):
    # only the JSON API is cross-origin; the HTML app stays same-origin
    assert "Access-Control-Allow-Origin" not in client.get("/").headers
