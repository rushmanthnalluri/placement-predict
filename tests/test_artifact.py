"""Build-time model artifact: save/load round-trip and validation."""

import os

import model


def test_artifact_roundtrip(tmp_path, default_df):
    # small slice keeps training fast; artifact must carry a usable model
    slice_path = tmp_path / "slice.csv"
    default_df.head(300).to_csv(slice_path, index=False)

    model.save_artifact(str(slice_path))
    artifact = tmp_path / "model_artifact.joblib"
    assert artifact.exists()

    # cold caches -> the load must come from the artifact, not a retrain
    model._bundle_cache.clear()
    model._fitted_cache.clear()
    bundle = model.get_model_bundle(str(slice_path))
    assert bundle["ok"] is True
    assert bundle["best"] in {"Logistic Regression", "Random Forest", "Gradient Boosting"}

    # and inference works off the restored champion
    proba = model.predict(str(slice_path), {c: 7.0 for c in model.FEATURES})
    assert 0.0 <= proba <= 1.0


def test_artifact_rejected_on_content_change(tmp_path, default_df):
    slice_path = tmp_path / "slice.csv"
    default_df.head(300).to_csv(slice_path, index=False)
    model.save_artifact(str(slice_path))

    # same path, different content -> hash mismatch -> no artifact served
    default_df.head(200).to_csv(slice_path, index=False)
    model._bundle_cache.clear()
    model._fitted_cache.clear()
    assert model._load_artifact(str(slice_path)) is None
    bundle = model.get_model_bundle(str(slice_path))  # retrains instead
    assert bundle["ok"] is True
    assert bundle["split"]["train"] == 160  # 80% of the 200-row file
