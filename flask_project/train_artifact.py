"""Pretrain the bundled dataset and write the model artifact.

Run at image/deploy build time so production never trains at request time:

    python flask_project/train_artifact.py

Writes flask_project/data/model_artifact.joblib (bundle + champion + salary
regressor), plus one model_artifact_<key>.joblib per candidate for on-demand
non-champion loads. model.get_model_bundle loads them after validating the
recipe version, training environment, and dataset content hash.
"""

import os

import model

DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "placement_predict_50k.csv"
)

if __name__ == "__main__":
    # Remove existing artifacts first: with a valid artifact on disk (e.g. the
    # committed one, which the Docker build context copies into the image),
    # get_model_bundle loads it instead of training, so the fitted cache holds
    # only the champion and save_artifact's completeness guard fails. The
    # artifacts are build outputs — the point of this script is a fresh train.
    stale = [model._artifact_path(DATA)] + [
        model._model_artifact_path(DATA, key) for key in model.MODEL_KEYS
    ]
    for path in stale:
        if os.path.exists(path):
            os.remove(path)
    model.save_artifact(DATA)
    total = os.path.getsize(model._artifact_path(DATA))
    for key in model.MODEL_KEYS:
        total += os.path.getsize(model._model_artifact_path(DATA, key))
    print(f"artifacts written next to {DATA} ({total / 1e6:.1f} MB total: "
          f"bundle + champion + salary regressor, plus one file per candidate)")
