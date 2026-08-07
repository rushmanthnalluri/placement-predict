"""Pretrain the bundled dataset and write the model artifact.

Run at image/deploy build time so production never trains at request time:

    python flask_project/train_artifact.py

Writes flask_project/data/model_artifact.joblib, which model.get_model_bundle
loads after validating the recipe version and the dataset's content hash.
"""

import os

import model

DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "placement_predict_50k.csv"
)

if __name__ == "__main__":
    model.save_artifact(DATA)
    size_mb = os.path.getsize(model._artifact_path(DATA)) / 1e6
    print(f"artifact written: {model._artifact_path(DATA)} ({size_mb:.1f} MB)")
