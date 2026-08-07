"""Shared pytest scaffolding.

Imports the Flask app from flask_project (the app expects cwd/module path of
its own directory), pins a deterministic SECRET_KEY for session signing, and
provides a test client whose upload folder lives in a per-test tmp dir.
"""

import io
import os
import sys

import pandas as pd
import pytest

PROJECT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "flask_project"
)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# Deterministic signing key for session cookies in tests. Must be set before
# app is imported, and must NOT be the retired 'dev-only-change-me' key.
os.environ["SECRET_KEY"] = "pytest-secret-key"

import app as app_module  # noqa: E402
import eda  # noqa: E402
import model  # noqa: E402

DEFAULT_DATASET = app_module.DEFAULT_DATASET
DEFAULT_DATASET_NAME = app_module.DEFAULT_DATASET_NAME

# The app keeps at most 2 dataset bundles in memory; the suite touches more
# than that (default + 300-row slice + 3 degenerate fixtures). Raising the
# ceilings for the test session keeps the trained default model from being
# evicted mid-run, which would cost a second ~20s cold train.
eda._CACHE_MAX = 16
model._CACHE_MAX = 16


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: trains/evaluates the model suite (~20s, paid once per session)",
    )


@pytest.fixture()
def client(tmp_path):
    """Flask test client with uploads redirected into a per-test tmp dir."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    app_module.app.config.update(TESTING=True, UPLOAD_FOLDER=str(upload_dir))
    with app_module.app.test_client() as test_client:
        yield test_client


@pytest.fixture()
def default_df():
    """Raw bundled dataset, exactly as shipped (sentinel row included)."""
    return pd.read_csv(DEFAULT_DATASET)


def make_csv(tmp_path, df, name="upload.csv"):
    """Materialise a DataFrame as a CSV under tmp_path and return its path."""
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def post_file(client, path, filename=None):
    """POST a real file to /upload through the multipart form field."""
    with open(path, "rb") as fh:
        payload = io.BytesIO(fh.read())
    data = {"dataset": (payload, filename or os.path.basename(path))}
    return client.post("/upload", data=data, content_type="multipart/form-data")
