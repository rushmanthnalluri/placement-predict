"""Upload intake: validation, schema enforcement, clearing, forged sessions."""

import io

import app as app_module
from conftest import DEFAULT_DATASET_NAME, make_csv, post_file


def test_upload_no_file_shows_error(client):
    resp = client.post("/upload", data={}, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert "Choose a CSV or Excel file before uploading." in resp.get_data(
        as_text=True
    )


def test_upload_wrong_extension_rejected(client):
    data = {"dataset": (io.BytesIO(b"not a dataset"), "notes.txt")}
    resp = client.post("/upload", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert "Only .csv or .xlsx files are accepted." in resp.get_data(as_text=True)


def test_upload_bad_schema_rejected(client, tmp_path):
    bad = io.BytesIO(b"a,b,c\n1,2,3\n")
    resp = client.post(
        "/upload",
        data={"dataset": (bad, "bad.csv")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    assert "missing required columns" in resp.get_data(as_text=True)
    # a rejected file is deleted, never becomes the active dataset
    assert list((tmp_path / "uploads").iterdir()) == []
    home = client.get("/").get_data(as_text=True)
    assert "50,000" in home
    assert DEFAULT_DATASET_NAME in home


def test_upload_valid_slice_activates(client, tmp_path, default_df):
    df = default_df[default_df["StudentID"] != 0].head(300)
    resp = post_file(client, make_csv(tmp_path, df, "slice.csv"))
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "slice.csv" in body
    assert '<span class="metric-value">300</span>' in body
    # every stage now reads the upload: /descriptive reports 300 rows
    desc = client.get("/descriptive").get_data(as_text=True)
    assert "300-row total" in desc


def test_overview_follows_the_uploaded_dataset(client, tmp_path, default_df):
    """The overview section and its API recompute from the active dataset —
    nothing is pinned to the bundled cohort."""
    df = default_df[default_df["StudentID"] != 0].head(300)
    post_file(client, make_csv(tmp_path, df, "slice.csv"))
    body = client.get("/api/dataset").get_json()
    assert body["dataset"] == "slice.csv"
    assert body["summary"]["total_records"] == 300
    home = client.get("/").get_data(as_text=True)
    assert "300 student records" in home


def test_clear_restores_default(client, tmp_path, default_df):
    df = default_df[default_df["StudentID"] != 0].head(300)
    post_file(client, make_csv(tmp_path, df, "slice.csv"))
    resp = client.post("/upload/clear")
    assert resp.status_code == 302
    assert list((tmp_path / "uploads").iterdir()) == []  # file removed from disk
    home = client.get("/").get_data(as_text=True)
    assert "50,000" in home
    assert DEFAULT_DATASET_NAME in home


def _sign_with(secret, payload):
    """Sign a session payload with an arbitrary key (forgery simulation)."""
    app = app_module.app
    original = app.secret_key
    app.secret_key = secret
    try:
        return app.session_interface.get_signing_serializer(app).dumps(payload)
    finally:
        app.secret_key = original


def test_forged_session_cookie_rejected(client, tmp_path):
    marker = "FORGED-SENTINEL-MARKER"
    sentinel = tmp_path / "sentinel.csv"
    sentinel.write_text(f"StudentID,Note\n1,{marker}\n", encoding="utf-8")
    forged = _sign_with(
        "dev-only-change-me",
        {"dataset_file": str(sentinel), "dataset_name": marker},
    )
    client.set_cookie("session", forged)
    resp = client.get("/upload")
    body = resp.get_data(as_text=True)
    assert resp.status_code == 200
    # the forged signature fails verification: session is empty, so the
    # sentinel outside the app tree is never read or rendered
    assert marker not in body
    assert DEFAULT_DATASET_NAME in body
