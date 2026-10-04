from __future__ import annotations

import importlib.metadata as md
import json
import os
import platform
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE_URL = os.environ.get("PLACEMENT_PREDICT_URL", "http://127.0.0.1:5000")
ROOT = Path(__file__).resolve().parents[1]
SCREENSHOTS = ROOT / "screenshots"
OUTPUT = Path(os.environ.get("VISUAL_OUTPUT", str(ROOT / "visuals_artifact")))
SCREENSHOTS.mkdir(exist_ok=True)
OUTPUT.mkdir(exist_ok=True)

ROUTES = [
    ("home", "/"),
    ("upload", "/upload"),
    ("features", "/features"),
    ("descriptive", "/descriptive"),
    ("missing", "/missing"),
    ("visualize", "/visualize"),
    ("preprocess", "/preprocess"),
    ("train", "/train"),
    ("evaluate", "/evaluate"),
    ("predict", "/predict"),
]


def wait_for_page(page):
    page.wait_for_load_state("domcontentloaded")
    page.wait_for_timeout(1200)
    page.wait_for_function("document.fonts ? document.fonts.status === 'loaded' : true")
    page.wait_for_timeout(600)


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(
            viewport={"width": 1500, "height": 950},
            device_scale_factor=1,
        )

        page.goto(BASE_URL + "/api/health", wait_until="domcontentloaded")
        health = page.locator("body").inner_text()
        print("health:", health)

        for name, route in ROUTES:
            page.goto(BASE_URL + route, wait_until="domcontentloaded")
            wait_for_page(page)

            if name == "predict":
                values = {
                    "CGPA": "8.2",
                    "AttendancePercent": "88",
                    "Internships": "2",
                    "Projects": "3",
                    "Workshops": "2",
                    "Certifications": "3",
                    "Publications": "1",
                    "AptitudeTestScore": "82",
                    "SoftSkillsRating": "8",
                    "CodingTestScore": "85",
                    "MockInterviewScore": "8",
                    "ExtraCurricular": "2",
                }
                for key, value in values.items():
                    page.locator(f'input[name="{key}"]').fill(value)
                page.locator('button[type="submit"]').click()
                wait_for_page(page)

            out = SCREENSHOTS / f"{name}.png"
            page.screenshot(path=str(out), full_page=False)
            print("captured", out)

        from PIL import Image

        gif_names = [
            "home", "features", "visualize", "missing",
            "train", "evaluate", "predict",
        ]
        frames = []
        for name in gif_names:
            image = Image.open(SCREENSHOTS / f"{name}.png").convert("RGB")
            image.thumbnail((1200, 760), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (1200, 760), "white")
            canvas.paste(
                image,
                ((1200 - image.width) // 2, (760 - image.height) // 2),
            )
            frames.append(canvas)

        frames[0].save(
            SCREENSHOTS / "demo.gif",
            save_all=True,
            append_images=frames[1:],
            duration=[1800, 1400, 1800, 1400, 1800, 1800, 2400],
            loop=0,
            optimize=True,
        )

        import model

        dataset = ROOT / "flask_project" / "data" / "placement_predict_50k.csv"
        bundle = model.get_model_bundle(str(dataset))
        if not bundle.get("ok"):
            raise RuntimeError(bundle.get("error", "model training failed"))

        packages = {}
        for name in [
            "Flask", "Jinja2", "Werkzeug", "numpy", "pandas",
            "joblib", "scikit-learn", "openpyxl", "gunicorn",
        ]:
            try:
                packages[name] = md.version(name)
            except md.PackageNotFoundError:
                packages[name] = None

        metrics = {
            "python": platform.python_version(),
            "packages": packages,
            "best": bundle["best"],
            "best_key": bundle["best_key"],
            "seed": bundle["seed"],
            "cv_folds": bundle["cv_folds"],
            "cv_rows": bundle["cv_rows"],
            "split": bundle["split"],
            "models": bundle["models"],
            "salary_model": bundle["salary_model"],
            "training_env": bundle["training_env"],
            "dataset_sha": model._dataset_sha(str(dataset)),
        }
        (OUTPUT / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(metrics, indent=2))
        browser.close()


if __name__ == "__main__":
    main()
