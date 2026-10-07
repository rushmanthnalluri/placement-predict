from __future__ import annotations

import importlib.metadata as md
import json
import os
import platform
from pathlib import Path
import sys

from playwright.sync_api import sync_playwright

BASE_URL = os.environ.get("PLACEMENT_PREDICT_URL", "http://127.0.0.1:5000")
ROOT = Path(__file__).resolve().parents[1]
FLASK_PROJECT = ROOT / "flask_project"
if str(FLASK_PROJECT) not in sys.path:
    sys.path.insert(0, str(FLASK_PROJECT))
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
    # Charts must be painted before the screenshot. Chart.js is optional now:
    # the app has a native canvas fallback when the CDN is unreachable.
    page.wait_for_timeout(600)
    page.wait_for_function(
        """() => {
            const canvases = [...document.querySelectorAll("canvas[data-chart]")];
            if (!canvases.length) return true;
            return canvases.every((canvas) => {
                const w = canvas.width, h = canvas.height;
                if (!w || !h) return false;
                const ctx = canvas.getContext("2d");
                if (!ctx) return false;
                const stepX = Math.max(1, Math.floor(w / 80));
                const stepY = Math.max(1, Math.floor(h / 40));
                const pixels = ctx.getImageData(0, 0, w, h).data;
                let painted = 0;
                for (let y = 0; y < h; y += stepY) {
                    for (let x = 0; x < w; x += stepX) {
                        const i = (y * w + x) * 4;
                        if (pixels[i + 3] > 20) painted++;
                    }
                }
                return painted >= 4;
            });
        }"""
    )


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
                    "SoftSkillsRating": "4.5",
                    "CodingTestScore": "85",
                    "MockInterviewScore": "80",
                    "ExtraCurricular": "1",
                }
                for key, value in values.items():
                    page.locator(f'input[name="{key}"]').fill(value)
                page.locator('button[type="submit"]').click()
                page.locator(".result-kicker").filter(has_text="Prediction").wait_for()
                if page.locator(".field-hint-error").count():
                    raise RuntimeError("visual predict fixture produced validation errors")
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
