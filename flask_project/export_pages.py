"""Export a static snapshot of the app into docs/ for GitHub Pages.

GitHub Pages serves static files only, so this renders every route with the
Flask test client (bundled dataset, no session), rewrites asset and page
links to relative paths, neutralizes the upload form, and copies static/.

Run from anywhere:  python flask_project/export_pages.py
"""

import os
import shutil

from app import app

DOCS = os.path.join(app.root_path, "..", "docs")
DOCS = os.path.abspath(DOCS)

ROUTES = {
    "/": "index.html",
    "/upload": "upload.html",
    "/features": "features.html",
    "/descriptive": "descriptive.html",
    "/missing": "missing.html",
    "/visualize": "visualize.html",
    "/preprocess": "preprocess.html",
    "/train": "train.html",
    "/evaluate": "evaluate.html",
    "/predict": "predict.html",
    # GitHub Pages serves a custom 404.html at the site root
    "/this-page-does-not-exist": "404.html",
}

EXPECTED_STATUS = {"/this-page-does-not-exist": 404}


def _rewrite(html):
    """Make app URLs work as static relative links."""
    # page links with anchors first (e.g. /features#group-identity)
    for route, filename in ROUTES.items():
        if route == "/":
            continue
        html = html.replace(f'href="{route}#', f'href="{filename}#')
    for route, filename in ROUTES.items():
        if route == "/":
            continue
        html = html.replace(f'href="{route}"', f'href="{filename}"')
        html = html.replace(f'action="{route}"', f'action="{filename}"')
    html = html.replace('href="/"', 'href="index.html"')
    # static assets + anchor-only links
    html = html.replace('href="/static/', 'href="static/')
    html = html.replace('src="/static/', 'src="static/')
    # neutralize forms: a static host cannot accept POSTs
    html = html.replace("<form ", '<form onsubmit="return false" ')
    html = html.replace('type="submit"', 'type="button"')
    return html


def main():
    if os.path.exists(DOCS):
        shutil.rmtree(DOCS)
    os.makedirs(DOCS)

    client = app.test_client()
    for route, filename in ROUTES.items():
        response = client.get(route)
        expected = EXPECTED_STATUS.get(route, 200)
        if response.status_code != expected:
            raise RuntimeError(f"{route} returned {response.status_code}")
        html = _rewrite(response.get_data(as_text=True))
        with open(os.path.join(DOCS, filename), "w", encoding="utf-8") as fh:
            fh.write(html)
        print(f"  {route:14s} -> docs/{filename}")

    shutil.copytree(
        os.path.join(app.root_path, "static"),
        os.path.join(DOCS, "static"),
    )
    # GitHub Pages: serve files as-is, no Jekyll processing
    open(os.path.join(DOCS, ".nojekyll"), "w").close()
    print("  static assets copied; .nojekyll written")


if __name__ == "__main__":
    main()
