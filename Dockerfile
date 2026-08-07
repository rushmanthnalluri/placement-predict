# Placement Predict System — one image runs the full app anywhere.
# Works on Render, Hugging Face Spaces (Docker SDK), Railway, Fly.io, or locally:
#   docker build -t placement-predict . && docker run -p 7860:7860 placement-predict
FROM python:3.12-slim

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY flask_project/ flask_project/

# Run as an unprivileged user.
RUN useradd --create-home appuser && chown -R appuser:appuser /srv
USER appuser

# Hosts inject $PORT (Render) or expect 7860 (HF Spaces); default to 7860.
EXPOSE 7860
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 4 --timeout 120 --chdir flask_project app:app"]
