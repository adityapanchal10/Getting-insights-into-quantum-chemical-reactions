FROM python:3.12-slim

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/serve.py app/serve.py
COPY model/champion.joblib model/champion.joblib

ENV MODEL_PATH=/srv/model/champion.joblib

EXPOSE 8000

# A plain process check isn't enough here — /health actually confirms the
# model artifact loaded, which is what the k8s readiness probe relies on.
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.serve:app", "--host", "0.0.0.0", "--port", "8000"]
