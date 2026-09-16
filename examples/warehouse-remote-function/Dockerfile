FROM python:3.11-slim@sha256:e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE /app/
COPY models.jsonl /app/models.jsonl
COPY openmed /app/openmed

RUN python -m pip install --no-cache-dir --upgrade \
        "pip==26.1.2" \
        "setuptools==83.0.0" \
        "wheel==0.47.0" \
        "jaraco.context==6.1.2" \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir ".[hf,service]"

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3)"

CMD ["uvicorn", "openmed.integrations.remote_function:app", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
