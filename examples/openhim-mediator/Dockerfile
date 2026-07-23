FROM python:3.11-slim@sha256:e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENMED_PROFILE=prod \
    OPENMED_SERVICE_KEEP_ALIVE=10m \
    OPENMED_OPENHIM_MEDIATOR_ENABLED=true \
    OPENMED_OPENHIM_CONFIG_PATH=/etc/openmed/mediator-config.json

WORKDIR /app

COPY pyproject.toml README.md LICENSE /app/
COPY openmed /app/openmed
COPY examples/openhim-mediator/mediator-config.json /etc/openmed/mediator-config.json

RUN python -m pip install --no-cache-dir --upgrade \
        "pip==26.1.2" \
        "setuptools==83.0.0" \
        "wheel==0.47.0" \
        "jaraco.context==6.1.2" \
    && pip install --no-cache-dir ".[service]"

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/openhim/heartbeat', timeout=2)"

CMD ["uvicorn", "openmed.service.app:app", "--host", "0.0.0.0", "--port", "8080"]
