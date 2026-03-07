FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENMED_PROFILE=prod

WORKDIR /app

COPY . /app

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir ".[hf,service]"

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import sys,urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3); sys.exit(0)"

CMD ["uvicorn", "openmed.service.app:app", "--host", "0.0.0.0", "--port", "8080"]
