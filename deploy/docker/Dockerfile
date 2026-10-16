# syntax=docker/dockerfile:1.7

FROM --platform=$TARGETPLATFORM python:3.11-slim@sha256:be1575ed968de893bd54f4c56315ff7c4736ce522c1bca08fd521731aafc0d76 AS python-runtime

FROM --platform=$TARGETPLATFORM debian:forky-slim@sha256:91b0aaebf7a1ccacfe7a9cbff6ab2d6be7d9b3b6cf1dfcf44b25f9095c0e0464

ARG BUILDPLATFORM
ARG TARGETARCH
ARG TARGETPLATFORM

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENMED_PROFILE=prod \
    OPENMED_SERVICE_KEEP_ALIVE=10m

WORKDIR /app

# Keep the immutable base's Python runtime while using Debian's fixed SQLite
# release. Runtime libraries stay exact and come from one Debian suite.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        "ca-certificates=20260816" \
        "libbz2-1.0=1.0.8-6+b2" \
        "libffi8=3.8.0-2" \
        "libgdbm6t64=1.26-1+b2" \
        "liblzma5=5.8.3-1" \
        "libncursesw6=6.6+20260608-2" \
        "libreadline8t64=8.3-4" \
        "libsqlite3-0=3.53.4-2" \
        "libssl3t64=3.6.5-1" \
        "libuuid1=2.42.4-1" \
        "netbase=6.6" \
        "openssl=3.6.5-1" \
        "openssl-provider-legacy=3.6.5-1" \
        "readline-common=8.3-4" \
        "tzdata=2026c-1" \
        "zlib1g=1:1.3.dfsg+really1.3.2-3" \
    && rm -rf /var/lib/apt/lists/*

COPY --from=python-runtime /usr/local /usr/local

COPY LICENSE NOTICE README.md pyproject.toml uv.lock models.jsonl /app/
COPY openmed /app/openmed
COPY gates/baseline.json gates/registry_state.json /app/gates/
COPY eval/redteam/corpus/adversarial_phi.jsonl /app/eval/redteam/corpus/

RUN python -m pip install --no-cache-dir --upgrade \
        "pip==26.2.0" \
        "setuptools==83.0.0" \
        "wheel==0.47.0" \
        "jaraco.context==6.1.2" \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.13.0,<3" \
    && pip install --no-cache-dir ".[hf,journey,mcp,service]" \
    && python -m pip check \
    && python -m pip uninstall --yes pip

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import sys,urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3); sys.exit(0)"

CMD ["python", "-m", "openmed.service.logging", "--host", "0.0.0.0", "--port", "8080"]
