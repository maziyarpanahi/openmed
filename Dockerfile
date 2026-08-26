FROM python:3.11-slim@sha256:be1575ed968de893bd54f4c56315ff7c4736ce522c1bca08fd521731aafc0d76

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENMED_PROFILE=prod \
    OPENMED_SERVICE_KEEP_ALIVE=10m

WORKDIR /app

# Keep the immutable base's OS security fixes explicit and reproducible.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        "bsdutils=1:2.41.5-0+deb13u1" \
        "libblkid1=2.41.5-0+deb13u1" \
        "liblastlog2-2=2.41.5-0+deb13u1" \
        "libmount1=2.41.5-0+deb13u1" \
        "libsmartcols1=2.41.5-0+deb13u1" \
        "libssl3t64=3.5.7-1~deb13u2" \
        "libuuid1=2.41.5-0+deb13u1" \
        "login=1:4.16.0-2+really2.41.5-0+deb13u1" \
        "mount=2.41.5-0+deb13u1" \
        "openssl=3.5.7-1~deb13u2" \
        "openssl-provider-legacy=3.5.7-1~deb13u2" \
        "util-linux=2.41.5-0+deb13u1" \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN python -m pip install --no-cache-dir --upgrade \
        "pip==26.1.2" \
        "setuptools==83.0.0" \
        "wheel==0.47.0" \
        "jaraco.context==6.1.2" \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir ".[hf,service]"

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import sys,urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3); sys.exit(0)"

CMD ["uvicorn", "openmed.service.app:app", "--host", "0.0.0.0", "--port", "8080"]
