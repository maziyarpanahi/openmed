FROM python:3.12-slim

ARG KOPF_VERSION=1.44.6

RUN python -m pip install --no-cache-dir "kopf==${KOPF_VERSION}"

WORKDIR /app
COPY --chown=65532:65532 deploy/operator/openmed_operator.py /app/openmed_operator.py

ENV HOME=/tmp \
    PYTHONUNBUFFERED=1

USER 65532:65532

ENTRYPOINT ["kopf"]
CMD ["run", "--standalone", "--all-namespaces", "--log-format=json", "--liveness=http://0.0.0.0:8080/healthz", "/app/openmed_operator.py"]
