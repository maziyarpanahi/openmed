# No-Telemetry Guarantee

OpenMed collects no telemetry and phones no data home. The library ships no
analytics or usage-tracking client, and the core inference path makes no
outbound network calls at runtime. This restates the roadmap non-goal S8.8 #4 —
*"no telemetry-by-default, no license server, no mandatory network calls
post-download"* — and the project's public promise of *"no telemetry, no
outbound calls at runtime"*.

This guarantee complements the OM-005 offline mode
([`OPENMED_OFFLINE`](../configuration.md)) and the OM-004 no-raw-PHI logging
policy ([`no-raw-phi-logging.md`](no-raw-phi-logging.md)).

## What is prohibited

- Analytics / product-telemetry / error-reporting SDKs (Segment, PostHog,
  Sentry, Mixpanel, Amplitude, Datadog, Bugsnag, Rollbar, New Relic, and the
  like) anywhere in the `openmed` package.
- Outbound calls to analytics or phone-home hosts.
- Telemetry that is on by default. An *opt-OUT* switch is itself prohibited,
  because it implies collection is enabled until the user disables it.

## What outbound traffic is allowed

Networking is confined to legitimate, explicitly user-initiated actions. The
only reviewed reasons a module in `openmed` may open a connection are:

- **Model downloads** — Hugging Face model/weight fetches via
  `huggingface_hub` (e.g. `core/hf_hub.py`, `onnx/inference.py`). These honour
  `HF_HOME`/`HF_HUB_CACHE`/`HF_ENDPOINT` and are skipped entirely in offline
  mode.
- **Opt-in evaluation dataset downloads** — `eval/` dataset and suite loaders.
- **Opt-in clinical vocabulary downloads** — `clinical/grounding/vocab.py`,
  offline-gated.
- **The opt-in REST service** — the HTTP client, smart-backend routing, privacy
  gateway, and user-configured webhooks under `service/`.
- **Opt-in distributed training** — rendezvous sockets under `training/`.
- **The offline guard itself** — `core/offline.py`, which blocks sockets when
  `OPENMED_OFFLINE`/`local_only` is set.

None of these run on the default de-identification path, and none contact an
OpenMed-operated or analytics endpoint.

### Distributed tracing (opt-in, off by default)

The REST service integrates OpenTelemetry in `service/tracing.py` for operators
who want request tracing in **their own** infrastructure. It is off by default
(`enabled = False`), enabled only via `OPENMED_SERVICE_TRACING_ENABLED`, and
exports to a user-supplied `OPENMED_SERVICE_OTLP_ENDPOINT`. Because it is opt-in
and points only at the operator's own collector, it does not constitute
telemetry-by-default or a phone-home.

## How the guarantee is enforced

[`tests/unit/test_no_telemetry.py`](https://github.com/maziyarpanahi/openmed/blob/master/tests/unit/test_no_telemetry.py)
enforces the guarantee on every test run:

- A **static scan** of `openmed/**/*.py` fails if a telemetry/analytics SDK is
  imported, a known phone-home host literal appears, or a telemetry opt-OUT
  environment variable is read.
- A **network-surface inventory** pins the exact set of modules allowed to
  import a raw-network library. A new module that opens the network fails the
  test until it is reviewed and added to the allowlist — this is how outbound
  additions are caught.
- A **self-test** plants synthetic violations to prove each detector fires.
- A **runtime check** reuses the OM-005 socket-blocking pattern to assert a
  representative `extract_pii` + `deidentify` path opens no socket.

Run the guard directly:

```bash
.venv/bin/python -m pytest tests/unit/test_no_telemetry.py -q
```

The full suite must also pass before release or pull request review:

```bash
.venv/bin/python -m pytest tests/ -q
```
