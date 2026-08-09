# Browser network-egress proof

The browser demo is local-first, but a UI assertion alone cannot prove that an
action stayed local. `scripts/web/network_egress_check.py` provides a small,
dependency-free request probe for the action phase of a synthetic redaction
session.

The probe only observes request events supplied by the browser test. It does
not create an HTTP client, resolve hosts, download assets, or inspect request
headers and bodies. Every HTTP(S) request fails unless its URL is explicitly
configured as a model asset. `data:`, `blob:`, `about:`, and `file:` resources
are treated as browser-internal and do not require a network allowlist.

## Capture a browser action

The probe accepts Playwright-style request objects without importing
Playwright, so the browser dependency remains optional:

```python
from scripts.web.network_egress_check import capture_browser_requests

MODEL_ASSETS = "http://127.0.0.1:8000/models/synthetic-redactor/"

with capture_browser_requests(
    page,
    allowed_model_assets=(MODEL_ASSETS,),
) as egress:
    page.get_by_role("button", name="Redact synthetic note").click()

egress.assert_clean()
```

Attach the probe after page and stylesheet setup when the test is proving an
application action. If the action loads a local model, list the narrowest
directory prefix or exact asset URL it is expected to request. Do not use a
host-only allowlist when a model directory can be named instead. A request to
an API, analytics endpoint, CDN, websocket, or other remote data service will
raise `NetworkEgressViolation`.

The same flow can be used with a synthetic event source in an offline unit
test:

```python
from scripts.web.network_egress_check import assert_no_unexpected_requests

assert_no_unexpected_requests(
    [
        {
            "method": "GET",
            "resource_type": "fetch",
            "url": "http://127.0.0.1:8000/models/synthetic-redactor/model.onnx",
        }
    ],
    allowed_model_assets=(MODEL_ASSETS,),
)
```

Only method, resource type, a classification, and SHA-256 digests of the URL
and origin are retained in `EgressReport`. Paths, query strings, fragments,
headers, bodies, and browser request objects are not included in the report or
exception text. This keeps the proof artifact useful for correlating repeated
events without placing synthetic note text or raw identifiers in logs and
test output.

## Inspect a saved request trace

The CLI reads a local JSON list, or an object with a `requests` list, and emits
the same safe report. It performs no network operation:

```bash
.venv/bin/python scripts/web/network_egress_check.py \
  /tmp/synthetic-browser-trace.json \
  --allow-model-asset 'http://127.0.0.1:8000/models/synthetic-redactor/' \
  --report /tmp/browser-egress-report.json
```

Use one `--allow-model-asset` option per expected model URL or prefix. Exit
status `0` means no unexpected request was observed; `1` means the trace
contains unexpected egress; `2` means the local trace or policy was invalid.
The checker does not evaluate de-identification quality and is not a
compliance certification or clinical decision guarantee.
