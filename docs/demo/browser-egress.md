# Browser network-egress proof

The browser demo is local-first, but a UI assertion alone cannot prove that an
action stayed local. `scripts/web/network_egress_check.py` provides a small,
dependency-free request probe for the action phase of a synthetic redaction
session.

The probe only observes request events supplied by the browser test. It does
not create an HTTP client, resolve hosts, download assets, or inspect request
headers and bodies. Every HTTP(S) request fails unless its URL is explicitly
configured as a model asset. `data:`, `blob:`, and `about:` resources are
treated as browser-internal and do not require a network allowlist. `file:`
requests fail closed because remote file shares can cross a machine boundary.

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
host-only allowlist; the checker rejects host-wide entries and wildcard
patterns. Exact URLs provide the strongest proof. Directory prefixes trust all
query-free `GET` paths below that directory, while rejecting dot segments,
encoded separators, and other path-reinterpretation forms. If a model host
requires a fixed cache query, allowlist that complete asset URL instead. A
request to an API, analytics endpoint, CDN, websocket, or other remote data
service will raise `NetworkEgressViolation`.

Every network request must also expose an explicit `GET` method. A URL-only
trace entry cannot prove its method or absence of an upload body, so it fails
closed even when the URL matches a configured model asset.

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
exception text. Each raw URL is summarized during the request callback and is
not retained in probe memory. This keeps the proof artifact useful for
correlating repeated events without placing synthetic note text or raw
identifiers in logs and test output.

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
Traces, URLs, allowlist entries, and event counts have fixed safety budgets;
inputs outside those budgets fail closed with a source-safe error. The checker
does not evaluate de-identification quality and is not a compliance
certification or clinical decision guarantee.
