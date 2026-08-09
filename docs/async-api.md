# Async API

OpenMed's model and PII pipelines are synchronous, while many web applications
already run an event loop. The async helpers keep that loop responsive by
delegating each synchronous call to a worker thread:

```python
from openmed import aanalyze_text, aextract_pii, adeidentify

analysis = await aanalyze_text(
    "Synthetic note: Casey Example reports a cough.",
    model_name="fixture-ner-model",
)
entities = await aextract_pii(
    "Synthetic contact: casey@example.test",
    model_name="fixture-pii-model",
)
redacted = await adeidentify(
    "Synthetic contact: casey@example.test",
    model_name="fixture-pii-model",
    method="mask",
)
```

The wrappers accept the same arguments and return the same result types as
`analyze_text`, `extract_pii`, and `deidentify`. They do not create an event
loop when `openmed` is imported, and they do not provide native async model
inference.

## FastAPI

An async route can await a wrapper directly. The following example uses
synthetic text and returns the typed de-identification payload:

```python
from fastapi import FastAPI

from openmed import adeidentify

app = FastAPI()


@app.post("/redact")
async def redact(text: str) -> dict:
    result = await adeidentify(text, method="mask")
    return result.to_dict()
```

For multiple independent inputs, `abatch` preserves input order and can cap
concurrency:

```python
from openmed import abatch, aextract_pii

results = await abatch(aextract_pii, ["Synthetic note one", "Synthetic note two"])
```

These helpers only offload work; they do not make clinical decisions. Use the
same local model and privacy configuration as the synchronous APIs.
