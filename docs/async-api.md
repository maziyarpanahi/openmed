# Async Python API

OpenMed provides first-class coroutine wrappers for its blocking Python
helpers. They run the existing synchronous implementation in asyncio's worker
thread pool, so an application event loop remains responsive while model
inference or batch processing is in progress.

```python
import openmed

result = await openmed.adeidentify(
    "Synthetic patient Casey Example called 555-0100.",
    method="mask",
)
print(result.deidentified_text)
```

The lazy top-level exports are:

| Async helper | Synchronous implementation |
| --- | --- |
| `openmed.aextract_pii(...)` | `openmed.extract_pii(...)` |
| `openmed.adeidentify(...)` | `openmed.deidentify(...)` |
| `openmed.aanalyze_text(...)` | `openmed.analyze_text(...)` |
| `openmed.abatch(...)` | `openmed.process_batch(...)` |

Each wrapper accepts the same arguments and returns the same result type as its
synchronous implementation. Exceptions also propagate unchanged. Importing
`openmed` alone does not import `asyncio` or create an event loop; the async
module is loaded only when one of these helpers is first accessed.

## FastAPI example

The wrappers are suitable for an async route when inference remains local and
the caller wants to avoid blocking the server event loop:

```python
from fastapi import FastAPI
from pydantic import BaseModel

import openmed

app = FastAPI()


class RedactionRequest(BaseModel):
    text: str


@app.post("/redact")
async def redact(request: RedactionRequest) -> dict[str, str]:
    result = await openmed.adeidentify(
        request.text,
        method="mask",
        use_safety_sweep=True,
    )
    return {"text": result.deidentified_text}
```

Do not log request text, model outputs, or exceptions containing source values.
Reuse a warmed loader when traffic is sustained, and apply application-level
concurrency limits so requests do not create unbounded worker pressure.

## Batch concurrency

For multiple independent inputs, `abatch` preserves input order and can cap
concurrency:

```python
from openmed import abatch, aextract_pii

results = await abatch(aextract_pii, ["Synthetic note one", "Synthetic note two"])
```

Pass `max_concurrency` to place a hard bound on simultaneously scheduled
operations. This is recommended when model sessions or input collections are
large:

```python
results = await abatch(
    aextract_pii,
    ["Synthetic note one", "Synthetic note two"],
    max_concurrency=2,
)
```

These helpers only offload work; they do not make clinical decisions. Use the
same local model and privacy configuration as the synchronous APIs.
## Cancellation and shutdown

Cancelling the awaiting task stops waiting for the result, but Python cannot
forcibly stop a synchronous function that is already running in a worker
thread. Use OpenMed request budgets for bounded work and allow the process to
finish in-flight worker calls during graceful shutdown.
