# Comparator benchmark harness

`openmed.eval.comparator` provides a local, reproducible comparison surface for
OpenMed and user-supplied de-identification baselines. The harness itself never
downloads a model, contacts a service, or writes telemetry. Every runner
executes behind OpenMed's outbound-socket guard, even when an adapter forgets
to declare that it requires a network connection. Network-dependent adapters
are reported as unavailable when `requires_network=True`.

## Adapter contract

Build a `ComparatorAdapter` with a stable name and a local runner. The preferred
runner receives the synthetic document text and language and returns span-like
records with `start`, `end`, and `label` fields:

```python
from openmed.eval.comparator import (
    ComparatorAdapter,
    ComparatorBudget,
    ComparatorFixture,
    run_comparator_benchmark,
)


def local_baseline(text: str, language: str):
    """Run an already-installed local detector."""
    del language
    return []  # Replace with local predictions; never call a remote service.


fixture = ComparatorFixture(
    fixture_id="synthetic-case-001",
    text="Synthetic note for SYNTHETIC_NAME; code SYNTHETIC_ID.",
    gold_spans=(),
    metadata={"synthetic": True, "phi_free": True},
)

report = run_comparator_benchmark(
    [fixture],
    [ComparatorAdapter(name="local-baseline", runner=local_baseline)],
    budget=ComparatorBudget(max_latency_ms=400.0, max_memory_bytes=900_000_000),
)
```

Existing OpenMed benchmark runners with the signature
`(fixture, model_name, device)` are also accepted. A runner must return spans;
the compatibility fixture contains text and language but deliberately replaces
gold spans and arbitrary metadata with empty/safety-only values. This keeps a
baseline from accidentally reading the answer key.

Only synthetic, PHI-free fixtures are in scope. JSON and JSONL fixture files can
be loaded with `load_comparator_fixtures`; loading is local-only and bounded.
Each mapping loaded from a file, or supplied directly as a mapping, must include
the JSON booleans `"synthetic": true` and `"phi_free": true` (either at the
top level or in `metadata`). Existing `BenchmarkFixture` values must carry the
same explicit metadata flags. Missing flags fail closed.

## Measurements

Each adapter sees the same fixture sequence and optional `ComparatorBudget`.
The report contains aggregate values only:

- `recall`, `precision`, and `f1` are exact label-and-offset span metrics.
- `character_recall` is the grapheme-aware coverage metric used by the existing
  evaluation primitives.
- `critical_leakage` is the grapheme-weighted residual leakage rate over
  critical identifier labels (`SSN`, `ID_NUM`, `API_KEY`, and related labels by
  default). The report also includes missed critical-span and denominator counts.
- `latency` contains p50, p95, p99, and sample count in milliseconds. The
  optional latency budget is checked against p95.
- `memory` records sampled process-memory baseline, peak, delta, and sample
  count. The optional memory budget is checked against peak bytes.

Pass a fixed `clock`, `memory_sampler`, and `generated_at` value in tests or
reproducibility jobs when byte-identical measurements are required. The report
also contains a stable input/configuration reproducibility hash.

## Privacy-safe output

JSON and Markdown reports contain fixture content digests, counts, labels,
metrics, and provenance digests. They do not contain fixture identifiers,
source text, predicted surfaces, arbitrary adapter metadata, or adapter
exception text. The harness raises source-safe errors if fixture or adapter
output validation fails. Report-facing suite, adapter, version, language, and
fixture identifiers are bounded and restricted to non-markup identifier
characters. Fixture files, case counts, text, spans, and adapter counts also
have defensive limits. Reports are evaluation evidence, not a compliance
certification or a clinical decision guarantee.
