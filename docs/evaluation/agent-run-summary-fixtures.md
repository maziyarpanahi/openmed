# Agent run-summary golden fixtures

The bundled `openmed.eval` vectors check compatibility of metadata-only agent
run summaries without a model, network connection, credentials or production
traces. They cover empty, success, abstention, denial, review, failure and mixed
summaries. The mixed vector exercises repeated workflows, shared artifact
digests, aggregation and stable ordering.

```python
from openmed.eval.agent_run_summary_fixtures import load_agent_run_summary_fixtures

vectors = load_agent_run_summary_fixtures()
assert len(vectors) == 7
for vector in vectors:
    assert vector.build_summary().to_json() == vector.expected_json
```

The versioned JSON resource is bundled in wheels and source distributions.
It contains synthetic opaque workflow references, closed outcomes, counts,
durations and digests. It contains no prompts, clinical text, credentials,
private paths or production identifiers.

Loading validates the fixture version, summary/outcome schemas and commitment
version, then rebuilds every vector from typed events. Canonical JSON must match
byte for byte; the Markdown SHA-256 digest and domain-separated commitment must
also match. Expectations are committed bytes, never regenerated at test time.
Event order may change without changing the canonical aggregate summary.

`AgentRunSummaryFixtureError` reports a controlled code only. Unknown fields,
undeclared case IDs, duplicate IDs, malformed digests, unsafe workflow strings,
invalid outcomes and stale expectations fail closed. Input is bounded to 1 MiB,
seven declared cases and 128 events per case; duplicate JSON keys and non-finite
numbers are rejected. An optional caller-owned file may contain a subset of the
declared cases. A full coverage test requires all seven in the bundled resource.

Run the offline compatibility check with:

```bash
.venv/bin/python -m pytest tests/unit/eval/test_agent_run_summary_fixtures.py -q
```

These are serialization and validation vectors. They do not execute tools,
certify deployment readiness or establish clinical performance. See
[run summaries](../agent/run-summaries.md),
[commitments](../agent/run-commitments.md) and
[governed-agent trace fixtures](governed-agent-fixtures.md).
