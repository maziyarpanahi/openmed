"""Verify that the evidence deprecation annotation preserves the 4.x wire shape."""

import json
from pathlib import Path

from openmed.core.schemas.span import OpenMedSpan, schema_fingerprint


def test_free_form_evidence_round_trips_under_deprecated_schema():
    root = Path(__file__).resolve().parents[3]
    schemas = root / "openmed/core/schemas/json"
    schema = json.loads((schemas / "span.schema.json").read_text())
    evidence = schema["properties"]["evidence"]
    assert evidence["deprecated"] is True
    assert evidence["type"] == "object"
    pins = json.loads((schemas / "schema-fingerprints.json").read_text())
    assert pins["span"]["fingerprint"] == schema_fingerprint(schema)
    span = OpenMedSpan(
        doc_id="example",
        start=0,
        end=1,
        text_hash="hmac-sha256:" + "0" * 64,
        entity_type="PERSON",
        canonical_label="PERSON",
        evidence={"legacy": {"rank": 1}},
    )
    assert OpenMedSpan.from_json(span.to_json()).evidence == span.evidence
