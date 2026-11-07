# Synthetic pipeline goldens

These three records are generated examples, not patient records. They cover an
English discharge note with synthetic identifiers and negation, a Spanish/English
code-mixed note, and a hypothetical condition. The clinical vocabulary samples
in `tests/fixtures/clinical/grounding/` are synthetic and local. The fixture
marker and source-rights field must remain present on every case.

The committed `expected_pipeline` fields pin entity offsets, assertion axes,
selected codes, synthetic vocabulary hashes, and FHIR bundle shape. Regenerate
them only after a reviewer has inspected an intentional behavior change:

```bash
.venv/bin/python tests/integration/test_pipeline_e2e.py --regenerate-golden
.venv/bin/python -m pytest tests/integration/test_pipeline_e2e.py -q
```

The regeneration command requires the explicit flag and rejects records without
`synthetic: true`. Review the JSON diff before committing. These fixtures are
engineering regression examples, not clinical evidence or treatment guidance.
