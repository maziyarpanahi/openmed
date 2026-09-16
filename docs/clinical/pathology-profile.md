# Pathology result profile

`openmed.clinical.pathology_profile` provides a small, deterministic contract
for explicitly reported pathology fields. It extracts specimen, diagnosis,
grade, and reported biomarker name/result pairs from labelled fields or
supported report sections.

The profile is intentionally conservative. It does not infer a diagnosis,
derive a grade, interpret a positive or negative biomarker, calculate stage,
normalize terminology, or make a treatment recommendation. It is not a
diagnostic decision engine and requires pathologist or clinician review.

## Python API

```python
from openmed.clinical import extract_pathology_result

result = extract_pathology_result(
    """SPECIMEN: Synthetic core biopsy
FINAL DIAGNOSIS: Explicitly reported lesion
HISTOLOGIC GRADE: 2 of 3
BIOMARKERS:
ER: positive
HER2 (IHC): 0 (negative)
"""
)
```

The result contains four lists and a fixed `advisory` string:

```python
result["specimen"]      # [{"value": ..., "span": {"start": ..., "end": ...}}]
result["diagnosis"]     # same span-linked field shape
result["grade"]         # explicitly reported grades only
result["biomarkers"]    # name/result plus name_span, result_span, and span
```

Values are kept in source order. A field is empty when it is absent or not
explicitly reported; an empty list never means that a negative result was
inferred. Every span is half-open and indexes the exact input string, so a
caller can display evidence from a protected source buffer without storing
the whole document in the result.

Supported section headings include `SPECIMEN`, `FINAL DIAGNOSIS`,
`HISTOLOGIC GRADE`, `GLEASON SCORE`, `BIOMARKERS`,
`IMMUNOHISTOCHEMISTRY`, and `MOLECULAR RESULTS`. Labelled forms such as
`Diagnosis: ...` and `Grade: ...` are also accepted. Biomarker results are
copied as reported; strings such as `positive`, `negative`, `CPS 10`, and
`not detected` are not clinically interpreted.

## Privacy and offline behavior

The parser imports no model, terminology package, or network client and makes
no mandatory network call. It returns only the selected structured values,
offsets, and advisory text. It does not return the source document, patient
metadata, accession identifiers, or logging payloads. Use synthetic fixtures
for tests and keep the caller-owned source buffer under the caller's privacy
controls.

This profile is assistive extraction only. It does not provide a compliance
certification, diagnostic guarantee, or clinical decision support.
