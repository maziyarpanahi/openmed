# spaCy Pipeline Component

OpenMed can attach local PII detections to a spaCy `Doc` so existing clinical NLP
pipelines can consume de-identification spans without leaving the spaCy runtime.
The integration is optional: importing `openmed` or `openmed.interop` does not
import spaCy, and the pipeline factory only needs the `spacy` extra when you add
the component.

```bash
pip install "openmed[spacy]"
```

## Add OpenMed PII spans to a pipeline

Import the component module once to register the `openmed_deid` factory, then add
it to any spaCy pipeline. Detected spans are projected from OpenMed character
offsets with `Doc.char_span(alignment_mode="expand")` and stored in
`doc.spans["openmed_pii"]`.

```python
import spacy

import openmed.interop.spacy_component  # noqa: F401

nlp = spacy.blank("en")
nlp.add_pipe("openmed_deid")

doc = nlp("Patient Jane Roe called 555-0100.")

for span in doc.spans["openmed_pii"]:
    print(span.text, span.label_, span.start_char, span.end_char)
```

The component also stores dependency-light raw spans on `Doc._.openmed_pii`.
Each item has `label`, `start`, `end`, and `score` fields for downstream
components that need OpenMed offsets instead of spaCy token spans.

## Configure detection

Pass component config through spaCy's `add_pipe` call. `model_name`,
`confidence_threshold`, `lang`, and `policy` are retained on the OpenMed config;
supported extraction arguments are forwarded to `openmed.extract_pii`.

```python
nlp.add_pipe(
    "openmed_deid",
    config={
        "model_name": "OpenMed/openmed-pii-redaction-phi",
        "confidence_threshold": 0.7,
        "lang": "en",
        "policy": "hipaa_safe_harbor",
        "target": "clinical_pii",
    },
)

doc = nlp(note)
assert "clinical_pii" in doc.spans
```

## Merge detections into `doc.ents`

Set `merge_ents=True` when later spaCy components expect PII detections in
`doc.ents`. OpenMed resolves overlapping entity spans before assignment so spaCy
does not raise on conflicting entity offsets.

```python
nlp.add_pipe("openmed_deid", config={"merge_ents": True})

doc = nlp("Jane Roe called Jane.")
assert all(ent.label_ for ent in doc.ents)
```
