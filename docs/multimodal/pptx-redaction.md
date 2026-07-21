# PPTX Slide and Speaker-Notes Redaction

OpenMed can extract text from PowerPoint slide shapes, table cells, and speaker
notes through the shared multimodal document contract. The extractor returns a
normalized text stream plus run-level provenance, so detections can be mapped
back to the presentation without placing the original PHI in audit metadata.

Install the multimodal extra first:

```bash
uv pip install ".[multimodal]"
```

## Extract text and offsets

```python
from openmed.multimodal import extract_pptx

document = extract_pptx("clinical-case.pptx")

print(document.text)
for source in document.spans:
    print(
        source.page,
        source.metadata["part"],
        source.metadata["slide_start"],
        source.metadata["slide_end"],
    )
```

`SourceSpan.page` is the zero-based slide index. Each span records whether the
text came from a slide shape, table cell, or speaker notes, together with its
paragraph/run location and character offsets within that slide. The document
metadata also includes a `slide_offsets` entry for every slide, including empty
slides.

## Project detections and write a redacted copy

Use offsets from `document.text` when supplying detected spans:

```python
from openmed.multimodal import (
    map_text_spans_to_pptx_runs,
    write_redacted_pptx,
)

name = "Jane Doe"
start = document.text.index(name)
entities = [
    {
        "start": start,
        "end": start + len(name),
        "label": "PERSON",
    }
]

provenance = map_text_spans_to_pptx_runs(document, entities)
write_redacted_pptx(
    "clinical-case.pptx",
    "clinical-case.redacted.pptx",
    entities,
)
```

The default replacement is label-based, such as `[PERSON]`. A span may cross
multiple styled runs; OpenMed inserts the replacement into the first covered
run and removes the covered text from the remaining runs. The returned
provenance contains offsets, labels, confidence values, source locations, and a
SHA-256 digest of the removed text, but not the plaintext PHI.

## Use the multimodal dispatcher

The `.pptx` handler is registered when `openmed.multimodal` is imported:

```python
from openmed.multimodal import redact_document

result = redact_document(
    "clinical-case.pptx",
    models={"detector": detector},
    policy={"output_path": "clinical-case.redacted.pptx"},
    lang="en",
)
```

Without an output path, the dispatcher extracts text and returns projected
redaction provenance without writing a presentation.

## Scope and limitations

- Text in slide shapes, grouped text shapes, table cells, and speaker notes is
  supported.
- Embedded images are not inspected; use the image/OCR redaction pipeline for
  image-based PHI.
- Write-back preserves the deck structure and unaffected run formatting, but
  exact layout fidelity after replacement is not guaranteed.
- SmartArt, charts, OLE objects, comments, masters, and layout-template text are
  outside this adapter's scope.
