# Haystack Redaction Component

OpenMed provides a Haystack 2.x component that redacts `Document.content`
locally before documents reach an index, retriever, prompt, or generator.
Haystack remains optional: importing `openmed` or `openmed.interop` does not
import it.

```bash
pip install "openmed[haystack]"
```

## Redact documents before indexing

Place `OpenMedRedactor` before the first component that persists or embeds
documents. The component returns copies, so the input documents are not
modified.

```python
from haystack import Document, Pipeline
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore

from openmed.interop.haystack import OpenMedRedactor

document_store = InMemoryDocumentStore()
indexing = Pipeline()
indexing.add_component("redactor", OpenMedRedactor())
indexing.add_component(
    "writer",
    DocumentWriter(document_store=document_store),
)
indexing.connect("redactor.documents", "writer.documents")

indexing.run(
    {
        "redactor": {
            "documents": [
                Document(
                    content=(
                        "Patient Jane Roe called jane.roe@example.com for follow-up."
                    ),
                    meta={"source": "synthetic-fixture"},
                )
            ]
        }
    }
)
```

Downstream query pipelines now retrieve the redacted content. Keep the redactor
before splitting, embedding, caching, or writing so those stages never receive
the original text. OpenMed performs de-identification on the local runtime; the
component does not call a hosted Haystack service.

## Configure de-identification

Masking and the deterministic safety sweep are enabled by default. Pass
`deidentify_kwargs` to override arguments accepted by
`openmed.core.pii.deidentify()`.

```python
redactor = OpenMedRedactor(
    deidentify_kwargs={
        "method": "mask",
        "confidence_threshold": 0.6,
        "lang": "en",
        "use_safety_sweep": True,
    }
)
```

Only `Document.content` is redacted. Treat document metadata, blobs, existing
embeddings, and external caches as separate privacy surfaces and sanitize them
before they enter the pipeline.
