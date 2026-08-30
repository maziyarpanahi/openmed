# LlamaIndex Redaction Transform

OpenMed provides an optional, local-first LlamaIndex ingestion transform. It
redacts node text and sensitive metadata before splitting, embedding, or
storage while keeping the LlamaIndex list-of-nodes contract.

```bash
pip install "openmed[llamaindex]"
```

## Redact before ingestion

Create the transform before the splitter so later chunks are derived from
protected text. The transform copies nodes, keeps their existing chunk
metadata, and replaces node and relationship identifiers with deterministic
UUID pseudonyms for storage-safe linkage.

```python
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter

from openmed.interop.llamaindex import (
    LlamaIndexRedactionConfig,
    create_redaction_transform,
)

redaction_transform = create_redaction_transform(
    config=LlamaIndexRedactionConfig(
        numeric_metadata_allowlist=("page_number",),
    )
)

pipeline = IngestionPipeline(
    transformations=[
        redaction_transform,
        SentenceSplitter(chunk_size=512, chunk_overlap=32),
        embed_model,
    ],
    disable_cache=True,
)
redacted_nodes = pipeline.run(documents=documents, store_doc_text=False)
```

`numeric_metadata_allowlist` is for reviewed, non-identifying values such as
page numbers. Other numeric metadata is replaced with deterministic tokens.
The original nodes are not mutated, and repeated source identifiers produce
the same pseudonymous identifier.

## Counts-only audit metadata

The transform still returns only nodes. After a call, read the safe summary from
the transform instance:

```python
audit = redaction_transform.audit_metadata
```

The summary contains node, changed-value, entity-category, and identifier
pseudonymization counts. It contains no source identifiers, offsets, input
text, replacement values, or arbitrary deidentifier metadata, and it is not
added to node metadata or sent to the embedding model. Category counts are
copied into immutable validated state. Optional detector entity metadata is
bounded to 10,000 observations per redacted value; malformed metadata is
ignored and cannot make redaction fail.

Use `LlamaIndexRedactionConfig` to tune the local OpenMed de-identification
method, language, confidence threshold, policy, and safety sweep. LlamaIndex
is an optional dependency; importing `openmed` or `openmed.interop` does not
import it. For retrieval-time node protection, see the
[LlamaIndex redaction postprocessor guide](../integrations-llamaindex.md).
