# LlamaIndex Redaction Postprocessor

OpenMed can redact retrieved LlamaIndex nodes locally before response synthesis.
The integration is optional: importing `openmed` or `openmed.interop` does not
import LlamaIndex, and the adapter loads `llama-index-core` only when you create
a postprocessor, ingestion transform, or framework tool.

```bash
pip install "openmed[llamaindex]"
```

## Redact retrieved nodes before synthesis

Pass the postprocessor to a LlamaIndex query engine. It copies each retrieved
node, runs `openmed.core.pii.deidentify()` over its text and string-valued
metadata, and preserves the node score and metadata structure. The original
retrieved node is not mutated. Metadata is included because LlamaIndex may add
it to the content sent for synthesis. The copied node excludes all metadata from
LLM and embedding content rendering after sanitizing the string values.

```python
from openmed.interop.llamaindex import create_redaction_postprocessor

redact_nodes = create_redaction_postprocessor()

query_engine = index.as_query_engine(
    node_postprocessors=[redact_nodes],
)
response = query_engine.query(
    "What follow-up is documented for this patient?"
)
```

`create_redaction_postprocessor()` uses masking defaults and the deterministic
safety sweep. Tune the de-identification settings with
`LlamaIndexRedactionConfig`.

```python
from openmed.interop.llamaindex import (
    LlamaIndexRedactionConfig,
    create_redaction_postprocessor,
)

redact_nodes = create_redaction_postprocessor(
    config=LlamaIndexRedactionConfig(
        method="mask",
        confidence_threshold=0.6,
        lang="en",
        redact_metadata=True,
        use_safety_sweep=True,
    )
)
```

This postprocessor protects the context sent to response synthesis. It does not
rewrite text already stored in an index or vector store. String values inside
nested metadata mappings and sequences, including string keys, are redacted by
default. Numbers, booleans, and nulls are preserved but excluded from LLM and
embedding content. Unsupported objects and cyclic metadata fail closed. Set
`redact_metadata=False` only when an upstream control guarantees that metadata
contains no personal data.

## Redact during ingestion

Use the optional transform when PHI must be removed before nodes are embedded or
stored. Place it before splitting, caching, embedding, or storage so every later
stage sees only the copied, redacted document.

```python
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter

from openmed.interop.llamaindex import (
    LlamaIndexRedactionConfig,
    create_redaction_transform,
)

redaction_config = LlamaIndexRedactionConfig(
    numeric_metadata_allowlist=("page_number",),
)

pipeline = IngestionPipeline(
    transformations=[
        create_redaction_transform(config=redaction_config),
        SentenceSplitter(chunk_size=512, chunk_overlap=32),
        embed_model,
    ],
    disable_cache=True,
)
redacted_nodes = pipeline.run(documents=documents, store_doc_text=False)
```

The transform also copies nodes before changing their text and string metadata.
For storage safety, it replaces node and relationship ids with deterministic
UUID pseudonyms and replaces numeric metadata with deterministic tokens. Put
only reviewed, non-identifying numeric fields such as page numbers in
`numeric_metadata_allowlist`; other numeric values are pseudonymized. These
deterministic values preserve linkage and therefore remain personal data rather
than establishing anonymization.

OpenMed performs the redaction; LlamaIndex remains an optional orchestration
consumer and is never a core dependency. Keep ingestion caching disabled unless
its persistence, retention, and access controls are approved for the input. Do
not attach a pipeline docstore to raw documents; if a docstore is required, keep
`store_doc_text=False` and write only the returned redacted nodes through a
separately reviewed storage path.
