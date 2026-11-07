# LangChain Redaction Node

OpenMed provides a local redaction node for LangChain chains. It redacts
document text and message content before a downstream prompt or model, while
preserving message order and copying metadata unchanged. LangChain remains an
optional dependency:

```bash
pip install "openmed[langchain]"
```

## Add the node before a chain step

`create_redaction_node()` loads `langchain-core` only when the node is created.
The default policy is explicit (`hipaa_safe_harbor`), masking is deterministic,
and the adapter does not call a hosted service.

```python
from openmed.interop.langchain import create_redaction_node

redact = create_redaction_node(policy="strict_no_leak")
safe_chain = redact | downstream_chain

result = safe_chain.invoke("Summarize the synthetic clinical note.")
```

The node accepts strings, LangChain `Document` objects, message objects, lists,
tuples, and mapping payloads. Document and message objects are copied; their
metadata and ordering are retained. Multimodal message content redacts text
blocks and leaves non-text blocks unchanged.

```python
from openmed.interop.langchain import create_redaction_node

redact = create_redaction_node(
    input_key="messages",
    policy="strict_no_leak",
)

payload = redact.invoke(
    {
        "messages": [
            {"role": "user", "content": "Synthetic Name needs a follow-up."},
            {"role": "assistant", "content": "The appointment is synthetic."},
        ],
        "metadata": {"source": "synthetic-fixture"},
    }
)
```

Use `create_redaction_transform()` when a LangChain dependency is not needed,
such as in an offline unit test. It exposes the same `invoke`, `batch`, and
`transform` shape without importing LangChain.

## Deterministic replacements

Masking is the safest default. If a reviewed workflow uses `replace` or
`format_preserve`, pass request-local replacement controls so separate messages
share deterministic surrogate behavior. The state stores only the seed and
aggregate counters; it never stores source text or a re-identification map.

```python
from openmed.interop.langchain import (
    LangChainRedactionConfig,
    LangChainRedactionState,
    create_redaction_node,
)

replacement_state = LangChainRedactionState(seed=17)
redact = create_redaction_node(
    config=LangChainRedactionConfig(
        method="replace",
        policy="strict_no_leak",
    ),
    replacement_state=replacement_state,
)
```

The optional adapter fails with an actionable `openmed[langchain]` installation
message when `langchain-core` is missing. Deidentifier failures are surfaced as
safe adapter errors without embedding input content in exception text.
