# LangChain Redaction Wrapper

OpenMed can run local de-identification as a chain stage so retrieved context is redacted before it reaches a model call. The integration is optional: importing `openmed` or `openmed.interop` does not import LangChain, and the runnable factory only needs the `langchain` extra when you create a runnable.

```bash
pip install "openmed[langchain]"
```

## Redact retrieved context before prompting

The wrapper accepts strings, LangChain `Document` objects, lists or tuples of those values, and mapping payloads. This makes it useful in RAG pipelines where a retriever returns documents and a downstream prompt consumes their `page_content`.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from openmed.interop.langchain import create_redaction_runnable

redact_context = create_redaction_runnable()

prompt = ChatPromptTemplate.from_template(
    "Use only the redacted clinical context below.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)

chain = (
    {
        "context": retriever | redact_context,
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
)

response = chain.invoke("What follow-up is documented for this patient?")
```

`create_redaction_runnable()` uses `openmed.core.pii.deidentify` with masking defaults. Tune those settings by passing `LangChainRedactionConfig`.

```python
from openmed.interop.langchain import (
    LangChainRedactionConfig,
    create_redaction_runnable,
)

redact_context = create_redaction_runnable(
    config=LangChainRedactionConfig(
        method="mask",
        confidence_threshold=0.6,
        lang="en",
        use_safety_sweep=True,
    )
)
```

## Redact a specific payload field

For chains that pass dictionaries between stages, set `input_key` to redact only the field that may contain PHI. Other fields, such as the user question, stay unchanged.

```python
from openmed.interop.langchain import create_redaction_runnable

redact_payload = create_redaction_runnable(input_key="context")

payload = redact_payload.invoke(
    {
        "context": "Patient Jane Roe called from jane.roe@example.com.",
        "question": "Summarize the follow-up.",
    }
)

assert payload["context"] == "Patient [PERSON] called from [EMAIL]."
assert payload["question"] == "Summarize the follow-up."
```

Use the dependency-light `create_redaction_transform()` in tests or non-LangChain orchestration. It exposes `invoke`, `batch`, and `transform` without importing optional chain packages.
