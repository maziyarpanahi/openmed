from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from openmed.interop import langchain as langchain_adapter


@dataclass
class MessageLike:
    content: str | list[dict[str, str]]
    type: str
    additional_kwargs: dict[str, object] = field(default_factory=dict)
    response_metadata: dict[str, object] = field(default_factory=dict)

    def copy(self, update=None):
        values = {
            "content": self.content,
            "type": self.type,
            "additional_kwargs": dict(self.additional_kwargs),
            "response_metadata": dict(self.response_metadata),
        }
        values.update(update or {})
        return MessageLike(**values)


def fake_deidentify(text: str, **kwargs):
    assert kwargs["policy"] == "strict_no_leak"
    assert kwargs["method"] == "mask"
    return SimpleNamespace(
        deidentified_text=(
            text.replace("Synthetic Name", "[PERSON]").replace(
                "synthetic@example.test", "[EMAIL]"
            )
        )
    )


def test_node_redacts_ordered_messages_without_mutating_metadata() -> None:
    original = [
        MessageLike(
            content="Synthetic Name can be reached at synthetic@example.test.",
            type="human",
            additional_kwargs={"trace_id": "synthetic-trace"},
            response_metadata={"source": "synthetic-fixture"},
        ),
        MessageLike(
            content=[
                {"type": "text", "text": "The follow-up is for Synthetic Name."},
                {"type": "image_url", "image_url": "synthetic://image"},
            ],
            type="ai",
        ),
    ]

    transform = langchain_adapter.create_redaction_transform(
        policy="strict_no_leak",
        deidentifier=fake_deidentify,
    )
    redacted = transform.invoke(original)

    assert [message.type for message in redacted] == ["human", "ai"]
    assert redacted[0].content == "[PERSON] can be reached at [EMAIL]."
    assert redacted[1].content[0]["text"] == "The follow-up is for [PERSON]."
    assert redacted[1].content[1] == {
        "type": "image_url",
        "image_url": "synthetic://image",
    }
    assert redacted[0].additional_kwargs == original[0].additional_kwargs
    assert redacted[0].response_metadata == original[0].response_metadata
    assert original[0].content == (
        "Synthetic Name can be reached at synthetic@example.test."
    )
    assert original[1].content[0]["text"] == "The follow-up is for Synthetic Name."


def test_replacement_state_forwards_deterministic_controls_without_raw_state() -> None:
    calls: list[dict[str, object]] = []

    def replace_deidentify(text: str, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(deidentified_text="[PERSON_SURROGATE]")

    state = langchain_adapter.LangChainRedactionState(seed=17)
    transform = langchain_adapter.create_redaction_transform(
        config=langchain_adapter.LangChainRedactionConfig(
            method="replace",
            policy="strict_no_leak",
        ),
        replacement_state=state,
        deidentifier=replace_deidentify,
    )

    assert transform.invoke("Synthetic Name") == "[PERSON_SURROGATE]"
    assert transform.invoke("Synthetic Name") == "[PERSON_SURROGATE]"
    assert all(call["policy"] == "strict_no_leak" for call in calls)
    assert all(call["consistent"] is True and call["seed"] == 17 for call in calls)
    assert state.redacted_items == 2
    assert state.replacement_items == 2
    assert "Synthetic Name" not in repr(state)


def test_deidentifier_failure_does_not_expose_input_text() -> None:
    def broken_deidentify(text: str, **kwargs):
        del kwargs
        raise ValueError(f"failure while processing {text}")

    transform = langchain_adapter.create_redaction_transform(
        deidentifier=broken_deidentify,
    )

    with pytest.raises(langchain_adapter.LangChainRedactionError) as error:
        transform.invoke("Synthetic Name")

    assert "Synthetic Name" not in str(error.value)
    assert "ValueError" in str(error.value)


def test_redaction_node_requires_langchain_extra(monkeypatch) -> None:
    def missing_dependency(name: str):
        raise ImportError(name)

    monkeypatch.setattr(langchain_adapter, "_import_module", missing_dependency)

    with pytest.raises(ImportError, match=r"openmed\[langchain\]"):
        langchain_adapter.create_redaction_node(deidentifier=fake_deidentify)


def test_extra_kwargs_cannot_override_explicit_policy() -> None:
    config = langchain_adapter.LangChainRedactionConfig(
        extra_kwargs={"policy": "unreviewed-policy"}
    )

    with pytest.raises(ValueError, match="cannot override named configuration fields"):
        config.to_deidentify_kwargs()
