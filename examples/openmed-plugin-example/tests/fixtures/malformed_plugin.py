"""Deliberately malformed, synthetic-only plugin conformance fixture."""

from __future__ import annotations


class MalformedRecognizer:
    """Declare an ambiguous network policy to demonstrate a specific error."""

    metadata = {
        "plugin_id": "openmed-malformed-example",
        "component_id": "toy-recognizer",
        "kind": "recognizer",
        "sdk_version": "1.0.0",
        "license": "Apache-2.0",
        "network_egress": "false",
        "labels": ("PERSON",),
        "languages": ("en",),
    }

    def recognize(self, text: str, **kwargs: object) -> tuple[()]:
        """Return no spans; metadata validation fails before this is probed."""

        del text, kwargs
        return ()


def plugin_components() -> tuple[MalformedRecognizer]:
    """Return the deliberately malformed component."""

    return (MalformedRecognizer(),)
