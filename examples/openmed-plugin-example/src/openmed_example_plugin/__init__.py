"""Deterministic, local-first example components for the OpenMed plugin SDK."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from openmed.core.schemas.span import OpenMedSpan, hmac_text_hash

SYNTHETIC_PERSON_MARKER = "OPENMED_SYNTHETIC_PERSON"
_HASH_KEY = "openmed-example-plugin-synthetic-only"


class ToyRecognizer:
    """Recognize a fictional marker without loading weights or using a network."""

    metadata = {
        "plugin_id": "openmed-example-plugin",
        "component_id": "toy-recognizer",
        "kind": "recognizer",
        "sdk_version": "1.0.0",
        "license": "Apache-2.0",
        "network_egress": False,
        "labels": ("PERSON",),
        "languages": ("en",),
        "name": "Toy synthetic marker recognizer",
        "description": "Recognizes one fictional conformance marker locally.",
        "metadata": {
            "fixture_policy": "synthetic_only",
            "local_first": True,
        },
    }

    def recognize(self, text: str, **kwargs: Any) -> Sequence[OpenMedSpan]:
        """Return canonical spans for every fictional marker in ``text``.

        Args:
            text: Source text containing zero or more synthetic markers.
            **kwargs: Reserved SDK extension options.

        Returns:
            Canonical PERSON spans with source-relative character offsets.
        """

        del kwargs
        spans: list[OpenMedSpan] = []
        cursor = 0
        while True:
            start = text.find(SYNTHETIC_PERSON_MARKER, cursor)
            if start < 0:
                break
            end = start + len(SYNTHETIC_PERSON_MARKER)
            spans.append(
                OpenMedSpan(
                    doc_id="openmed-example-synthetic",
                    start=start,
                    end=end,
                    text_hash=hmac_text_hash(SYNTHETIC_PERSON_MARKER, _HASH_KEY),
                    entity_type="person",
                    canonical_label="PERSON",
                    score=1.0,
                    detector="plugin:openmed-example-plugin:toy-recognizer",
                    evidence={"source": "synthetic_literal_marker"},
                    metadata={"fixture_policy": "synthetic_only"},
                )
            )
            cursor = end
        return tuple(spans)


class ToyExporter:
    """Export span metadata without source surfaces or external I/O."""

    metadata = {
        "plugin_id": "openmed-example-plugin",
        "component_id": "toy-exporter",
        "kind": "exporter",
        "sdk_version": "1.0.0",
        "license": "Apache-2.0",
        "network_egress": False,
        "labels": ("PERSON",),
        "languages": ("*",),
        "name": "Toy privacy-safe exporter",
        "description": "Exports canonical offsets, hashes, and labels locally.",
        "metadata": {
            "fixture_policy": "synthetic_only",
            "local_first": True,
        },
    }

    def export(
        self,
        spans: Sequence[OpenMedSpan],
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        """Serialize canonical span records without adding source text.

        Args:
            spans: Canonical OpenMed spans.
            **kwargs: Reserved SDK extension options.

        Returns:
            A JSON-compatible mapping containing privacy-safe span fields.
        """

        del kwargs
        return {
            "schema": "openmed.example-plugin.spans.v1",
            "spans": [span.to_dict() for span in spans],
        }


def plugin_components() -> tuple[ToyRecognizer, ToyExporter]:
    """Return the components loaded from the package entry point."""

    return ToyRecognizer(), ToyExporter()


__all__ = [
    "SYNTHETIC_PERSON_MARKER",
    "ToyExporter",
    "ToyRecognizer",
    "plugin_components",
]
