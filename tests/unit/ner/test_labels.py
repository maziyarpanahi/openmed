from __future__ import annotations

from pathlib import Path

from openmed.ner.labels import (
    available_domains,
    get_default_labels,
    load_default_label_map,
    reload_default_label_map,
)


def test_load_default_label_map_returns_expected_domains() -> None:
    label_map = load_default_label_map()
    domains = available_domains(label_map)
    assert "biomedical" in domains
    assert "generic" in domains


def test_get_default_labels_known_domain() -> None:
    labels = get_default_labels("biomedical")
    assert "Disease" in labels


def test_get_default_labels_fallback_to_generic(tmp_path) -> None:
    override = tmp_path / "defaults.json"
    override.write_text("{\"generic\": [\"Person\"]}", encoding="utf-8")
    label_map = load_default_label_map(override)
    labels = get_default_labels("unknown", label_map=label_map)
    assert labels == ["Person"]
