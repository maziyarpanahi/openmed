from __future__ import annotations

import builtins
import json
import socket

import pytest

from openmed.training import (
    QLORA_CONFIG_SCHEMA_VERSION,
    QLORA_SMOKE_PRESET,
    QloraRecipeConfig,
    RecipeConfigError,
    load_qlora_preset,
    run_qlora_smoke,
)


def test_qlora_smoke_preset_validates_local_4bit_contract():
    config = load_qlora_preset()

    assert config.schema_version == QLORA_CONFIG_SCHEMA_VERSION
    assert config.preset_name == QLORA_SMOKE_PRESET
    assert config.base_model.load_in_4bit is True
    assert config.base_model.quantization_type == "nf4"
    assert config.base_model.local_files_only is True
    assert config.data.local_files_only is True
    assert set(config.heads) == {"token_classification", "generative_pii"}


def test_qlora_smoke_dry_run_passes_and_returns_phi_free_deltas():
    result = run_qlora_smoke()
    payload = result.to_dict()
    encoded = json.dumps(payload, sort_keys=True)

    assert result.gate_passed is True
    assert result.contains_raw_phi is False
    assert result.base_overall_recall < result.adapted_overall_recall
    assert result.base_overall_leakage > result.adapted_overall_leakage
    assert set(result.labels) == {"DATE", "EMAIL", "ID_NUM", "PERSON", "PHONE"}
    assert set(payload["per_entity_deltas"]) == set(result.labels)
    assert payload["network"]["egress_blocked"] is True

    for label, delta in payload["per_entity_deltas"].items():
        assert delta["label"] == label
        assert {"recall_delta", "leakage_delta", "passed"} <= set(delta)

    for raw_value in (
        "Alice",
        "alice@example.test",
        "SYN123",
        "5550100",
        "2026-01-02",
    ):
        assert raw_value not in encoded
    assert {"text", "start", "end", "offset", "offsets"}.isdisjoint(
        set(_walk_keys(payload))
    )


def test_qlora_smoke_seeded_regression_fails_on_leakage_increase():
    result = run_qlora_smoke(seeded_regression=True)
    payload = result.to_dict()

    assert result.gate_passed is False
    assert payload["gate"]["violations"]
    leakage_labels = {
        item["label"]
        for item in payload["gate"]["violations"]
        if item["metric"] == "leakage_delta"
    }
    assert {"EMAIL", "ID_NUM", "PHONE"} <= leakage_labels
    assert payload["per_entity_deltas"]["EMAIL"]["leakage_delta"] > 0
    assert payload["per_entity_deltas"]["PHONE"]["recall_delta"] < 0


def test_qlora_smoke_blocks_socket_egress_probe():
    def probe() -> None:
        socket.create_connection(("example.com", 443), timeout=0.01)

    result = run_qlora_smoke(egress_probe=probe)

    assert result.network_egress_blocked is True
    assert result.egress_probe_blocked is True


def test_qlora_smoke_rejects_remote_corpus_references():
    raw = load_qlora_preset().to_dict()
    raw["data"]["corpus_path"] = "https://example.test/smoke.jsonl"

    with pytest.raises(RecipeConfigError, match="corpus_path must be local-only"):
        QloraRecipeConfig.from_mapping(raw)


def test_qlora_smoke_does_not_import_optional_training_backends(monkeypatch):
    blocked_roots = {"bitsandbytes", "peft", "torch", "transformers"}
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in blocked_roots:
            raise AssertionError(f"unexpected optional backend import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    assert run_qlora_smoke().gate_passed is True


def test_qlora_smoke_full_training_is_out_of_scope_for_child_issue():
    with pytest.raises(NotImplementedError, match="Full QLoRA training"):
        run_qlora_smoke(dry_run=False)


def _walk_keys(value):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield key
            yield from _walk_keys(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)
