"""Tests for the optional Health Universe PHI replacement agent."""

from __future__ import annotations

import asyncio
import json

import pytest

from openmed.integrations.health_universe_phi import (
    OpenMedReplacementEngine,
    PhiReplacementSettings,
    ReplacementResult,
    invalid_ip_surrogate,
    seed_for_scope,
)


class FakeEngine:
    """Avoid model loading while exercising SDK document flow."""

    def replace(self, markdown: str, *, seed_scope: str) -> ReplacementResult:
        assert seed_scope
        return ReplacementResult(
            markdown=markdown.replace("Synthetic Person", "Morgan Example"),
            entity_count=1,
            action_counts={"replace": 1},
            entity_label_counts={"person": 1},
            replacement_collision_count=0,
            replacement_collision_label_counts={},
            ip_surrogates_checked=0,
            invalid_ip_surrogates=0,
        )


def test_settings_reject_unsafe_bounds() -> None:
    with pytest.raises(ValueError, match="confidence_threshold"):
        PhiReplacementSettings(confidence_threshold=1.1)
    with pytest.raises(ValueError, match="maximum_documents"):
        PhiReplacementSettings(maximum_documents=0)


def test_seed_is_stable_and_scope_specific() -> None:
    first = seed_for_scope(42, "thread-a")
    assert first == seed_for_scope(42, "thread-a")
    assert first != seed_for_scope(42, "thread-b")


def test_ip_surrogate_validation_is_value_free() -> None:
    assert invalid_ip_surrogate("ipv4", "192.0.2.1") is False
    assert invalid_ip_surrogate("ipv4", "not-an-ip") is True
    assert invalid_ip_surrogate("email", "not-an-ip") is None


def test_engine_forces_offline_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENMED_OFFLINE", "0")
    OpenMedReplacementEngine(PhiReplacementSettings())
    assert __import__("os").environ["OPENMED_OFFLINE"] == "1"


def test_local_sdk_flow_writes_opaque_draft_and_report(tmp_path) -> None:
    sdk_local = pytest.importorskip("health_universe_a2a.local")
    agent_module = pytest.importorskip("openmed.integrations.health_universe_agent")

    data_dir = tmp_path / "data"
    source_dir = data_dir / "source"
    source_dir.mkdir(parents=True)
    source_dir.joinpath("Synthetic Person record.md").write_text(
        "Patient: Synthetic Person\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "outputs"
    context = sdk_local.create_local_context(str(data_dir), str(output_dir))
    agent = agent_module.PhiReplacementAgent(
        settings=PhiReplacementSettings(),
        engine=FakeEngine(),
    )

    message = asyncio.run(agent.process_message("replace", context))

    assert "Created 1" in message
    draft = output_dir / "Deidentified_Document_0001.md"
    draft_text = draft.read_text(encoding="utf-8")
    assert "HUMAN REVIEW REQUIRED" in draft_text
    assert "Morgan Example" in draft_text
    assert "Synthetic Person record" not in draft.name
    report = json.loads(
        output_dir.joinpath("Deidentification_Safety_Report.json").read_text(
            encoding="utf-8"
        )
    )
    serialized = json.dumps(report)
    assert report["draft_outputs_created"] == 1
    assert report["contains_raw_phi"] is False
    assert "Synthetic Person" not in serialized
