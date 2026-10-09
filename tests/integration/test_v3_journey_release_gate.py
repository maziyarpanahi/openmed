"""Integration proof from the five-source golden Journey to release evidence."""

from __future__ import annotations

import json
from pathlib import Path

from openmed.eval.golden_journey import (
    load_golden_journey_scenario,
    run_golden_journey,
    semantic_diff,
    verify_golden_journey_privacy,
    verify_journey_privacy_consistency,
)
from openmed.eval.journey_release import (
    JOURNEY_RELEASE_NOT_READY,
    JOURNEY_RELEASE_READY,
    evaluate_journey_release,
)
from tests.fixtures.journey_release import (
    gate_report,
    make_release_repository,
    privacy_control_processor,
    privacy_control_sources,
)

SIGNING_KEY = "synthetic-release-signing-key-32-bytes-minimum"


def test_five_source_golden_journey_is_frozen_into_signed_release_packet(
    tmp_path: Path,
) -> None:
    scenario_path = Path("tests/fixtures/journey/v3/scenario.json")
    golden_path = Path("tests/fixtures/journey/v3/golden.json")
    scenario = load_golden_journey_scenario(scenario_path)
    observed = run_golden_journey(scenario, work_dir=tmp_path / "journey-run")
    expected = json.loads(golden_path.read_text(encoding="utf-8"))
    assert semantic_diff(expected, observed) == []

    root, _commit, manifest = make_release_repository(
        tmp_path,
        source_files={
            "inputs/five-source-scenario.json": scenario_path,
            "inputs/five-source-golden.json": golden_path,
        },
    )
    consistency = verify_golden_journey_privacy(
        scenario,
        patient_key="synthetic-patient",
        processors={},
        date_shift_secret=b"synthetic-shift-key-at-least-32-bytes",
        proof_secret=b"synthetic-proof-key-at-least-32-bytes",
    )
    privacy = gate_report(manifest, "privacy")
    privacy["metrics"].update(consistency.lane_metrics())
    privacy["state"] = consistency.state
    packet = evaluate_journey_release(
        manifest,
        repo_root=root,
        signing_key=SIGNING_KEY,
        key_id="v3-integration-release",
    )

    assert packet.decision == JOURNEY_RELEASE_NOT_READY
    assert packet.verify(SIGNING_KEY)
    assert packet.reproduction["input_count"] == 3
    assert all(item["verified"] for item in packet.frozen_inputs)
    result = next(item for item in packet.gates if item.gate == "privacy")
    assert result.metrics["date_shift_inconsistency_count"] == 0
    assert result.metrics["consistency_unsupported_format_count"] == 5


def test_five_format_privacy_controls_feed_signed_ready_packet(tmp_path: Path) -> None:
    sources = privacy_control_sources()
    consistency = verify_journey_privacy_consistency(
        sources,
        patient_key="synthetic-patient",
        date_shift_secret=b"synthetic-shift-key-at-least-32-bytes",
        proof_secret=b"synthetic-proof-key-at-least-32-bytes",
        processors={source.format: privacy_control_processor for source in sources},
    )
    root, _commit, manifest = make_release_repository(tmp_path)
    privacy = gate_report(manifest, "privacy")
    privacy["metrics"].update(consistency.lane_metrics())
    privacy["state"] = consistency.state
    packet = evaluate_journey_release(manifest, repo_root=root, signing_key=SIGNING_KEY)
    assert packet.decision == JOURNEY_RELEASE_READY
    assert packet.verify(SIGNING_KEY)


def test_one_wrong_format_shift_blocks_signed_release_packet(tmp_path: Path) -> None:
    from dataclasses import replace
    from datetime import date, timedelta

    def different_shift(source, context):
        transformed = privacy_control_processor(source, context)
        if source.format != "csv":
            return transformed
        text = transformed.text
        for span in reversed(transformed.dates):
            original = date.fromisoformat(
                text[span.replacement_start : span.replacement_end]
            )
            shifted = (original + timedelta(days=1)).isoformat()
            text = (
                text[: span.replacement_start] + shifted + text[span.replacement_end :]
            )
        return replace(transformed, text=text)

    sources = privacy_control_sources()
    consistency = verify_journey_privacy_consistency(
        sources,
        patient_key="synthetic-patient",
        date_shift_secret=b"synthetic-shift-key-at-least-32-bytes",
        proof_secret=b"synthetic-proof-key-at-least-32-bytes",
        processors={source.format: different_shift for source in sources},
    )
    root, _commit, manifest = make_release_repository(tmp_path)
    privacy = gate_report(manifest, "privacy")
    privacy["metrics"].update(consistency.lane_metrics())
    privacy["state"] = "success"  # An optimistic lane state cannot bypass counts.
    packet = evaluate_journey_release(manifest, repo_root=root, signing_key=SIGNING_KEY)
    assert packet.decision == JOURNEY_RELEASE_NOT_READY
    result = next(item for item in packet.gates if item.gate == "privacy")
    assert "nonzero:date_shift_inconsistency_count" in result.blocking_codes
