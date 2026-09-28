"""Integration proof from the five-source golden Journey to release evidence."""

from __future__ import annotations

import json
from pathlib import Path

from openmed.eval.golden_journey import (
    load_golden_journey_scenario,
    run_golden_journey,
    semantic_diff,
)
from openmed.eval.journey_release import (
    JOURNEY_RELEASE_READY,
    evaluate_journey_release,
)
from tests.fixtures.journey_release import make_release_repository

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
    packet = evaluate_journey_release(
        manifest,
        repo_root=root,
        signing_key=SIGNING_KEY,
        key_id="v3-integration-release",
    )

    assert packet.decision == JOURNEY_RELEASE_READY
    assert packet.verify(SIGNING_KEY)
    assert packet.reproduction["input_count"] == 3
    assert all(item["verified"] for item in packet.frozen_inputs)
