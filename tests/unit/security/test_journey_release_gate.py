"""Security boundaries for signed Journey release evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from openmed.eval.journey_release import (
    JOURNEY_RELEASE_NOT_READY,
    JourneyReleaseError,
    evaluate_journey_release,
    main,
)
from tests.fixtures.journey_release import make_release_repository

SIGNING_KEY = "synthetic-release-signing-key-32-bytes-minimum"


def test_input_path_cannot_escape_repository(tmp_path: Path) -> None:
    root, _commit, manifest = make_release_repository(tmp_path)
    manifest["frozen_inputs"][0]["path"] = "../outside.json"

    with pytest.raises(JourneyReleaseError, match="portable relative path"):
        evaluate_journey_release(
            manifest,
            repo_root=root,
            signing_key=SIGNING_KEY,
        )


def test_symlinked_evidence_is_not_followed(tmp_path: Path) -> None:
    root, _commit, manifest = make_release_repository(tmp_path)
    source = root / "inputs" / "synthetic.json"
    original = root / "inputs" / "original.json"
    source.rename(original)
    source.symlink_to(original.name)

    packet = evaluate_journey_release(
        manifest,
        repo_root=root,
        signing_key=SIGNING_KEY,
    )

    assert packet.decision == JOURNEY_RELEASE_NOT_READY
    provenance = next(item for item in packet.gates if item.gate == "provenance")
    assert "frozen_input_mismatch:golden_scenario" in provenance.blocking_codes


def test_packet_is_value_free_and_does_not_disclose_paths(tmp_path: Path) -> None:
    root, _commit, manifest = make_release_repository(tmp_path)
    packet = evaluate_journey_release(
        manifest,
        repo_root=root,
        signing_key=SIGNING_KEY,
    )
    rendered = json.dumps(packet.to_dict(), sort_keys=True)

    for forbidden in (
        str(root),
        "inputs/synthetic.json",
        "journey-release-contract",
        '"source_text":',
        '"raw_value":',
        '"path"',
    ):
        assert forbidden not in rendered


def test_short_or_missing_signing_key_never_writes_packet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _commit, manifest = make_release_repository(tmp_path)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = root / "packet.json"
    monkeypatch.delenv("OPENMED_JOURNEY_RELEASE_KEY", raising=False)

    assert (
        main(
            [
                "--manifest",
                str(manifest_path),
                "--repo-root",
                str(root),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert not output.exists()

    key_file = root / "short.key"
    key_file.write_bytes(b"short")
    os.chmod(key_file, 0o600)
    assert (
        main(
            [
                "--manifest",
                str(manifest_path),
                "--repo-root",
                str(root),
                "--output",
                str(output),
                "--signing-key-file",
                str(key_file),
            ]
        )
        == 2
    )
    assert not output.exists()
