"""Privacy boundaries for the signed v4.0 OpenMedKit release packet."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from openmed.eval import kit_release as kr
from tests.fixtures.kit_release import (
    TEST_ONLY_KEY_ID,
    TEST_ONLY_SIGNING_KEY,
    entry,
    head_resolver,
    write_evidence,
)

SENTINELS = (
    "Sentinel Patient Jane Roe",
    "MRN-SENTINEL-004217",
    "SERIAL-F2LXSENTINEL9",
    "iPhone15,3",
    "00008110-001A2B3C4D5E801E",
    "sentinel.patient@example.org",
    "sk-live-SENTINELTOKEN0042",
    "SentinelDirectory4477",
)
_LOCAL_PATH = re.compile(r"/(?:Users|home|private|tmp|var)/")


def _planted_manifest(root: Path) -> dict:
    text = "\n".join(SENTINELS).encode("utf-8")
    manifest = write_evidence(
        root,
        prefix="SentinelDirectory4477/evidence",
        contents={
            kr.BUNDLE_MANIFEST: b'{"patient": "' + SENTINELS[0].encode() + b'"}',
            kr.PARITY_FIXTURE: text,
            kr.LICENSE_INVENTORY: text,
            kr.THREAT_MODEL: b"# Threat model\n" + text,
        },
    )
    manifest["moved_items"].append(
        {
            "issue": "#3943",
            "priority": "P1",
            "destination": SENTINELS[1],
            "rationale": SENTINELS[0],
            "narrowed_claims": [],
        }
    )
    return manifest


def test_sentinel_scan_finds_no_raw_text_identifiers_paths_or_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "checkout"
    manifest = _planted_manifest(root)

    packet = kr.evaluate_kit_release(
        manifest,
        repo_root=root,
        signing_key=TEST_ONLY_SIGNING_KEY,
        key_id=TEST_ONLY_KEY_ID,
        resolve_head=head_resolver(),
    )
    rendered = json.dumps(packet.to_dict(), sort_keys=True)

    assert packet.decision == kr.KIT_RELEASE_NOT_READY
    assert set(packet.blocking_codes) == {
        "moved_item_destination_invalid:#3943",
        "moved_item_rationale_unknown:#3943",
    }
    for sentinel in (*SENTINELS, TEST_ONLY_SIGNING_KEY.decode(), str(tmp_path)):
        assert sentinel not in rendered
    for row in manifest["evidence"]:
        assert row["path"] not in rendered
    assert _LOCAL_PATH.search(rendered) is None
    assert '"path"' not in rendered

    # The command line prints the same value-free codes and nothing else.
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv(kr.KIT_RELEASE_SIGNING_ENV, TEST_ONLY_SIGNING_KEY.decode())
    status = kr.main(
        [
            "--manifest",
            str(manifest_path),
            "--repo-root",
            str(root),
            "--output",
            str(tmp_path / "packet.json"),
        ]
    )
    output = capsys.readouterr()
    assert status == 1
    for sentinel in (*SENTINELS, TEST_ONLY_SIGNING_KEY.decode(), str(tmp_path)):
        assert sentinel not in output.out + output.err


def test_evidence_outside_the_checkout_or_behind_symlinks_is_never_read(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkout"
    manifest = write_evidence(root)
    threat = root / entry(manifest, "kit-threat-model")["path"]
    outside = tmp_path / "outside"
    outside.mkdir()
    # Same name and bytes as the pinned file, reachable only through a link.
    (outside / threat.name).write_bytes(threat.read_bytes())
    threat.unlink()
    threat.parent.rmdir()
    threat.parent.symlink_to(outside, target_is_directory=True)

    packet = kr.evaluate_kit_release(
        manifest,
        repo_root=root,
        signing_key=TEST_ONLY_SIGNING_KEY,
        resolve_head=head_resolver(),
    )

    assert packet.blocking_codes == ("evidence_missing:kit-threat-model",)
    assert str(outside) not in json.dumps(packet.to_dict())


def test_oversized_typed_evidence_is_digested_but_not_parsed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = write_evidence(tmp_path)
    monkeypatch.setattr(kr, "MAX_TYPED_EVIDENCE_BYTES", 64)

    packet = kr.evaluate_kit_release(
        manifest,
        repo_root=tmp_path,
        signing_key=TEST_ONLY_SIGNING_KEY,
        resolve_head=head_resolver(),
    )

    matrix = next(row for row in packet.evidence if row["id"] == "support-matrix")
    assert matrix["digest"] == entry(manifest, "support-matrix")["sha256"]
    assert "evidence_too_large:support-matrix" in packet.blocking_codes
    assert "evidence_too_large:device-qualification" in packet.blocking_codes
