"""OpenMedKit Swift bundles a byte-identical copy of the English cue table."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PARITY = ROOT / "tests" / "fixtures" / "parity"
SHARED = PARITY / "context_cues_en_v1.json"
SWIFT_COPY = (
    ROOT
    / "swift"
    / "OpenMedKit"
    / "Sources"
    / "OpenMedKit"
    / "Resources"
    / "clinical"
    / "context_cues_en_v1.json"
)


def test_swift_cue_table_is_byte_identical_to_the_shared_table() -> None:
    assert SWIFT_COPY.read_bytes() == SHARED.read_bytes(), (
        "swift/OpenMedKit context_cues_en_v1.json is stale; copy "
        "tests/fixtures/parity/context_cues_en_v1.json after regenerating it"
    )


def test_swift_cue_table_digest_matches_the_assertion_vectors() -> None:
    vectors = json.loads((PARITY / "clinical_assertion_v1.json").read_text("utf-8"))
    digest = hashlib.sha256(SWIFT_COPY.read_bytes()).hexdigest()
    assert vectors["cue_table_sha256"] == digest
