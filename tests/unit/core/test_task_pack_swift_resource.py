"""OpenMedKit Swift bundles a byte-identical copy of the task pack manifest."""

from __future__ import annotations

from pathlib import Path

from openmed.core.task_packs import TASK_PACKS_DATA_FILE

ROOT = Path(__file__).resolve().parents[3]
SWIFT_COPY = (
    ROOT
    / "swift"
    / "OpenMedKit"
    / "Sources"
    / "OpenMedKit"
    / "Resources"
    / "task_packs"
    / "task_packs.json"
)


def test_swift_copy_is_byte_identical_to_the_generated_manifest() -> None:
    assert SWIFT_COPY.read_bytes() == TASK_PACKS_DATA_FILE.read_bytes(), (
        "swift/OpenMedKit task_packs.json is stale; copy "
        "openmed/core/data/task_packs.json after regenerating it"
    )


def test_swift_resource_directory_holds_only_the_manifest() -> None:
    assert sorted(path.name for path in SWIFT_COPY.parent.glob("*.json")) == [
        "task_packs.json"
    ]
