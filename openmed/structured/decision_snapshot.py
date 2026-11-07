"""Verify operator-approved local model snapshots before executing custom code."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from openmed.structured.decision import DecisionUnsupportedError


class DecisionSnapshotError(DecisionUnsupportedError):
    """Controlled local snapshot refusal; never includes paths or upstream errors."""


@dataclass(frozen=True, slots=True)
class SnapshotPin:
    """Trusted manifest of executable sources and runtime assets for one revision."""

    repository: str
    revision: str
    python_files: tuple[str, ...]
    runtime_files: tuple[str, ...]
    python_sha256: str
    runtime_sha256: str
    auxiliary: frozenset[str]


def _refuse(code: str) -> None:
    raise DecisionSnapshotError(code)


def _entries(
    root: Path, pin: SnapshotPin
) -> tuple[tuple[str, Path, tuple[int, ...]], ...]:
    files = set(pin.python_files + pin.runtime_files)
    names = {p.name for p in root.iterdir()}
    if not files <= names or names - files - pin.auxiliary:
        _refuse("decision_snapshot_inventory_mismatch")
    items = []
    for name in sorted(files):
        path = root / name
        target = path.resolve(strict=True)
        # Standard Hub snapshots use relative links to the same repository's
        # blob store. A standalone snapshot must contain its own regular files.
        if target.parent != root and not (
            root.parent.name == "snapshots"
            and target.parent == root.parent.parent / "blobs"
        ):
            _refuse("decision_snapshot_external_file")
        st = target.stat()
        if not stat.S_ISREG(st.st_mode) or not 0 < st.st_size <= 4 * 1024**3:
            _refuse("decision_snapshot_invalid_file")
        items.append(
            (
                name,
                target,
                (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns),
            )
        )
    return tuple(items)


@dataclass(frozen=True, slots=True)
class VerifiedSnapshot:
    """Content-verified local files and their original filesystem identities."""

    root: Path = field(repr=False)
    pin: SnapshotPin
    _entries: tuple[tuple[str, Path, tuple[int, ...]], ...] = field(repr=False)

    def check(self) -> None:
        """Withhold inference if the verified local inventory has changed."""
        try:
            same = _entries(self.root, self.pin) == self._entries
        except (OSError, RuntimeError):
            same = False
        if not same:
            _refuse("decision_snapshot_changed")


def verify_snapshot(path: Path, revision: str, pin: SnapshotPin) -> VerifiedSnapshot:
    """Verify an exact revision, custom-code allowlist and all runtime assets."""
    from openmed.torch.privacy_filter import is_trusted_for_remote_code

    if revision != pin.revision:
        _refuse("decision_snapshot_revision_mismatch")
    # Require explicit operator consent for this repository, even if an unrelated
    # local directory has been named after it.
    allowed = {
        v.strip().lower()
        for v in os.getenv("OPENMED_TRUSTED_REMOTE_CODE_MODELS", "").split(",")
    }
    if pin.repository.lower() not in allowed or not is_trusted_for_remote_code(
        pin.repository
    ):
        _refuse("decision_custom_code_not_allowed")
    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or str(path).startswith(("//", "\\\\"))
    ):
        _refuse("decision_local_snapshot_required")
    try:
        root = path.resolve(strict=True)
        if not root.is_dir() or (
            root.parent.name == "snapshots" and root.name != revision
        ):
            _refuse("decision_snapshot_revision_mismatch")
        entries = _entries(root, pin)
        if sum(st[2] for _, _, st in entries) > 5 * 1024**3:
            _refuse("decision_snapshot_size_limit")
        code: list[list[object]] = []
        assets: list[list[object]] = []
        for name, target, identity in entries:
            digest = hashlib.sha256()
            with target.open("rb") as handle:
                left = identity[2]
                while left:
                    chunk = handle.read(min(left, 2**20))
                    if not chunk:
                        _refuse("decision_snapshot_changed")
                    digest.update(chunk)
                    left -= len(chunk)
                if handle.read(1):
                    _refuse("decision_snapshot_changed")
            row: list[object] = [name, identity[2], digest.hexdigest()]
            (code if name in pin.python_files else assets).append(row)
        for rows, expected, reason in (
            (code, pin.python_sha256, "decision_snapshot_code_mismatch"),
            (assets, pin.runtime_sha256, "decision_snapshot_assets_mismatch"),
        ):
            if (
                hashlib.sha256(
                    json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode()
                ).hexdigest()
                != expected
            ):
                _refuse(reason)
        snapshot = VerifiedSnapshot(root, pin, entries)
        snapshot.check()
        return snapshot
    except (OSError, RuntimeError):
        pass
    raise DecisionSnapshotError("decision_snapshot_unavailable")


def require_runtime(expected: str) -> None:
    """Never import custom model code in a mismatched or online environment."""
    try:
        matches = version("transformers") == expected
    except PackageNotFoundError:
        matches = False
    if not matches:
        _refuse("decision_transformers_version_mismatch")
    try:
        from huggingface_hub.constants import HF_HUB_OFFLINE
    except ImportError:
        HF_HUB_OFFLINE = False
    if not HF_HUB_OFFLINE:
        _refuse("decision_offline_runtime_required")
