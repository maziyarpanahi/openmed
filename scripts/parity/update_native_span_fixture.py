#!/usr/bin/env python3
"""Validate or refresh the shared Swift-Kotlin span-equivalence fixture.

The canonical native-kit golden lives in ``fixtures/parity/``:

* ``expected_spans.json`` holds the synthetic cases, their expected Unicode
  scalar spans, the derived UTF-8 and UTF-16 offsets, the policy actions, and
  the expected de-identified text.
* ``span_parity_manifest.json`` pins a SHA-256 digest of every fixture file and
  names each consumer that must pass when the fixture changes.

``--check`` (the default) validates the schema, recomputes every derived field
and digest, and fails with a value-free reason code when anything is stale.
``--write`` recomputes the derived fields and digests in place; review the diff
and run every consumer listed in the manifest afterwards.

Failure messages carry reason codes, case identifiers, span indexes, and
offsets only. They never echo fixture text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = ROOT / "fixtures" / "parity"
SPANS_FILE = "expected_spans.json"
MANIFEST_FILE = "span_parity_manifest.json"

FIXTURE_SCHEMA = "openmed.native_span_parity"
FIXTURE_VERSION = 2
MANIFEST_SCHEMA = "openmed.native_span_parity_manifest"
MANIFEST_VERSION = 1
OFFSET_UNIT = "unicode_scalar"
UPDATE_COMMAND = "python scripts/parity/update_native_span_fixture.py --write"
CHECK_COMMAND = "python scripts/parity/update_native_span_fixture.py --check"

POLICY_ACTIONS = frozenset({"keep", "mask", "replace", "remove", "hash", "redact"})
CASE_ID = re.compile(r"[a-z][a-z0-9_]*\Z")

CONSUMERS: tuple[Mapping[str, str], ...] = (
    {
        "surface": "python",
        "test": "tests/unit/parity/test_native_span_fixture.py",
        "workflow": ".github/workflows/ci.yml",
        "gate": "pytest",
    },
    {
        "surface": "swift",
        "test": (
            "swift/OpenMedKit/Tests/OpenMedKitTests/SpanEquivalenceParityTests.swift"
        ),
        "workflow": ".github/workflows/swift-test.yml",
        "gate": "swift test --filter SpanEquivalenceParityTests",
    },
    {
        "surface": "kotlin",
        "test": (
            "android/openmedkit/src/test/kotlin/com/openmed/openmedkit/parity/"
            "SpanEquivalenceTest.kt"
        ),
        "workflow": ".github/workflows/android-ci.yml",
        "gate": "--tests '*SpanEquivalenceTest'",
    },
)

_TOP_LEVEL_KEYS = frozenset(
    {"schema", "version", "description", "offset_unit", "cases"}
)
_CASE_KEYS = frozenset(
    {"id", "policy", "spans", "actions", "expected_deidentified_text"}
)
_SPAN_KEYS = frozenset(
    {"label", "start", "end", "utf8_start", "utf8_end", "utf16_start", "utf16_end"}
)
_ACTION_KEYS = frozenset(
    {"label", "canonical_label", "action", "start", "end", "replacement"}
)


class FixtureError(ValueError):
    """A value-free fixture validation failure.

    Attributes:
        code: Stable reason code such as ``digest_mismatch``.
        case_id: Fixture case identifier, when the failure belongs to a case.
        detail: Offsets, indexes, or file names only; never fixture text.
    """

    def __init__(self, code: str, *, case_id: str | None = None, detail: str = ""):
        self.code = code
        self.case_id = case_id
        self.detail = detail
        parts = [code]
        if case_id is not None:
            parts.append(f"case={case_id}")
        if detail:
            parts.append(detail)
        super().__init__(": ".join(parts))


@dataclass(frozen=True)
class FixtureCase:
    """One loaded fixture case with its resolved text."""

    case_id: str
    text: str
    payload: Mapping[str, Any]


def scalar_to_utf8(text: str, offset: int) -> int:
    """Return the UTF-8 byte offset of a Unicode scalar offset.

    Args:
        text: Source text.
        offset: Unicode scalar (code point) offset into ``text``.

    Returns:
        The number of UTF-8 bytes that precede ``offset``.
    """
    return len(text[:offset].encode("utf-8"))


def scalar_to_utf16(text: str, offset: int) -> int:
    """Return the UTF-16 code-unit offset of a Unicode scalar offset.

    Args:
        text: Source text.
        offset: Unicode scalar (code point) offset into ``text``.

    Returns:
        The number of UTF-16 code units that precede ``offset``.
    """
    return len(text[:offset].encode("utf-16-le")) // 2


def apply_actions(text: str, actions: Sequence[Mapping[str, Any]]) -> str:
    """Apply policy replacements right to left at original scalar offsets.

    Args:
        text: Source text.
        actions: Ordered, non-overlapping action records. A ``None``
            replacement keeps the original span.

    Returns:
        The expected de-identified text.
    """
    result = text
    for action in reversed(actions):
        replacement = action["replacement"]
        if replacement is None:
            continue
        result = result[: action["start"]] + replacement + result[action["end"] :]
    return result


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Return the committed serialization for a fixture JSON payload.

    ASCII-only escapes keep combining marks and emoji visible in review and
    stop editors from silently normalizing them.

    Args:
        payload: JSON-compatible mapping.

    Returns:
        Two-space indented, ASCII-escaped JSON with a trailing newline.
    """
    return json.dumps(payload, indent=2, ensure_ascii=True) + "\n"


def sha256_file(path: Path) -> str:
    """Return the lowercase hex SHA-256 digest of a file's bytes.

    Args:
        path: File to hash.

    Returns:
        Hex digest string.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_spans_payload(fixture_dir: Path = FIXTURE_DIR) -> dict[str, Any]:
    """Load ``expected_spans.json`` from a fixture directory.

    Args:
        fixture_dir: Directory holding the parity fixture files.

    Returns:
        The decoded fixture payload.

    Raises:
        FixtureError: If the file is missing or is not a JSON object.
    """
    path = fixture_dir / SPANS_FILE
    if not path.is_file():
        raise FixtureError("fixture_missing", detail=SPANS_FILE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise FixtureError("fixture_not_json", detail=SPANS_FILE) from error
    if not isinstance(payload, dict):
        raise FixtureError("fixture_schema_invalid", detail="top level")
    return payload


def resolve_cases(
    payload: Mapping[str, Any], fixture_dir: Path = FIXTURE_DIR
) -> list[FixtureCase]:
    """Resolve each case's text from inline ``text`` or a sibling ``text_file``.

    Args:
        payload: Decoded fixture payload.
        fixture_dir: Directory holding the parity fixture files.

    Returns:
        Cases in fixture order.

    Raises:
        FixtureError: If a case is malformed or its text cannot be read.
    """
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise FixtureError("fixture_schema_invalid", detail="cases")
    resolved: list[FixtureCase] = []
    seen: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise FixtureError("case_schema_invalid", detail=f"index={index}")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not CASE_ID.match(case_id):
            raise FixtureError("case_id_invalid", detail=f"index={index}")
        if case_id in seen:
            raise FixtureError("case_id_duplicate", case_id=case_id)
        seen.add(case_id)
        has_inline = "text" in case
        has_file = "text_file" in case
        if has_inline == has_file:
            raise FixtureError("case_text_source_invalid", case_id=case_id)
        if has_inline:
            text = case["text"]
            if not isinstance(text, str):
                raise FixtureError("case_text_source_invalid", case_id=case_id)
        else:
            name = case["text_file"]
            if not isinstance(name, str) or Path(name).name != name:
                raise FixtureError("case_text_source_invalid", case_id=case_id)
            path = fixture_dir / name
            if not path.is_file():
                raise FixtureError("case_text_file_missing", case_id=case_id)
            text = path.read_bytes().decode("utf-8")
        resolved.append(FixtureCase(case_id=case_id, text=text, payload=case))
    return resolved


def derive_payload(
    payload: Mapping[str, Any], fixture_dir: Path = FIXTURE_DIR
) -> dict[str, Any]:
    """Return a copy of the fixture with every derived field recomputed.

    UTF-8 and UTF-16 offsets are recomputed from the scalar offsets, and the
    expected de-identified text is recomputed from the action replacements.

    Args:
        payload: Decoded fixture payload.
        fixture_dir: Directory holding the parity fixture files.

    Returns:
        A new payload with refreshed derived fields.
    """
    derived = json.loads(json.dumps(payload))
    for case, fixture_case in zip(
        derived["cases"], resolve_cases(payload, fixture_dir), strict=True
    ):
        text = fixture_case.text
        for span in case.get("spans", []):
            start, end = span["start"], span["end"]
            span["utf8_start"] = scalar_to_utf8(text, start)
            span["utf8_end"] = scalar_to_utf8(text, end)
            span["utf16_start"] = scalar_to_utf16(text, start)
            span["utf16_end"] = scalar_to_utf16(text, end)
        case["expected_deidentified_text"] = apply_actions(
            text, case.get("actions", [])
        )
    return derived


def validate_payload(
    payload: Mapping[str, Any], fixture_dir: Path = FIXTURE_DIR
) -> list[FixtureCase]:
    """Validate the fixture schema and every derived field.

    Args:
        payload: Decoded fixture payload.
        fixture_dir: Directory holding the parity fixture files.

    Returns:
        The resolved cases.

    Raises:
        FixtureError: On the first schema, ordering, offset, or output problem.
    """
    missing = _TOP_LEVEL_KEYS - payload.keys()
    if missing:
        raise FixtureError("fixture_schema_invalid", detail="missing top-level key")
    if payload["schema"] != FIXTURE_SCHEMA:
        raise FixtureError("fixture_schema_invalid", detail="schema")
    if payload["version"] != FIXTURE_VERSION:
        raise FixtureError("fixture_version_unsupported")
    if payload["offset_unit"] != OFFSET_UNIT:
        raise FixtureError("fixture_offset_unit_invalid")

    cases = resolve_cases(payload, fixture_dir)
    for fixture_case in cases:
        _validate_case(fixture_case)
    return cases


def _validate_case(fixture_case: FixtureCase) -> None:
    case_id = fixture_case.case_id
    case = fixture_case.payload
    text = fixture_case.text
    if not _CASE_KEYS <= case.keys():
        raise FixtureError("case_schema_invalid", case_id=case_id)
    if not isinstance(case["policy"], str) or not case["policy"]:
        raise FixtureError("case_policy_invalid", case_id=case_id)

    length = len(text)
    spans = case["spans"]
    if not isinstance(spans, list):
        raise FixtureError("case_schema_invalid", case_id=case_id, detail="spans")
    previous: tuple[int, int] | None = None
    span_keys: set[tuple[str, int, int]] = set()
    for index, span in enumerate(spans):
        if not isinstance(span, dict) or set(span) != _SPAN_KEYS:
            raise FixtureError(
                "span_schema_invalid", case_id=case_id, detail=f"span={index}"
            )
        start, end = span["start"], span["end"]
        if not (0 <= start < end <= length):
            raise FixtureError(
                "span_out_of_bounds",
                case_id=case_id,
                detail=f"span={index} start={start} end={end} length={length}",
            )
        if previous is not None and (start, end) < previous:
            raise FixtureError(
                "span_order_invalid", case_id=case_id, detail=f"span={index}"
            )
        previous = (start, end)
        span_keys.add((span["label"], start, end))
        for field, expected in (
            ("utf8_start", scalar_to_utf8(text, start)),
            ("utf8_end", scalar_to_utf8(text, end)),
            ("utf16_start", scalar_to_utf16(text, start)),
            ("utf16_end", scalar_to_utf16(text, end)),
        ):
            if span[field] != expected:
                raise FixtureError(
                    "span_offset_mismatch",
                    case_id=case_id,
                    detail=(
                        f"span={index} field={field} "
                        f"expected={expected} actual={span[field]}"
                    ),
                )

    actions = case["actions"]
    if not isinstance(actions, list):
        raise FixtureError("case_schema_invalid", case_id=case_id, detail="actions")
    previous_end = 0
    for index, action in enumerate(actions):
        if not isinstance(action, dict) or set(action) != _ACTION_KEYS:
            raise FixtureError(
                "action_schema_invalid", case_id=case_id, detail=f"action={index}"
            )
        if action["action"] not in POLICY_ACTIONS:
            raise FixtureError(
                "action_unknown", case_id=case_id, detail=f"action={index}"
            )
        if (action["replacement"] is None) != (action["action"] == "keep"):
            raise FixtureError(
                "action_replacement_invalid", case_id=case_id, detail=f"action={index}"
            )
        start, end = action["start"], action["end"]
        if (action["label"], start, end) not in span_keys:
            raise FixtureError(
                "action_without_span",
                case_id=case_id,
                detail=f"action={index} start={start} end={end}",
            )
        if start < previous_end:
            raise FixtureError(
                "action_overlap_invalid", case_id=case_id, detail=f"action={index}"
            )
        previous_end = end

    expected_text = apply_actions(text, actions)
    if case["expected_deidentified_text"] != expected_text:
        raise FixtureError(
            "deidentified_text_mismatch",
            case_id=case_id,
            detail=_first_utf8_difference(
                expected_text, case["expected_deidentified_text"]
            ),
        )


def _first_utf8_difference(expected: str, actual: str) -> str:
    left = expected.encode("utf-8")
    right = actual.encode("utf-8")
    offset = next(
        (index for index, pair in enumerate(zip(left, right)) if pair[0] != pair[1]),
        min(len(left), len(right)),
    )
    return (
        f"first_differing_utf8_byte={offset} "
        f"expected_bytes={len(left)} actual_bytes={len(right)}"
    )


def build_manifest(payload: Mapping[str, Any], fixture_dir: Path) -> dict[str, Any]:
    """Return the digest manifest for the fixture files on disk.

    Args:
        payload: Decoded fixture payload, used to find referenced text files.
        fixture_dir: Directory holding the parity fixture files.

    Returns:
        Manifest mapping each fixture file to its SHA-256 digest.
    """
    names = {SPANS_FILE}
    names.update(case["text_file"] for case in payload["cases"] if "text_file" in case)
    return {
        "schema": MANIFEST_SCHEMA,
        "version": MANIFEST_VERSION,
        "fixture_schema": FIXTURE_SCHEMA,
        "fixture_version": FIXTURE_VERSION,
        "algorithm": "sha256",
        "files": [
            {"path": name, "sha256": sha256_file(fixture_dir / name)}
            for name in sorted(names)
        ],
        "update_command": UPDATE_COMMAND,
        "check_command": CHECK_COMMAND,
        "consumers": [dict(consumer) for consumer in CONSUMERS],
    }


def check_fixture(fixture_dir: Path = FIXTURE_DIR) -> list[FixtureCase]:
    """Validate the committed fixture, its serialization, and its digests.

    Args:
        fixture_dir: Directory holding the parity fixture files.

    Returns:
        The resolved cases.

    Raises:
        FixtureError: When the fixture is invalid, not canonically serialized,
            or does not match the digest manifest.
    """
    payload = load_spans_payload(fixture_dir)
    cases = validate_payload(payload, fixture_dir)
    spans_text = (fixture_dir / SPANS_FILE).read_text(encoding="utf-8")
    if spans_text != canonical_json(payload):
        raise FixtureError("fixture_not_canonical", detail=SPANS_FILE)

    manifest_path = fixture_dir / MANIFEST_FILE
    if not manifest_path.is_file():
        raise FixtureError("manifest_missing", detail=MANIFEST_FILE)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise FixtureError("manifest_not_json", detail=MANIFEST_FILE) from error
    expected = build_manifest(payload, fixture_dir)
    if not isinstance(manifest, dict):
        raise FixtureError("manifest_schema_invalid")
    actual_files = {
        entry.get("path"): entry.get("sha256")
        for entry in manifest.get("files", [])
        if isinstance(entry, dict)
    }
    for entry in expected["files"]:
        if actual_files.get(entry["path"]) != entry["sha256"]:
            raise FixtureError("digest_mismatch", detail=entry["path"])
    if manifest != expected:
        raise FixtureError("manifest_stale", detail=MANIFEST_FILE)
    return cases


def write_fixture(fixture_dir: Path = FIXTURE_DIR) -> None:
    """Recompute derived fields and digests, then rewrite the fixture files.

    Args:
        fixture_dir: Directory holding the parity fixture files.

    Raises:
        FixtureError: When the refreshed fixture is still invalid.
    """
    payload = derive_payload(load_spans_payload(fixture_dir), fixture_dir)
    validate_payload(payload, fixture_dir)
    (fixture_dir / SPANS_FILE).write_text(canonical_json(payload), encoding="utf-8")
    manifest = build_manifest(payload, fixture_dir)
    (fixture_dir / MANIFEST_FILE).write_text(canonical_json(manifest), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fixture validator or updater.

    Args:
        argv: Command-line arguments, excluding the program name.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(
        description="Validate or refresh the shared Swift-Kotlin span fixture."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="validate (default)")
    mode.add_argument(
        "--write", action="store_true", help="refresh derived fields and digests"
    )
    parser.add_argument(
        "--fixture-dir", type=Path, default=FIXTURE_DIR, help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    try:
        if args.write:
            write_fixture(args.fixture_dir)
        cases = check_fixture(args.fixture_dir)
    except FixtureError as error:
        print(f"native span fixture invalid: {error}", file=sys.stderr)
        return 1
    print(f"native span fixture ok: {len(cases)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
