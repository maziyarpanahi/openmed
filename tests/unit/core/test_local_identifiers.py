"""Focused tests for deterministic local-path username detection."""

from __future__ import annotations

import pytest

from openmed.core.labels import USERNAME, normalize_label
from openmed.core.local_identifiers import (
    LOCAL_IDENTIFIER_SOURCE,
    LocalIdentifierDetector,
    detect_local_identifiers,
)


def _spans(text: str):
    return [
        (entity.text, entity.label, entity.start, entity.end)
        for entity in detect_local_identifiers(text)
    ]


def test_detects_posix_and_windows_home_usernames_with_exact_offsets() -> None:
    text = (
        'POSIX "/home/fixture_user/project/trace.log"; '
        r'Windows "C:\Users\fixture_runner\project\trace.log".'
    )

    matches = _spans(text)

    assert matches == [
        (
            "fixture_user",
            USERNAME,
            text.index("fixture_user"),
            text.index("fixture_user") + len("fixture_user"),
        ),
        (
            "fixture_runner",
            USERNAME,
            text.index("fixture_runner"),
            text.index("fixture_runner") + len("fixture_runner"),
        ),
    ]


def test_supports_macos_and_legacy_windows_home_markers() -> None:
    text = (
        "macOS /Users/synthetic_user/Documents/note.txt; "
        r"legacy C:\Documents and Settings\synthetic_runner\trace.txt; "
        r"portable C:/Users/synthetic_portable/trace.txt"
    )

    matches = _spans(text)

    assert [match[0] for match in matches] == [
        "synthetic_user",
        "synthetic_runner",
        "synthetic_portable",
    ]
    assert all(match[1] == USERNAME for match in matches)


def test_metadata_is_deterministic_and_does_not_copy_username() -> None:
    text = "trace at /home/synthetic_user/project/app.py"

    first = detect_local_identifiers(text)
    second = detect_local_identifiers(text)

    assert first == second
    assert first[0].label == normalize_label(USERNAME)
    assert first[0].metadata == {
        "source": LOCAL_IDENTIFIER_SOURCE,
        "detector": LOCAL_IDENTIFIER_SOURCE,
        "canonical_label": USERNAME,
        "normalized_label": USERNAME,
        "path_kind": "posix_home",
    }
    assert "synthetic_user" not in repr(first[0].metadata)


def test_ignores_generic_system_paths_shared_accounts_and_relative_fixtures() -> None:
    text = (
        "/usr/local/bin/python /var/log/openmed.log /tmp/fixture.txt "
        "/Users/Shared/cache /home/default/output.txt "
        "fixtures/Users/relative_user/data.json "
        r"relative\Users\relative_runner\data.json"
    )

    assert detect_local_identifiers(text) == []


def test_detector_adapter_matches_function() -> None:
    text = "trace /home/adapter_user/project"
    detector = LocalIdentifierDetector()

    assert detector.detect_entities(text) == detect_local_identifiers(text)
    assert detector.detect(text) == detect_local_identifiers(text)
    assert detector(text) == detect_local_identifiers(text)


def test_rejects_non_text_input_without_echoing_a_value() -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        detect_local_identifiers(None)  # type: ignore[arg-type]
