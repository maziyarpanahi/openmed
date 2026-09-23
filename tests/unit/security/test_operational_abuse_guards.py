"""Operational abuse guards reject work before expensive processing."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest
from jsonschema import Draft202012Validator

from openmed.guard.operational_limits import (
    LimitState,
    OperationalLimitError,
    OperationalLimits,
    inspect_zip_payload,
    parse_bounded_json,
    validate_page_size,
    validate_payload_size,
)

ROOT = Path(__file__).resolve().parents[3]
LIMIT_SCHEMA = (
    ROOT
    / "openmed"
    / "core"
    / "schemas"
    / "json"
    / "operational_limit_decision.schema.json"
)
ARCHIVE_SCHEMA = (
    ROOT
    / "openmed"
    / "core"
    / "schemas"
    / "json"
    / "archive_limit_decision.schema.json"
)


def _zip_payload(content: bytes, *, name: str = "records/item.txt") -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, content)
    return output.getvalue()


def test_oversized_payload_and_pagination_fail_closed() -> None:
    limits = OperationalLimits(max_request_bytes=8, max_page_size=2)

    payload = validate_payload_size(9, limits=limits)
    page = validate_page_size(3, limits=limits)

    assert payload.state is LimitState.DENIED
    assert payload.code == "payload_size_exceeded"
    assert page.state is LimitState.DENIED
    assert page.code == "page_size_exceeded"
    assert payload.to_dict()["observed_count"] == 8
    schema = json.loads(LIMIT_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(payload.to_dict())
    Draft202012Validator(schema).validate(page.to_dict())


def test_bounded_json_rejects_depth_nodes_and_strings_without_echoing_values() -> None:
    canary = "synthetic-sensitive-canary-5521"

    with pytest.raises(OperationalLimitError) as depth_error:
        parse_bounded_json(
            b'{"a":{"b":{"c":1}}}',
            limits=OperationalLimits(max_json_depth=2),
        )
    with pytest.raises(OperationalLimitError) as node_error:
        parse_bounded_json(
            b"[1,2,3]",
            limits=OperationalLimits(max_json_nodes=3),
        )
    with pytest.raises(OperationalLimitError) as string_error:
        parse_bounded_json(
            ('{"value":"' + canary + '"}').encode(),
            limits=OperationalLimits(max_json_string_bytes=8),
        )

    assert depth_error.value.code == "json_depth_limit_exceeded"
    assert node_error.value.code == "json_node_limit_exceeded"
    assert string_error.value.code == "json_string_limit_exceeded"
    assert canary not in repr(string_error.value)


@pytest.mark.parametrize(
    "payload",
    (
        b'"\\ud800"',
        b'{"\\ud800":1}',
        b"NaN",
        b"Infinity",
        b"1e999",
    ),
)
def test_bounded_json_rejects_invalid_unicode_and_nonfinite_values(
    payload: bytes,
) -> None:
    with pytest.raises(OperationalLimitError) as error:
        parse_bounded_json(payload)

    assert error.value.code == "json_invalid"


def test_zip_bomb_and_traversal_are_denied_without_member_decompression() -> None:
    bomb = _zip_payload(b"x" * 100_000)
    traversal = _zip_payload(b"safe", name="../synthetic-secret.txt")
    limits = OperationalLimits(
        max_request_bytes=1_000_000,
        max_archive_member_bytes=1_024,
        max_archive_total_bytes=1_024,
        max_archive_expansion_ratio=2,
    )

    with patch.object(
        zipfile.ZipFile,
        "open",
        side_effect=AssertionError("archive member must not be opened"),
    ):
        bomb_decision = inspect_zip_payload(bomb, limits=limits)
        traversal_decision = inspect_zip_payload(traversal, limits=limits)

    assert bomb_decision.state is LimitState.DENIED
    assert bomb_decision.code == "archive_limit_denied"
    assert traversal_decision.state is LimitState.DENIED
    assert traversal_decision.report is not None
    assert traversal_decision.report.reason_counts == {"path_traversal": 1}
    assert "synthetic-secret" not in repr(traversal_decision.to_dict())
    schema = json.loads(ARCHIVE_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(bomb_decision.to_dict())
    Draft202012Validator(schema).validate(traversal_decision.to_dict())


def test_malformed_archive_is_a_typed_content_free_failure() -> None:
    canary = b"PK\x03\x04synthetic-sensitive-canary-8890"

    result = inspect_zip_payload(canary)

    assert result.state is LimitState.FAILURE
    assert result.code == "archive_invalid"
    assert "synthetic-sensitive" not in repr(result)
