"""Streaming JSON structure scans bound work before any parser runs."""

from __future__ import annotations

import json
import time
import tracemalloc

import pytest

from openmed.guard.operational_limits import (
    JsonStructureLimitError,
    JsonStructureScanner,
    OperationalLimitError,
    OperationalLimits,
    parse_bounded_json,
)

_SAMPLES = (
    b'{"alpha": [1, 2.5e3, -0, true, false, null], "beta": {"gamma": "x"}}',
    b'[[], {}, [[]], {"a": {}}, ""]',
    b'{"esc": "a\\n\\"\\\\\\/\\b\\f\\r\\t", "u": "\\u00e9\\u20ac\\ud83d\\ude00"}',
    b'{"lone": "\\ud800x\\udc00", "pair_then_lone": "\\ud83d\\ude00\\ud83d"}',
    '{"raw": "é€\U0001f600", "é": 1}'.encode(),
    b'  "scalar root string"  ',
    b"12345",
    b'{"k\\u0041": [{"n": [1, [2, [3]]]}], "z": "\\\\u0041"}',
)


def _expected(sample: bytes) -> tuple[int, int, int]:
    value = json.loads(sample)
    nodes = 0
    longest = 0
    deepest = 0
    stack = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if isinstance(current, str):
            longest = max(longest, len(current.encode("utf-8", "surrogatepass")))
        elif isinstance(current, dict):
            deepest = max(deepest, depth + 1)
            for key, item in current.items():
                longest = max(longest, len(key.encode("utf-8", "surrogatepass")))
                stack.append((item, depth + 1))
        elif isinstance(current, list):
            deepest = max(deepest, depth + 1)
            stack.extend((item, depth + 1) for item in current)
    return nodes, deepest, longest


def _scan(chunks: list[bytes], limits: OperationalLimits | None = None):
    scanner = JsonStructureScanner(limits)
    for chunk in chunks:
        scanner.feed(chunk)
    scanner.finish()
    return scanner


@pytest.mark.parametrize("sample", _SAMPLES)
def test_counts_match_decoded_json_for_every_chunk_split(sample: bytes) -> None:
    expected = _expected(sample)

    for split in range(len(sample) + 1):
        scanner = _scan([sample[:split], sample[split:]])
        observed = (
            scanner.node_count,
            scanner.max_depth,
            scanner.longest_string_bytes,
        )
        assert observed == expected, split

    bytewise = _scan([sample[index : index + 1] for index in range(len(sample))])
    assert bytewise.node_count == expected[0]
    assert bytewise.max_depth == expected[1]
    assert bytewise.longest_string_bytes == expected[2]


@pytest.mark.parametrize(
    ("limits", "payload", "category", "ceiling"),
    (
        (OperationalLimits(max_json_depth=2), b"[[[]]]", "depth", 2),
        (OperationalLimits(max_json_nodes=2), b'{"a":1,"b":2}', "nodes", 2),
        (
            OperationalLimits(max_json_string_bytes=3),
            b'["\\u20ac!"]',
            "string_bytes",
            3,
        ),
    ),
)
def test_violations_carry_only_category_and_ceiling(
    limits: OperationalLimits, payload: bytes, category: str, ceiling: int
) -> None:
    scanner = JsonStructureScanner(limits)

    with pytest.raises(JsonStructureLimitError) as error:
        scanner.feed(payload)
        scanner.finish()

    assert isinstance(error.value, OperationalLimitError)
    assert error.value.code == "json_structure_exceeded"
    assert error.value.to_dict() == {"category": category, "ceiling": ceiling}
    assert str(error.value) == "json_structure_exceeded"


def test_scan_stops_at_first_violation_and_stays_failed() -> None:
    scanner = JsonStructureScanner(OperationalLimits(max_json_nodes=3))

    with pytest.raises(JsonStructureLimitError):
        scanner.feed(b"[1,2,3" + b",4" * 1_000)
    assert scanner.node_count == 4
    with pytest.raises(JsonStructureLimitError):
        scanner.feed(b"]")


@pytest.mark.parametrize(
    "payload",
    (
        "[[[[1]]]]".encode("utf-16"),
        "[[[[1]]]]".encode("utf-16-le"),
        "[[[[1]]]]".encode("utf-32-be"),
        b"\xef\xbb\xbf[[[[1]]]]",
    ),
)
def test_alternate_json_encodings_are_scanned(payload: bytes) -> None:
    scanner = JsonStructureScanner(OperationalLimits(max_json_depth=3))

    with pytest.raises(JsonStructureLimitError) as error:
        for index in range(len(payload)):
            scanner.feed(payload[index : index + 1])
        scanner.finish()

    assert error.value.to_dict() == {"category": "depth", "ceiling": 3}


def test_invalid_escape_halts_scanning_for_the_parser_to_reject() -> None:
    scanner = JsonStructureScanner(OperationalLimits(max_json_depth=1))

    scanner.feed(b'["\\x", [[[[[[')
    scanner.finish()

    assert scanner.max_depth == 1
    assert not scanner.is_complete


def test_unbalanced_documents_are_reported_incomplete() -> None:
    for payload in (b"[1", b'"open', b"]", b'["\\'):
        scanner = _scan([payload])
        assert not scanner.is_complete
    assert _scan([b' {"a": [1]} ']).is_complete


def test_escape_dense_strings_scan_fast_with_small_bounded_memory() -> None:
    body = b'["' + b"\\n\\u00e9" * 400_000 + b'"]'
    scanner = JsonStructureScanner()

    tracemalloc.start()
    try:
        started = time.perf_counter()
        for offset in range(0, len(body), 65_536):
            scanner.feed(body[offset : offset + 65_536])
        scanner.finish()
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert scanner.longest_string_bytes == 1_200_000
    assert peak < 1_000_000
    assert elapsed < 5.0


@pytest.mark.parametrize(
    "flood",
    (
        b"[]" + b"],]," * 100_000,
        b"{" + b'"",' * 100_000 + b"}",
        b"{" + b'"k":,' * 100_000 + b"}",
        b'{"k": }' * 100_000,
    ),
)
def test_malformed_token_floods_halt_at_the_first_grammar_error(flood: bytes) -> None:
    scanner = JsonStructureScanner(OperationalLimits(max_json_nodes=10))

    started = time.perf_counter()
    scanner.feed(flood)
    scanner.finish()

    assert time.perf_counter() - started < 0.5
    assert not scanner.is_complete
    assert scanner.node_count <= 2


def test_token_budget_bounds_keys_that_never_receive_values() -> None:
    scanner = JsonStructureScanner(OperationalLimits(max_json_nodes=10))

    with pytest.raises(JsonStructureLimitError) as error:
        scanner.feed(b"{" + b'"\\n":,' * 1_000 + b"}")

    assert scanner.node_count == 1
    assert error.value.to_dict() == {"category": "nodes", "ceiling": 10}


def test_trickled_valid_bodies_never_exhaust_the_token_budget() -> None:
    body = json.dumps({"k": [[{}], [], " " * 40], "v": 1.25}, indent=8).encode()
    scanner = _scan(
        [body[index : index + 1] for index in range(len(body))],
        OperationalLimits(max_json_nodes=_expected(body)[0]),
    )

    assert scanner.is_complete
    assert scanner.node_count == _expected(body)[0]


def test_parse_bounded_json_preflight_reuses_the_streaming_scanner() -> None:
    with pytest.raises(OperationalLimitError) as nodes:
        parse_bounded_json(
            b"[" + b"1," * 50 + b"1]", limits=OperationalLimits(max_json_nodes=10)
        )
    with pytest.raises(OperationalLimitError) as total:
        parse_bounded_json(
            b'["abcdef", "ghijkl"]',
            limits=OperationalLimits(max_json_string_bytes=10),
        )
    with pytest.raises(OperationalLimitError) as invalid:
        parse_bounded_json(b"[1]]")

    assert nodes.value.code == "json_node_limit_exceeded"
    assert total.value.code == "json_string_limit_exceeded"
    assert invalid.value.code == "json_invalid"
    assert parse_bounded_json(b'{"a": ["\\u00e9"]}') == {"a": ["é"]}
