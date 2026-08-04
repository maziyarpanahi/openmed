from __future__ import annotations

import io
from pathlib import Path

import pytest

from openmed.multimodal.documents_html import extract_html

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_phi.html"


def _source_range(document, offset: int) -> tuple[int, int]:
    span = document.location_at(offset)
    assert span is not None
    start = int(span.metadata["source_start"])
    end = int(span.metadata["source_end"])
    if span.metadata["source_map_mode"] == "linear":
        start += offset - span.start
        end = start + 1
    return start, end


def _flatten(value: object) -> list[str]:
    if isinstance(value, dict):
        return [
            *map(str, value.keys()),
            *sum((_flatten(item) for item in value.values()), []),
        ]
    if isinstance(value, (list, tuple, set)):
        return sum((_flatten(item) for item in value), [])
    return [str(value)]


def test_extracts_exact_fixture_with_total_safe_source_map() -> None:
    raw = FIXTURE.read_text(encoding="utf-8")
    document = extract_html(FIXTURE)

    assert document.text == "Patient Jane & Roe"
    assert all(
        document.location_at(index) is not None for index in range(len(document.text))
    )
    amp_offset = document.text.index("&")
    start, end = _source_range(document, amp_offset)
    assert raw[start:end] == "&amp;"
    flattened = _flatten(document.metadata) + sum(
        (_flatten(span.metadata) for span in document.spans), []
    )
    assert raw not in flattened
    assert "Hidden Jane" not in flattened
    assert "display:none" not in flattened


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("<head><title>x</title><body><p>Jane</p>", "Jane"),
        ("<head><script>Hidden Jane<body><p>Jane</p>", ""),
        ("<head><style>Hidden Jane<body><p>Jane</p>", ""),
        ("<p>Patient Jane", "Patient Jane"),
        ("<!-- x --><!doctype html><head>x</head><script>x</script>", ""),
        ("", ""),
    ],
)
def test_suppression_and_malformed_inputs(source: str, expected: str) -> None:
    document = extract_html(source)
    assert document.text == expected
    assert all(
        document.location_at(index) is not None for index in range(len(document.text))
    )


def test_entities_use_callback_bounded_atomic_and_linear_ranges() -> None:
    source = (
        "<p>&amp;copycat &amp copy &copycat &ampersand &boguscat; "
        "&#38; &#x26; &NotEqualTilde; José 李</p>"
    )
    document = extract_html(source)

    assert document.text == "&copycat & copy ©cat &ersand &boguscat; & & ≂̸ José 李"

    first_amp = document.text.index("&")
    start, end = _source_range(document, first_amp)
    assert source[start:end] == "&amp;"
    suffix = document.text.index("copycat")
    assert all(
        source[slice(*_source_range(document, offset))] == document.text[offset]
        for offset in range(suffix, suffix + len("copycat"))
    )

    copy_symbol = document.text.index("©")
    start, end = _source_range(document, copy_symbol)
    assert source[start:end] == "&copy"
    combining = document.text.index("≂")
    assert _source_range(document, combining) == _source_range(document, combining + 1)
    start, end = _source_range(document, combining)
    assert source[start:end] == "&NotEqualTilde;"


def test_inline_and_block_whitespace_mapping() -> None:
    document = extract_html("<p>A <em>B</em></p><p>C</p>")
    assert document.text == "A B\nC"
    newline = document.location_at(document.text.index("\n"))
    assert newline is not None
    assert newline.metadata["source_map_mode"] == "atomic"
    assert newline.metadata["replaceable"] is False


def test_path_raw_and_file_like_inputs_match_and_preserve_newline_offsets(
    tmp_path: Path,
) -> None:
    source = "<p>A\r\nB\nC\rD</p>"
    path = tmp_path / "mixed.html"
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(source)

    path_document = extract_html(path)
    raw_document = extract_html(source)
    file_document = extract_html(io.StringIO(source))

    assert path_document.text == raw_document.text == file_document.text
    assert path_document.spans == raw_document.spans == file_document.spans
    assert path_document.metadata == {"format": "html", "source_path": str(path)}
    assert raw_document.metadata == file_document.metadata == {"format": "html"}
    for index, character in enumerate(path_document.text):
        start, end = _source_range(path_document, index)
        if path_document.location_at(index).metadata["source_map_mode"] == "linear":
            assert source[start:end] == character
