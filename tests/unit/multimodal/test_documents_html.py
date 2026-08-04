from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

import openmed.multimodal as multimodal
from openmed.multimodal import base
from openmed.multimodal.documents_html import extract_html, write_redacted_html

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


def test_writer_redacts_exact_fixture_and_preserves_surrounding_source(
    tmp_path: Path,
) -> None:
    raw = FIXTURE.read_text(encoding="utf-8")
    output = tmp_path / "redacted.html"

    result = write_redacted_html(FIXTURE, output, [(8, 18, "[PERSON]")])

    redacted = output.read_text(encoding="utf-8")
    assert result == output
    assert "<p>Patient [PERSON]</p>" in redacted
    start = raw.index("Jane", raw.index("<body>"))
    end = raw.index(" Roe", start) + len(" Roe")
    assert redacted == raw[:start] + "[PERSON]" + raw[end:]


def test_writer_preserves_mixed_newlines_and_escapes_replacement(
    tmp_path: Path,
) -> None:
    raw = "<!doctype html>\r\n<!--keep-->\n<p>Jane\rRoe</p>"
    source = tmp_path / "source.html"
    noop = tmp_path / "noop.html"
    redacted = tmp_path / "redacted.html"
    source.write_bytes(raw.encode())

    write_redacted_html(source, noop, [])
    document = extract_html(source)
    start = document.text.index("Jane")
    write_redacted_html(source, redacted, [(start, start + 4, "A&B")])

    assert noop.read_bytes() == source.read_bytes()
    expected = raw[: raw.index("Jane")] + "A&amp;B" + raw[raw.index("Jane") + 4 :]
    assert redacted.read_bytes() == expected.encode()


def test_writer_projects_linear_suffix_atomic_entity_and_cross_tag_ranges(
    tmp_path: Path,
) -> None:
    source = tmp_path / "entities.html"
    suffix_output = tmp_path / "suffix.html"
    entity_output = tmp_path / "entity.html"
    cross_output = tmp_path / "cross.html"
    source.write_text("<p>&amp;copycat</p>", encoding="utf-8")

    document = extract_html(source)
    suffix = document.text.index("copycat")
    write_redacted_html(source, suffix_output, [(suffix, suffix + 7, "word")])
    write_redacted_html(source, entity_output, [(0, 1, "and")])
    assert suffix_output.read_text(encoding="utf-8") == "<p>&amp;word</p>"
    assert entity_output.read_text(encoding="utf-8") == "<p>andcopycat</p>"

    cross_source = tmp_path / "cross-source.html"
    cross_source.write_text("<p>Jane <em>Roe</em></p>", encoding="utf-8")
    write_redacted_html(cross_source, cross_output, [(0, 8, "[PERSON]")])
    assert cross_output.read_text(encoding="utf-8") == "<p>[PERSON]<em></em></p>"


@pytest.mark.parametrize(
    "replacements",
    [
        [(-1, 1, "x")],
        [(0, 0, "x")],
        [(0, 99, "x")],
        [(0, 2, "x"), (1, 3, "y")],
    ],
)
def test_writer_rejects_invalid_ranges_before_output(
    tmp_path: Path, replacements: list[tuple[int, int, str]]
) -> None:
    source = tmp_path / "source.html"
    output = tmp_path / "output.html"
    source.write_text("<p>Jane</p>", encoding="utf-8")
    with pytest.raises(ValueError):
        write_redacted_html(source, output, replacements)
    assert not output.exists()


def test_writer_deduplicates_exact_requests_and_rejects_atomic_collisions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.html"
    duplicate_output = tmp_path / "duplicate.html"
    collision_output = tmp_path / "collision.html"
    source.write_text("<p>&NotEqualTilde;</p>", encoding="utf-8")

    write_redacted_html(source, duplicate_output, [(0, 2, "x"), (0, 2, "x")])
    assert duplicate_output.read_text(encoding="utf-8") == "<p>x</p>"
    with pytest.raises(ValueError):
        write_redacted_html(
            source,
            collision_output,
            [(0, 1, "x"), (1, 2, "x")],
        )
    assert not collision_output.exists()


def test_writer_rejects_no_replaceable_text_and_source_aliases(tmp_path: Path) -> None:
    source = tmp_path / "source.html"
    source.write_text("<p>Jane</p><p>Roe</p>", encoding="utf-8")
    document = extract_html(source)
    separator = document.text.index("\n")
    with pytest.raises(ValueError):
        write_redacted_html(
            source,
            tmp_path / "break.html",
            [(separator, separator + 1, "x")],
        )

    for alias in (source, source.resolve()):
        with pytest.raises(ValueError):
            write_redacted_html(source, alias, [(0, 4, "x")])
    symlink = tmp_path / "alias.html"
    symlink.symlink_to(source)
    with pytest.raises(ValueError):
        write_redacted_html(source, symlink, [(0, 4, "x")])
    hardlink = tmp_path / "hardlink.html"
    os.link(source, hardlink)
    with pytest.raises(ValueError):
        write_redacted_html(source, hardlink, [(0, 4, "x")])
    assert source.read_text(encoding="utf-8") == "<p>Jane</p><p>Roe</p>"


def test_explicit_module_import_registers_stdlib_handlers_and_safe_dispatch(
    tmp_path: Path,
) -> None:
    assert base._HANDLERS[".html"][-1].requires_multimodal is False
    assert base._HANDLERS[".htm"][-1].requires_multimodal is False
    output = tmp_path / "redacted.html"
    observed: dict[str, object] = {}

    def detector(text: str, *, lang: str | None = None):
        observed.update(text=text, lang=lang)
        return {"entities": [{"start": 8, "end": 18, "label": "PERSON"}]}

    document = base._HANDLERS[".html"][-1].handler(
        FIXTURE,
        policy={"output_path": output},
        models={"detector": detector},
        lang="en",
    )

    assert observed == {"text": "Patient Jane & Roe", "lang": "en"}
    assert "<p>Patient [PERSON]</p>" in output.read_text(encoding="utf-8")
    assert document.metadata == {
        "format": "html",
        "source_path": str(FIXTURE),
        "detected_span_count": 1,
        "redacted_html_path": str(output),
    }
    assert all(
        value not in _flatten(document.metadata)
        for value in (
            FIXTURE.read_text(encoding="utf-8"),
            "Hidden Jane",
            "display:none",
        )
    )


def test_handler_without_entities_does_not_create_output(tmp_path: Path) -> None:
    output = tmp_path / "unused.html"
    document = base._HANDLERS[".htm"][-1].handler(
        FIXTURE,
        policy={"output_path": output},
        models=lambda text, **kwargs: [],
    )
    assert document.text == "Patient Jane & Roe"
    assert document.metadata["detected_span_count"] == 0
    assert not output.exists()


def test_clean_package_import_activates_handlers_without_optional_extra() -> None:
    command = (
        "import openmed.multimodal as m; "
        "from openmed.multimodal import base; "
        "print(m.extract_html.__module__); "
        "print(base._HANDLERS['.html'][-1].requires_multimodal); "
        "print(base._HANDLERS['.htm'][-1].requires_multimodal)"
    )
    result = subprocess.run(
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "openmed.multimodal.documents_html",
        "False",
        "False",
    ]


def test_package_exports_are_unique_identical_and_documented() -> None:
    assert multimodal.extract_html is extract_html
    assert multimodal.write_redacted_html is write_redacted_html
    assert multimodal.__all__.count("extract_html") == 1
    assert multimodal.__all__.count("write_redacted_html") == 1
    assert extract_html.__doc__ and "source" in extract_html.__doc__.lower()
    assert (
        write_redacted_html.__doc__
        and "replacement" in write_redacted_html.__doc__.lower()
    )


@pytest.mark.parametrize("suffix", [".html", ".HTM"])
def test_real_dispatcher_redacts_both_extensions(tmp_path: Path, suffix: str) -> None:
    source = tmp_path / f"synthetic{suffix}"
    output = tmp_path / f"redacted{suffix}"
    source.write_bytes(FIXTURE.read_bytes())

    def detector(text: str, *, lang: str | None = None):
        assert text == "Patient Jane & Roe"
        assert lang == "en"
        return [(8, 18, "PERSON")]

    document = multimodal.redact_document(
        source,
        models=detector,
        lang="en",
        policy={"output_path": output, "replacement": "[REDACTED]"},
    )

    assert document.metadata["detected_span_count"] == 1
    assert "<p>Patient [REDACTED]</p>" in output.read_text(encoding="utf-8")
