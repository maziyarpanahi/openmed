"""Bounded property fuzzing for untrusted healthcare document parsers.

The corpus contains only synthetic records.  Each available parser runs in a
persistent child process so a single malformed input cannot hang the test
runner.  Successfully parsed samples must satisfy the shared character-offset
contract; declared format-rejection errors are acceptable, while every other
exception is reported as a parser crash.

The target list is capability-driven.  It covers the currently registered
Markdown, CDA, and EPUB handlers plus the HL7 parser, and automatically picks
up EML/MSG, RTF, and ODT handlers or the X12 parser when those modules land.
"""

from __future__ import annotations

import importlib.util
import multiprocessing
import os
import queue
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis import target as hypothesis_target

import openmed.multimodal  # noqa: F401  (populate the lazy handler registry)
import openmed.multimodal.base as multimodal_base
from openmed.interop.hl7v2 import parse_hl7v2
from openmed.multimodal.base import ExtractedDocument, SourceSpan
from openmed.multimodal.exceptions import (
    MissingDependencyError,
    UnsupportedDocumentError,
)

# Ensure the bounded/nightly profile is registered even if this module is
# collected before the package conftest side effects run.
from . import conftest as _fuzz_conftest  # noqa: F401  (import for side effects)

pytestmark = pytest.mark.fuzz

_SEED_DIR = Path(__file__).with_name("seeds")
_PARSER_TIMEOUT_SECONDS = float(
    os.environ.get("OPENMED_FORMAT_PARSER_TIMEOUT_SECONDS", "0.5")
)
_WORKER_START_TIMEOUT_SECONDS = 15.0

_EXPECTED_SEEDS = {
    "email": "minimal.eml",
    "markdown": "minimal.md",
    "hl7": "minimal.hl7",
    "cda": "minimal.cda.xml",
    "rtf": "minimal.rtf",
    "odt": "minimal.odt",
    "epub": "minimal.epub",
    "x12": "minimal.x12",
}


@dataclass(frozen=True)
class _ParserTarget:
    key: str
    format_name: str
    seed_name: str
    suffix: str | None = None
    seed_must_parse: bool = True


_DOCUMENT_FORMATS = (
    ("eml", ".eml", "minimal.eml", True),
    # EML/MSG is one scope family.  The RFC 5322 seed is valid for EML and is
    # still useful malformed input for an optional MSG handler.
    ("msg", ".msg", "minimal.eml", False),
    ("markdown", ".md", "minimal.md", True),
    ("markdown", ".markdown", "minimal.md", True),
    ("cda", ".xml", "minimal.cda.xml", True),
    ("rtf", ".rtf", "minimal.rtf", True),
    ("odt", ".odt", "minimal.odt", True),
    ("epub", ".epub", "minimal.epub", True),
)

_EXPECTED_REJECTIONS = (
    EOFError,
    MissingDependencyError,
    OSError,
    UnicodeError,
    UnsupportedDocumentError,
    ValueError,
)


class _RejectedInput(Exception):
    """A malformed sample rejected through a parser's documented error path."""


def _available_targets() -> tuple[_ParserTarget, ...]:
    targets = [
        _ParserTarget(
            key=f"document:{format_name}:{suffix}",
            format_name=format_name,
            seed_name=seed_name,
            suffix=suffix,
            seed_must_parse=seed_must_parse,
        )
        for format_name, suffix, seed_name, seed_must_parse in _DOCUMENT_FORMATS
        if suffix in multimodal_base._HANDLERS
    ]
    targets.append(
        _ParserTarget(
            key="hl7",
            format_name="hl7",
            seed_name="minimal.hl7",
        )
    )
    if importlib.util.find_spec("openmed.interop.x12_837") is not None:
        targets.append(
            _ParserTarget(
                key="x12",
                format_name="x12",
                seed_name="minimal.x12",
            )
        )
    return tuple(targets)


_TARGETS = _available_targets()
_TARGETS_BY_KEY = {target.key: target for target in _TARGETS}


def _identity_model(text: str, **_: Any) -> str:
    """Return parser text unchanged so the fuzz target stays offline."""
    return text


def _seed_bytes(target: _ParserTarget) -> bytes:
    return (_SEED_DIR / target.seed_name).read_bytes()


def _assert_document_invariants(document: ExtractedDocument) -> None:
    """Validate the common text and character-offset contract."""
    assert isinstance(document.text, str)
    assert isinstance(document.spans, tuple)

    previous_end = 0
    for span in document.spans:
        assert isinstance(span, SourceSpan)
        assert isinstance(span.start, int) and not isinstance(span.start, bool)
        assert isinstance(span.end, int) and not isinstance(span.end, bool)
        assert 0 <= span.start < span.end <= len(document.text)
        assert span.start >= previous_end
        assert document.text_for(span) == document.text[span.start : span.end]
        assert document.location_at(span.start) == span
        previous_end = span.end

        source_start = span.metadata.get("source_start")
        source_end = span.metadata.get("source_end")
        if source_start is not None or source_end is not None:
            assert isinstance(source_start, int)
            assert isinstance(source_end, int)
            assert 0 <= source_start < source_end

            source_text = document.metadata.get("source_text")
            if isinstance(source_text, str):
                assert source_end <= len(source_text)

    sections = document.metadata.get("sections", ())
    if isinstance(sections, (list, tuple)):
        previous_section_end = 0
        for section in sections:
            assert isinstance(section, dict)
            start = section.get("start")
            end = section.get("end")
            assert isinstance(start, int)
            assert isinstance(end, int)
            assert 0 <= start <= end <= len(document.text)
            assert start >= previous_section_end
            previous_section_end = end


def _parse_document_target(target: _ParserTarget, data: bytes) -> None:
    assert target.suffix is not None
    with tempfile.TemporaryDirectory(prefix="openmed-format-fuzz-") as directory:
        path = Path(directory) / f"input{target.suffix}"
        path.write_bytes(data)

        # The Markdown implementation is pure Python; its optional dependency
        # guard is tested elsewhere and third-party parsers are outside this
        # fuzz target's scope.
        if target.format_name == "markdown":
            from openmed.multimodal import documents_markdown

            documents_markdown._ensure_markup_parser_available = lambda _flavor: None

        try:
            specs = multimodal_base._HANDLERS.get(target.suffix, ())
            handler = multimodal_base._select_handler(path, specs)
            if handler is None:
                raise UnsupportedDocumentError(
                    f"No {target.format_name} handler accepted this sample"
                )

            document = handler.handler(
                path,
                policy=None,
                models=_identity_model,
                lang="en",
            )
        except _EXPECTED_REJECTIONS as exc:
            raise _RejectedInput(type(exc).__name__) from exc
        assert isinstance(document, ExtractedDocument)
        _assert_document_invariants(document)


def _parse_hl7(data: bytes) -> None:
    try:
        source = data.decode("utf-8")
        parsed = parse_hl7v2(source)
    except _EXPECTED_REJECTIONS as exc:
        raise _RejectedInput(type(exc).__name__) from exc
    rendered = parsed.serialize()
    canonical_source = source.lstrip("\ufeff")
    assert rendered == canonical_source

    spans: list[SourceSpan] = []
    cursor = 0
    for segment_index, segment in enumerate(parsed.segments):
        segment_text = segment.serialize()
        start = rendered.find(segment_text, cursor)
        assert start >= cursor
        end = start + len(segment_text)
        if start < end:
            spans.append(
                SourceSpan(
                    start=start,
                    end=end,
                    metadata={
                        "format": "hl7",
                        "segment_index": segment_index,
                        "segment_name": segment.name,
                        "source_start": start,
                        "source_end": end,
                    },
                )
            )
        cursor = end

    _assert_document_invariants(
        ExtractedDocument(
            text=rendered,
            spans=tuple(spans),
            metadata={"format": "hl7", "source_text": rendered},
        )
    )


def _parse_x12(data: bytes) -> None:
    from openmed.interop.x12_837 import parse_x12_837

    try:
        source = data.decode("utf-8")
        parsed = parse_x12_837(source)
    except _EXPECTED_REJECTIONS as exc:
        raise _RejectedInput(type(exc).__name__) from exc
    rendered, offset_map = parsed.serialize_with_offset_map()
    assert rendered == source

    spans: list[SourceSpan] = []
    for entry in offset_map.entries:
        assert 0 <= entry.source_start <= entry.source_end <= len(source)
        assert 0 <= entry.output_start <= entry.output_end <= len(rendered)
        assert (
            source[entry.source_start : entry.source_end]
            == rendered[entry.output_start : entry.output_end]
        )
        assert offset_map.source_to_output(entry.source_start, entry.source_end) == (
            entry.output_start,
            entry.output_end,
        )
        assert offset_map.output_to_source(entry.output_start, entry.output_end) == (
            entry.source_start,
            entry.source_end,
        )
        if entry.output_start < entry.output_end:
            spans.append(
                SourceSpan(
                    start=entry.output_start,
                    end=entry.output_end,
                    metadata={
                        "format": "x12",
                        "segment_index": entry.segment_index,
                        "segment_tag": entry.segment_tag,
                        "element_position": entry.element_position,
                        "source_start": entry.source_start,
                        "source_end": entry.source_end,
                    },
                )
            )

    _assert_document_invariants(
        ExtractedDocument(
            text=rendered,
            spans=tuple(spans),
            metadata={"format": "x12", "source_text": source},
        )
    )


def _execute_target(target_key: str, data: bytes) -> None:
    if target_key == "__broken_offset_map__":
        _assert_document_invariants(
            ExtractedDocument(
                text="ok",
                spans=(SourceSpan(start=0, end=3),),
            )
        )
        return
    if target_key == "__hang__":
        time.sleep(60)
        return
    if target_key == "hl7":
        _parse_hl7(data)
        return
    if target_key == "x12":
        _parse_x12(data)
        return

    target = _TARGETS_BY_KEY[target_key]
    _parse_document_target(target, data)


def _worker_main(requests: Any, responses: Any) -> None:
    responses.put(("ready", ""))
    while True:
        request = requests.get()
        if request is None:
            return
        target_key, data = request
        try:
            _execute_target(target_key, data)
        except _RejectedInput as exc:
            responses.put(("rejected", str(exc)))
        except BaseException as exc:  # noqa: BLE001 - crashes are the fuzz result.
            detail = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
            responses.put(("failure", detail))
        else:
            responses.put(("ok", ""))


class _ParserWorker:
    """Persistent killable parser process with a per-request wall-clock bound."""

    def __init__(self, timeout: float = _PARSER_TIMEOUT_SECONDS) -> None:
        self.timeout = timeout
        self._context = multiprocessing.get_context("spawn")
        self._requests: Any = None
        self._responses: Any = None
        self._process: Any = None

    def __enter__(self) -> "_ParserWorker":
        self._requests = self._context.Queue()
        self._responses = self._context.Queue()
        self._process = self._context.Process(
            target=_worker_main,
            args=(self._requests, self._responses),
            daemon=True,
        )
        self._process.start()
        try:
            status, detail = self._responses.get(timeout=_WORKER_START_TIMEOUT_SECONDS)
        except queue.Empty as exc:
            self._terminate()
            raise AssertionError("format-parser worker did not start") from exc
        assert (status, detail) == ("ready", "")
        return self

    def __exit__(self, *_: object) -> None:
        if self._process is not None and self._process.is_alive():
            self._requests.put(None)
            self._process.join(timeout=1)
        self._terminate()
        for channel in (self._requests, self._responses):
            if channel is not None:
                channel.close()
                channel.join_thread()

    def run(self, target_key: str, data: bytes) -> tuple[str, str]:
        assert self._process is not None and self._process.is_alive()
        self._requests.put((target_key, data))
        try:
            return self._responses.get(timeout=self.timeout)
        except queue.Empty as exc:
            self._terminate()
            raise AssertionError(
                f"{target_key} exceeded the {self.timeout:.3f}s parser budget"
            ) from exc

    def _terminate(self) -> None:
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2)


def _assert_no_crash(
    worker: _ParserWorker,
    target: _ParserTarget,
    data: bytes,
) -> str:
    status, detail = worker.run(target.key, data)
    if status == "failure":
        pytest.fail(f"{target.key} crashed or returned a broken offset map:\n{detail}")
    assert status in {"ok", "rejected"}
    return status


@st.composite
def _mutated_inputs(draw: st.DrawFn) -> tuple[_ParserTarget, bytes]:
    parser_target = draw(st.sampled_from(_TARGETS))
    seed = _seed_bytes(parser_target)
    operation = draw(
        st.sampled_from(("seed", "truncate", "delete", "insert", "flip", "replace"))
    )

    if operation == "seed":
        return parser_target, seed
    if operation == "truncate":
        end = draw(st.integers(min_value=0, max_value=len(seed)))
        return parser_target, seed[:end]
    if operation == "delete" and seed:
        start = draw(st.integers(min_value=0, max_value=len(seed) - 1))
        end = draw(st.integers(min_value=start + 1, max_value=len(seed)))
        return parser_target, seed[:start] + seed[end:]
    if operation == "insert":
        at = draw(st.integers(min_value=0, max_value=len(seed)))
        inserted = draw(st.binary(min_size=1, max_size=128))
        return parser_target, seed[:at] + inserted + seed[at:]
    if operation == "flip" and seed:
        at = draw(st.integers(min_value=0, max_value=len(seed) - 1))
        bit = draw(st.sampled_from((1, 2, 4, 8, 16, 32, 64, 128)))
        return parser_target, seed[:at] + bytes((seed[at] ^ bit,)) + seed[at + 1 :]

    replacement = draw(st.binary(max_size=max(1024, min(len(seed) * 2, 8192))))
    return parser_target, replacement


def test_seed_corpus_declares_every_scope_format() -> None:
    assert set(_EXPECTED_SEEDS) == {
        "email",
        "markdown",
        "hl7",
        "cda",
        "rtf",
        "odt",
        "epub",
        "x12",
    }
    for seed_name in _EXPECTED_SEEDS.values():
        seed = _SEED_DIR / seed_name
        assert seed.is_file()
        assert seed.stat().st_size > 0


def test_minimal_seeds_parse_for_available_targets() -> None:
    with _ParserWorker() as worker:
        for parser_target in _TARGETS:
            if not parser_target.seed_must_parse:
                continue
            status = _assert_no_crash(
                worker,
                parser_target,
                _seed_bytes(parser_target),
            )
            assert status == "ok", f"{parser_target.key} rejected its valid seed"


def test_truncated_seeds_do_not_crash_registered_parsers() -> None:
    with _ParserWorker() as worker:
        for parser_target in _TARGETS:
            seed = _seed_bytes(parser_target)
            cutoffs = sorted({0, 1, len(seed) // 4, len(seed) // 2, len(seed) - 1})
            for cutoff in cutoffs:
                _assert_no_crash(worker, parser_target, seed[: max(cutoff, 0)])


def test_registered_format_parsers_resist_mutated_input() -> None:
    with _ParserWorker() as worker:

        @given(case=_mutated_inputs())
        def exercise(case: tuple[_ParserTarget, bytes]) -> None:
            parser_target, data = case
            seed = _seed_bytes(parser_target)
            hypothesis_target(
                len(data), label=f"{parser_target.format_name}:input-size"
            )
            hypothesis_target(
                len(set(data)),
                label=f"{parser_target.format_name}:byte-diversity",
            )
            hypothesis_target(
                abs(len(data) - len(seed)),
                label=f"{parser_target.format_name}:size-delta",
            )
            _assert_no_crash(worker, parser_target, data)

        exercise()


def test_harness_catches_deliberately_broken_offset_map() -> None:
    with _ParserWorker() as worker:
        status, detail = worker.run("__broken_offset_map__", b"synthetic")

    assert status == "failure"
    assert "AssertionError" in detail


def test_harness_terminates_parser_that_exceeds_time_budget() -> None:
    with _ParserWorker(timeout=0.05) as worker:
        with pytest.raises(AssertionError, match="exceeded the 0.050s parser budget"):
            worker.run("__hang__", b"synthetic")
