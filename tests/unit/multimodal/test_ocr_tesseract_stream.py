"""Bounded native pipe execution and strict, synthetic Tesseract TSV parsing."""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import replace
from io import BytesIO

import pytest

from openmed.multimodal import ocr_tesseract_stream as stream
from openmed.multimodal.ocr import OcrResult, ocr

HEADER = "\t".join(stream._TSV_HEADER) + "\n"
ROWS = [
    "1\t1\t0\t0\t0\t0\t0\t0\t200\t150\t-1\t",
    "5\t1\t1\t1\t1\t1\t10\t20\t80\t10\t96.2\tBefund:",
    "5\t1\t1\t1\t2\t1\t10\t40\t30\t10\t88.5\tKeine",
    "5\t1\t1\t1\t2\t2\t44\t40\t60\t10\t92\tDyspnoe.",
]


def parse(rows=ROWS, **kwargs):
    return stream._parse_tsv(
        (HEADER + "\n".join(rows)).encode(),
        width=200,
        height=150,
        max_words=kwargs.get("max_words", 20),
        max_text_chars=kwargs.get("max_text_chars", 1000),
    )


def test_tsv_words_preserve_layout_geometry_confidence_and_preview():
    result = parse()
    assert result.text == "Befund: Keine Dyspnoe."
    assert result.words[0].bbox == (10, 20, 90, 30)
    assert result.words[0].confidence == pytest.approx(0.962)
    assert result.metadata["line_ids"] == ((1, 1, 1), (1, 1, 2), (1, 1, 2))
    assert result.metadata["qualification"] == "preview"
    document = result.to_document(preserve_lines=True)
    assert document.text == "Befund:\nKeine Dyspnoe."
    assert document.spans == result.to_document().spans
    assert document.metadata["line_breaks_preserved"] is True
    assert document.location_at(document.text.index("Dyspnoe")).bbox == (
        44,
        40,
        104,
        50,
    )
    assert "Dyspnoe" not in repr(result.metadata)


def test_native_lines_keep_pages_unicode_and_custom_separator_offsets():
    result = parse()
    words = (replace(result.words[0], text="Übersicht:"), *result.words[1:])
    result = OcrResult(
        words + tuple(replace(word, page=1) for word in words),
        {"line_ids": result.metadata["line_ids"] * 2},
    )
    document = result.to_document(preserve_lines=True, separator=" / ")
    assert document.text == "Übersicht:\nKeine / Dyspnoe.\nÜbersicht:\nKeine / Dyspnoe."
    assert [document.text_for(span) for span in document.spans] == [
        word.text for word in result.words
    ]
    assert [span.page for span in document.spans] == [0, 0, 0, 1, 1, 1]
    assert all(
        document.location_at(i) is None
        for i, c in enumerate(document.text)
        if c == "\n"
    )


@pytest.mark.parametrize(
    "line_ids",
    [
        None,
        [],
        [(1, 1, 1)],
        [(1, 1, 0)] * 3,
        [(True, 1, 1)] * 3,
        [(1, 1, "private")] * 3,
        [(1, 1, 1), (1, 1, 2), (1, 1, 1)],
    ],
)
def test_line_preservation_requires_valid_contiguous_native_ids(line_ids):
    result = replace(parse(), metadata={"line_ids": line_ids})
    with pytest.raises(
        ValueError,
        match="^ocr_(line_ids_required|invalid_line_ids|noncontiguous_line_ids)$",
    ):
        result.to_document(preserve_lines=True)
    assert result.to_document().text == result.text


def test_empty_native_result_preserves_explicit_empty_lines():
    result = parse([])
    assert result.to_document(preserve_lines=True).text == ""
    with pytest.raises(TypeError, match="preserve_lines"):
        result.to_document(preserve_lines="yes")


def test_duplicate_native_word_ids_fail():
    with pytest.raises(ValueError, match="^ocr_invalid_output$"):
        parse([ROWS[1], ROWS[1]])


@pytest.mark.parametrize(
    "column,value",
    [
        (0, "6"),
        (1, "2"),
        (2, "0"),
        (3, "-1"),
        (4, "0"),
        (5, "0"),
        (6, "-1"),
        (7, "-1"),
        (8, "0"),
        (9, "-5"),
        (8, "500"),
        (9, "500"),
        (10, "nan"),
        (10, "inf"),
        (10, "101"),
        (10, "-1"),
        (11, "private\tvalue"),
        (11, "private\t"),
        (11, "private\x7f"),
    ],
)
def test_tsv_rejects_invalid_word_records_without_echoing_text(column, value):
    row = ROWS[1].split("\t")
    row[column] = value
    with pytest.raises(ValueError, match="^ocr_invalid_output$"):
        parse(["\t".join(row)])


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"bad header\npatient secret",
        HEADER.encode() + b"\xff",
        HEADER.encode() + b"5\t1",
    ],
)
def test_malformed_native_output_is_rejected(data):
    with pytest.raises(ValueError, match="^ocr_invalid_output$"):
        stream._parse_tsv(data, width=200, height=150, max_words=10, max_text_chars=100)


def test_word_and_text_budgets_are_enforced():
    for limits in ({"max_words": 2}, {"max_text_chars": 8}):
        with pytest.raises(RuntimeError, match="ocr_text_limit"):
            parse(**limits)


def execute(script, *, payload=b"", timeout=2, limit=200_000, cancel=None):
    return stream._execute(
        [sys.executable, "-c", script],
        payload,
        timeout=timeout,
        max_output_bytes=limit,
        cancel_check=cancel,
    )


def test_native_pipes_drain_output_while_feeding_large_input(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "synthetic-private-environment-value")
    output = execute(
        "import sys,os; sys.stdout.buffer.write(b'x'*100000); sys.stdout.flush(); "
        "data=sys.stdin.buffer.read(); print(len(data), 'HF_TOKEN' in os.environ)",
        payload=b"p" * 3_000_000,
    )
    trailer = b"3000000 False" + os.linesep.encode()
    assert len(output) == 100_000 + len(trailer)
    assert output[:100_000].count(b"x") == 100_000
    assert output[100_000:] == trailer


@pytest.mark.parametrize("mode", ["timeout", "cancel", "output", "exit"])
def test_native_failures_reap_processes_and_threads(monkeypatch, mode):
    processes = []
    original = stream.subprocess.Popen

    def launch(*args, **kwargs):
        child = original(*args, **kwargs)
        processes.append(child)
        return child

    monkeypatch.setattr(stream.subprocess, "Popen", launch)
    started = time.monotonic()
    scripts = {
        "timeout": "import time; time.sleep(30)",
        "cancel": "import time; time.sleep(30)",
        "output": "import sys; sys.stdout.buffer.write(b'x'*1000000); sys.stdout.flush()",
        "exit": "import sys; sys.stderr.write('synthetic identifier'); sys.exit(7)",
    }
    errors = {
        "timeout": "ocr_timeout",
        "cancel": "ocr_cancelled",
        "output": "ocr_output_limit",
        "exit": "ocr_execution_failed",
    }
    with pytest.raises(RuntimeError, match="^" + errors[mode] + "$"):
        execute(
            scripts[mode],
            payload=b"p" * 3_000_000,
            timeout=0.1 if mode == "timeout" else 3,
            limit=100 if mode == "output" else 200_000,
            cancel=(lambda: time.monotonic() - started > 0.1)
            if mode == "cancel"
            else None,
        )
    assert time.monotonic() - started < 3
    assert all(process.poll() is not None for process in processes)
    assert not any(
        thread.name.startswith("openmed-ocr-") for thread in threading.enumerate()
    )
    assert execute("print('recovered')") == b"recovered" + os.linesep.encode()


def test_cancelled_work_never_spawns(monkeypatch):
    monkeypatch.setattr(
        stream.subprocess, "Popen", lambda *a, **k: pytest.fail("must not spawn")
    )
    with pytest.raises(RuntimeError, match="ocr_cancelled"):
        execute("pass", cancel=lambda: True)


@pytest.mark.parametrize("name", ["openmed-ocr-input", "openmed-ocr-watchdog"])
def test_thread_start_failure_reaps_native_process(monkeypatch, name):
    original_start = threading.Thread.start
    original_popen = stream.subprocess.Popen
    processes = []

    def launch(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        processes.append(child)
        return child

    def start(thread):
        if thread.name == name:
            raise RuntimeError("synthetic thread failure")
        original_start(thread)

    monkeypatch.setattr(stream.subprocess, "Popen", launch)
    monkeypatch.setattr(threading.Thread, "start", start)
    with pytest.raises(RuntimeError, match="^ocr_process_unavailable$"):
        execute("import time; time.sleep(30)", payload=b"p" * 3_000_000)
    assert len(processes) == 1 and processes[0].poll() is not None
    assert not any(
        thread.name.startswith("openmed-ocr-") for thread in threading.enumerate()
    )


def test_broken_cancel_callback_stops_owned_process():
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise ValueError("synthetic callback error")
        return False

    with pytest.raises(RuntimeError, match="^ocr_cancelled$"):
        execute("import time; time.sleep(30)", cancel=cancelled)


def test_alpha_is_composited_on_white_without_mutating_source():
    Image = pytest.importorskip("PIL.Image")
    with Image.new("RGBA", (10, 10), (0, 0, 0, 0)) as source:
        source.putpixel((5, 5), (0, 0, 0, 255))
        encoded, _, _ = stream._png(source, max_pixels=100)
        with Image.open(BytesIO(encoded)) as output:
            assert output.getpixel((0, 0)) == (255, 255, 255)
            assert output.getpixel((5, 5)) == (0, 0, 0)
        assert source.getpixel((0, 0)) == (0, 0, 0, 0)


def test_streaming_engine_sends_png_bytes_and_explicit_language_controls(monkeypatch):
    Image = pytest.importorskip("PIL.Image")
    image = Image.new("RGB", (200, 150), "white")
    monkeypatch.setattr(stream.shutil, "which", lambda name: "/trusted/tesseract")
    calls = []

    def run(command, payload, **kwargs):
        assert payload.startswith(b"\x89PNG")
        assert Image.open(BytesIO(payload)).size == image.size
        calls.append(command)
        return (HEADER + "\n".join(ROWS)).encode()

    monkeypatch.setattr(stream, "_execute", run)
    result = ocr(
        image,
        engine=stream.StreamingTesseractEngine(tessdata_dir="/trusted/data"),
        languages=["de", "en"],
    )
    assert result.text == "Befund: Keine Dyspnoe."
    assert calls == [
        [
            "/trusted/tesseract",
            "stdin",
            "stdout",
            "-l",
            "deu+eng",
            "--tessdata-dir",
            "/trusted/data",
            "-c",
            "tessedit_write_images=0",
            "tsv",
        ]
    ]
    assert image.getpixel((0, 0)) == (255, 255, 255)
    image.close()


def test_image_limits_are_checked_before_ocr_and_failures_are_safe(monkeypatch):
    Image = pytest.importorskip("PIL.Image")
    monkeypatch.setattr(stream.shutil, "which", lambda name: "/trusted/tesseract")
    monkeypatch.setattr(
        stream, "_execute", lambda *a, **k: pytest.fail("must not start OCR")
    )
    with Image.new("RGB", (20, 20)) as image:
        with pytest.raises(ValueError, match="ocr_pixel_limit"):
            stream.StreamingTesseractEngine(max_pixels=100).recognize(image)
    with pytest.raises(ValueError, match="ocr_invalid_image"):
        stream.StreamingTesseractEngine().recognize(
            b"not an image: synthetic private value"
        )
    with pytest.raises(ValueError, match="ocr_unsupported_language"):
        stream.StreamingTesseractEngine().recognize(b"", languages=["../../private"])


@pytest.mark.parametrize(
    "options",
    [
        {"timeout_seconds": True},
        {"timeout_seconds": float("nan")},
        {"timeout_seconds": 121},
        {"max_pixels": False},
        {"max_words": 0},
        {"max_output_bytes": -1},
        {"cancel_check": "invalid"},
    ],
)
def test_invalid_runtime_limits_fail_before_work(options):
    with pytest.raises(ValueError):
        stream.StreamingTesseractEngine(**options)
