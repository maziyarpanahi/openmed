"""Bounded property tests for multilingual byte decoding."""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from openmed.core.script_detect import (
    ALLOWED_INGESTION_ENCODINGS,
    ConfusableIngestionWarning,
    EncodingIngestionError,
    EncodingInputLimitError,
    UnsupportedIngestionEncodingError,
    decode_ingestion_bytes,
    decode_legacy_text,
)


def _fuzz_examples() -> int:
    value = os.environ.get("OPENMED_INGESTION_FUZZ_EXAMPLES", "500")
    try:
        return max(500, int(value))
    except ValueError as exc:  # pragma: no cover - explicit local misuse
        raise RuntimeError(
            "OPENMED_INGESTION_FUZZ_EXAMPLES must be an integer"
        ) from exc


FUZZ_EXAMPLES = _fuzz_examples()
FUZZ_ENCODINGS = sorted(
    ALLOWED_INGESTION_ENCODINGS
    | {"utf-7", "utf-16", "utf-32", "rot-13", "unknown-codec"}
)


def test_encoding_allow_list_rejects_ambiguous_or_executable_codecs() -> None:
    for encoding in ("utf-7", "utf-16", "utf-32", "rot-13"):
        with pytest.raises(UnsupportedIngestionEncodingError):
            decode_ingestion_bytes(b"synthetic", encoding=encoding)


def test_strict_decoder_warns_with_content_free_confusable_codes() -> None:
    text = "SyntheticP\u0430tient"

    with pytest.warns(ConfusableIngestionWarning) as warning_info:
        decoded = decode_ingestion_bytes(text.encode(), encoding="utf-8")

    assert decoded.text == text
    assert decoded.warning_codes == ("mixed_script", "confusable_characters")
    assert text not in str(warning_info[0].message)


def test_legacy_adapter_returns_only_decoded_text() -> None:
    assert decode_legacy_text(b"caf\xe9", encoding="windows-1252") == "caf\xe9"


def test_oversized_conversion_is_rejected_before_decode() -> None:
    with pytest.raises(EncodingInputLimitError):
        decode_ingestion_bytes(b"x" * 129, encoding="utf-8", max_bytes=128)


def test_encoding_rejection_logs_only_file_metadata(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "SyntheticPatientBob"
    path = tmp_path / f"{secret}-legacy.txt"
    payload = f"{secret}-payload".encode()

    with caplog.at_level(logging.WARNING, logger="openmed.core.script_detect"):
        with pytest.raises(UnsupportedIngestionEncodingError):
            decode_ingestion_bytes(
                payload,
                encoding="utf-7",
                source_path=path,
            )

    assert secret not in caplog.text
    assert str(path) not in caplog.text
    assert "size_bytes=" in caplog.text
    assert "entry_count=0" in caplog.text
    assert "reason=encoding_not_allowed" in caplog.text


@pytest.mark.fuzz
@pytest.mark.timeout(30)
@settings(max_examples=FUZZ_EXAMPLES, deadline=100, derandomize=True)
@example(data=b"\xe2\x82", encoding="utf-8")
@example(data=b"\xff", encoding="utf-16-le")
@example(data=b"x" * 129, encoding="utf-8")
@example(data="Latin\u0430Mixed".encode(), encoding="utf-8")
@example(data=b"+ADw-script+AD4-", encoding="utf-7")
@given(
    data=st.binary(max_size=160),
    encoding=st.sampled_from(FUZZ_ENCODINGS),
)
def test_encoding_conversion_fuzz_target_is_fail_closed(
    data: bytes,
    encoding: str,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConfusableIngestionWarning)
        try:
            decoded = decode_ingestion_bytes(
                data,
                encoding=encoding,
                max_bytes=128,
            )
        except EncodingIngestionError:
            return

    assert decoded.byte_length == len(data)
    assert decoded.encoding in ALLOWED_INGESTION_ENCODINGS
    assert isinstance(decoded.text, str)
    assert set(decoded.warning_codes) <= {
        "confusable_characters",
        "mixed_script",
    }
