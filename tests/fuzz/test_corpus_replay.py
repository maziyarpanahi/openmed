"""Fast deterministic replay for the adversarial de-identification corpus."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

from openmed.core.custom_recognizer import (
    MAX_CUSTOM_RECOGNIZER_CONFIG_BYTES,
    MAX_CUSTOM_RECOGNIZER_RULE_BYTES,
    MAX_CUSTOM_RECOGNIZER_RULES,
    CustomRecognizer,
    CustomRecognizerInputError,
    CustomRecognizerLimitError,
)
from openmed.core.pii import DeidentificationResult
from openmed.core.script_detect import normalize_for_pii_detection
from openmed.processing.text import (
    MAX_PII_COMBINING_SEQUENCE,
    MAX_PII_CONTROL_SEQUENCE,
    MAX_PII_FORMAT_SEQUENCE,
    InputComplexityError,
    InputEncodingError,
    InputError,
    InputSizeError,
    normalize_indic_text,
    validate_pii_input,
)
from openmed.processing.tokenization import (
    grapheme_tokenize,
    indic_word_tokenize,
    medical_tokenize,
)

from .fuzz_deidentify import (
    CORPUS_DIRECTORY,
    decode_fuzz_case,
    exercise_fuzz_input,
    iter_corpus_files,
    replay_corpus,
)

CORPUS_FILES = iter_corpus_files()


@pytest.mark.parametrize(
    "corpus_path",
    CORPUS_FILES,
    ids=[path.stem for path in CORPUS_FILES],
)
def test_corpus_case_returns_valid_result_or_typed_input_error(
    corpus_path: Path,
) -> None:
    data = corpus_path.read_bytes()
    try:
        result = exercise_fuzz_input(data)
    except InputError:
        return
    assert isinstance(result, DeidentificationResult)


def test_corpus_covers_every_required_adversarial_shape() -> None:
    assert {path.stem for path in CORPUS_FILES} == {
        "c0_c1_controls",
        "mixed_rtl_ltr",
        "oversized_custom_dictionary",
        "ten_megabytes",
        "truncated_utf8",
        "zero_width_joiners",
    }


@pytest.mark.parametrize(
    "corpus_path",
    CORPUS_FILES,
    ids=[f"{path.stem}-crlf" for path in CORPUS_FILES],
)
def test_corpus_directives_decode_identically_with_crlf(corpus_path: Path) -> None:
    data = corpus_path.read_bytes()

    assert decode_fuzz_case(data.replace(b"\n", b"\r\n")) == decode_fuzz_case(data)


@pytest.mark.timeout(2)
def test_ten_megabyte_case_rejects_before_pipeline_allocation() -> None:
    data = (CORPUS_DIRECTORY / "ten_megabytes.case").read_bytes()
    assert len(decode_fuzz_case(data).text) == 10 * 1024 * 1024

    with pytest.raises(InputSizeError):
        exercise_fuzz_input(data)


def test_truncated_utf8_raises_typed_content_free_error() -> None:
    data = (CORPUS_DIRECTORY / "truncated_utf8.case").read_bytes()

    with pytest.raises(InputEncodingError) as exc_info:
        exercise_fuzz_input(data)

    assert "utf-8" in str(exc_info.value).lower()
    assert "f09f92" not in str(exc_info.value).lower()


def test_oversized_custom_dictionary_raises_shared_input_error() -> None:
    data = (CORPUS_DIRECTORY / "oversized_custom_dictionary.case").read_bytes()

    with pytest.raises(CustomRecognizerLimitError):
        exercise_fuzz_input(data)


@pytest.mark.parametrize(
    "payload",
    [
        "\u0300" * (MAX_PII_COMBINING_SEQUENCE + 1),
        "\u200d" * (MAX_PII_FORMAT_SEQUENCE + 1),
        "\x00" * (MAX_PII_CONTROL_SEQUENCE + 1),
    ],
    ids=["combining", "format", "control"],
)
def test_unicode_complexity_guards_are_shared_by_normalizers_and_tokenizers(
    payload: str,
) -> None:
    entry_points = (
        validate_pii_input,
        normalize_for_pii_detection,
        normalize_indic_text,
        grapheme_tokenize,
        indic_word_tokenize,
        medical_tokenize,
    )

    for entry_point in entry_points:
        with pytest.raises(InputComplexityError):
            entry_point(payload)


def test_custom_recognizer_rejects_work_and_rule_size_before_scanning() -> None:
    with pytest.raises(CustomRecognizerLimitError):
        CustomRecognizer(
            deny_terms=[
                {"term": f"SyntheticRule{index}", "label": "NAME"}
                for index in range(MAX_CUSTOM_RECOGNIZER_RULES + 1)
            ]
        )

    with pytest.raises(CustomRecognizerLimitError):
        CustomRecognizer(
            deny_terms=[
                {
                    "term": "x" * (MAX_CUSTOM_RECOGNIZER_RULE_BYTES + 1),
                    "label": "NAME",
                }
            ]
        )

    recognizer = CustomRecognizer(
        deny_terms=[
            {"term": f"SyntheticRule{index}", "label": "NAME"}
            for index in range(MAX_CUSTOM_RECOGNIZER_RULES)
        ]
    )
    with pytest.raises(CustomRecognizerLimitError):
        recognizer.detect_entities("x" * 32_769)


def test_custom_recognizer_file_reads_are_bounded_and_malformed_json_is_typed(
    tmp_path: Path,
) -> None:
    oversized_path = tmp_path / "oversized.json"
    oversized_path.write_bytes(b"x" * (MAX_CUSTOM_RECOGNIZER_CONFIG_BYTES + 1))
    with pytest.raises(CustomRecognizerLimitError):
        CustomRecognizer.from_config(oversized_path)

    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text('{"deny_terms": [', encoding="utf-8")
    with pytest.raises(CustomRecognizerInputError):
        CustomRecognizer.from_config(malformed_path)


def test_observation_log_contains_only_length_hash_and_outcome(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    secret = b"SyntheticPrivateMarker"
    data = b"OPENMED_FUZZ_V1\ntext\n" + secret
    (tmp_path / "synthetic.case").write_bytes(data)

    with caplog.at_level(logging.INFO, logger="tests.fuzz.fuzz_deidentify"):
        observations = replay_corpus(tmp_path)

    assert len(observations) == 1
    observation = observations[0]
    assert secret.decode("ascii") not in caplog.text
    assert re.fullmatch(r"[0-9a-f]{64}", observation.sha256)
    assert observation.outcome in {"valid", "rejected:input_rejected"}
    assert caplog.messages == [
        "fuzz_case "
        f"length_bytes={len(secret)} "
        f"sha256={observation.sha256} "
        f"outcome={observation.outcome}"
    ]
