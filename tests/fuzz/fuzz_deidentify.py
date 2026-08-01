"""Coverage-guided and deterministic fuzz target for :func:`deidentify`.

The harness is intentionally offline. A synthetic empty detector keeps the
real de-identification pipeline, normalization, deterministic recognizers, and
redaction stages in scope without loading model weights or using a network.

When Atheris is installed, running this module starts its coverage-guided
engine. Without Atheris, the same command deterministically replays the
committed corpus once. Add ``--replay`` to force deterministic replay.
"""

from __future__ import annotations

import binascii
import hashlib
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import patch

# Direct script execution does not receive pytest's repository-root path setup.
# Pin imports to this isolated checkout instead of any unrelated editable install
# that may share the active virtual environment.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

try:  # pragma: no cover - Atheris is intentionally an optional dependency.
    import atheris
except ImportError:  # pragma: no cover - deterministic CI fallback is tested.
    atheris = None  # type: ignore[assignment]

if atheris is not None:  # pragma: no cover - exercised by the local fuzz gate.
    with atheris.instrument_imports(include=["openmed"]):
        from openmed.core.custom_recognizer import MAX_CUSTOM_RECOGNIZER_RULES
        from openmed.core.pii import DeidentificationResult, deidentify
        from openmed.processing.outputs import PredictionResult
        from openmed.processing.text import InputError
else:
    from openmed.core.custom_recognizer import MAX_CUSTOM_RECOGNIZER_RULES
    from openmed.core.pii import DeidentificationResult, deidentify
    from openmed.processing.outputs import PredictionResult
    from openmed.processing.text import InputError


logger = logging.getLogger(__name__)

CORPUS_DIRECTORY = Path(__file__).with_name("corpus")
_CORPUS_HEADER = b"OPENMED_FUZZ_V1"
_MAX_GENERATED_INPUT_BYTES = 10 * 1024 * 1024
_MAX_GENERATED_CUSTOM_RULES = MAX_CUSTOM_RECOGNIZER_RULES + 1
_MODEL_NAME = "synthetic-offline-fuzz-model"


@dataclass(frozen=True)
class FuzzCase:
    """One bounded target input decoded from fuzz-engine bytes."""

    text: bytes
    custom_recognizer: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class FuzzObservation:
    """Content-free replay metadata safe to write to a test log."""

    length_bytes: int
    sha256: str
    outcome: str


def iter_corpus_files(directory: Path = CORPUS_DIRECTORY) -> tuple[Path, ...]:
    """Return the committed synthetic corpus in deterministic order."""

    return tuple(sorted(directory.glob("*.case")))


def decode_fuzz_case(data: bytes) -> FuzzCase:
    """Decode a bounded corpus directive, falling back to raw input bytes.

    Coverage-guided mutations commonly damage a seed's directive syntax. Such
    values remain useful: an invalid directive is treated as the raw byte input
    to :func:`deidentify`, so arbitrary and truncated UTF-8 still reach the
    production validation path.
    """

    header, separator, directive = _partition_line(data)
    if not separator or header != _CORPUS_HEADER:
        return FuzzCase(text=data)

    operation, separator, payload = _partition_line(directive)
    if not separator:
        return FuzzCase(text=data)

    if operation == b"text":
        return FuzzCase(text=payload)

    if operation == b"hex":
        try:
            return FuzzCase(text=binascii.unhexlify(payload.strip()))
        except (binascii.Error, ValueError):
            return FuzzCase(text=data)

    if operation == b"repeat":
        count_line, count_separator, unit_hex = _partition_line(payload)
        if not count_separator:
            return FuzzCase(text=data)
        try:
            count = int(count_line.decode("ascii"), 10)
            unit = binascii.unhexlify(unit_hex.strip())
        except (UnicodeError, ValueError, binascii.Error):
            return FuzzCase(text=data)
        if not unit:
            return FuzzCase(text=data)
        bounded_count = min(max(count, 0), _MAX_GENERATED_INPUT_BYTES)
        repeats = (bounded_count + len(unit) - 1) // len(unit)
        return FuzzCase(text=(unit * repeats)[:bounded_count])

    if operation == b"custom-rule-count":
        try:
            count = int(payload.strip().decode("ascii"), 10)
        except (UnicodeError, ValueError):
            return FuzzCase(text=data)
        bounded_count = min(max(count, 0), _MAX_GENERATED_CUSTOM_RULES)
        rules = [
            {"term": f"SyntheticRule{index}", "label": "NAME"}
            for index in range(bounded_count)
        ]
        return FuzzCase(
            text=b"Synthetic offline note",
            custom_recognizer={"deny_terms": rules},
        )

    return FuzzCase(text=data)


def _partition_line(data: bytes) -> tuple[bytes, bytes, bytes]:
    """Partition one metadata line while accepting LF and CRLF checkouts."""

    line, separator, remainder = data.partition(b"\n")
    if separator and line.endswith(b"\r"):
        line = line[:-1]
    return line, separator, remainder


def _empty_prediction(
    text: str,
    model_name: str = _MODEL_NAME,
    **_kwargs: Any,
) -> PredictionResult:
    return PredictionResult(
        text=text,
        entities=[],
        model_name=model_name,
        timestamp=datetime(2026, 1, 1).isoformat(),
    )


def exercise_fuzz_input(data: bytes) -> DeidentificationResult:
    """Run one engine input through the real, offline de-identification path.

    Expected malformed-input rejections remain visible as :class:`InputError`.
    Any other exception escapes so Atheris treats it as a crash.
    """

    case = decode_fuzz_case(data)
    with patch("openmed.analyze_text", _empty_prediction):
        result = deidentify(
            case.text,
            method="mask",
            model_name=_MODEL_NAME,
            use_smart_merging=False,
            use_safety_sweep=False,
            custom_recognizer=case.custom_recognizer,
        )
    if not isinstance(result, DeidentificationResult):
        raise TypeError("deidentify returned an unexpected result type")
    return result


def test_one_input(data: bytes) -> None:
    """Atheris entry point: accept valid results and typed input rejections."""

    try:
        exercise_fuzz_input(data)
    except InputError:
        return


def observe_fuzz_input(data: bytes) -> FuzzObservation:
    """Replay one input and return content-free diagnostics."""

    case = decode_fuzz_case(data)
    try:
        exercise_fuzz_input(data)
    except InputError as exc:
        outcome = f"rejected:{exc.reason}"
    else:
        outcome = "valid"
    return FuzzObservation(
        length_bytes=len(case.text),
        sha256=hashlib.sha256(case.text).hexdigest(),
        outcome=outcome,
    )


def replay_corpus(directory: Path = CORPUS_DIRECTORY) -> tuple[FuzzObservation, ...]:
    """Replay every committed case and log only safe outcome metadata."""

    observations = []
    for corpus_path in iter_corpus_files(directory):
        observation = observe_fuzz_input(corpus_path.read_bytes())
        logger.info(
            "fuzz_case length_bytes=%d sha256=%s outcome=%s",
            observation.length_bytes,
            observation.sha256,
            observation.outcome,
        )
        observations.append(observation)
    return tuple(observations)


def _replay_directories(arguments: list[str]) -> tuple[Path, ...]:
    directories = tuple(
        Path(argument)
        for argument in arguments
        if argument != "--replay" and not argument.startswith("-")
    )
    return directories or (CORPUS_DIRECTORY,)


def main() -> None:
    """Start Atheris when available, otherwise replay the corpus once."""

    arguments = sys.argv[1:]
    if atheris is not None and "--replay" not in arguments:
        fuzzer_arguments = list(sys.argv)
        if not any(not argument.startswith("-") for argument in arguments):
            fuzzer_arguments.append(str(CORPUS_DIRECTORY))
        atheris.Setup(fuzzer_arguments, test_one_input)
        atheris.Fuzz()
        return

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for directory in _replay_directories(arguments):
        replay_corpus(directory)


if __name__ == "__main__":  # pragma: no cover - exercised as a local harness.
    main()
