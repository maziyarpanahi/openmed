"""Synthetic Chinese and Indic throughput benchmark.

The benchmark intentionally uses only local segmentation and deterministic
PII-pattern paths. It never loads model weights and emits aggregate counts,
durations, and corpus hashes rather than source text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

I18N_THROUGHPUT_ARTIFACT = "openmed.eval.i18n_throughput"
I18N_THROUGHPUT_SCHEMA_VERSION = 1
I18N_THROUGHPUT_LANGUAGES = ("zh", "hi", "ta")
I18N_THROUGHPUT_MIN_CHARS = 100_000
I18N_THROUGHPUT_MAX_SECONDS = 300.0
I18N_THROUGHPUT_DEFAULT_ITERATIONS = 1
I18N_THROUGHPUT_FIXTURE_SCHEMA_VERSION = 1
I18N_THROUGHPUT_DEID_CHUNK_CHARS = 4_096

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE_DIR = _REPO_ROOT / "tests" / "fixtures" / "i18n"
_FIXTURE_NAMES = {
    language: f"{language}_throughput.json" for language in I18N_THROUGHPUT_LANGUAGES
}
_FAKER_LOCALES = {"zh": "zh_CN", "hi": "hi_IN", "ta": "ta_IN"}
_FAKER_SEEDS = {"zh": 698_001, "hi": 698_002, "ta": 698_003}

Segmenter = Callable[[str], Any]
SegmenterFactory = Callable[[], Segmenter]
Deidentifier = Callable[[str], int]
DeidentifierFactory = Callable[[], Deidentifier]


class _PatternOnlyLoader:
    """Model-loader adapter that deliberately produces no model predictions."""

    config = None

    def create_pipeline(self, model_name: str, **kwargs: Any) -> Callable[..., Any]:
        del model_name, kwargs

        def empty_pipeline(text: str | list[str], **call_kwargs: Any) -> Any:
            del call_kwargs
            if isinstance(text, list):
                return [[] for _ in text]
            return []

        return empty_pipeline

    def get_max_sequence_length(
        self,
        model_name: str,
        tokenizer: Any = None,
    ) -> int:
        del model_name, tokenizer
        return 1_000_000


def generate_synthetic_corpus(
    language: str,
    *,
    target_chars: int = I18N_THROUGHPUT_MIN_CHARS,
    seed: int | None = None,
) -> dict[str, Any]:
    """Generate a deterministic Faker corpus for one benchmark language.

    Args:
        language: One of ``zh``, ``hi``, or ``ta``.
        target_chars: Minimum source-code-point count.
        seed: Optional Faker seed. The committed language seed is the default.

    Returns:
        A JSON-serializable fixture containing only synthetic text and
        reproducibility metadata.
    """

    language = _require_language(language)
    if target_chars < 1:
        raise ValueError("target_chars must be positive")

    from faker import Faker

    from openmed.core.anonymizer.providers.clinical_ids import (
        register_clinical_providers,
    )

    fixture_seed = _FAKER_SEEDS[language] if seed is None else int(seed)
    faker = Faker(_FAKER_LOCALES[language])
    register_clinical_providers(faker)
    faker.seed_instance(fixture_seed)

    records: list[str] = []
    char_count = 0
    record_index = 0
    while char_count < target_chars:
        record = _synthetic_record(language, faker, record_index)
        records.append(record)
        char_count += len(record) + int(bool(records[:-1]))
        record_index += 1

    text = "\n".join(records)
    text_sha256 = _sha256_text(text)
    return {
        "schema_version": I18N_THROUGHPUT_FIXTURE_SCHEMA_VERSION,
        "language": language,
        "metadata": {
            "synthetic": True,
            "generated_only": True,
            "generator": "Faker",
            "faker_version": version("faker"),
            "faker_locale": _FAKER_LOCALES[language],
            "seed": fixture_seed,
            "target_chars": target_chars,
            "char_count": len(text),
            "record_count": len(records),
            "sha256": text_sha256,
        },
        "text": text,
    }


def write_synthetic_corpora(
    output_dir: str | Path = DEFAULT_FIXTURE_DIR,
    *,
    target_chars: int = I18N_THROUGHPUT_MIN_CHARS,
) -> tuple[Path, ...]:
    """Write all deterministic throughput corpora as reviewed JSON fixtures."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for language in I18N_THROUGHPUT_LANGUAGES:
        path = destination / _FIXTURE_NAMES[language]
        path.write_text(
            json.dumps(
                generate_synthetic_corpus(language, target_chars=target_chars),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        paths.append(path)
    return tuple(paths)


def load_synthetic_corpus(
    language: str,
    *,
    fixture_dir: str | Path = DEFAULT_FIXTURE_DIR,
) -> dict[str, Any]:
    """Load and validate one committed synthetic benchmark corpus."""

    language = _require_language(language)
    path = Path(fixture_dir) / _FIXTURE_NAMES[language]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"benchmark fixture is unavailable: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"benchmark fixture is not valid JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"benchmark fixture must be a JSON object: {path}")
    if payload.get("schema_version") != I18N_THROUGHPUT_FIXTURE_SCHEMA_VERSION:
        raise ValueError(f"benchmark fixture has an unsupported schema: {path}")
    if payload.get("language") != language:
        raise ValueError(f"benchmark fixture language mismatch: {path}")

    metadata = payload.get("metadata")
    text = payload.get("text")
    if not isinstance(metadata, Mapping) or metadata.get("synthetic") is not True:
        raise ValueError(f"benchmark fixture must be marked synthetic: {path}")
    if metadata.get("generated_only") is not True:
        raise ValueError(f"benchmark fixture must be generated-only: {path}")
    if metadata.get("generator") != "Faker":
        raise ValueError(f"benchmark fixture must record Faker provenance: {path}")
    if not isinstance(text, str) or len(text) < I18N_THROUGHPUT_MIN_CHARS:
        raise ValueError(
            f"benchmark fixture must contain at least "
            f"{I18N_THROUGHPUT_MIN_CHARS} characters: {path}"
        )
    if metadata.get("char_count") != len(text):
        raise ValueError(f"benchmark fixture char_count mismatch: {path}")
    if metadata.get("sha256") != _sha256_text(text):
        raise ValueError(f"benchmark fixture hash mismatch: {path}")
    return {
        "language": language,
        "path": str(path),
        "text": text,
        "metadata": dict(metadata),
    }


def benchmark_language(
    language: str,
    corpus: str,
    *,
    iterations: int = I18N_THROUGHPUT_DEFAULT_ITERATIONS,
    segmenter_factory: SegmenterFactory | None = None,
    deidentifier_factory: DeidentifierFactory | None = None,
) -> dict[str, Any]:
    """Measure cold-start and steady-state throughput for one language."""

    language = _require_language(language)
    if not corpus:
        raise ValueError("corpus must not be empty")
    if iterations < 1:
        raise ValueError("iterations must be positive")

    build_segmenter = segmenter_factory or _segmenter_factory(language)
    build_deidentifier = deidentifier_factory or _deidentifier_factory(language)
    cold_start_sample = next(
        _iter_corpus_chunks(
            corpus,
            max_chars=I18N_THROUGHPUT_DEID_CHUNK_CHARS,
        )
    )

    cold_started = time.perf_counter()
    segmenter = build_segmenter()
    cold_token_count = _result_count(segmenter(cold_start_sample))
    segmentation_cold_ms = (time.perf_counter() - cold_started) * 1000.0

    steady_started = time.perf_counter()
    steady_token_count = 0
    for _ in range(iterations):
        steady_token_count += _result_count(segmenter(corpus))
    segmentation_seconds = time.perf_counter() - steady_started

    cold_started = time.perf_counter()
    deidentifier = build_deidentifier()
    cold_span_count = _result_count(deidentifier(cold_start_sample))
    deidentify_cold_ms = (time.perf_counter() - cold_started) * 1000.0

    steady_started = time.perf_counter()
    steady_span_count = 0
    for _ in range(iterations):
        steady_span_count += _result_count(deidentifier(corpus))
    deidentify_seconds = time.perf_counter() - steady_started

    if cold_token_count < 1 or steady_token_count < 1:
        raise RuntimeError(f"{language} segmenter emitted no tokens")
    if cold_span_count < 1 or steady_span_count < 1:
        raise RuntimeError(f"{language} pattern de-identification emitted no spans")

    return {
        "char_count": len(corpus),
        "cold_start_char_count": len(cold_start_sample),
        "deidentify_chunk_chars": I18N_THROUGHPUT_DEID_CHUNK_CHARS,
        "iterations": iterations,
        "segmentation_cold_start_ms": round(segmentation_cold_ms, 3),
        "segmentation_chars_per_second": round(
            (len(corpus) * iterations) / max(segmentation_seconds, 1e-12),
            3,
        ),
        "segmentation_token_count": steady_token_count // iterations,
        "deidentify_cold_start_ms": round(deidentify_cold_ms, 3),
        "deidentify_spans_per_second": round(
            steady_span_count / max(deidentify_seconds, 1e-12),
            3,
        ),
        "deidentify_span_count": steady_span_count // iterations,
    }


def run_benchmark(
    *,
    fixture_dir: str | Path = DEFAULT_FIXTURE_DIR,
    languages: Sequence[str] = I18N_THROUGHPUT_LANGUAGES,
    iterations: int = I18N_THROUGHPUT_DEFAULT_ITERATIONS,
    max_duration_seconds: float = I18N_THROUGHPUT_MAX_SECONDS,
) -> dict[str, Any]:
    """Run the complete aggregate-only benchmark and enforce its time budget."""

    if not languages:
        raise ValueError("at least one language is required")
    if max_duration_seconds <= 0:
        raise ValueError("max_duration_seconds must be positive")

    started = time.perf_counter()
    results: dict[str, Any] = {}
    corpora: dict[str, Any] = {}
    for language_value in languages:
        language = _require_language(language_value)
        fixture = load_synthetic_corpus(language, fixture_dir=fixture_dir)
        metadata = fixture["metadata"]
        corpora[language] = {
            "path": fixture["path"],
            "synthetic": True,
            "generator": metadata["generator"],
            "faker_locale": metadata["faker_locale"],
            "seed": metadata["seed"],
            "char_count": metadata["char_count"],
            "sha256": metadata["sha256"],
        }
        results[language] = benchmark_language(
            language,
            fixture["text"],
            iterations=iterations,
        )

    duration_seconds = time.perf_counter() - started
    if duration_seconds >= max_duration_seconds:
        raise RuntimeError(
            f"i18n throughput benchmark took {duration_seconds:.3f}s; "
            f"limit is {max_duration_seconds:.3f}s"
        )

    return {
        "schema_version": I18N_THROUGHPUT_SCHEMA_VERSION,
        "artifact_type": I18N_THROUGHPUT_ARTIFACT,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "duration_seconds": round(duration_seconds, 3),
        "max_duration_seconds": float(max_duration_seconds),
        "backend": {
            "segmentation": {
                "zh": "jieba",
                "hi": "grapheme-safe-indic",
                "ta": "grapheme-safe-indic",
            },
            "deidentify": "pattern-only",
            "model_weights_required": False,
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "corpora": corpora,
        "languages": results,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the local benchmark CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark synthetic zh/hi/ta segmentation and pattern-only "
            "de-identification throughput."
        )
    )
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=DEFAULT_FIXTURE_DIR,
        help="Directory containing the committed synthetic corpus JSON files.",
    )
    parser.add_argument(
        "--language",
        action="append",
        choices=I18N_THROUGHPUT_LANGUAGES,
        dest="languages",
        help="Language to benchmark. Repeat to select multiple languages.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=I18N_THROUGHPUT_DEFAULT_ITERATIONS,
        help="Steady-state iterations per operation and language.",
    )
    parser.add_argument(
        "--max-duration-seconds",
        type=float,
        default=I18N_THROUGHPUT_MAX_SECONDS,
        help="Fail when the complete local benchmark reaches this duration.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the machine-readable JSON report.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark CLI."""

    args = build_arg_parser().parse_args(argv)
    try:
        report = run_benchmark(
            fixture_dir=args.fixtures_dir,
            languages=args.languages or I18N_THROUGHPUT_LANGUAGES,
            iterations=args.iterations,
            max_duration_seconds=args.max_duration_seconds,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"i18n throughput benchmark failed: {exc}", file=sys.stderr)
        return 2

    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0


def _synthetic_record(language: str, faker: Any, index: int) -> str:
    name = str(faker.name()).replace("\n", " ")
    address = str(faker.address()).replace("\n", ", ")
    visit_date = faker.date_between(
        start_date="-5y",
        end_date="-1d",
    ).isoformat()
    if language == "zh":
        phone = faker.chinese_mobile_number()
        identity = faker.chinese_resident_id()
        return (
            f"合成病例{index:05d}：患者{name}，证件号码{identity}，"
            f"电话{phone}，于{visit_date}在{address}复诊。"
            "此记录完全由测试数据生成，不含真实患者信息。"
        )
    phone = faker.indian_phone_number()
    identity = faker.aadhaar()
    if language == "hi":
        return (
            f"कृत्रिम मामला {index:05d}: रोगी {name}, आधार {identity}, "
            f"फ़ोन {phone}, ने {visit_date} को {address} में जाँच कराई। "
            "यह केवल कृत्रिम परीक्षण डेटा है।"
        )
    return (
        f"செயற்கை பதிவு {index:05d}: நோயாளி {name}, ஆதார் {identity}, "
        f"தொலைபேசி {phone}, {visit_date} அன்று {address} இல் பரிசோதிக்கப்பட்டார். "
        "இது முழுவதும் செயற்கை சோதனைத் தரவு."
    )


def _segmenter_factory(language: str) -> SegmenterFactory:
    if language == "zh":

        def build_zh_segmenter() -> Segmenter:
            from openmed.processing.zh_segmentation import create_chinese_segmenter

            return create_chinese_segmenter("jieba").segment

        return build_zh_segmenter

    def build_indic_segmenter() -> Segmenter:
        from openmed.processing.tokenization import indic_word_tokenize

        return indic_word_tokenize

    return build_indic_segmenter


def _deidentifier_factory(language: str) -> DeidentifierFactory:
    def build_deidentifier() -> Deidentifier:
        from openmed.core.pii import deidentify

        loader = _PatternOnlyLoader()

        def pattern_deidentify(text: str) -> int:
            span_count = 0
            for chunk in _iter_corpus_chunks(
                text,
                max_chars=I18N_THROUGHPUT_DEID_CHUNK_CHARS,
            ):
                result = deidentify(
                    chunk,
                    model_name="pattern-only",
                    method="mask",
                    lang=language,
                    loader=loader,
                    use_smart_merging=False,
                    use_safety_sweep=True,
                )
                span_count += len(result.pii_entities)
            return span_count

        return pattern_deidentify

    return build_deidentifier


def _result_count(result: Any) -> int:
    if isinstance(result, bool):
        raise TypeError("benchmark operation count must not be boolean")
    if isinstance(result, int):
        return result
    try:
        return len(result)
    except TypeError as exc:
        raise TypeError("benchmark operation must return a countable result") from exc


def _iter_corpus_chunks(text: str, *, max_chars: int) -> Iterator[str]:
    """Yield newline-aligned chunks while covering the entire corpus."""

    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    chunk_parts: list[str] = []
    chunk_length = 0
    for line in text.splitlines(keepends=True):
        if chunk_parts and chunk_length + len(line) > max_chars:
            yield "".join(chunk_parts)
            chunk_parts = []
            chunk_length = 0
        if len(line) <= max_chars:
            chunk_parts.append(line)
            chunk_length += len(line)
            continue
        if chunk_parts:
            yield "".join(chunk_parts)
            chunk_parts = []
            chunk_length = 0
        for start in range(0, len(line), max_chars):
            yield line[start : start + max_chars]
    if chunk_parts:
        yield "".join(chunk_parts)
    elif not text:
        yield ""


def _require_language(language: str) -> str:
    normalized = str(language).strip().casefold()
    if normalized not in I18N_THROUGHPUT_LANGUAGES:
        choices = ", ".join(I18N_THROUGHPUT_LANGUAGES)
        raise ValueError(f"unsupported throughput language {language!r}; use {choices}")
    return normalized


def _sha256_text(text: str) -> str:
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


if __name__ == "__main__":
    raise SystemExit(main())
