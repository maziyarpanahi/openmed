"""Offline translation and back-translation augmentation for span NER data."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final, Iterable, Mapping, Protocol, Sequence

from .offset_projection import (
    SpanAnnotation,
    SpanProjectionError,
    normalize_span_annotations,
    realign_translated_spans,
    validate_span_integrity,
)

SYNTHETIC_SOURCE: Final = "translation_backtranslation"
DEFAULT_TARGET_LANGUAGES: Final[tuple[str, ...]] = ("hi", "te")
EVAL_SPLIT_NAMES: Final[frozenset[str]] = frozenset(
    {"dev", "eval", "heldout", "test", "validation"}
)

Lexicon = Mapping[tuple[str, str], Sequence[tuple[str, str]]]
TranslateCallable = Callable[[str, str, str], str]

_DEFAULT_LEXICON: Final[dict[tuple[str, str], tuple[tuple[str, str], ...]]] = {
    ("en", "hi"): (
        ("chest pain", "सीने में दर्द"),
        ("high blood pressure", "उच्च रक्तचाप"),
        ("metformin", "मेटफॉर्मिन"),
        ("aspirin", "एस्पिरिन"),
        ("diabetes", "मधुमेह"),
        ("hypertension", "उच्च रक्तचाप"),
        ("fever", "बुखार"),
        ("cough", "खांसी"),
        ("patient", "रोगी"),
        ("reports", "बताता है"),
        ("takes", "लेता है"),
        ("uses", "उपयोग करता है"),
        ("has", "है"),
        ("with", "साथ"),
        ("and", "और"),
    ),
    ("hi", "en"): (
        ("सीने में दर्द", "chest pain"),
        ("उच्च रक्तचाप", "hypertension"),
        ("मेटफॉर्मिन", "metformin"),
        ("एस्पिरिन", "aspirin"),
        ("मधुमेह", "diabetes"),
        ("बुखार", "fever"),
        ("खांसी", "cough"),
        ("उपयोग करता है", "uses"),
        ("लेता है", "uses"),
        ("बताता है", "reports"),
        ("रोगी", "case"),
        ("है", "shows"),
        ("साथ", "with"),
        ("और", "and"),
    ),
    ("en", "te"): (
        ("chest pain", "ఛాతి నొప్పి"),
        ("high blood pressure", "అధిక రక్తపోటు"),
        ("metformin", "మెట్ఫార్మిన్"),
        ("aspirin", "ఆస్పిరిన్"),
        ("diabetes", "మధుమేహం"),
        ("hypertension", "అధిక రక్తపోటు"),
        ("fever", "జ్వరం"),
        ("cough", "దగ్గు"),
        ("patient", "రోగి"),
        ("reports", "తెలియజేస్తాడు"),
        ("takes", "తీసుకుంటాడు"),
        ("uses", "ఉపయోగిస్తాడు"),
        ("has", "ఉంది"),
        ("with", "తో"),
        ("and", "మరియు"),
    ),
    ("te", "en"): (
        ("ఛాతి నొప్పి", "chest pain"),
        ("అధిక రక్తపోటు", "hypertension"),
        ("మెట్ఫార్మిన్", "metformin"),
        ("ఆస్పిరిన్", "aspirin"),
        ("మధుమేహం", "diabetes"),
        ("జ్వరం", "fever"),
        ("దగ్గు", "cough"),
        ("ఉపయోగిస్తాడు", "uses"),
        ("తీసుకుంటాడు", "uses"),
        ("తెలియజేస్తాడు", "reports"),
        ("రోగి", "case"),
        ("ఉంది", "shows"),
        ("తో", "with"),
        ("మరియు", "and"),
    ),
}


class Translator(Protocol):
    """Translation backend contract used by the augmentation pipeline."""

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate ``text`` from ``source_lang`` to ``target_lang``."""


@dataclass(frozen=True)
class DictionaryTranslator:
    """Deterministic offline lexicon translator.

    The backend performs local longest-match replacement only. It never opens
    sockets, downloads models, or calls hosted translation APIs.
    """

    lexicon: Lexicon = field(default_factory=lambda: dict(_DEFAULT_LEXICON))

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text with the configured local lexicon."""

        if source_lang == target_lang:
            return text
        result = text
        entries = self.lexicon.get((source_lang, target_lang), ())
        for source, target in sorted(
            entries, key=lambda item: len(item[0]), reverse=True
        ):
            result = _replace_lexeme(result, source, target)
        return result


@dataclass(frozen=True)
class ModelBackedTranslator:
    """Adapter for user-supplied local translation models.

    The caller owns model loading. This adapter only invokes the supplied
    callable and therefore does not introduce network behavior by itself.
    """

    translate_fn: TranslateCallable
    backend_name: str = "model"

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text with the injected model callable."""

        return str(self.translate_fn(text, source_lang, target_lang))


@dataclass(frozen=True)
class TranslationAugmentedExample:
    """One translated or back-translated span-annotated training example."""

    example_id: str
    text: str
    gold_spans: tuple[SpanAnnotation, ...]
    language: str
    source_language: str
    source_id: str
    transform: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_training_item(self) -> dict[str, Any]:
        """Return a JSONL-ready item compatible with train and eval loaders."""

        spans = [span.to_dict() for span in self.gold_spans]
        metadata = dict(self.metadata)
        metadata.update(
            {
                "augmentation_only": True,
                "language": self.language,
                "source_id": self.source_id,
                "source_language": self.source_language,
                "synthetic": True,
                "synthetic_source": SYNTHETIC_SOURCE,
                "transform": self.transform,
            }
        )
        return {
            "gold_spans": spans,
            "id": self.example_id,
            "is_synthetic": True,
            "labels": spans,
            "language": self.language,
            "metadata": metadata,
            "source_id": self.source_id,
            "source_language": self.source_language,
            "synthetic_source": SYNTHETIC_SOURCE,
            "text": self.text,
        }


def augment_span_annotated_examples(
    examples: Iterable[Mapping[str, Any]],
    *,
    translator: Translator | None = None,
    target_languages: Sequence[str] = DEFAULT_TARGET_LANGUAGES,
    pivot_language: str = "en",
    include_translations: bool = True,
    include_backtranslations: bool = True,
    heldout_eval_ids: Iterable[str] = (),
    heldout_eval_texts: Iterable[str] = (),
    drop_identical: bool = True,
) -> tuple[TranslationAugmentedExample, ...]:
    """Create translated and round-trip back-translated NER examples.

    Eval-derived seeds are skipped before augmentation, and generated examples
    are filtered against held-out text hashes before they can enter training.
    """

    backend = translator or DictionaryTranslator()
    heldout_ids = {str(value) for value in heldout_eval_ids}
    heldout_text_values = {_normalized_text(str(value)) for value in heldout_eval_texts}
    heldout_hashes = {_normalized_text_hash(value) for value in heldout_text_values}
    augmented: list[TranslationAugmentedExample] = []
    seen: set[tuple[Any, ...]] = set()

    for index, raw_example in enumerate(examples):
        seed = _normalize_example(raw_example, fallback_id=f"seed-{index}")
        if _is_eval_derived(seed, heldout_ids, heldout_text_values, heldout_hashes):
            continue

        for target_language in target_languages:
            if target_language == seed.language:
                continue

            translated = _safe_translate_example(
                seed,
                translator=backend,
                target_language=target_language,
                transform=f"translation:{seed.language}->{target_language}",
            )
            if include_translations and translated is not None:
                _append_if_trainable(
                    translated,
                    seed=seed,
                    seen=seen,
                    output=augmented,
                    heldout_texts=heldout_text_values,
                    heldout_hashes=heldout_hashes,
                    drop_identical=drop_identical,
                )

            if not include_backtranslations or translated is None:
                continue
            backtranslation_target = seed.language or pivot_language
            backtranslated = _safe_translate_example(
                _SeedExample(
                    example_id=translated.example_id,
                    text=translated.text,
                    spans=translated.gold_spans,
                    language=translated.language,
                    metadata=translated.metadata,
                ),
                translator=backend,
                target_language=backtranslation_target,
                transform=(
                    f"backtranslation:{seed.language}->{target_language}->"
                    f"{backtranslation_target}"
                ),
                root_source_id=seed.example_id,
                root_source_language=seed.language,
                root_source_text_hash=_text_hash(seed.text),
            )
            if backtranslated is not None:
                _append_if_trainable(
                    backtranslated,
                    seed=seed,
                    seen=seen,
                    output=augmented,
                    heldout_texts=heldout_text_values,
                    heldout_hashes=heldout_hashes,
                    drop_identical=drop_identical,
                )

    return tuple(augmented)


def load_span_jsonl(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Load JSONL span examples from disk."""

    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("span JSONL row must be an object")
            rows.append(payload)
    return tuple(rows)


def write_augmented_jsonl(
    examples: Iterable[TranslationAugmentedExample],
    path: str | Path,
) -> None:
    """Write augmented examples as newline-delimited JSON."""

    rows = [
        json.dumps(example.to_training_item(), ensure_ascii=False, sort_keys=True)
        for example in examples
    ]
    Path(path).write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


@dataclass(frozen=True)
class _SeedExample:
    example_id: str
    text: str
    spans: tuple[SpanAnnotation, ...]
    language: str
    metadata: Mapping[str, Any]


def _normalize_example(
    raw_example: Mapping[str, Any],
    *,
    fallback_id: str,
) -> _SeedExample:
    text = str(raw_example.get("text", ""))
    if not text:
        raise ValueError("span example text is required")
    language = str(raw_example.get("language") or raw_example.get("lang") or "en")
    spans = normalize_span_annotations(raw_example, source_text=text)
    validate_span_integrity(text, spans)
    metadata = raw_example.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        raise ValueError("span example metadata must be a mapping")
    return _SeedExample(
        example_id=str(
            raw_example.get("id") or raw_example.get("fixture_id") or fallback_id
        ),
        text=text,
        spans=spans,
        language=language,
        metadata=dict(metadata),
    )


def _safe_translate_example(
    seed: _SeedExample,
    *,
    translator: Translator,
    target_language: str,
    transform: str,
    root_source_id: str | None = None,
    root_source_language: str | None = None,
    root_source_text_hash: str | None = None,
) -> TranslationAugmentedExample | None:
    try:
        translated_text = translator.translate(
            seed.text,
            seed.language,
            target_language,
        ).strip()
        if not translated_text:
            return None
        projected = realign_translated_spans(
            source_text=seed.text,
            translated_text=translated_text,
            source_spans=seed.spans,
            translator=translator,
            source_language=seed.language,
            target_language=target_language,
        )
    except SpanProjectionError:
        return None

    source_id = root_source_id or seed.example_id
    source_language = root_source_language or seed.language
    source_text_hash = root_source_text_hash or _text_hash(seed.text)
    metadata = {
        "contains_real_phi": bool(seed.metadata.get("contains_real_phi", False)),
        "provenance": {
            "source_id": source_id,
            "source_language": source_language,
            "source_text_hash": source_text_hash,
            "transform": transform,
        },
        "source_text_hash": source_text_hash,
    }
    example_id = _augmented_id(source_id, target_language, transform, translated_text)
    return TranslationAugmentedExample(
        example_id=example_id,
        text=translated_text,
        gold_spans=projected,
        language=target_language,
        source_language=source_language,
        source_id=source_id,
        transform=transform,
        metadata=metadata,
    )


def _append_if_trainable(
    example: TranslationAugmentedExample,
    *,
    seed: _SeedExample,
    seen: set[tuple[Any, ...]],
    output: list[TranslationAugmentedExample],
    heldout_texts: set[str],
    heldout_hashes: set[str],
    drop_identical: bool,
) -> None:
    if drop_identical and _normalized_text(example.text) == _normalized_text(seed.text):
        return
    normalized_text = _normalized_text(example.text)
    if (
        normalized_text in heldout_texts
        or _normalized_text_hash(normalized_text) in heldout_hashes
    ):
        return
    key = _dedupe_key(example)
    if key in seen:
        return
    seen.add(key)
    output.append(example)


def _is_eval_derived(
    seed: _SeedExample,
    heldout_ids: set[str],
    heldout_texts: set[str],
    heldout_hashes: set[str],
) -> bool:
    split = str(seed.metadata.get("split") or seed.metadata.get("source_split") or "")
    if split.casefold() in EVAL_SPLIT_NAMES:
        return True
    if seed.example_id in heldout_ids:
        return True
    normalized_text = _normalized_text(seed.text)
    if normalized_text in heldout_texts:
        return True
    return _normalized_text_hash(normalized_text) in heldout_hashes


def _dedupe_key(example: TranslationAugmentedExample) -> tuple[Any, ...]:
    return (
        example.language,
        _normalized_text(example.text),
        tuple((span.label, span.text.casefold()) for span in example.gold_spans),
    )


def _augmented_id(
    source_id: str,
    language: str,
    transform: str,
    text: str,
) -> str:
    digest = hashlib.blake2b(
        f"{source_id}|{language}|{transform}|{text}".encode("utf-8"),
        digest_size=6,
    ).hexdigest()
    safe_transform = re.sub(r"[^a-z0-9]+", "-", transform.casefold()).strip("-")
    return f"{source_id}-{safe_transform}-{digest}"


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalized_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _normalized_text_hash(text: str) -> str:
    return _text_hash(_normalized_text(text))


def _replace_lexeme(text: str, source: str, target: str) -> str:
    flags = re.IGNORECASE if source.isascii() else 0
    pattern = re.compile(
        rf"(?<!\w){re.escape(source)}(?!\w)",
        flags=flags | re.UNICODE,
    )
    return pattern.sub(target, text)


__all__ = [
    "DEFAULT_TARGET_LANGUAGES",
    "EVAL_SPLIT_NAMES",
    "SYNTHETIC_SOURCE",
    "DictionaryTranslator",
    "ModelBackedTranslator",
    "TranslationAugmentedExample",
    "Translator",
    "augment_span_annotated_examples",
    "load_span_jsonl",
    "write_augmented_jsonl",
]
