"""Pluggable Chinese word segmentation with exact source offsets.

The default backend is :mod:`jieba`, which is a lightweight core dependency.
The pkuseg and HanLP adapters are imported only when selected and never download
model files on OpenMed's behalf.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .tokenization import (
    DEFAULT_DICTIONARY_LIMITS,
    DictionaryLimits,
    SpanToken,
    UserDictionaryEntry,
)
from .tokenization import (
    load_user_dictionary as _load_bounded_user_dictionary,
)

DEFAULT_MEDICAL_USER_DICTIONARY = (
    Path(__file__).with_name("data") / "zh_medical_terms.txt"
)
SUPPORTED_CHINESE_SEGMENTATION_BACKENDS = frozenset({"jieba", "pkuseg", "hanlp"})


@runtime_checkable
class ChineseSegmenter(Protocol):
    """Protocol implemented by Chinese word-segmentation backends."""

    def segment(self, text: str) -> list[SpanToken]:
        """Return monotonic word tokens over exact Python string offsets."""


class ChineseSegmentationConfig(Protocol):
    """Configuration fields consumed by the segmenter factory."""

    chinese_segmentation_backend: str
    chinese_user_dict_path: str | None
    chinese_pkuseg_domain: str


class JiebaSegmenter:
    """Chinese segmenter backed by jieba's DAG and HMM/Viterbi implementation.

    Args:
        user_dict_path: Optional additional jieba-format dictionary. Each line is
            ``term``, ``term frequency``, or ``term frequency POS``.
        hmm: Whether jieba may use its HMM/Viterbi unknown-word model.
    """

    def __init__(
        self,
        *,
        user_dict_path: str | Path | None = None,
        hmm: bool = True,
    ) -> None:
        jieba = _import_optional_dependency(
            "jieba",
            extra=None,
            license_name="MIT",
        )
        self._tokenizer = jieba.Tokenizer()
        self._tokenizer.load_userdict(str(DEFAULT_MEDICAL_USER_DICTIONARY))
        if user_dict_path is not None:
            for entry in load_user_dictionary(user_dict_path):
                self._tokenizer.add_word(
                    entry.term,
                    freq=entry.frequency,
                    tag=entry.pos,
                )
        self._hmm = bool(hmm)

    def segment(self, text: str) -> list[SpanToken]:
        """Segment ``text`` while preserving jieba's exact code-point offsets."""

        _require_text(text)
        if not text:
            return []
        tokens = [
            SpanToken(word, start, end)
            for word, start, end in self._tokenizer.tokenize(text, HMM=self._hmm)
            if word and not word.isspace()
        ]
        validate_segmentation(text, tokens)
        return tokens


class PkusegSegmenter:
    """Chinese segmenter backed by an optional pkuseg domain model.

    Args:
        model_name: Installed pkuseg model name or user-supplied model path.
            ``"medicine"`` is the clinical default.
        user_dict_path: Optional additional user dictionary.
    """

    def __init__(
        self,
        *,
        model_name: str = "medicine",
        user_dict_path: str | Path | None = None,
    ) -> None:
        self._pkuseg = _import_optional_dependency(
            "pkuseg",
            extra="zh-pkuseg",
            license_name="MIT",
        )
        if not str(model_name).strip():
            raise ValueError("pkuseg model_name must be a non-empty name or path")
        self._model_name = str(model_name)
        entries = list(load_user_dictionary())
        if user_dict_path is not None:
            entries.extend(load_user_dictionary(user_dict_path))
        self._user_terms = _unique_terms(entries)
        self._segmenter: Any | None = None

    def segment(self, text: str) -> list[SpanToken]:
        """Segment ``text`` and align pkuseg words to exact source offsets."""

        _require_text(text)
        if not text:
            return []
        words = self._get_segmenter().cut(text)
        tokens = _align_words_to_text(text, words)
        validate_segmentation(text, tokens)
        return tokens

    def _get_segmenter(self) -> Any:
        if self._segmenter is None:
            model_name = _local_pkuseg_model(self._pkuseg, self._model_name)
            self._segmenter = self._pkuseg.pkuseg(
                model_name=model_name,
                user_dict=self._user_terms,
            )
        return self._segmenter


class HanLPSegmenter:
    """Chinese segmenter backed by an optional user-supplied HanLP model.

    Args:
        model: A preloaded callable tokenizer or a user-supplied local model path.
            OpenMed does not choose or download HanLP model weights.
        user_dict_path: Optional dictionary whose terms are applied as a
            deterministic longest-match overlay before returning tokens.
    """

    def __init__(
        self,
        *,
        model: str | Path | Callable[[str], Any] | None = None,
        user_dict_path: str | Path | None = None,
    ) -> None:
        self._hanlp = _import_optional_dependency(
            "hanlp",
            extra="zh-hanlp",
            license_name="Apache-2.0",
        )
        self._model_source = model
        self._model: Callable[[str], Any] | None = model if callable(model) else None
        entries = list(load_user_dictionary())
        if user_dict_path is not None:
            entries.extend(load_user_dictionary(user_dict_path))
        self._user_terms = _unique_terms(entries)

    def segment(self, text: str) -> list[SpanToken]:
        """Segment ``text`` and align HanLP words to exact source offsets."""

        _require_text(text)
        if not text:
            return []
        output = self._get_model()(text)
        words = _hanlp_words(output)
        tokens = _align_words_to_text(text, words)
        tokens = _apply_dictionary_overlay(text, tokens, self._user_terms)
        validate_segmentation(text, tokens)
        return tokens

    def _get_model(self) -> Callable[[str], Any]:
        if self._model is not None:
            return self._model
        if self._model_source is None:
            raise ValueError(
                "HanLP requires a preloaded tokenizer or a user-supplied local "
                "model path; OpenMed does not download model weights implicitly"
            )
        model_path = Path(self._model_source).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(
                f"HanLP model path does not exist: {model_path}. "
                "Supply an installed local model or a preloaded callable."
            )
        self._model = self._hanlp.load(str(model_path))
        return self._model


def create_chinese_segmenter(
    backend: str = "jieba",
    *,
    user_dict_path: str | Path | None = None,
    pkuseg_domain: str = "medicine",
    hanlp_model: str | Path | Callable[[str], Any] | None = None,
) -> ChineseSegmenter:
    """Create a Chinese segmenter without importing unselected backends.

    Args:
        backend: One of ``"jieba"``, ``"pkuseg"``, or ``"hanlp"``.
        user_dict_path: Optional additional user dictionary.
        pkuseg_domain: Installed pkuseg domain model name or model path.
        hanlp_model: Preloaded HanLP tokenizer or local model path.

    Returns:
        A backend implementing :class:`ChineseSegmenter`.

    Raises:
        ValueError: If ``backend`` is not supported.
        ImportError: If a selected optional backend is not installed.
    """

    normalized = str(backend).strip().lower()
    if normalized == "jieba":
        return JiebaSegmenter(user_dict_path=user_dict_path)
    if normalized == "pkuseg":
        return PkusegSegmenter(
            model_name=pkuseg_domain,
            user_dict_path=user_dict_path,
        )
    if normalized == "hanlp":
        return HanLPSegmenter(model=hanlp_model, user_dict_path=user_dict_path)
    choices = ", ".join(sorted(SUPPORTED_CHINESE_SEGMENTATION_BACKENDS))
    raise ValueError(
        f"Unsupported Chinese segmentation backend {backend!r}; use {choices}"
    )


def create_chinese_segmenter_from_config(
    config: ChineseSegmentationConfig,
    *,
    hanlp_model: str | Path | Callable[[str], Any] | None = None,
) -> ChineseSegmenter:
    """Create a segmenter from ``OpenMedConfig`` Chinese settings.

    Args:
        config: Configuration exposing the Chinese segmentation fields.
        hanlp_model: Preloaded HanLP tokenizer or local model path when HanLP is
            selected. Model weights are never downloaded implicitly.

    Returns:
        Configured Chinese segmenter backend.
    """

    return create_chinese_segmenter(
        config.chinese_segmentation_backend,
        user_dict_path=config.chinese_user_dict_path,
        pkuseg_domain=config.chinese_pkuseg_domain,
        hanlp_model=hanlp_model,
    )


def load_user_dictionary(
    path: str | Path = DEFAULT_MEDICAL_USER_DICTIONARY,
    *,
    limits: DictionaryLimits = DEFAULT_DICTIONARY_LIMITS,
) -> tuple[UserDictionaryEntry, ...]:
    """Load a jieba-format medical dictionary through bounded validation.

    Args:
        path: Plain UTF-8 dictionary or single-member ZIP archive.
        limits: Positive resource and entry limits.

    Returns:
        Parsed immutable dictionary entries.

    Raises:
        DictionaryIngestionError: If the source or an entry is unsafe.
    """
    return _load_bounded_user_dictionary(path, limits=limits)


def validate_segmentation(text: str, tokens: Sequence[SpanToken]) -> None:
    """Validate exact offsets, ordering, and non-whitespace coverage.

    Args:
        text: Original source string.
        tokens: Candidate segmentation tokens.

    Raises:
        ValueError: If a token is invalid, overlaps another token, or leaves a
            non-whitespace source code point uncovered.
    """

    _require_text(text)
    cursor = 0
    for index, token in enumerate(tokens):
        if not isinstance(token, SpanToken):
            raise ValueError(f"Segmentation token {index} is not a SpanToken")
        if token.start < cursor:
            raise ValueError(f"Segmentation token {index} overlaps or is out of order")
        if token.start < 0 or token.end <= token.start or token.end > len(text):
            raise ValueError(f"Segmentation token {index} has invalid offsets")
        if any(not char.isspace() for char in text[cursor : token.start]):
            raise ValueError(
                f"Segmentation leaves non-whitespace text uncovered before token {index}"
            )
        if text[token.start : token.end] != token.text:
            raise ValueError(f"Segmentation token {index} does not match source text")
        cursor = token.end
    if any(not char.isspace() for char in text[cursor:]):
        raise ValueError("Segmentation leaves trailing non-whitespace text uncovered")


def segmentation_boundary_f1(
    gold_tokens: Sequence[SpanToken],
    predicted_tokens: Sequence[SpanToken],
) -> float:
    """Compute word-boundary F1 on character offsets.

    The terminal document boundary is excluded because every valid segmentation
    shares it and including it would inflate short examples.

    Args:
        gold_tokens: Hand-cut reference tokens.
        predicted_tokens: Tokens returned by a segmenter.

    Returns:
        Boundary F1 in ``[0.0, 1.0]``.
    """

    gold = _interior_boundaries(gold_tokens)
    predicted = _interior_boundaries(predicted_tokens)
    if not gold and not predicted:
        return 1.0
    if not gold or not predicted:
        return 0.0
    true_positives = len(gold & predicted)
    return 2.0 * true_positives / (len(gold) + len(predicted))


def _import_optional_dependency(
    module_name: str,
    *,
    extra: str | None,
    license_name: str,
) -> Any:
    try:
        return importlib.import_module(module_name)
    except (ImportError, OSError) as exc:
        if extra is None:
            install = "Install or repair the core openmed package"
        else:
            install = (
                f'Install the optional extra with `pip install "openmed[{extra}]"`'
            )
        raise ImportError(
            f"The {module_name} Chinese segmentation backend is unavailable. "
            f"{install}. Backend license: {license_name}."
        ) from exc


def _align_words_to_text(text: str, words: Iterable[Any]) -> list[SpanToken]:
    tokens: list[SpanToken] = []
    cursor = 0
    for raw_word in words:
        word = str(raw_word)
        if not word or word.isspace():
            continue
        start = text.find(word, cursor)
        if start < 0:
            raise ValueError(
                f"Backend token {word!r} could not be aligned after offset {cursor}"
            )
        tokens.extend(_tokens_for_uncovered_text(text, cursor, start))
        end = start + len(word)
        tokens.append(SpanToken(word, start, end))
        cursor = end
    tokens.extend(_tokens_for_uncovered_text(text, cursor, len(text)))
    return tokens


def _tokens_for_uncovered_text(text: str, start: int, end: int) -> list[SpanToken]:
    tokens: list[SpanToken] = []
    cursor = start
    while cursor < end:
        if text[cursor].isspace():
            cursor += 1
            continue
        token_start = cursor
        cursor += 1
        while cursor < end and not text[cursor].isspace():
            cursor += 1
        tokens.append(SpanToken(text[token_start:cursor], token_start, cursor))
    return tokens


def _apply_dictionary_overlay(
    text: str,
    tokens: Sequence[SpanToken],
    terms: Sequence[str],
) -> list[SpanToken]:
    boundaries = {token.start for token in tokens} | {token.end for token in tokens}
    matches: list[tuple[int, int]] = []
    for term in sorted(terms, key=len, reverse=True):
        start = 0
        while True:
            match = text.find(term, start)
            if match < 0:
                break
            matches.append((match, match + len(term)))
            start = match + len(term)
    selected: list[tuple[int, int]] = []
    for start, end in sorted(matches, key=lambda item: (item[0], -(item[1] - item[0]))):
        if any(
            start < chosen_end and end > chosen_start
            for chosen_start, chosen_end in selected
        ):
            continue
        selected.append((start, end))
    for start, end in selected:
        boundaries.add(start)
        boundaries.add(end)
        boundaries.difference_update(range(start + 1, end))
    ordered = sorted(boundaries)
    result: list[SpanToken] = []
    for start, end in zip(ordered, ordered[1:]):
        if start < end and not text[start:end].isspace():
            result.append(SpanToken(text[start:end], start, end))
    return result


def _hanlp_words(output: Any) -> Sequence[Any]:
    if isinstance(output, Mapping):
        for key in ("tok/fine", "tok/coarse", "tok"):
            if key in output:
                return _flatten_words(output[key])
    if isinstance(output, Sequence) and not isinstance(output, (str, bytes)):
        return _flatten_words(output)
    raise ValueError("HanLP tokenizer returned an unsupported token structure")


def _flatten_words(words: Any) -> list[Any]:
    if isinstance(words, Sequence) and not isinstance(words, (str, bytes)):
        flattened: list[Any] = []
        for word in words:
            if isinstance(word, Sequence) and not isinstance(word, (str, bytes)):
                flattened.extend(_flatten_words(word))
            else:
                flattened.append(word)
        return flattened
    raise ValueError("HanLP tokenizer returned an unsupported token sequence")


def _unique_terms(entries: Iterable[UserDictionaryEntry]) -> list[str]:
    return list(dict.fromkeys(entry.term for entry in entries))


def _local_pkuseg_model(pkuseg_module: Any, model_name: str) -> str:
    config = getattr(pkuseg_module, "config", None)
    available_models = set(getattr(config, "available_models", ()))
    if model_name not in available_models or model_name == "default":
        return model_name
    model_path = Path(config.pkuseg_home).expanduser() / model_name
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"pkuseg model {model_name!r} is not installed at {model_path}. "
            "Download the domain model explicitly or pass a user-supplied local "
            "model path; OpenMed does not download model files implicitly."
        )
    return str(model_path)


def _interior_boundaries(tokens: Sequence[SpanToken]) -> set[int]:
    if not tokens:
        return set()
    terminal = max(token.end for token in tokens)
    return {token.end for token in tokens if token.end != terminal}


def _require_text(text: str) -> None:
    if not isinstance(text, str):
        raise TypeError("Chinese segmentation input must be a string")


__all__ = [
    "ChineseSegmentationConfig",
    "ChineseSegmenter",
    "DEFAULT_MEDICAL_USER_DICTIONARY",
    "HanLPSegmenter",
    "JiebaSegmenter",
    "PkusegSegmenter",
    "SUPPORTED_CHINESE_SEGMENTATION_BACKENDS",
    "UserDictionaryEntry",
    "create_chinese_segmenter",
    "create_chinese_segmenter_from_config",
    "load_user_dictionary",
    "segmentation_boundary_f1",
    "validate_segmentation",
]
