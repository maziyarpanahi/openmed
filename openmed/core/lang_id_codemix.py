"""Deterministic token language identification for Hinglish clinical text.

The fallback identifier is intentionally stdlib-only and emits offset-based
records without retaining token surfaces.  A caller may provide a model hook
for Latin-script disambiguation; Devanagari and universal tokens are still
resolved deterministically so script routing cannot be weakened by the hook.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, cast, runtime_checkable

from .script_detect import detect_script

TokenLanguageLabel = Literal["hi", "en", "ne", "univ", "other"]

VALID_TOKEN_LANGUAGE_LABELS = frozenset({"hi", "en", "ne", "univ", "other"})

_TOKEN_RE = re.compile(
    r"\d+(?:[./:-]\d+)*|[\u0900-\u097f]+|"
    r"[^\W\d_]+(?:[\N{RIGHT SINGLE QUOTATION MARK}'-][^\W\d_]+)*|\S",
    re.UNICODE,
)

# High-frequency Roman Hindi and clinical-routing cues.  The set is kept
# deliberately conservative: unknown Latin words remain ``other`` instead of
# being guessed as Hindi.
_ROMAN_HINDI_WORDS = frozenset(
    {
        "aaj",
        "aadhaar",
        "adhar",
        "adhaar",
        "admit",
        "aap",
        "aaya",
        "aayi",
        "abhi",
        "aur",
        "bahut",
        "bharti",
        "bimari",
        "bukhar",
        "dard",
        "dawai",
        "din",
        "hai",
        "hain",
        "hua",
        "hui",
        "ilaaj",
        "ilaj",
        "janam",
        "ka",
        "kar",
        "ke",
        "ki",
        "ko",
        "liye",
        "mahina",
        "mein",
        "mera",
        "meri",
        "milna",
        "naam",
        "nahi",
        "number",
        "par",
        "pata",
        "pehle",
        "phir",
        "pin",
        "raha",
        "rahi",
        "sampark",
        "se",
        "tabiyat",
        "tareekh",
        "teen",
        "tha",
        "thi",
        "wale",
        "wali",
    }
)

_ENGLISH_WORDS = frozenset(
    {
        "address",
        "admitted",
        "age",
        "allergy",
        "and",
        "appointment",
        "birth",
        "blood",
        "born",
        "call",
        "called",
        "clinic",
        "contact",
        "date",
        "diagnosis",
        "discharged",
        "doctor",
        "dr",
        "email",
        "fever",
        "follow",
        "follow-up",
        "for",
        "from",
        "has",
        "hospital",
        "id",
        "is",
        "mobile",
        "name",
        "note",
        "of",
        "on",
        "patient",
        "phone",
        "recorded",
        "reports",
        "since",
        "the",
        "to",
        "was",
        "with",
    }
)


@dataclass(frozen=True)
class TokenSpan:
    """One token boundary passed to an optional LID model hook."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("token offsets must satisfy 0 <= start < end")


@dataclass(frozen=True)
class TokenLanguage:
    """A raw-text-free token language decision with exact character offsets."""

    start: int
    end: int
    label: TokenLanguageLabel
    source: str = "heuristic"

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("token offsets must satisfy 0 <= start < end")
        if self.label not in VALID_TOKEN_LANGUAGE_LABELS:
            raise ValueError(f"unsupported token language label: {self.label!r}")

    def to_metadata(self, source_text: str) -> dict[str, object]:
        """Return offset/hash evidence suitable for logs and audit metadata."""
        if self.end > len(source_text):
            raise ValueError("token offsets exceed source text length")
        surface = source_text[self.start : self.end]
        return {
            "start": self.start,
            "end": self.end,
            "length": self.end - self.start,
            "label": self.label,
            "source": self.source,
            "token_hash": (
                "sha256:" + hashlib.sha256(surface.encode("utf-8")).hexdigest()
            ),
        }


@dataclass(frozen=True)
class TokenLanguageRun:
    """A contiguous run of adjacent token decisions with one LID label."""

    start: int
    end: int
    label: TokenLanguageLabel
    token_start: int
    token_end: int


@runtime_checkable
class TokenLIDModel(Protocol):
    """Protocol for a user-supplied token language classification model."""

    def predict(
        self,
        text: str,
        spans: Sequence[TokenSpan],
    ) -> Sequence[str]:
        """Return one language label for every supplied token span."""


TokenLIDHook = TokenLIDModel | Callable[[str, Sequence[TokenSpan]], Sequence[str]]


class TokenLanguageIdentifier:
    """Identify Hinglish token languages with an optional model hook."""

    def __init__(self, model: TokenLIDHook | None = None) -> None:
        self.model = model

    def identify(self, text: str) -> tuple[TokenLanguage, ...]:
        """Return deterministic token labels with offsets into ``text``."""
        spans = tuple(TokenSpan(*match.span()) for match in _TOKEN_RE.finditer(text))
        if not spans:
            return ()

        model_labels = self._model_labels(text, spans)
        decisions: list[TokenLanguage] = []
        for index, span in enumerate(spans):
            surface = text[span.start : span.end]
            guarded = _guarded_label(surface)
            if guarded is not None:
                label = guarded
                source = "script" if guarded == "hi" else "universal"
            elif model_labels is not None:
                label = model_labels[index]
                source = "model_hook"
            else:
                label = _latin_fallback_label(surface)
                source = "heuristic"
            decisions.append(
                TokenLanguage(
                    start=span.start,
                    end=span.end,
                    label=label,
                    source=source,
                )
            )
        return tuple(decisions)

    def _model_labels(
        self,
        text: str,
        spans: Sequence[TokenSpan],
    ) -> tuple[TokenLanguageLabel, ...] | None:
        if self.model is None:
            return None
        if isinstance(self.model, TokenLIDModel):
            raw_labels = self.model.predict(text, spans)
        else:
            raw_labels = self.model(text, spans)
        if len(raw_labels) != len(spans):
            raise ValueError(
                "token LID model hook must return one label per token span"
            )

        labels: list[TokenLanguageLabel] = []
        for raw_label in raw_labels:
            label = str(raw_label).strip().casefold()
            if label not in VALID_TOKEN_LANGUAGE_LABELS:
                raise ValueError(
                    "token LID model hook returned unsupported label "
                    f"{raw_label!r}; expected one of "
                    f"{sorted(VALID_TOKEN_LANGUAGE_LABELS)}"
                )
            labels.append(cast(TokenLanguageLabel, label))
        return tuple(labels)


def identify_token_languages(
    text: str,
    model: TokenLIDHook | None = None,
) -> tuple[TokenLanguage, ...]:
    """Convenience wrapper for token-level Hinglish language identification."""
    return TokenLanguageIdentifier(model=model).identify(text)


def token_language_runs(
    tokens: Sequence[TokenLanguage],
) -> tuple[TokenLanguageRun, ...]:
    """Merge adjacent, same-label token decisions into offset-preserving runs."""
    if not tokens:
        return ()

    runs: list[TokenLanguageRun] = []
    token_start = 0
    run_start = tokens[0].start
    run_end = tokens[0].end
    run_label = tokens[0].label
    for index, token in enumerate(tokens[1:], start=1):
        if token.start < run_end:
            raise ValueError("token language decisions must be ordered and disjoint")
        if token.label == run_label:
            run_end = token.end
            continue
        runs.append(
            TokenLanguageRun(
                start=run_start,
                end=run_end,
                label=run_label,
                token_start=token_start,
                token_end=index,
            )
        )
        token_start = index
        run_start = token.start
        run_end = token.end
        run_label = token.label
    runs.append(
        TokenLanguageRun(
            start=run_start,
            end=run_end,
            label=run_label,
            token_start=token_start,
            token_end=len(tokens),
        )
    )
    return tuple(runs)


def _guarded_label(surface: str) -> TokenLanguageLabel | None:
    script = detect_script(surface)
    if script == "Devanagari":
        return "hi"
    if not any(char.isalpha() for char in surface):
        return "univ"
    if script != "Latin":
        return "other"
    return None


def _latin_fallback_label(surface: str) -> TokenLanguageLabel:
    folded = unicodedata.normalize("NFKC", surface).casefold()
    if folded in _ROMAN_HINDI_WORDS:
        return "hi"
    if folded in _ENGLISH_WORDS:
        return "en"
    if _looks_like_named_entity(surface):
        return "ne"
    if folded.endswith(("ing", "tion", "ment", "ness", "ity", "ive", "ous")):
        return "en"
    return "other"


def _looks_like_named_entity(surface: str) -> bool:
    letters = [char for char in surface if char.isalpha()]
    if not letters:
        return False
    return letters[0].isupper() and any(char.islower() for char in letters[1:])


__all__ = [
    "TokenLIDHook",
    "TokenLIDModel",
    "TokenLanguage",
    "TokenLanguageIdentifier",
    "TokenLanguageLabel",
    "TokenLanguageRun",
    "TokenSpan",
    "VALID_TOKEN_LANGUAGE_LABELS",
    "identify_token_languages",
    "token_language_runs",
]
