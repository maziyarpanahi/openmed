"""Bounded local abbreviation mining with metadata-only review exports."""

from __future__ import annotations

import hashlib
import hmac
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import islice

_WORD = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", re.UNICODE)
_PARENS = re.compile(r"\(([^()\r\n]{1,160})\)")


def _short(value: str) -> str | None:
    value = value.strip()
    letters = value.replace(".", "").replace("-", "")
    if 2 <= len(letters) <= 10 and letters.isalpha() and letters.isupper():
        return letters
    return None


def _expansion(words: list[str], abbreviation: str) -> str | None:
    # Prefer the shortest suffix whose initials match the entire short form.
    for size in range(len(abbreviation), min(len(words), len(abbreviation) * 2) + 1):
        candidate = words[-size:]
        if "".join(word[0] for word in candidate).casefold() == abbreviation.casefold():
            return " ".join(candidate)
    return None


def _fingerprint(key: bytes, domain: str, value: str) -> str:
    return hmac.new(
        key, (domain + "\0" + value).encode("utf-8"), hashlib.sha256
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class AbbreviationCandidate:
    """An in-memory local term pair with a safe metadata-only export method."""

    abbreviation: str = field(repr=False)
    expansion: str = field(repr=False)
    candidate_id: str
    count: int
    concordance_fingerprints: tuple[str, ...]

    def __post_init__(self) -> None:
        for value, maximum in ((self.abbreviation, 10), (self.expansion, 512)):
            if not isinstance(value, str) or not value or len(value) > maximum:
                raise ValueError("invalid local abbreviation term")
        if not isinstance(self.candidate_id, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.candidate_id
        ):
            raise ValueError("invalid abbreviation candidate fingerprint")
        if type(self.count) is not int or not 1 <= self.count <= 10000000:
            raise ValueError("invalid abbreviation candidate count")
        if isinstance(self.concordance_fingerprints, (str, bytes)):
            raise ValueError("invalid abbreviation concordance collection")
        try:
            fingerprints = tuple(islice(self.concordance_fingerprints, 10001))
        except (TypeError, ValueError):
            raise ValueError("invalid abbreviation concordance collection") from None
        if (
            not fingerprints
            or len(fingerprints) > 10000
            or len(fingerprints) > self.count
        ):
            raise ValueError("invalid abbreviation concordance count")
        if any(
            not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in fingerprints
        ):
            raise ValueError("invalid abbreviation concordance fingerprint")
        if len(set(fingerprints)) != len(fingerprints):
            raise ValueError("duplicate abbreviation concordance fingerprint")
        object.__setattr__(
            self, "concordance_fingerprints", tuple(sorted(fingerprints))
        )

    @property
    def frequency_band(self) -> str:
        """Return a closed frequency band rather than exposing exact counts."""
        if self.count == 1:
            return "once"
        if self.count < 5:
            return "few"
        if self.count < 20:
            return "repeated"
        return "frequent"

    def to_dict(self) -> dict[str, object]:
        """Return keyed fingerprints and frequency bands, never source snippets."""
        return {
            "candidate_id": self.candidate_id,
            "frequency_band": self.frequency_band,
            "concordance_fingerprints": list(self.concordance_fingerprints),
            "review_status": "unreviewed",
        }


def mine_abbreviations(
    documents: Iterable[str],
    *,
    fingerprint_key: bytes,
    max_documents: int = 1000,
    max_characters: int = 1000000,
    max_candidates: int = 10000,
    context_characters: int = 80,
) -> tuple[AbbreviationCandidate, ...]:
    """Mine forward/reverse parenthetical clinical candidates entirely locally.

    The caller must supply a private high-entropy key for concordance and term
    fingerprints. Candidate spellings remain in memory for authorized native
    review; only ``to_dict`` should be exported. Matching initials is a lexical
    hypothesis, not a medical interpretation or validated terminology mapping.
    """
    if not isinstance(fingerprint_key, bytes) or len(fingerprint_key) < 32:
        raise ValueError("abbreviation fingerprint key must contain at least 32 bytes")
    if isinstance(documents, (str, bytes)):
        raise ValueError("abbreviation input must be a document collection")
    for value, upper in (
        (max_documents, 10000),
        (max_characters, 10000000),
        (max_candidates, 10000),
        (context_characters, 1024),
    ):
        if type(value) is not int or not 1 <= value <= upper:
            raise ValueError("invalid abbreviation mining limit")
    rows: dict[tuple[str, str], dict[str, object]] = {}
    total = 0
    for document_index, text in enumerate(documents):
        if document_index >= max_documents:
            raise ValueError("abbreviation document limit exceeded")
        if not isinstance(text, str):
            raise ValueError("invalid abbreviation document")
        total += len(text)
        if total > max_characters:
            raise ValueError("abbreviation character limit exceeded")
        for match in _PARENS.finditer(text):
            inner = match.group(1).strip()
            before = text[max(0, match.start() - 512) : match.start()]
            # A sentence/line boundary prevents borrowing an expansion from another sentence.
            before = re.split(r"[.!?\r\n]", before)[-1]
            words = _WORD.findall(before)
            abbreviation = _short(inner)
            expansion = _expansion(words, abbreviation) if abbreviation else None
            if expansion is None and words:
                abbreviation = _short(words[-1])
                if abbreviation:
                    inner_words = _WORD.findall(inner)
                    expansion = _expansion(inner_words, abbreviation)
                    # Reverse candidates must use the entire parenthetical expansion.
                    if expansion and len(_WORD.findall(expansion)) != len(inner_words):
                        expansion = None
            if expansion is None or abbreviation is None:
                continue
            canonical = (
                unicodedata.normalize("NFC", abbreviation).casefold(),
                unicodedata.normalize("NFC", expansion).casefold(),
            )
            if canonical not in rows:
                if len(rows) >= max_candidates:
                    raise ValueError("abbreviation candidate limit exceeded")
                rows[canonical] = {"count": 0, "fingerprints": set()}
            row = rows[canonical]
            row["count"] += 1
            context = text[
                max(0, match.start() - context_characters) : min(
                    len(text), match.end() + context_characters
                )
            ]
            row["fingerprints"].add(
                _fingerprint(fingerprint_key, "concordance", context)
            )
    candidates = []
    for (abbreviation, expansion), row in rows.items():
        candidates.append(
            AbbreviationCandidate(
                abbreviation,
                expansion,
                _fingerprint(
                    fingerprint_key, "candidate", abbreviation + "\0" + expansion
                ),
                row["count"],
                tuple(sorted(row["fingerprints"])),
            )
        )
    return tuple(sorted(candidates, key=lambda candidate: candidate.candidate_id))
