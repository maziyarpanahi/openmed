"""Offline script tokenization qualification over caller-supplied synthetic terms."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from openmed.core.decoding.spans import iter_grapheme_clusters

_CODE = re.compile(r"[a-z][a-z0-9_]{0,31}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class ScriptTerm:
    """An explicitly synthetic term with a controlled script category."""

    script: str
    text: str
    synthetic: bool = True

    def __post_init__(self) -> None:
        if self.synthetic is not True:
            raise ValueError("tokenization terms must be explicitly synthetic")
        if not isinstance(self.script, str) or not _CODE.fullmatch(self.script):
            raise ValueError("invalid script code")
        if (
            not isinstance(self.text, str)
            or not self.text.strip()
            or len(self.text) > 4096
        ):
            raise ValueError("synthetic terms must contain 1 to 4096 characters")


@dataclass(frozen=True)
class TokenizationObservation:
    """Token identities and code-point offsets from an injected local tokenizer."""

    token_ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.token_ids, Sequence)
            or not isinstance(self.offsets, Sequence)
            or len(self.token_ids) > 8192
            or len(self.offsets) > 8192
            or any(
                not isinstance(pair, Sequence) or len(pair) != 2
                for pair in self.offsets
            )
        ):
            raise ValueError("provide bounded token identity and offset sequences")
        object.__setattr__(self, "token_ids", tuple(self.token_ids))
        object.__setattr__(self, "offsets", tuple(tuple(pair) for pair in self.offsets))
        if len(self.token_ids) != len(self.offsets) or len(self.token_ids) > 8192:
            raise ValueError(
                "token identities and offsets must have bounded matching lengths"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.token_ids
        ):
            raise ValueError("token identities must be non-negative integers")
        if any(
            len(pair) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) for value in pair
            )
            for pair in self.offsets
        ):
            raise ValueError("offset pairs must contain integer code-point coordinates")


@dataclass(frozen=True)
class ScriptTokenizationLimits:
    """Pinned metric limits; they qualify tokenization, not model entity recall."""

    max_tokens_per_grapheme: float = 4.0
    max_unknown_rate: float = 0.01
    max_grapheme_split_rate: float = 0.0
    min_offset_recovery_rate: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.max_tokens_per_grapheme,
            self.max_unknown_rate,
            self.max_grapheme_split_rate,
            self.min_offset_recovery_rate,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ):
            raise ValueError("tokenization limits must be finite numbers")
        if self.max_tokens_per_grapheme <= 0 or any(
            not 0 <= value <= 1 for value in values[1:]
        ):
            raise ValueError("tokenization limits outside allowed range")


def audit_script_tokenization(
    terms: Sequence[ScriptTerm],
    *,
    tokenizer: Callable[[str], TokenizationObservation],
    model_digest: str,
    unknown_token_id: int,
    limits: ScriptTokenizationLimits = ScriptTokenizationLimits(),
) -> dict[str, object]:
    """Audit synthetic script samples without returning text or token identities.

    The adapter must omit special tokens and use source code-point offsets.
    Invalid or overlapping offsets reduce recovery; grapheme-internal endpoints
    are scored separately. Empty output cannot qualify a script. Thresholds
    are evaluated on unrounded pooled counts, with inclusive boundaries.
    """
    if not isinstance(model_digest, str) or not _DIGEST.fullmatch(model_digest):
        raise ValueError("model identity must be a SHA-256 digest")
    if (
        isinstance(unknown_token_id, bool)
        or not isinstance(unknown_token_id, int)
        or unknown_token_id < 0
    ):
        raise ValueError("unknown token identity must be a non-negative integer")
    if not isinstance(limits, ScriptTokenizationLimits):
        raise ValueError("expected pinned tokenization limits")
    if (
        not terms
        or len(terms) > 10_000
        or not all(isinstance(term, ScriptTerm) for term in terms)
    ):
        raise ValueError("provide 1 to 10000 validated synthetic terms")
    counts = defaultdict(lambda: defaultdict(int))
    for term in terms:
        try:
            result = tokenizer(term.text)
        except Exception:
            raise ValueError("local tokenization adapter failed") from None
        if not isinstance(result, TokenizationObservation):
            raise ValueError("tokenizer must return validated observations")
        row = counts[term.script]
        row["term_count"] += 1
        graphemes = list(iter_grapheme_clusters(term.text))
        row["grapheme_count"] += sum(
            not term.text[start:end].isspace() for start, end in graphemes
        )
        boundaries = {0, len(term.text)} | {
            point for pair in graphemes for point in pair
        }
        row["token_count"] += len(result.token_ids)
        row["unknown_count"] += sum(
            token == unknown_token_id for token in result.token_ids
        )
        row["empty_output_count"] += not bool(result.token_ids)
        covered: set[int] = set()
        previous_end = 0
        for start, end in result.offsets:
            if not 0 <= start < end <= len(term.text) or start < previous_end:
                row["invalid_offset_count"] += 1
                continue
            previous_end = end
            row["split_token_count"] += start not in boundaries or end not in boundaries
            covered.update(range(start, end))
        positions = {
            index for index, char in enumerate(term.text) if not char.isspace()
        }
        row["source_codepoint_count"] += len(positions)
        row["recovered_codepoint_count"] += len(positions & covered)
    scripts = {}
    for script, row in sorted(counts.items()):
        tokens = row["token_count"]
        fragmentation = tokens / row["grapheme_count"]
        unknown = row["unknown_count"] / tokens if tokens else 1.0
        split = row["split_token_count"] / tokens if tokens else 1.0
        recovery = row["recovered_codepoint_count"] / row["source_codepoint_count"]
        reasons = []
        for failed, reason in (
            (bool(row["empty_output_count"]), "empty_tokenization"),
            (bool(row["invalid_offset_count"]), "invalid_offsets"),
            (fragmentation > limits.max_tokens_per_grapheme, "fragmentation_limit"),
            (unknown > limits.max_unknown_rate, "unknown_limit"),
            (split > limits.max_grapheme_split_rate, "grapheme_split_limit"),
            (recovery < limits.min_offset_recovery_rate, "offset_recovery_limit"),
        ):
            if failed:
                reasons.append(reason)
        scripts[script] = {
            "counts": dict(sorted(row.items())),
            "tokens_per_grapheme": fragmentation,
            "unknown_rate": unknown,
            "grapheme_split_rate": split,
            "offset_recovery_rate": recovery,
            "verdict": "unsupported" if reasons else "qualified_tokenization",
            "reasons": reasons,
        }
    return {
        "schema_version": 1,
        "synthetic": True,
        "model_digest": model_digest,
        "limits": {name: getattr(limits, name) for name in limits.__dataclass_fields__},
        "scripts": scripts,
    }
