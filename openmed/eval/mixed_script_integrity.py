"""Deterministic mixed-script span-integrity regression fixtures.

The evaluator keeps fixture text and surrogate values in memory, but reports
only offsets, labels, stable hashes, counts, and failure codes.  It is intended
for offline regression checks around Unicode grapheme and script boundaries.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from openmed.core.decoding.spans import iter_grapheme_cluster_spans

MIXED_SCRIPT_INTEGRITY_SCHEMA_VERSION: Final = "openmed.eval.mixed_script_integrity.v1"

GRAPHEME_BOUNDARY_FAILURE: Final = "grapheme-boundary"
SOURCE_DIGEST_FAILURE: Final = "source-digest"
SPAN_ORDER_FAILURE: Final = "span-order"
SPAN_MISMATCH_FAILURE: Final = "span-mismatch"
SURROGATE_STABILITY_FAILURE: Final = "surrogate-stability"
RUNNER_FAILURE: Final = "runner-error"
NONDETERMINISTIC_RUN_FAILURE: Final = "nondeterministic-run"

_SOURCE_HASH_DOMAIN: Final = b"openmed-mixed-script-source-v1\x00"
_SURROGATE_HASH_DOMAIN: Final = b"openmed-mixed-script-surrogate-v1\x00"
_ENTITY_HASH_DOMAIN: Final = b"openmed-mixed-script-entity-v1\x00"


def _content_hash(domain: bytes, value: str) -> str:
    return f"sha256:{hashlib.sha256(domain + value.encode('utf-8')).hexdigest()}"


@dataclass(frozen=True)
class MixedScriptSpan:
    """One expected or observed half-open span.

    ``entity_key`` links repeated occurrences whose replacement must remain
    stable. ``surrogate`` is deliberately excluded from ``repr`` and every
    serialized report.
    """

    start: int
    end: int
    label: str
    entity_key: str
    source_hash: str
    surrogate: str = field(repr=False)

    @classmethod
    def from_source(
        cls,
        text: str,
        source: str,
        *,
        label: str,
        entity_key: str,
        surrogate: str,
        occurrence: int = 0,
    ) -> MixedScriptSpan:
        """Locate a synthetic source occurrence and build its expected span."""

        if occurrence < 0:
            raise ValueError("occurrence must be non-negative")
        cursor = 0
        start = -1
        for _ in range(occurrence + 1):
            start = text.find(source, cursor)
            if start < 0:
                raise ValueError("synthetic source occurrence was not found")
            cursor = start + len(source)
        return cls(
            start=start,
            end=start + len(source),
            label=label,
            entity_key=entity_key,
            source_hash=_content_hash(_SOURCE_HASH_DOMAIN, source),
            surrogate=surrogate,
        )


@dataclass(frozen=True)
class MixedScriptFixture:
    """Synthetic document with ordered expected spans."""

    fixture_id: str
    text: str = field(repr=False)
    spans: tuple[MixedScriptSpan, ...]
    coverage: tuple[str, ...]


@dataclass(frozen=True)
class MixedScriptSpanVerdict:
    """Privacy-safe verdict for one observed span."""

    index: int
    start: int
    end: int
    label: str
    entity_hash: str
    source_hash: str
    surrogate_hash: str
    grapheme_aligned: bool
    source_matches: bool

    @property
    def passed(self) -> bool:
        """Return whether source and grapheme boundaries are intact."""

        return self.grapheme_aligned and self.source_matches

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic mapping without raw fixture values."""

        return {
            "end": self.end,
            "entity_hash": self.entity_hash,
            "grapheme_aligned": self.grapheme_aligned,
            "index": self.index,
            "label": self.label,
            "passed": self.passed,
            "source_hash": self.source_hash,
            "source_matches": self.source_matches,
            "start": self.start,
            "surrogate_hash": self.surrogate_hash,
        }


@dataclass(frozen=True)
class MixedScriptFixtureResult:
    """Integrity result for one fixture."""

    fixture_id: str
    coverage: tuple[str, ...]
    spans: tuple[MixedScriptSpanVerdict, ...]
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Return whether every fixture check passed."""

        return not self.failures and all(span.passed for span in self.spans)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, raw-value-free fixture result."""

        return {
            "coverage": list(self.coverage),
            "failures": list(self.failures),
            "fixture_id": self.fixture_id,
            "passed": self.passed,
            "span_count": len(self.spans),
            "spans": [span.to_dict() for span in self.spans],
        }


@dataclass(frozen=True)
class MixedScriptIntegrityReport:
    """Aggregate mixed-script span-integrity report."""

    fixture_results: tuple[MixedScriptFixtureResult, ...]
    iterations: int
    deterministic: bool
    failures: tuple[str, ...]
    schema_version: str = MIXED_SCRIPT_INTEGRITY_SCHEMA_VERSION

    @property
    def fixture_count(self) -> int:
        """Return the number of evaluated fixtures."""

        return len(self.fixture_results)

    @property
    def span_count(self) -> int:
        """Return the number of observed spans."""

        return sum(len(result.spans) for result in self.fixture_results)

    @property
    def passed(self) -> bool:
        """Return whether all checks passed identically across iterations."""

        return (
            self.deterministic
            and not self.failures
            and all(result.passed for result in self.fixture_results)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic report containing no source or surrogate text."""

        return {
            "deterministic": self.deterministic,
            "failures": list(self.failures),
            "fixture_count": self.fixture_count,
            "fixtures": [result.to_dict() for result in self.fixture_results],
            "iterations": self.iterations,
            "passed": self.passed,
            "schema_version": self.schema_version,
            "span_count": self.span_count,
        }


SpanRunner = Callable[[MixedScriptFixture], Sequence[MixedScriptSpan]]


def default_mixed_script_fixtures() -> tuple[MixedScriptFixture, ...]:
    """Return deterministic synthetic fixtures for Unicode boundary risks."""

    latin_value = "Jose\u0301 Alvarez"
    latin_text = f"Patient {latin_value}; repeat {latin_value}."
    latin_surrogate = "Avery Morgan"

    indic_value = "क्षिति राव"
    indic_text = f"नाम: {indic_value}; पुनः {indic_value}।"
    indic_surrogate = "अनन्या सेन"

    cjk_value = "青山李"
    cjk_text = f"ID-A{cjk_value}-ID-B；再診:{cjk_value}。"
    cjk_surrogate = "高橋森"

    rtl_value = "ليان صالح"
    rtl_text = f"ملف:{rtl_value}(ID-7)؛ إعادة:{rtl_value}."
    rtl_surrogate = "نور أمين"

    emoji_value = "👩🏽\u200d⚕️Mira"
    emoji_text = f"Alias:{emoji_value}; repeat:{emoji_value}."
    emoji_surrogate = "👩🏽\u200d⚕️Avery"

    return (
        MixedScriptFixture(
            fixture_id="latin-combining",
            text=latin_text,
            spans=tuple(
                MixedScriptSpan.from_source(
                    latin_text,
                    latin_value,
                    label="NAME",
                    entity_key="latin-person-1",
                    surrogate=latin_surrogate,
                    occurrence=index,
                )
                for index in range(2)
            ),
            coverage=("Latin", "combining-mark", "grapheme"),
        ),
        MixedScriptFixture(
            fixture_id="indic-conjunct",
            text=indic_text,
            spans=tuple(
                MixedScriptSpan.from_source(
                    indic_text,
                    indic_value,
                    label="NAME",
                    entity_key="indic-person-1",
                    surrogate=indic_surrogate,
                    occurrence=index,
                )
                for index in range(2)
            ),
            coverage=("Devanagari", "Indic-conjunct", "grapheme"),
        ),
        MixedScriptFixture(
            fixture_id="cjk-transition",
            text=cjk_text,
            spans=tuple(
                MixedScriptSpan.from_source(
                    cjk_text,
                    cjk_value,
                    label="NAME",
                    entity_key="cjk-person-1",
                    surrogate=cjk_surrogate,
                    occurrence=index,
                )
                for index in range(2)
            ),
            coverage=("Han", "Latin", "script-transition"),
        ),
        MixedScriptFixture(
            fixture_id="rtl-transition",
            text=rtl_text,
            spans=tuple(
                MixedScriptSpan.from_source(
                    rtl_text,
                    rtl_value,
                    label="NAME",
                    entity_key="rtl-person-1",
                    surrogate=rtl_surrogate,
                    occurrence=index,
                )
                for index in range(2)
            ),
            coverage=("Arabic", "bidi", "script-transition"),
        ),
        MixedScriptFixture(
            fixture_id="emoji-zwj-transition",
            text=emoji_text,
            spans=tuple(
                MixedScriptSpan.from_source(
                    emoji_text,
                    emoji_value,
                    label="NAME",
                    entity_key="emoji-person-1",
                    surrogate=emoji_surrogate,
                    occurrence=index,
                )
                for index in range(2)
            ),
            coverage=("emoji-ZWJ", "grapheme", "script-transition"),
        ),
    )


def evaluate_mixed_script_integrity(
    fixtures: Sequence[MixedScriptFixture] | None = None,
    *,
    runner: SpanRunner | None = None,
    iterations: int = 3,
) -> MixedScriptIntegrityReport:
    """Evaluate ordered spans and surrogate stability without network access.

    Args:
        fixtures: Synthetic fixtures to evaluate. Defaults to the built-in set.
        runner: Optional candidate span producer. It receives one fixture and
            must return spans in source order. The default replays the fixture's
            expected spans, making this useful as a regression-fixture audit.
        iterations: Number of identical evaluations used to detect drift.

    Returns:
        A privacy-safe aggregate report.

    Raises:
        ValueError: If ``iterations`` is not positive.
    """

    if iterations < 1:
        raise ValueError("iterations must be positive")
    selected = tuple(default_mixed_script_fixtures() if fixtures is None else fixtures)
    span_runner = runner or _expected_span_runner

    runs = tuple(_evaluate_once(selected, span_runner) for _ in range(iterations))
    signatures = tuple(_results_signature(results) for results in runs)
    deterministic = len(set(signatures)) == 1
    failures = () if deterministic else (NONDETERMINISTIC_RUN_FAILURE,)
    return MixedScriptIntegrityReport(
        fixture_results=runs[0],
        iterations=iterations,
        deterministic=deterministic,
        failures=failures,
    )


def _expected_span_runner(fixture: MixedScriptFixture) -> Sequence[MixedScriptSpan]:
    return fixture.spans


def _evaluate_once(
    fixtures: Sequence[MixedScriptFixture], runner: SpanRunner
) -> tuple[MixedScriptFixtureResult, ...]:
    results: list[MixedScriptFixtureResult] = []
    for fixture in fixtures:
        try:
            observed = tuple(runner(fixture))
        except Exception:  # noqa: BLE001 - raw runner exceptions must not leak text
            results.append(
                MixedScriptFixtureResult(
                    fixture_id=fixture.fixture_id,
                    coverage=fixture.coverage,
                    spans=(),
                    failures=(RUNNER_FAILURE,),
                )
            )
            continue
        results.append(_evaluate_fixture(fixture, observed))
    return tuple(results)


def _evaluate_fixture(
    fixture: MixedScriptFixture, observed: Sequence[MixedScriptSpan]
) -> MixedScriptFixtureResult:
    boundaries = {0, len(fixture.text)}
    for start, end in iter_grapheme_cluster_spans(fixture.text):
        boundaries.update((start, end))

    failures: list[str] = []
    verdicts: list[MixedScriptSpanVerdict] = []
    previous_end = 0
    surrogate_hashes: defaultdict[str, set[str]] = defaultdict(set)

    expected_keys = tuple(_span_comparison_key(span) for span in fixture.spans)
    observed_keys = tuple(_span_comparison_key(span) for span in observed)
    if observed_keys != expected_keys:
        failures.append(SPAN_MISMATCH_FAILURE)

    for index, span in enumerate(observed):
        in_bounds = (
            isinstance(span.start, int)
            and not isinstance(span.start, bool)
            and isinstance(span.end, int)
            and not isinstance(span.end, bool)
            and 0 <= span.start < span.end <= len(fixture.text)
        )
        grapheme_aligned = (
            in_bounds and span.start in boundaries and span.end in boundaries
        )
        if not grapheme_aligned:
            failures.append(f"span-{index}:{GRAPHEME_BOUNDARY_FAILURE}")

        source_matches = False
        if in_bounds:
            source_matches = (
                _content_hash(_SOURCE_HASH_DOMAIN, fixture.text[span.start : span.end])
                == span.source_hash
            )
        if not source_matches:
            failures.append(f"span-{index}:{SOURCE_DIGEST_FAILURE}")

        if index and span.start < previous_end:
            failures.append(f"span-{index}:{SPAN_ORDER_FAILURE}")
        previous_end = max(previous_end, span.end) if in_bounds else previous_end

        surrogate_hash = _content_hash(_SURROGATE_HASH_DOMAIN, span.surrogate)
        surrogate_hashes[span.entity_key].add(surrogate_hash)
        verdicts.append(
            MixedScriptSpanVerdict(
                index=index,
                start=span.start,
                end=span.end,
                label=span.label,
                entity_hash=_content_hash(_ENTITY_HASH_DOMAIN, span.entity_key),
                source_hash=span.source_hash,
                surrogate_hash=surrogate_hash,
                grapheme_aligned=grapheme_aligned,
                source_matches=source_matches,
            )
        )

    for entity_key, digests in surrogate_hashes.items():
        if len(digests) > 1:
            entity_hash = _content_hash(_ENTITY_HASH_DOMAIN, entity_key)
            failures.append(f"entity-{entity_hash}:{SURROGATE_STABILITY_FAILURE}")

    return MixedScriptFixtureResult(
        fixture_id=fixture.fixture_id,
        coverage=fixture.coverage,
        spans=tuple(verdicts),
        failures=tuple(sorted(set(failures))),
    )


def _span_comparison_key(span: MixedScriptSpan) -> tuple[int, int, str, str, str]:
    return (
        span.start,
        span.end,
        span.label,
        span.entity_key,
        span.source_hash,
    )


def _results_signature(results: Sequence[MixedScriptFixtureResult]) -> str:
    payload = json.dumps(
        [result.to_dict() for result in results],
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "GRAPHEME_BOUNDARY_FAILURE",
    "MIXED_SCRIPT_INTEGRITY_SCHEMA_VERSION",
    "NONDETERMINISTIC_RUN_FAILURE",
    "RUNNER_FAILURE",
    "SOURCE_DIGEST_FAILURE",
    "SPAN_MISMATCH_FAILURE",
    "SPAN_ORDER_FAILURE",
    "SURROGATE_STABILITY_FAILURE",
    "MixedScriptFixture",
    "MixedScriptFixtureResult",
    "MixedScriptIntegrityReport",
    "MixedScriptSpan",
    "MixedScriptSpanVerdict",
    "SpanRunner",
    "default_mixed_script_fixtures",
    "evaluate_mixed_script_integrity",
]
