"""Token- and document-level language and script routing.

The default path is stdlib-only and deterministic.  An optional ``pycld2``
adapter supplies Apache-2.0 CLD2 language identification when installed and is
imported only on the first routing call. OpenMed does not bundle language-ID
weights. CLD3 is outside this router's approved dependency scope, and
non-commercial language-ID assets must not be added.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .language_pack import LanguagePack
from .language_pack_catalog import LANGUAGE_PACK_ADAPTERS
from .script_detect import (
    UNKNOWN_SCRIPT,
    candidate_languages_for_script,
    candidate_languages_for_text,
    segment_by_script,
)


@dataclass(frozen=True, slots=True)
class LanguagePrediction:
    """One language-ID backend prediction."""

    language: str
    confidence: float

    def __post_init__(self) -> None:
        """Validate the prediction probability."""

        if not self.language:
            raise ValueError("language must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


@runtime_checkable
class LanguageIdentifier(Protocol):
    """Interface implemented by optional on-device language-ID backends."""

    name: str

    def identify(
        self,
        text: str,
        candidates: Sequence[str],
    ) -> LanguagePrediction | None:
        """Return the best candidate language for ``text``, if available."""


class PyCLD2LanguageIdentifier:
    """Lazy adapter for the Apache-2.0 ``pycld2`` package.

    Missing optional dependencies are represented by ``None`` so callers can
    continue through the deterministic fallback without an import failure.
    """

    name = "pycld2"

    def __init__(self) -> None:
        """Create an unloaded adapter."""

        self._module: object | None = None
        self._load_attempted = False

    def _load(self) -> object | None:
        if not self._load_attempted:
            self._load_attempted = True
            try:
                self._module = importlib.import_module("pycld2")
            except ImportError:
                self._module = None
        return self._module

    def identify(
        self,
        text: str,
        candidates: Sequence[str],
    ) -> LanguagePrediction | None:
        """Identify ``text`` using CLD2 and restrict output to ``candidates``."""

        module = self._load()
        if module is None or not text.strip():
            return None

        try:
            _reliable, _text_bytes, details = module.detect(  # type: ignore[attr-defined]
                text,
                bestEffort=True,
            )
        except Exception:
            # Optional LID must never prevent the deterministic router from
            # handling an input that CLD2 rejects or cannot classify.
            return None
        candidate_set = set(candidates)
        ranked: list[tuple[float, str]] = []
        for _name, raw_code, percent, _score in details:
            code = str(raw_code).lower().split("-", maxsplit=1)[0]
            if code in candidate_set:
                ranked.append((max(0.0, min(float(percent) / 100.0, 1.0)), code))
        if not ranked:
            return None
        confidence, language = max(ranked)
        return LanguagePrediction(language=language, confidence=confidence)


@dataclass(frozen=True, slots=True)
class LanguageRun:
    """Exact-offset routing decision for one contiguous script run."""

    start: int
    end: int
    script: str
    language: str
    confidence: float
    source: str


@dataclass(frozen=True, slots=True)
class DocumentLanguageDecision:
    """Dominant document pack plus every per-run routing override."""

    dominant_pack: LanguagePack
    dominant_script: str
    confidence: float
    runs: tuple[LanguageRun, ...]
    overrides: tuple[LanguageRun, ...]
    lid_backend: str | None

    @property
    def language(self) -> str:
        """Return the dominant pack's language code."""

        return self.dominant_pack.code


class LanguageRouter:
    """Route exact script runs to language packs without mandatory dependencies."""

    def __init__(
        self,
        *,
        packs: Sequence[LanguagePack] | None = None,
        language_identifier: LanguageIdentifier | None = None,
        use_optional_lid: bool = True,
    ) -> None:
        """Create a router.

        Args:
            packs: Candidate packs. Built-in packs are used when omitted.
            language_identifier: Injected local LID backend. When omitted and
                ``use_optional_lid`` is true, the lazy CLD2 adapter is used.
            use_optional_lid: Try the optional CLD2 adapter when no backend was
                injected. Set false to force the stdlib-only path.
        """

        self.packs = tuple(
            LANGUAGE_PACK_ADAPTERS.registry.iter_packs() if packs is None else packs
        )
        if not self.packs:
            raise ValueError("packs must contain at least one LanguagePack")
        if len({pack.code for pack in self.packs}) != len(self.packs):
            raise ValueError("packs must not contain duplicate language codes")
        if language_identifier is not None and not isinstance(
            language_identifier,
            LanguageIdentifier,
        ):
            raise TypeError("language_identifier must implement LanguageIdentifier")

        self.language_identifier = language_identifier
        if language_identifier is None and use_optional_lid:
            self.language_identifier = PyCLD2LanguageIdentifier()
        self._packs_by_code = {pack.code: pack for pack in self.packs}
        self._packs_by_script = {
            script: tuple(
                sorted(
                    (pack for pack in self.packs if script in pack.scripts),
                    key=lambda pack: self._candidate_sort_key(pack, script),
                )
            )
            for script in {script for pack in self.packs for script in pack.scripts}
        }

    @staticmethod
    def _candidate_sort_key(pack: LanguagePack, script: str) -> tuple[int, int, str]:
        hints = candidate_languages_for_script(script)
        hint_order = {code: index for index, code in enumerate(hints)}
        return (
            -pack.priority_for(script),
            hint_order.get(pack.code, len(hint_order)),
            pack.code,
        )

    def route_runs(self, text: str) -> tuple[LanguageRun, ...]:
        """Return exact-offset language decisions that tile ``text``."""

        script_runs = tuple(segment_by_script(text))
        routed: list[LanguageRun] = []
        for index, (start, end, script) in enumerate(script_runs):
            neighboring_scripts = {
                script_runs[neighbor][2]
                for neighbor in (index - 1, index + 1)
                if 0 <= neighbor < len(script_runs)
            }
            pack, confidence, source = self._select_pack(
                text[start:end],
                script,
                neighboring_scripts,
            )
            routed.append(
                LanguageRun(
                    start=start,
                    end=end,
                    script=script,
                    language=pack.code,
                    confidence=confidence,
                    source=source,
                )
            )

        self._validate_tiling(routed, len(text))
        return tuple(routed)

    def route(self, text: str) -> DocumentLanguageDecision:
        """Return a document-level pack and all per-run decisions."""

        runs = self.route_runs(text)
        fallback_pack = self._packs_by_code.get("en", self.packs[0])
        if not runs:
            return DocumentLanguageDecision(
                dominant_pack=fallback_pack,
                dominant_script=UNKNOWN_SCRIPT,
                confidence=0.0,
                runs=(),
                overrides=(),
                lid_backend=self._lid_name,
            )

        totals: dict[str, int] = {}
        first_seen: dict[str, int] = {}
        for index, run in enumerate(runs):
            totals[run.language] = totals.get(run.language, 0) + run.end - run.start
            first_seen.setdefault(run.language, index)
        language = max(
            totals,
            key=lambda code: (totals[code], -first_seen[code]),
        )
        dominant_pack = self._packs_by_code[language]
        dominant_runs = [run for run in runs if run.language == language]
        dominant_script = max(
            dominant_runs,
            key=lambda run: (run.end - run.start, -run.start),
        ).script
        total_length = sum(run.end - run.start for run in runs)
        confidence = sum(run.confidence * (run.end - run.start) for run in runs) / max(
            total_length, 1
        )
        overrides = tuple(run for run in runs if run.language != language)
        return DocumentLanguageDecision(
            dominant_pack=dominant_pack,
            dominant_script=dominant_script,
            confidence=confidence,
            runs=runs,
            overrides=overrides,
            lid_backend=self._lid_name,
        )

    @property
    def _lid_name(self) -> str | None:
        backend = self.language_identifier
        return str(backend.name) if backend is not None else None

    def _select_pack(
        self,
        text: str,
        script: str,
        neighboring_scripts: set[str],
    ) -> tuple[LanguagePack, float, str]:
        candidates = self._packs_by_script.get(script, ())
        if not candidates:
            fallback = self._packs_by_code.get("en", self.packs[0])
            return fallback, 0.5, "stdlib:unknown-script"

        context_matches = tuple(
            pack
            for pack in candidates
            if neighboring_scripts.intersection(pack.context_scripts)
        )
        if context_matches:
            return context_matches[0], 0.99, "stdlib:context-script"

        text_hints = candidate_languages_for_text(text, script)
        if text_hints != candidate_languages_for_script(script):
            candidates_by_code = {pack.code: pack for pack in candidates}
            for code in text_hints:
                selected = candidates_by_code.get(code)
                if selected is not None:
                    return selected, 0.99, "stdlib:assamese-cues"

        if len(candidates) == 1:
            return candidates[0], 0.99, "stdlib:script"

        backend = self.language_identifier
        if backend is not None:
            prediction = backend.identify(text, [pack.code for pack in candidates])
            if prediction is not None:
                selected = self._packs_by_code.get(prediction.language)
                if selected in candidates:
                    return selected, prediction.confidence, backend.name

        return candidates[0], 0.8, "stdlib:pack-priority"

    @staticmethod
    def _validate_tiling(runs: Sequence[LanguageRun], text_length: int) -> None:
        cursor = 0
        for run in runs:
            if run.start != cursor or run.end <= run.start:
                raise RuntimeError(
                    "language runs must tile text without gaps or overlaps"
                )
            cursor = run.end
        if cursor != text_length:
            raise RuntimeError("language runs must cover the entire input")


__all__ = [
    "DocumentLanguageDecision",
    "LanguageIdentifier",
    "LanguagePrediction",
    "LanguageRouter",
    "LanguageRun",
    "PyCLD2LanguageIdentifier",
]
