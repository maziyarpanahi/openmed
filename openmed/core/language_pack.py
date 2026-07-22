"""Typed language-pack registration for multilingual PII capabilities.

This module owns the process-local contract and registry used to describe a
complete OpenMed language integration. Downstream adapters intentionally live
outside this module so importing the registry does not create cycles with PII,
anonymizer, script-routing, policy, or threshold modules.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from threading import RLock
from types import MappingProxyType

_LANGUAGE_CODE = re.compile(r"^[a-z]{2}$")


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _freeze_names(value: Iterable[str], field_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise TypeError(f"{field_name} must be an iterable of strings, not a string")

    names = tuple(_require_text(item, field_name) for item in value)
    if not names:
        raise ValueError(f"{field_name} must contain at least one value")
    if len(set(names)) != len(names):
        raise ValueError(f"{field_name} must not contain duplicate values")
    return names


def _freeze_optional_names(
    value: Iterable[str],
    field_name: str,
) -> tuple[str, ...]:
    if isinstance(value, str):
        raise TypeError(f"{field_name} must be an iterable of strings, not a string")

    names = tuple(_require_text(item, field_name) for item in value)
    if len(set(names)) != len(names):
        raise ValueError(f"{field_name} must not contain duplicate values")
    return names


def _freeze_text_mapping(
    value: Mapping[str, str],
    field_name: str,
) -> Mapping[str, str]:
    frozen: dict[str, str] = {}
    for key, item in value.items():
        frozen[_require_text(key, f"{field_name} key")] = _require_text(
            item,
            f"{field_name} value",
        )
    return MappingProxyType(frozen)


def _freeze_recall_floors(
    value: Mapping[str, float],
) -> Mapping[str, float]:
    frozen: dict[str, float] = {}
    for key, item in value.items():
        label = _require_text(key, "recall_floor_overrides key")
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(
                "recall_floor_overrides values must be numeric probabilities"
            )
        floor = float(item)
        if not math.isfinite(floor) or not 0.0 <= floor <= 1.0:
            raise ValueError(
                "recall_floor_overrides values must be finite probabilities"
            )
        frozen[label] = floor
    return MappingProxyType(frozen)


def _freeze_candidate_priorities(
    value: Mapping[str, int],
    scripts: tuple[str, ...],
) -> Mapping[str, int]:
    frozen: dict[str, int] = {}
    for key, item in value.items():
        script = _require_text(key, "candidate_priority key")
        if script not in scripts:
            raise ValueError(
                f"candidate priority script {script!r} is not declared by the pack"
            )
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError("candidate priorities must be integers")
        frozen[script] = item
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True)
class LanguagePack:
    """Describe one language's model, routing, surrogate, and safety hooks.

    Args:
        code: Lowercase ISO 639-1 language code.
        scripts: Unicode script identifiers used to route text to the pack.
        default_model: Default PII model repository or local model identifier.
        segmenter_id: Registered sentence or token segmenter identifier.
        recognizers: Registered recognizer identifiers enabled for the pack.
        surrogate_locale: Conceptual Faker/surrogate locale for replacements.
        national_id_providers: Mapping of national-ID provider method names to
            the locales that supply them.
        policy_overrides: Mapping of policy keys to pack-specific values.
        recall_floor_overrides: Mapping of canonical labels to probabilities.
        candidate_priority: Per-script deterministic routing priorities. Higher
            values win when a script maps to more than one registered pack.
        context_scripts: Neighboring scripts that strongly identify this pack.

    Collection inputs are copied into immutable tuples or read-only mappings,
    so callers cannot mutate a registered pack through an object they retained.
    """

    code: str
    scripts: tuple[str, ...]
    default_model: str
    segmenter_id: str
    recognizers: tuple[str, ...]
    surrogate_locale: str
    national_id_providers: Mapping[str, str] = field(default_factory=dict)
    policy_overrides: Mapping[str, str] = field(default_factory=dict)
    recall_floor_overrides: Mapping[str, float] = field(default_factory=dict)
    candidate_priority: Mapping[str, int] = field(default_factory=dict)
    context_scripts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate and freeze the complete pack declaration."""

        code = _require_text(self.code, "code")
        if not _LANGUAGE_CODE.fullmatch(code):
            raise ValueError("code must be a lowercase ISO 639-1 language code")

        object.__setattr__(self, "scripts", _freeze_names(self.scripts, "scripts"))
        object.__setattr__(
            self,
            "recognizers",
            _freeze_names(self.recognizers, "recognizers"),
        )
        object.__setattr__(
            self,
            "default_model",
            _require_text(self.default_model, "default_model"),
        )
        object.__setattr__(
            self,
            "segmenter_id",
            _require_text(self.segmenter_id, "segmenter_id"),
        )
        object.__setattr__(
            self,
            "surrogate_locale",
            _require_text(self.surrogate_locale, "surrogate_locale"),
        )
        object.__setattr__(
            self,
            "national_id_providers",
            _freeze_text_mapping(
                self.national_id_providers,
                "national_id_providers",
            ),
        )
        object.__setattr__(
            self,
            "policy_overrides",
            _freeze_text_mapping(self.policy_overrides, "policy_overrides"),
        )
        object.__setattr__(
            self,
            "recall_floor_overrides",
            _freeze_recall_floors(self.recall_floor_overrides),
        )
        object.__setattr__(
            self,
            "candidate_priority",
            _freeze_candidate_priorities(self.candidate_priority, self.scripts),
        )
        object.__setattr__(
            self,
            "context_scripts",
            _freeze_optional_names(self.context_scripts, "context_scripts"),
        )

    def priority_for(self, script: str) -> int:
        """Return the pack's deterministic routing priority for ``script``."""

        return self.candidate_priority.get(script, 0)


class LanguagePackRegistry:
    """Store language packs in a thread-safe, process-local registry."""

    def __init__(self) -> None:
        """Create an empty registry."""

        self._packs: dict[str, LanguagePack] = {}
        self._listeners: list[Callable[[], None]] = []
        self._lock = RLock()

    def register(
        self,
        pack: LanguagePack,
        *,
        replace: bool = False,
    ) -> LanguagePack:
        """Register and return a language pack.

        Args:
            pack: Fully validated language-pack declaration.
            replace: Replace an existing pack with the same code when true.

        Returns:
            The registered pack.

        Raises:
            TypeError: If ``pack`` is not a :class:`LanguagePack`.
            ValueError: If the code is already registered and replacement was
                not requested.
        """

        if not isinstance(pack, LanguagePack):
            raise TypeError("pack must be a LanguagePack")

        with self._lock:
            if pack.code in self._packs and not replace:
                raise ValueError(f"language pack {pack.code!r} is already registered")
            self._packs[pack.code] = pack
            listeners = tuple(self._listeners)
        for listener in listeners:
            listener()
        return pack

    def _add_listener(
        self,
        listener: Callable[[], None],
        *,
        replay: bool = True,
    ) -> None:
        """Subscribe an internal live adapter to registry changes.

        Args:
            listener: Zero-argument callback that refreshes derived state.
            replay: Invoke the callback immediately for the current snapshot.

        Raises:
            TypeError: If ``listener`` is not callable.
        """

        if not callable(listener):
            raise TypeError("listener must be callable")
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)
        if replay:
            listener()

    def get(self, code: str) -> LanguagePack:
        """Return the pack registered for ``code``.

        Args:
            code: Lowercase ISO 639-1 language code.

        Returns:
            The registered language pack.

        Raises:
            KeyError: If no pack is registered for ``code``.
        """

        with self._lock:
            return self._packs[code]

    def find(self, code: str) -> LanguagePack | None:
        """Return the pack registered for ``code``, if any."""

        with self._lock:
            return self._packs.get(code)

    def iter_codes(self) -> Iterator[str]:
        """Iterate over a stable, alphabetically sorted snapshot of codes."""

        with self._lock:
            codes = tuple(sorted(self._packs))
        return iter(codes)

    def iter_packs(self) -> Iterator[LanguagePack]:
        """Iterate over a stable, code-sorted snapshot of registered packs."""

        with self._lock:
            packs = tuple(self._packs[code] for code in sorted(self._packs))
        return iter(packs)


LANGUAGE_PACK_REGISTRY = LanguagePackRegistry()


def register_language_pack(
    pack: LanguagePack,
    *,
    replace: bool = False,
) -> LanguagePack:
    """Register a pack in OpenMed's process-local language-pack registry.

    Args:
        pack: Fully declared language pack.
        replace: Replace an existing pack with the same code when true.

    Returns:
        The registered pack.
    """

    return LANGUAGE_PACK_REGISTRY.register(pack, replace=replace)


def get_language_pack(code: str) -> LanguagePack | None:
    """Return a process-local language pack without raising for unknown codes."""

    return LANGUAGE_PACK_REGISTRY.find(code)


__all__ = [
    "LANGUAGE_PACK_REGISTRY",
    "LanguagePack",
    "LanguagePackRegistry",
    "get_language_pack",
    "register_language_pack",
]
