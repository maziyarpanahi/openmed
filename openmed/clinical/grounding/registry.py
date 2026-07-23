"""Registries for grounding linkers and free-vocabulary loaders.

Linker factories keep their established short system keys.  Vocabulary loaders
use canonical system URIs and are accepted only when they explicitly declare
that their source is redistributable.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Callable, Protocol, cast, runtime_checkable

from .matcher import AbbreviationMap, LexicalMatcher, VocabularyTerms

__all__ = [
    "InvalidVocabularyLoaderError",
    "LinkerFactory",
    "RestrictedVocabularyLoaderError",
    "VocabularyLoader",
    "VocabularyLoaderRegistry",
    "VocabularyRegistryError",
    "available_linkers",
    "available_loaders",
    "get_linker",
    "get_loader",
    "register_linker",
    "register_loader",
    "validate_vocabulary_loader",
]

LinkerFactory = Callable[..., Any]
_URI_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")

_LINKERS: dict[str, LinkerFactory] = {}


def register_linker(system: str, factory: LinkerFactory) -> None:
    """Register a linker factory under a vocabulary ``system`` key."""
    _LINKERS[system] = factory


def get_linker(system: str) -> LinkerFactory:
    """Return the linker factory registered for ``system`` (raises KeyError)."""
    return _LINKERS[system]


def available_linkers() -> list[str]:
    """Return the sorted list of registered linker system keys."""
    return sorted(_LINKERS)


class VocabularyRegistryError(ValueError):
    """Base error raised by the free-vocabulary loader registry."""


class InvalidVocabularyLoaderError(TypeError, VocabularyRegistryError):
    """Raised when an object does not satisfy the loader contract."""


class RestrictedVocabularyLoaderError(VocabularyRegistryError):
    """Raised when a restricted-license loader is registered in-process."""


@runtime_checkable
class VocabularyLoader(Protocol):
    """Contract implemented by loaders for redistributable vocabularies.

    Implementations parse a caller-supplied or freely redistributable snapshot
    into terms consumed by :class:`~openmed.clinical.grounding.LexicalMatcher`.
    Registration never calls :meth:`load`, so configuring a loader has no I/O
    or network side effects.
    """

    system_uri: str
    redistributable: bool

    def load(self) -> VocabularyTerms:
        """Load local vocabulary terms keyed by lexical surface."""
        ...


class VocabularyLoaderRegistry:
    """Registry of redistributable vocabulary loaders keyed by system URI."""

    def __init__(self) -> None:
        self._loaders: dict[str, VocabularyLoader] = {}

    def register(self, loader: VocabularyLoader, *, replace: bool = False) -> None:
        """Register ``loader`` after validating its license declaration.

        Args:
            loader: Object implementing :class:`VocabularyLoader`.
            replace: Replace an existing loader for the same URI when true.

        Raises:
            InvalidVocabularyLoaderError: If the loader contract is incomplete.
            RestrictedVocabularyLoaderError: If ``redistributable`` is false.
            VocabularyRegistryError: If the URI is already registered.
        """

        validated = validate_vocabulary_loader(loader)
        system_uri = validated.system_uri.strip()
        if not replace and system_uri in self._loaders:
            raise VocabularyRegistryError(
                f"A vocabulary loader is already registered for {system_uri!r}."
            )
        self._loaders[system_uri] = validated

    def get(self, system_uri: str) -> VocabularyLoader:
        """Return the loader registered for ``system_uri``."""

        normalized = _validate_registry_system_uri(system_uri)
        try:
            return self._loaders[normalized]
        except KeyError:
            raise KeyError(
                f"No vocabulary loader is registered for {normalized!r}."
            ) from None

    def available(self) -> tuple[str, ...]:
        """Return registered system URIs in deterministic order."""

        return tuple(sorted(self._loaders))

    def matcher(
        self,
        system_uri: str,
        *,
        abbreviations: AbbreviationMap | None = None,
    ) -> LexicalMatcher:
        """Load ``system_uri`` and build its offline lexical matcher."""

        loader = self.get(system_uri)
        terms = loader.load()
        if not isinstance(terms, Mapping):
            raise InvalidVocabularyLoaderError(
                f"Vocabulary loader for {loader.system_uri!r} must return a mapping."
            )
        return LexicalMatcher(
            terms,
            system_uri=loader.system_uri,
            abbreviations=abbreviations,
        )

    def __contains__(self, system_uri: object) -> bool:
        return isinstance(system_uri, str) and system_uri.strip() in self._loaders

    def __len__(self) -> int:
        return len(self._loaders)


def validate_vocabulary_loader(loader: object) -> VocabularyLoader:
    """Validate and return an object implementing :class:`VocabularyLoader`."""

    system_uri = getattr(loader, "system_uri", None)
    redistributable = getattr(loader, "redistributable", None)
    restricted_license = getattr(loader, "restricted_license", False)
    load = getattr(loader, "load", None)
    if not isinstance(system_uri, str):
        raise InvalidVocabularyLoaderError(
            "Vocabulary loaders must declare a string system_uri."
        )
    _validate_registry_system_uri(system_uri)
    if not isinstance(restricted_license, bool):
        raise InvalidVocabularyLoaderError(
            f"Vocabulary loader for {system_uri!r} must declare "
            "restricted_license as a boolean when provided."
        )
    if restricted_license:
        raise RestrictedVocabularyLoaderError(
            f"Refusing to register vocabulary loader for {system_uri!r}: it is "
            "flagged as restricted-license and must remain user-controlled and "
            "out of process."
        )
    if not isinstance(redistributable, bool):
        raise InvalidVocabularyLoaderError(
            f"Vocabulary loader for {system_uri!r} must declare a boolean "
            "redistributable flag."
        )
    if not callable(load):
        raise InvalidVocabularyLoaderError(
            f"Vocabulary loader for {system_uri!r} must define load()."
        )
    if not redistributable:
        raise RestrictedVocabularyLoaderError(
            f"Refusing to register vocabulary loader for {system_uri!r}: "
            "redistributable must be true. Restricted-license vocabularies "
            "must remain user-controlled and out of process."
        )
    return cast(VocabularyLoader, loader)


_VOCABULARY_LOADERS = VocabularyLoaderRegistry()


def register_loader(loader: VocabularyLoader, *, replace: bool = False) -> None:
    """Register a loader in the process-wide free-vocabulary registry."""

    _VOCABULARY_LOADERS.register(loader, replace=replace)


def get_loader(system_uri: str) -> VocabularyLoader:
    """Return a loader from the process-wide free-vocabulary registry."""

    return _VOCABULARY_LOADERS.get(system_uri)


def available_loaders() -> list[str]:
    """Return system URIs in the process-wide loader registry."""

    return list(_VOCABULARY_LOADERS.available())


def _validate_registry_system_uri(system_uri: object) -> str:
    if not isinstance(system_uri, str):
        raise InvalidVocabularyLoaderError("system_uri must be a string")
    normalized = system_uri.strip()
    if (
        not normalized
        or not _URI_SCHEME_RE.match(normalized)
        or any(character.isspace() for character in normalized)
    ):
        raise InvalidVocabularyLoaderError(
            "system_uri must be an absolute URI with no whitespace"
        )
    return normalized
