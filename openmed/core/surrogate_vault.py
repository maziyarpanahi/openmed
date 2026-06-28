"""Stable surrogate vaults for cross-document pseudonymization.

The vault stores mappings from ``(canonical_label, lang, text_hash)`` to the
selected surrogate value. ``text_hash`` is an HMAC-SHA256 digest of the source
surface; raw source identifiers are never stored. Vault files are still
pseudonymous linkage artifacts, so protect them as sensitive data.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Iterable, Mapping

from .labels import normalize_label
from .schemas.span import hmac_text_hash

SCHEMA_VERSION = 1
HMAC_SCHEME = "hmac-sha256"


@dataclass(frozen=True, order=True)
class SurrogateKey:
    """Privacy-safe key for one source identifier."""

    canonical_label: str
    lang: str
    text_hash: str

    def __post_init__(self) -> None:
        if not self.canonical_label:
            raise ValueError("canonical_label must be non-empty")
        if not self.lang:
            raise ValueError("lang must be non-empty")
        if not self.text_hash.startswith(f"{HMAC_SCHEME}:"):
            raise ValueError(f"text_hash must use {HMAC_SCHEME}")

    def to_dict(self) -> dict[str, str]:
        """Serialize this key to JSON-compatible fields."""

        return {
            "canonical_label": self.canonical_label,
            "lang": self.lang,
            "text_hash": self.text_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SurrogateKey":
        """Deserialize a key from persisted JSON fields."""

        return cls(
            canonical_label=str(payload["canonical_label"]),
            lang=str(payload["lang"]),
            text_hash=str(payload["text_hash"]),
        )


@dataclass(frozen=True)
class SurrogateEntry:
    """Persisted vault entry."""

    key: SurrogateKey
    surrogate: str

    def __post_init__(self) -> None:
        if not self.surrogate:
            raise ValueError("surrogate must be non-empty")

    def to_dict(self) -> dict[str, str]:
        """Serialize this entry to deterministic JSON-compatible fields."""

        return {
            **self.key.to_dict(),
            "surrogate": self.surrogate,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SurrogateEntry":
        """Deserialize one persisted vault entry."""

        allowed = {"canonical_label", "lang", "text_hash", "surrogate"}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(
                "surrogate vault entries may not contain fields: "
                + ", ".join(sorted(map(str, unknown)))
            )
        return cls(
            key=SurrogateKey.from_dict(payload),
            surrogate=str(payload["surrogate"]),
        )


class InMemorySurrogateStore:
    """In-memory store for surrogate vault entries."""

    def __init__(self, entries: Iterable[SurrogateEntry] = ()) -> None:
        self._entries: dict[SurrogateKey, str] = {}
        self._lock = RLock()
        for entry in entries:
            self.set(entry.key, entry.surrogate)

    def get(self, key: SurrogateKey) -> str | None:
        """Return the surrogate for ``key``, if present."""

        with self._lock:
            return self._entries.get(key)

    def set(self, key: SurrogateKey, surrogate: str) -> None:
        """Store ``surrogate`` for ``key``."""

        if not surrogate:
            raise ValueError("surrogate must be non-empty")
        with self._lock:
            current = self._entries.get(key)
            if current is not None and current != surrogate:
                raise ValueError("surrogate key already maps to a different value")
            self._entries[key] = surrogate

    def entries(self) -> tuple[SurrogateEntry, ...]:
        """Return entries sorted by their privacy-safe key."""

        with self._lock:
            return tuple(
                SurrogateEntry(key, self._entries[key]) for key in sorted(self._entries)
            )

    def used_surrogates(self, *, canonical_label: str, lang: str) -> set[str]:
        """Return surrogates already used for the label/language bucket."""

        with self._lock:
            return {
                surrogate
                for key, surrogate in self._entries.items()
                if key.canonical_label == canonical_label and key.lang == lang
            }

    def to_payload(self) -> dict[str, Any]:
        """Serialize the store without any raw source surfaces."""

        return {
            "schema_version": SCHEMA_VERSION,
            "hmac_scheme": HMAC_SCHEME,
            "entries": [entry.to_dict() for entry in self.entries()],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "InMemorySurrogateStore":
        """Load a store from a JSON-compatible payload."""

        schema_version = int(payload.get("schema_version", 0))
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported surrogate vault schema_version {schema_version!r}"
            )
        hmac_scheme = str(payload.get("hmac_scheme", ""))
        if hmac_scheme != HMAC_SCHEME:
            raise ValueError(f"unsupported surrogate vault hmac_scheme {hmac_scheme!r}")

        entries_payload = payload.get("entries", [])
        if not isinstance(entries_payload, list):
            raise ValueError("surrogate vault entries must be a list")

        entries = [SurrogateEntry.from_dict(entry) for entry in entries_payload]
        return cls(entries)


class JsonFileSurrogateStore(InMemorySurrogateStore):
    """JSON file-backed surrogate store with deterministic atomic writes."""

    def __init__(
        self,
        path: str | Path,
        entries: Iterable[SurrogateEntry] = (),
        *,
        autosave: bool = True,
    ) -> None:
        self.path = Path(path)
        self.autosave = bool(autosave)
        self._hydrating = True
        super().__init__(entries)
        self._hydrating = False

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        create: bool = True,
        autosave: bool = True,
    ) -> "JsonFileSurrogateStore":
        """Load a JSON-backed store from ``path``."""

        vault_path = Path(path)
        if not vault_path.exists():
            if not create:
                raise FileNotFoundError(vault_path)
            return cls(vault_path, autosave=autosave)

        payload = json.loads(vault_path.read_text(encoding="utf-8"))
        loaded = InMemorySurrogateStore.from_payload(payload)
        return cls(vault_path, loaded.entries(), autosave=autosave)

    def set(self, key: SurrogateKey, surrogate: str) -> None:
        """Store an entry and save it atomically when autosave is enabled."""

        super().set(key, surrogate)
        if getattr(self, "autosave", False) and not getattr(
            self,
            "_hydrating",
            False,
        ):
            self.save()

    def save(self) -> None:
        """Atomically write the deterministic JSON vault payload."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(
            self.to_payload(),
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=str(self.path.parent),
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(data)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
                tmp_path = Path(handle.name)
            os.replace(tmp_path, self.path)
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()


SurrogateFactory = Callable[[int], str]


def _contains_source_surface(candidate: str, source_text: str) -> bool:
    if candidate == source_text:
        return True
    return len(source_text) >= 4 and source_text in candidate


class SurrogateVault:
    """Stable cross-document surrogate mapper.

    Args:
        hmac_secret: Secret used to HMAC source surfaces. The secret is required
            to reopen a vault consistently and is never persisted.
        store: Optional backing store. Defaults to an in-memory store.
    """

    def __init__(
        self,
        hmac_secret: str | bytes,
        *,
        store: InMemorySurrogateStore | None = None,
    ) -> None:
        if isinstance(hmac_secret, str):
            if not hmac_secret:
                raise ValueError("hmac_secret must be non-empty")
        elif isinstance(hmac_secret, bytes):
            if not hmac_secret:
                raise ValueError("hmac_secret must be non-empty")
        else:
            raise TypeError("hmac_secret must be str or bytes")
        self.hmac_secret = hmac_secret
        self.store = store or InMemorySurrogateStore()

    @classmethod
    def in_memory(cls, hmac_secret: str | bytes) -> "SurrogateVault":
        """Create a vault backed by memory only."""

        return cls(hmac_secret)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        hmac_secret: str | bytes,
        create: bool = True,
        autosave: bool = True,
    ) -> "SurrogateVault":
        """Create a vault backed by a deterministic JSON file."""

        return cls(
            hmac_secret,
            store=JsonFileSurrogateStore.load(
                path,
                create=create,
                autosave=autosave,
            ),
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"entries={len(self.entries())}, hmac_scheme={HMAC_SCHEME!r})"
        )

    def text_hash(self, source_text: str | bytes) -> str:
        """Return the vault HMAC for ``source_text``."""

        return hmac_text_hash(source_text, self.hmac_secret)

    def key_for(
        self,
        source_text: str,
        *,
        label: str,
        lang: str = "en",
    ) -> SurrogateKey:
        """Build a privacy-safe vault key for one source identifier."""

        effective_lang = str(lang or "en")
        canonical_label = normalize_label(str(label), effective_lang)
        return SurrogateKey(
            canonical_label=canonical_label,
            lang=effective_lang,
            text_hash=self.text_hash(source_text),
        )

    def get(
        self,
        source_text: str,
        *,
        label: str,
        lang: str = "en",
    ) -> str | None:
        """Return an existing surrogate for a source identifier."""

        return self.store.get(self.key_for(source_text, label=label, lang=lang))

    def get_or_create(
        self,
        source_text: str,
        *,
        label: str,
        lang: str = "en",
        create_surrogate: SurrogateFactory,
    ) -> str:
        """Return a stable surrogate, creating and storing it if needed."""

        key = self.key_for(source_text, label=label, lang=lang)
        existing = self.store.get(key)
        if existing is not None:
            return existing

        used = self.store.used_surrogates(
            canonical_label=key.canonical_label,
            lang=key.lang,
        )
        surrogate = ""
        for attempt in range(10):
            candidate = create_surrogate(attempt)
            if not candidate:
                continue
            if _contains_source_surface(candidate, source_text) or candidate in used:
                continue
            surrogate = candidate
            break

        if not surrogate:
            suffix = key.text_hash.rsplit(":", 1)[-1][:8]
            base = create_surrogate(0) or key.canonical_label
            if _contains_source_surface(base, source_text) or base in used:
                base = key.canonical_label
            surrogate = f"{base}_{suffix}"

        self.store.set(key, surrogate)
        return surrogate

    def entries(self) -> tuple[SurrogateEntry, ...]:
        """Return the current sorted vault entries."""

        return self.store.entries()

    def to_payload(self) -> dict[str, Any]:
        """Serialize the vault store without persisting the HMAC secret."""

        return self.store.to_payload()

    def save(self) -> None:
        """Save the backing store if it supports persistence."""

        save = getattr(self.store, "save", None)
        if save is None:
            raise TypeError("this surrogate vault store is not file-backed")
        save()


__all__ = [
    "HMAC_SCHEME",
    "SCHEMA_VERSION",
    "InMemorySurrogateStore",
    "JsonFileSurrogateStore",
    "SurrogateEntry",
    "SurrogateKey",
    "SurrogateVault",
]
