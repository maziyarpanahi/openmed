"""Local BiomedCLIP image/text embedding with pinned, digest-verified assets.

OpenMed bundles no weights and makes no diagnostic claim. The caller supplies a
local checkpoint directory (never a Hub ID or URL) plus the SHA-256 digest they
pinned for it. Embedding runs under an offline guard that blocks network
connections. Embeddings are wrapped in :class:`ProtectedEmbeddings`, whose
``repr``/``to_json`` expose only digests and dimensions, never vector values.

The default backend needs the optional ``biomedclip`` extra
(``pip install openmed[biomedclip]``); tests inject a fake backend instead.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import socket
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

SCHEMA_VERSION = "openmed.biomedclip.embeddings.v1"
_MAX_FILE_BYTES = 8 * 1024**3
_INSTALL_HINT = (
    "BiomedCLIP embedding needs the optional extra: pip install 'openmed[biomedclip]'"
)


class BiomedCLIPError(ValueError):
    """Content-free failure at the local BiomedCLIP boundary."""


class EmbeddingBackend(Protocol):
    """Minimal model surface; returns raw (unnormalized) float arrays."""

    dimension: int

    def encode_images(self, images: Sequence[Any]) -> np.ndarray: ...

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray: ...


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            count += len(chunk)
            if count > _MAX_FILE_BYTES:
                raise BiomedCLIPError("local BiomedCLIP assets rejected")
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class BiomedCLIPAssets:
    """Verified local checkpoint directory.

    Args:
        directory: Local, owner-controlled directory; symlinks are rejected.
        pinned_digest: Caller-pinned SHA-256 over the sorted (name, file digest)
            list, as returned by :func:`compute_assets_digest`.
        license_metadata: Local license/model-card snapshot, hashed without
            interpreting terms. The caller is responsible for permission to use
            the weights (the published checkpoint is MIT-licensed).
    """

    directory: Path = field(repr=False)
    pinned_digest: str
    license_metadata: bytes = field(repr=False)
    assets_digest: str = field(init=False)
    license_digest: str = field(init=False)

    def __post_init__(self) -> None:
        failed = False
        try:
            if (
                not isinstance(self.directory, Path)
                or type(self.pinned_digest) is not str
                or len(self.pinned_digest) != 64
                or type(self.license_metadata) is not bytes
                or not 0 < len(self.license_metadata) <= 128 * 1024
            ):
                raise ValueError
            actual = compute_assets_digest(self.directory)
            if actual != self.pinned_digest:
                raise ValueError
            object.__setattr__(self, "assets_digest", actual)
            object.__setattr__(
                self,
                "license_digest",
                hashlib.sha256(self.license_metadata).hexdigest(),
            )
        except Exception:
            failed = True
        if failed:
            raise BiomedCLIPError("local BiomedCLIP assets rejected")

    def verify_current(self) -> None:
        """Rehash before use; reject a checkpoint changed since verification."""
        failed = False
        try:
            failed = compute_assets_digest(self.directory) != self.assets_digest
        except Exception:
            failed = True
        if failed:
            raise BiomedCLIPError("local BiomedCLIP assets changed")


def compute_assets_digest(directory: Path) -> str:
    """SHA-256 over sorted ``(relative name, file sha256)`` pairs of ``directory``.

    Use this once on a trusted copy to obtain the value to pin. Only regular,
    non-symlink files directly inside ``directory`` are accepted.
    """
    if not isinstance(directory, Path) or not directory.is_dir():
        raise BiomedCLIPError("local BiomedCLIP assets rejected")
    entries = []
    for path in sorted(directory.iterdir()):
        if path.is_symlink() or not path.is_file():
            raise BiomedCLIPError("local BiomedCLIP assets rejected")
        entries.append((path.name, _file_digest(path)))
    if not entries:
        raise BiomedCLIPError("local BiomedCLIP assets rejected")
    payload = json.dumps(entries, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


class ProtectedEmbeddings:
    """L2-normalized embeddings that never reveal values via repr/JSON/pickle."""

    __slots__ = ("_array", "digest", "dimension", "count")

    def __init__(self, array: np.ndarray) -> None:
        array = np.asarray(array, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
            raise BiomedCLIPError("embedding shape rejected")
        if not np.isfinite(array).all():
            raise BiomedCLIPError("embedding values rejected")
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        if (norms == 0).any():
            raise BiomedCLIPError("embedding values rejected")
        normalized = np.ascontiguousarray(array / norms, dtype=np.float32)
        normalized.setflags(write=False)
        self._array = normalized
        self.count, self.dimension = normalized.shape
        self.digest = hashlib.sha256(normalized.tobytes()).hexdigest()

    def unprotect(self) -> np.ndarray:
        """Return the read-only array; the caller takes over protection."""
        return self._array

    def __repr__(self) -> str:
        return f"ProtectedEmbeddings(count={self.count}, dimension={self.dimension})"

    def __reduce__(self) -> Any:
        raise TypeError("ProtectedEmbeddings cannot be pickled")

    def to_json(self) -> str:
        """Digest and dimensions only; no vector values."""
        return json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "count": self.count,
                "dimension": self.dimension,
                "digest": self.digest,
            },
            sort_keys=True,
            separators=(",", ":"),
        )


@contextlib.contextmanager
def offline_guard() -> Iterator[None]:
    """Block socket connections and Hub access for the duration of the block.

    Process-wide and not a sandbox: it stops accidental egress, not hostile code.
    """
    env_keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    saved_env = {k: os.environ.get(k) for k in env_keys}
    saved_connect = socket.socket.connect
    saved_create = socket.create_connection

    def _blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise BiomedCLIPError("network access blocked during embedding")

    for key in env_keys:
        os.environ[key] = "1"
    socket.socket.connect = _blocked  # type: ignore[method-assign]
    socket.create_connection = _blocked  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket.connect = saved_connect  # type: ignore[method-assign]
        socket.create_connection = saved_create  # type: ignore[assignment]
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _open_clip_backend(assets: BiomedCLIPAssets) -> EmbeddingBackend:
    """Load the checkpoint with ``open_clip`` from the local directory only."""
    try:
        import open_clip  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:
        raise BiomedCLIPError(_INSTALL_HINT) from exc

    with offline_guard():
        model, _, preprocess = open_clip.create_model_and_transforms(
            f"local-dir:{assets.directory}"
        )
        tokenizer = open_clip.get_tokenizer(f"local-dir:{assets.directory}")
    model.eval()

    class _Backend:
        dimension = int(model.text_projection.shape[1])

        def encode_images(self, images: Sequence[Any]) -> np.ndarray:
            batch = torch.stack([preprocess(image) for image in images])
            with torch.no_grad():
                return model.encode_image(batch).float().cpu().numpy()

        def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
            with torch.no_grad():
                return model.encode_text(tokenizer(list(texts))).float().cpu().numpy()

    return _Backend()


class BiomedCLIPEmbedder:
    """Embed images and texts locally with a digest-verified checkpoint.

    Args:
        assets: Verified :class:`BiomedCLIPAssets`.
        backend_factory: Builds the model backend from the assets; defaults to an
            ``open_clip`` loader behind the ``biomedclip`` extra. Injectable so
            tests run without weights or heavy dependencies.
    """

    def __init__(
        self,
        assets: BiomedCLIPAssets,
        backend_factory: Callable[[BiomedCLIPAssets], EmbeddingBackend] | None = None,
    ) -> None:
        if not isinstance(assets, BiomedCLIPAssets):
            raise BiomedCLIPError("local BiomedCLIP assets rejected")
        assets.verify_current()
        self._assets = assets
        self._backend = (backend_factory or _open_clip_backend)(assets)

    @property
    def dimension(self) -> int:
        return int(self._backend.dimension)

    def embed_images(self, images: Sequence[Any]) -> ProtectedEmbeddings:
        return self._embed(self._backend.encode_images, images)

    def embed_texts(self, texts: Sequence[str]) -> ProtectedEmbeddings:
        if any(type(t) is not str for t in texts):
            raise BiomedCLIPError("text input rejected")
        return self._embed(self._backend.encode_texts, texts)

    def _embed(
        self, encode: Callable[[Any], np.ndarray], items: Sequence[Any]
    ) -> ProtectedEmbeddings:
        if len(items) == 0:
            raise BiomedCLIPError("empty input rejected")
        self._assets.verify_current()
        with offline_guard():
            raw = np.asarray(encode(items))
        if raw.ndim != 2 or raw.shape != (len(items), self.dimension):
            raise BiomedCLIPError("embedding shape rejected")
        return ProtectedEmbeddings(raw)

    def provenance_json(self) -> str:
        """Digest-only provenance; no local paths or license text."""
        return json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "assets_digest": self._assets.assets_digest,
                "license_digest": self._assets.license_digest,
                "dimension": self.dimension,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
