"""Alias encoders for dense clinical concept grounding.

Dense candidate generation needs a fixed-width vector for every vocabulary
alias and for the surface mention being grounded. This module defines the small
encoder contract the index consumes and two concrete encoders:

* :class:`HashingAliasEncoder` is a deterministic, dependency-free feature
  hashing encoder over character n-grams. It requires no downloaded weights and
  produces byte-for-byte identical vectors across runs, so it doubles as the
  synthetic reference encoder in tests and a local development stand-in.
* :class:`MLXSapBERTEncoder` wraps a local MLX SapBERT-class sentence encoder.
  It is the production encoder, gated behind the optional ``mlx`` extra and a
  user-supplied local weights directory. It never reaches the network and
  raises :class:`EncoderUnavailableError` when its backend or weights are
  absent, so the default install never requires it.

:func:`load_encoder` returns ``None`` when no encoder weights are configured,
which callers treat as the documented no-op that falls back to sparse-only
retrieval rather than raising.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from .vocab import normalize_alias

__all__ = [
    "AliasEncoder",
    "EncoderUnavailableError",
    "HashingAliasEncoder",
    "MLXSapBERTEncoder",
    "load_encoder",
]

#: Default dimensionality for the dependency-free hashing encoder.
DEFAULT_EMBEDDING_DIM = 64
_DEFAULT_NGRAM_SIZE = 3


class EncoderUnavailableError(RuntimeError):
    """Raised when a requested alias encoder backend or weights are absent."""


@runtime_checkable
class AliasEncoder(Protocol):
    """Contract implemented by alias encoders consumed by the index.

    Implementations expose a stable ``encoder_id`` (folded into the index cache
    key so a swapped encoder invalidates the persisted index), a fixed
    ``dimension``, and an :meth:`encode` method returning one unit-norm vector
    per input text in order.
    """

    encoder_id: str
    dimension: int

    def encode(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        """Return one embedding vector per text, in input order."""
        ...


class HashingAliasEncoder:
    """Deterministic, offline feature-hashing encoder over character n-grams.

    Every alias is normalized, expanded into boundary-padded character n-grams,
    and hashed into a fixed-width vector via SHA-256 bucketing. The vector is
    L2-normalized so cosine similarity lies in ``[0.0, 1.0]`` and an identical
    surface embeds to a vector whose cosine with itself is ``1.0``. No weights
    are downloaded and no randomness is used, so results are byte-for-byte
    reproducible.

    This encoder requires no model files; it is the synthetic reference encoder
    for tests and a local development stand-in for the production SapBERT-class
    encoder.
    """

    def __init__(
        self,
        dimension: int = DEFAULT_EMBEDDING_DIM,
        *,
        ngram_size: int = _DEFAULT_NGRAM_SIZE,
    ) -> None:
        if dimension < 1:
            raise ValueError("dimension must be a positive integer")
        if ngram_size < 1:
            raise ValueError("ngram_size must be a positive integer")
        self.dimension = dimension
        self._ngram_size = ngram_size
        self.encoder_id = f"hashing-v1-d{dimension}-n{ngram_size}"

    def encode(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        """Return one unit-norm hashed vector per text, in input order."""

        return tuple(self.encode_one(text) for text in texts)

    def encode_one(self, text: str) -> tuple[float, ...]:
        """Return the unit-norm hashed vector for a single surface."""

        vector = [0.0] * self.dimension
        for gram in self._char_ngrams(normalize_alias(text)):
            digest = hashlib.sha256(gram.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            vector[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return tuple(vector)
        return tuple(value / norm for value in vector)

    def _char_ngrams(self, text: str) -> tuple[str, ...]:
        if not text:
            return ()
        padded = f"#{text}#"
        size = self._ngram_size
        if len(padded) <= size:
            return (padded,)
        return tuple(
            padded[index : index + size] for index in range(len(padded) - size + 1)
        )


class MLXSapBERTEncoder:
    """Local MLX SapBERT-class sentence encoder (optional ``mlx`` extra).

    The encoder loads a user-supplied local weights directory through MLX and
    mean-pools token embeddings into a unit-norm sentence vector. It performs no
    network I/O: the weights must already exist on disk. When MLX is not
    installed or the weights are missing it raises
    :class:`EncoderUnavailableError`, so importing this module never forces the
    heavy dependency and the default install degrades to sparse-only retrieval.
    """

    def __init__(
        self, model_path: str | Path, *, encoder_id: str | None = None
    ) -> None:
        path = Path(model_path).expanduser()
        if not path.exists():
            raise EncoderUnavailableError(
                f"No SapBERT-class encoder weights found at {path}. Pre-stage a "
                "local MLX model directory or fall back to sparse retrieval."
            )
        try:  # pragma: no cover - exercised only with the optional extra installed
            import mlx.core as mx  # noqa: F401
            from mlx_lm import load as mlx_load
        except ImportError as exc:  # pragma: no cover - default install path
            raise EncoderUnavailableError(
                "Install openmed[mlx] to use the local SapBERT-class encoder."
            ) from exc

        model, tokenizer = mlx_load(str(path))  # pragma: no cover - needs weights
        self._model = model  # pragma: no cover
        self._tokenizer = tokenizer  # pragma: no cover
        self.encoder_id = encoder_id or f"mlx-sapbert:{path.name}"  # pragma: no cover
        self.dimension = int(getattr(model, "hidden_size", 0))  # pragma: no cover

    def encode(
        self, texts: Sequence[str]
    ) -> tuple[tuple[float, ...], ...]:  # pragma: no cover - needs weights
        import mlx.core as mx

        vectors: list[tuple[float, ...]] = []
        for text in texts:
            token_ids = self._tokenizer.encode(text)
            hidden = self._model(mx.array([token_ids]))
            pooled = mx.mean(hidden, axis=1)[0]
            norm = mx.sqrt(mx.sum(pooled * pooled))
            unit = pooled / norm if float(norm) != 0.0 else pooled
            vectors.append(tuple(float(value) for value in unit.tolist()))
        return tuple(vectors)


def load_encoder(
    model_path: str | Path | None = None,
    *,
    encoder_id: str | None = None,
) -> AliasEncoder | None:
    """Return an alias encoder, or ``None`` when no encoder weights exist.

    A ``model_path`` selects the local MLX SapBERT-class encoder. When it is
    omitted, or the weights/backend are unavailable, this returns ``None`` to
    signal the documented no-op: callers fall back to sparse-only retrieval
    instead of raising. This function never downloads weights.
    """

    if model_path is None:
        return None
    try:
        return MLXSapBERTEncoder(model_path, encoder_id=encoder_id)
    except EncoderUnavailableError:
        return None
