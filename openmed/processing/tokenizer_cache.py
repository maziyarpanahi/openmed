"""Process-level caching for Hugging Face tokenizers."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from os import PathLike
from threading import RLock
from typing import Any

try:
    from transformers import AutoTokenizer
except (ImportError, OSError):
    AutoTokenizer = None  # type: ignore[assignment]


DEFAULT_TOKENIZER_CACHE_SIZE = 32

_TOKENIZER_CACHE: "OrderedDict[tuple[Any, ...], Any]" = OrderedDict()
_TOKENIZER_CACHE_LOCK = RLock()


def get_tokenizer(
    name: str | PathLike[str], revision: str | None = None, **kwargs: Any
) -> Any:
    """Return a cached tokenizer loaded from ``name``.

    Args:
        name: Hugging Face model/tokenizer id or local tokenizer path.
        revision: Optional model revision to load.
        **kwargs: Additional ``AutoTokenizer.from_pretrained`` keyword arguments.

    Returns:
        A shared tokenizer instance for the same ``name``, ``revision``, and kwargs.
    """
    return get_tokenizer_with_loader(
        name, _default_tokenizer_loader(), revision, **kwargs
    )


def get_tokenizer_with_loader(
    name: str | PathLike[str],
    loader: Callable[..., Any],
    revision: str | None = None,
    *,
    refresh_cache: bool = False,
    **kwargs: Any,
) -> Any:
    """Return a cached tokenizer using an explicit ``from_pretrained`` loader."""
    cache_key = _cache_key(name, revision, kwargs)
    with _TOKENIZER_CACHE_LOCK:
        cached = _TOKENIZER_CACHE.get(cache_key)
        if cached is not None and not refresh_cache:
            _TOKENIZER_CACHE.move_to_end(cache_key)
            return cached

        load_kwargs = dict(kwargs)
        if revision is not None:
            load_kwargs["revision"] = revision

        tokenizer = loader(name, **load_kwargs)
        _TOKENIZER_CACHE[cache_key] = tokenizer
        _TOKENIZER_CACHE.move_to_end(cache_key)
        while len(_TOKENIZER_CACHE) > DEFAULT_TOKENIZER_CACHE_SIZE:
            _TOKENIZER_CACHE.popitem(last=False)
        return tokenizer


def clear_tokenizer_cache() -> None:
    """Clear all cached tokenizer instances."""
    with _TOKENIZER_CACHE_LOCK:
        _TOKENIZER_CACHE.clear()


def _default_tokenizer_loader() -> Callable[..., Any]:
    if AutoTokenizer is None:
        raise ImportError(
            "HuggingFace transformers is required to load tokenizers. "
            "Install with: pip install transformers"
        )
    return AutoTokenizer.from_pretrained


def _cache_key(
    name: str | PathLike[str],
    revision: str | None,
    kwargs: Mapping[str, Any],
) -> tuple[Any, ...]:
    return (
        str(name),
        revision,
        tuple(
            sorted(
                (str(key), _freeze_cache_value(value)) for key, value in kwargs.items()
            )
        ),
    )


def _freeze_cache_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(
            sorted((str(key), _freeze_cache_value(item)) for key, item in value.items())
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_cache_value(item) for item in value)
    if isinstance(value, set):
        return tuple(
            sorted(
                (_freeze_cache_value(item) for item in value),
                key=repr,
            )
        )
    if isinstance(value, PathLike):
        return str(value)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return repr(value)
