"""Deprecation helpers for intentionally retired public APIs."""

from __future__ import annotations

import warnings
from functools import wraps
from typing import Any, Callable, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])

__all__ = ["deprecated"]


def deprecated(
    *,
    since: str,
    remove_in: str,
    replacement: str | None = None,
) -> Callable[[F], F]:
    """Mark a callable as deprecated and warn when it is used.

    The metadata is intentionally available on the decorated object for local
    documentation tools. The release API-surface differ recognizes the static
    ``@deprecated(...)`` decorator directly, without importing OpenMed.

    Args:
        since: First OpenMed version that warns about the callable.
        remove_in: Earliest release in which removal is planned.
        replacement: Optional fully qualified replacement callable.

    Returns:
        A decorator that preserves the wrapped callable's identity and signature.

    Raises:
        ValueError: If a required version string is empty.
    """

    if not since.strip():
        raise ValueError("since must not be empty")
    if not remove_in.strip():
        raise ValueError("remove_in must not be empty")

    def decorator(target: F) -> F:
        qualified_name = getattr(
            target, "__qualname__", getattr(target, "__name__", "API")
        )
        message = (
            f"{qualified_name} is deprecated since OpenMed {since} and is planned "
            f"for removal in {remove_in}."
        )
        if replacement:
            message += f" Use {replacement} instead."
        metadata = {
            "since": since,
            "remove_in": remove_in,
            "replacement": replacement,
        }

        if isinstance(target, type):
            original_init = target.__init__

            @wraps(original_init)
            def warned_init(self: Any, *args: Any, **kwargs: Any) -> None:
                warnings.warn(message, DeprecationWarning, stacklevel=2)
                original_init(self, *args, **kwargs)

            target.__init__ = warned_init  # type: ignore[method-assign]
            setattr(target, "__openmed_deprecated__", metadata)
            return target

        @wraps(target)
        def warned(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return target(*args, **kwargs)

        setattr(warned, "__openmed_deprecated__", metadata)
        return cast(F, warned)

    return decorator
