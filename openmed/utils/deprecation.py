"""Deprecation helpers for intentionally retired public APIs."""

from __future__ import annotations

import inspect
import warnings
from functools import wraps
from importlib import import_module
from typing import Any, Callable, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])

__all__ = [
    "deprecated",
    "deprecated_attribute",
    "deprecated_module",
    "deprecated_parameter",
    "warn_deprecated",
]


def _validate_versions(since: str, remove_in: str) -> None:
    if not since.strip():
        raise ValueError("since must not be empty")
    if not remove_in.strip():
        raise ValueError("remove_in must not be empty")


def _message(name: str, since: str, remove_in: str, replacement: str | None) -> str:
    message = (
        f"{name} is deprecated since OpenMed {since} and is planned "
        f"for removal in {remove_in}."
    )
    if replacement:
        message += f" Use {replacement} instead."
    return message


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

    _validate_versions(since, remove_in)

    def decorator(target: F) -> F:
        qualified_name = getattr(
            target, "__qualname__", getattr(target, "__name__", "API")
        )
        message = _message(qualified_name, since, remove_in, replacement)
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


def deprecated_parameter(
    parameter: str,
    /,
    *,
    since: str,
    remove_in: str,
    replacement: str | None = None,
) -> Callable[[F], F]:
    """Warn when a caller explicitly supplies a deprecated parameter.

    Args:
        parameter: Name in the wrapped function or method's signature.
        since: First OpenMed version that warns about the parameter.
        remove_in: Earliest release in which removal is planned.
        replacement: Optional replacement parameter or API name.

    Returns:
        A decorator that preserves the signature and the wrapped callable.

    Raises:
        ValueError: If either version is empty or the parameter does not exist.
    """
    _validate_versions(since, remove_in)

    def decorator(target: F) -> F:
        signature = inspect.signature(target)
        if parameter not in signature.parameters:
            raise ValueError(f"Unknown deprecated parameter {parameter!r}")
        qualified_name = getattr(target, "__qualname__", type(target).__qualname__)
        message = _message(
            f"Parameter {parameter!r} of {qualified_name}",
            since,
            remove_in,
            replacement,
        )

        def warn_if_supplied(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            try:
                arguments = signature.bind(*args, **kwargs).arguments
            except TypeError:
                # Let the wrapped function raise its usual argument error.
                return
            if parameter in arguments:
                warnings.warn(message, DeprecationWarning, stacklevel=3)

        if inspect.iscoroutinefunction(target):

            @wraps(target)
            async def warned_async(*args: Any, **kwargs: Any) -> Any:
                warn_if_supplied(args, kwargs)
                return await target(*args, **kwargs)

            return cast(F, warned_async)

        @wraps(target)
        def warned(*args: Any, **kwargs: Any) -> Any:
            warn_if_supplied(args, kwargs)
            return target(*args, **kwargs)

        return cast(F, warned)

    return decorator


class _DeprecatedAttribute:
    def __init__(self, replacement: str, since: str, remove_in: str) -> None:
        self.replacement = replacement
        self.since = since
        self.remove_in = remove_in

    def __set_name__(self, owner: type, name: str) -> None:
        if name == self.replacement:
            raise ValueError("Deprecated attribute must alias a different attribute")
        self.message = _message(
            f"{owner.__qualname__}.{name}", self.since, self.remove_in, self.replacement
        )

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        if instance is None:
            return self
        warnings.warn(self.message, DeprecationWarning, stacklevel=2)
        return getattr(instance, self.replacement)

    def __set__(self, instance: Any, value: Any) -> None:
        warnings.warn(self.message, DeprecationWarning, stacklevel=2)
        setattr(instance, self.replacement, value)

    def __delete__(self, instance: Any) -> None:
        warnings.warn(self.message, DeprecationWarning, stacklevel=2)
        delattr(instance, self.replacement)


def deprecated_attribute(
    replacement: str, /, *, since: str, remove_in: str
) -> _DeprecatedAttribute:
    """Create a warning alias for another instance attribute on the same class.

    Args:
        replacement: Name of the attribute that handles reads, writes and deletion.
        since: First OpenMed version that warns about the alias.
        remove_in: Earliest release in which removal is planned.

    Returns:
        A descriptor to assign to the old attribute name in the class body.

    Raises:
        ValueError: If either version is empty or replacement is not a name.
    """
    _validate_versions(since, remove_in)
    if not replacement.isidentifier():
        raise ValueError("replacement must be an attribute name")
    return _DeprecatedAttribute(replacement, since, remove_in)


def deprecated_module(
    module: str, /, *, since: str, remove_in: str, replacement: str
) -> Callable[[str], Any]:
    """Build a module ``__getattr__`` that warns and forwards to a new module.

    Args:
        module: Name of the old module, typically ``__name__``.
        since: First OpenMed version that warns about the old import path.
        remove_in: Earliest release in which removal is planned.
        replacement: Absolute import path of the new module.

    Returns:
        A lazy getter that returns the original object from the new module.

    Raises:
        ValueError: If either version is empty or the replacement path is invalid.
    """
    _validate_versions(since, remove_in)
    if replacement == module or not all(
        part.isidentifier() for part in replacement.split(".")
    ):
        raise ValueError("replacement must name a different absolute module")
    message = _message(module, since, remove_in, replacement)

    def __getattr__(name: str) -> Any:
        if not name.startswith("__"):
            target = import_module(replacement)
            try:
                value = getattr(target, name)
            except AttributeError:
                pass
            else:
                warnings.warn(message, DeprecationWarning, stacklevel=2)
                return value
        raise AttributeError(f"module {module!r} has no attribute {name!r}")

    return __getattr__


def warn_deprecated(
    code: str, since: str, remove_in: str, *, replacement: str | None = None
) -> None:
    """Emit a coded warning for a setting, response shape or other non-symbol API.

    Args:
        code: Stable, application-assigned identifier, prefixed as ``[code]``.
        since: First OpenMed version that emits this warning.
        remove_in: Earliest release in which removal is planned.
        replacement: Optional replacement setting or API name.

    Raises:
        ValueError: If the code or either version is empty.
    """
    _validate_versions(since, remove_in)
    if not code.strip():
        raise ValueError("code must not be empty")
    warnings.warn(
        _message(f"[{code}] This behavior", since, remove_in, replacement),
        DeprecationWarning,
        stacklevel=2,
    )
