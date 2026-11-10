"""Privacy gateway wrappers for external text model calls.

The gateway keeps the reversible redaction mapping in memory, sends only
redacted text to user-supplied callables, and restores identifiers locally from
the callable response.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any

Deidentifier = Callable[..., Any]
Reidentifier = Callable[[str, Mapping[str, str]], str]
TextModelCall = Callable[..., str]


class RedactionMapping(Mapping[str, str]):
    """In-memory redaction mapping with a PHI-safe representation."""

    __slots__ = ("_data",)

    def __init__(self, mapping: Mapping[str, str] | None = None) -> None:
        self._data = dict(mapping or {})

    def __getitem__(self, key: str) -> str:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        entry = "entry" if len(self) == 1 else "entries"
        return f"{self.__class__.__name__}(<{len(self)} {entry}>)"

    __str__ = __repr__

    def to_dict(self) -> dict[str, str]:
        """Return a shallow copy for APIs that require a concrete ``dict``."""

        return dict(self._data)


@dataclass(frozen=True)
class PrivacyGatewayConfig:
    """Runtime options forwarded to ``openmed.core.pii.deidentify``."""

    method: str = "mask"
    model_name: str | None = None
    confidence_threshold: float = 0.7
    keep_year: bool = False
    shift_dates: bool | None = None
    date_shift_days: int | None = None
    patient_key: str | bytes | None = None
    date_shift_max_days: int | None = None
    date_shift_secret: str | bytes | None = None
    use_smart_merging: bool = True
    lang: str = "en"
    normalize_accents: bool | None = None
    use_safety_sweep: bool = True
    consistent: bool = False
    seed: int | None = None
    locale: str | None = None
    policy: str | None = None
    calibration_thresholds_path: str | Path | None = None
    extra_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def to_deidentify_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments for a reversible gateway redaction pass."""

        kwargs: dict[str, Any] = {
            "method": self.method,
            "confidence_threshold": float(self.confidence_threshold),
            "keep_year": self.keep_year,
            "shift_dates": self.shift_dates,
            "date_shift_days": self.date_shift_days,
            "patient_key": self.patient_key,
            "date_shift_max_days": self.date_shift_max_days,
            "date_shift_secret": self.date_shift_secret,
            "keep_mapping": True,
            "use_smart_merging": self.use_smart_merging,
            "lang": self.lang,
            "normalize_accents": self.normalize_accents,
            "use_safety_sweep": self.use_safety_sweep,
            "consistent": self.consistent,
            "seed": self.seed,
            "locale": self.locale,
            "policy": self.policy,
            "calibration_thresholds_path": self.calibration_thresholds_path,
        }
        if self.model_name is not None:
            kwargs["model_name"] = self.model_name

        kwargs.update(dict(self.extra_kwargs))
        kwargs["keep_mapping"] = True
        kwargs["audit"] = False
        return {key: value for key, value in kwargs.items() if value is not None}


class PrivacyGateway:
    """Run redact -> external callable -> restore for text model calls."""

    def __init__(
        self,
        *,
        config: PrivacyGatewayConfig | None = None,
        deidentifier: Deidentifier | None = None,
        reidentifier: Reidentifier | None = None,
    ) -> None:
        self.config = config or PrivacyGatewayConfig()
        self._deidentifier = deidentifier
        self._reidentifier = reidentifier

    def redact(self, text: str) -> tuple[str, RedactionMapping]:
        """Return redacted text plus an in-memory restore mapping."""

        result = self._deidentifier_or_default()(
            text,
            **self.config.to_deidentify_kwargs(),
        )
        clean_text = _result_text(result)
        mapping = _result_mapping(result)
        return clean_text, RedactionMapping(mapping)

    def restore(self, text: str, mapping: Mapping[str, str]) -> str:
        """Restore redacted identifiers in ``text`` from ``mapping``."""

        if not mapping:
            return text
        restored = self._reidentifier_or_default()(text, RedactionMapping(mapping))
        if not isinstance(restored, str):
            raise TypeError("reidentifier must return a string")
        return restored

    def assert_redacted(self, text: str, mapping: Mapping[str, str]) -> str:
        """Return ``text`` if no mapped original identifiers are present."""

        return assert_redacted(text, mapping)

    def input_guardrail(self, mapping: Mapping[str, str]) -> Callable[[str], str]:
        """Create a pre-call guardrail that rejects known identifier leaks."""

        protected_mapping = RedactionMapping(mapping)

        def guardrail(text: str) -> str:
            return self.assert_redacted(text, protected_mapping)

        return guardrail

    def output_guardrail(self, mapping: Mapping[str, str]) -> Callable[[str], str]:
        """Create a post-call guardrail that restores local identifiers."""

        protected_mapping = RedactionMapping(mapping)

        def guardrail(text: str) -> str:
            return self.restore(text, protected_mapping)

        return guardrail

    def guarded(self, call: TextModelCall) -> TextModelCall:
        """Wrap ``call`` so it only receives redacted text."""

        @wraps(call)
        def wrapper(text: str, *args: Any, **kwargs: Any) -> str:
            clean_text, mapping = self.redact(text)
            self.assert_redacted(clean_text, mapping)
            response = call(clean_text, *args, **kwargs)
            if not isinstance(response, str):
                raise TypeError("guarded callable must return a string")
            return self.restore(response, mapping)

        return wrapper

    def _deidentifier_or_default(self) -> Deidentifier:
        if self._deidentifier is not None:
            return self._deidentifier

        from openmed.core.pii import deidentify

        return deidentify

    def _reidentifier_or_default(self) -> Reidentifier:
        if self._reidentifier is not None:
            return self._reidentifier

        from openmed.core.pii import reidentify

        return reidentify


def assert_redacted(text: str, mapping: Mapping[str, str]) -> str:
    """Raise if ``text`` still contains any original mapped identifier."""

    leak_count = _known_identifier_leak_count(text, mapping)
    if leak_count:
        noun = "identifier" if leak_count == 1 else "identifiers"
        raise ValueError(
            f"text contains {leak_count} original {noun}; redact before external call"
        )
    return text


def restore_text(
    text: str,
    mapping: Mapping[str, str],
    *,
    reidentifier: Reidentifier | None = None,
) -> str:
    """Restore ``text`` using ``mapping`` without constructing a gateway."""

    restore = reidentifier
    if restore is None:
        from openmed.core.pii import reidentify

        restore = reidentify
    return restore(text, RedactionMapping(mapping))


def _result_text(result: Any) -> str:
    if isinstance(result, str):
        raise TypeError("deidentifier must return an object with deidentified_text")
    try:
        clean_text = result.deidentified_text
    except AttributeError as exc:
        raise TypeError(
            "deidentifier must return an object with deidentified_text"
        ) from exc
    if not isinstance(clean_text, str):
        raise TypeError("deidentifier result deidentified_text must be a string")
    return clean_text


def _result_mapping(result: Any) -> Mapping[str, str]:
    mapping = getattr(result, "mapping", None) or {}
    if not isinstance(mapping, Mapping):
        raise TypeError("deidentifier result mapping must be a mapping")
    return {str(key): str(value) for key, value in mapping.items()}


def _known_identifier_leak_count(
    text: str,
    mapping: Mapping[str, str],
) -> int:
    originals = {
        original
        for original in mapping.values()
        if isinstance(original, str) and original
    }
    return sum(1 for original in originals if original in text)


__all__ = [
    "Deidentifier",
    "PrivacyGateway",
    "PrivacyGatewayConfig",
    "RedactionMapping",
    "Reidentifier",
    "TextModelCall",
    "assert_redacted",
    "restore_text",
]
