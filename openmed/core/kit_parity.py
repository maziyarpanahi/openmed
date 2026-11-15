"""Versioned OpenMedKit kit-parity result profile.

Every OpenMedKit surface (Python, Swift, Kotlin, the TypeScript web package,
and the React Native and Electron projections) can project its native
de-identification result into one small, value-free record and serialize it to
the same canonical JSON bytes. ``tests/fixtures/parity/kit_parity_vectors.json``
holds the shared golden vectors; this module is the Python reference
implementation of the profile.

Profile ``openmed.kit_parity_result`` version 1:

* Offsets are half-open ``[start, end)`` Unicode scalar (code point) offsets
  into the exact, unnormalized source text. UTF-8 and UTF-16 offsets are
  derived through :func:`scalar_to_utf8_offset` and
  :func:`scalar_to_utf16_offset` and never stored in a result.
* ``label`` is the detector label, preserved byte for byte; ``canonical_label``
  must be one of :data:`KIT_CANONICAL_LABELS`; ``action`` must be one of
  :data:`KIT_ACTIONS`.
* Spans are ordered by ``(start, end)``, non-empty, and non-overlapping.
* ``confidence_ppm`` is the confidence in parts per million, computed as
  ``floor(confidence * 1_000_000 + 0.5)`` in double precision. It is omitted
  when a surface has no score.
* ``replacement`` is omitted for ``keep`` and required for every other action.
  Absent optional fields are omitted; ``null`` is never canonical.
* Canonical JSON uses lexicographically sorted keys, no insignificant
  whitespace, integers only, and RFC 8785 string escaping: ``"`` and ``\\``,
  the short escapes ``\\b \\f \\n \\r \\t``, ``\\u00xx`` (lowercase hex) for the
  remaining C0 controls, and raw UTF-8 for everything else.

The record carries no source text, span surface text, hashes of source values,
or reversible mappings. :class:`KitParityError` reports a stable reason code and
a JSON field path only; unknown keys are reported at the enclosing object path
so a key name is never echoed.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .pii import DeidentificationResult, PIIEntity

PROFILE_NAME: Final = "openmed.kit_parity_result"
PROFILE_VERSION: Final = 1
SUPPORTED_PROFILE_VERSIONS: Final[frozenset[int]] = frozenset({PROFILE_VERSION})
OFFSET_UNIT: Final = "unicode_scalar"
CONFIDENCE_SCALE: Final = 1_000_000

#: Offset-contract versions (``tests/fixtures/parity/offset_contract.json``)
#: whose scalar and grapheme semantics this profile accepts.
OFFSET_CONTRACT_VERSIONS: Final[frozenset[int]] = frozenset({1})
#: Span decoder contract versions (the BIO span decoder emitting span schema
#: version 1) whose output this profile accepts.
DECODER_CONTRACT_VERSIONS: Final[frozenset[int]] = frozenset({1})

#: Policy actions a kit result may carry.
KIT_ACTIONS: Final[tuple[str, ...]] = (
    "hash",
    "keep",
    "mask",
    "redact",
    "remove",
    "replace",
)

#: Canonical labels a kit result may carry: the OpenMedKit policy taxonomy.
KIT_CANONICAL_LABELS: Final[tuple[str, ...]] = (
    "ACCOUNT_NUMBER",
    "AGE",
    "AMOUNT",
    "ANTIBIOTIC",
    "API_KEY",
    "BIC",
    "BITCOIN_ADDRESS",
    "BODY_SITE",
    "BUILDING_NUMBER",
    "CONDITION",
    "CREDIT_CARD",
    "CREDIT_CARD_ISSUER",
    "CURRENCY",
    "CVV",
    "DATE",
    "DATE_OF_BIRTH",
    "DIET_TYPE",
    "EMAIL",
    "ETHEREUM_ADDRESS",
    "EYE_COLOR",
    "FEEDING_ROUTE",
    "FIRST_NAME",
    "GENDER",
    "GPS_COORDINATES",
    "HEIGHT",
    "IBAN",
    "ID_NUM",
    "IMEI",
    "IP_ADDRESS",
    "JOB_DEPARTMENT",
    "JOB_TITLE",
    "LAB_TEST",
    "LAST_NAME",
    "LITECOIN_ADDRESS",
    "LOCATION",
    "MAC_ADDRESS",
    "MASKED_NUMBER",
    "MEDICATION",
    "MICROORGANISM",
    "MIDDLE_NAME",
    "NUTRITIONAL_STATUS",
    "NUTRITION_TARGET",
    "OCCUPATION",
    "ORDINAL_DIRECTION",
    "ORGANIZATION",
    "OTHER",
    "PASSWORD",
    "PERSON",
    "PHONE",
    "PIN",
    "PREFIX",
    "PROCEDURE",
    "SSN",
    "STREET_ADDRESS",
    "SUSCEPTIBILITY",
    "TIME",
    "URL",
    "USERNAME",
    "USER_AGENT",
    "VEHICLE_REGISTRATION",
    "VIN",
    "ZIPCODE",
)

#: Detector labels: ASCII, starting with a letter, at most 80 characters.
LABEL_PATTERN: Final = re.compile(r"[A-Za-z][A-Za-z0-9_:-]{0,79}\Z", re.ASCII)

_ACTIONS: Final = frozenset(KIT_ACTIONS)
_LABELS: Final = frozenset(KIT_CANONICAL_LABELS)
_TOP_LEVEL_KEYS: Final = frozenset(
    {"deidentified_text", "offset_unit", "profile", "profile_version", "spans"}
)
_SPAN_KEYS: Final = frozenset(
    {
        "action",
        "canonical_label",
        "confidence_ppm",
        "end",
        "label",
        "replacement",
        "start",
    }
)
_MISSING: Final = object()


class KitParityError(ValueError):
    """A value-free kit-parity profile violation.

    Attributes:
        code: Stable reason code such as ``action_unknown``.
        field_path: JSON path of the offending field, for example
            ``$.spans[2].end``. Unknown keys are reported at the enclosing
            object path.
    """

    def __init__(self, code: str, field_path: str = "$") -> None:
        self.code = code
        self.field_path = field_path
        super().__init__(f"{code} at {field_path}")


@dataclass(frozen=True)
class KitParitySpan:
    """One applied span in a kit-parity result.

    Attributes:
        label: Detector label, preserved byte for byte.
        canonical_label: Canonical label from :data:`KIT_CANONICAL_LABELS`.
        action: Policy action from :data:`KIT_ACTIONS`.
        start: Inclusive Unicode scalar offset into the source text.
        end: Exclusive Unicode scalar offset into the source text.
        confidence_ppm: Confidence in parts per million, or ``None`` when the
            surface has no score.
        replacement: Replacement text, or ``None`` for ``keep``.
    """

    label: str
    canonical_label: str
    action: str
    start: int
    end: int
    confidence_ppm: int | None = None
    replacement: str | None = None


@dataclass(frozen=True)
class KitParityResult:
    """A kit-parity projection of one de-identification result.

    Attributes:
        deidentified_text: The exact de-identified output text.
        spans: Applied spans ordered by ``(start, end)``.
    """

    deidentified_text: str
    spans: tuple[KitParitySpan, ...] = ()


def scalar_length(text: str) -> int:
    """Return the number of Unicode scalar values in ``text``.

    Args:
        text: Source text.

    Returns:
        The scalar (code point) length.
    """
    return len(text)


def scalar_to_utf8_offset(text: str, offset: int) -> int:
    """Convert a Unicode scalar offset into a UTF-8 byte offset.

    Args:
        text: Source text.
        offset: Scalar offset in ``[0, len(text)]``.

    Returns:
        The number of UTF-8 bytes preceding ``offset``.

    Raises:
        KitParityError: ``offset_out_of_bounds`` when ``offset`` is outside
            the text.
    """
    _check_scalar_offset(text, offset)
    return len(text[:offset].encode("utf-8"))


def scalar_to_utf16_offset(text: str, offset: int) -> int:
    """Convert a Unicode scalar offset into a UTF-16 code-unit offset.

    Args:
        text: Source text.
        offset: Scalar offset in ``[0, len(text)]``.

    Returns:
        The number of UTF-16 code units preceding ``offset``.

    Raises:
        KitParityError: ``offset_out_of_bounds`` when ``offset`` is outside
            the text.
    """
    _check_scalar_offset(text, offset)
    return len(text[:offset].encode("utf-16-le")) // 2


def utf8_to_scalar_offset(text: str, offset: int) -> int:
    """Convert a UTF-8 byte offset into a Unicode scalar offset.

    Args:
        text: Source text.
        offset: UTF-8 byte offset.

    Returns:
        The scalar offset at ``offset``.

    Raises:
        KitParityError: ``offset_out_of_bounds`` outside the text, or
            ``offset_not_scalar_boundary`` inside a multi-byte sequence.
    """
    return _native_to_scalar(text, offset, lambda char: len(char.encode("utf-8")))


def utf16_to_scalar_offset(text: str, offset: int) -> int:
    """Convert a UTF-16 code-unit offset into a Unicode scalar offset.

    Args:
        text: Source text.
        offset: UTF-16 code-unit offset.

    Returns:
        The scalar offset at ``offset``.

    Raises:
        KitParityError: ``offset_out_of_bounds`` outside the text, or
            ``offset_not_scalar_boundary`` between the halves of a surrogate
            pair.
    """
    return _native_to_scalar(text, offset, lambda char: 2 if ord(char) > 0xFFFF else 1)


def confidence_to_ppm(confidence: float) -> int:
    """Encode a confidence in ``[0, 1]`` as integer parts per million.

    Args:
        confidence: Model confidence. Single-precision values are widened to
            double precision before scaling.

    Returns:
        ``floor(confidence * 1_000_000 + 0.5)``.

    Raises:
        KitParityError: ``confidence_invalid`` for non-finite values or values
            outside ``[0, 1]``.
    """
    value = float(confidence)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise KitParityError("confidence_invalid", "$.confidence")
    return math.floor(value * CONFIDENCE_SCALE + 0.5)


def check_compatibility(
    *, offset_contract_version: int, decoder_contract_version: int
) -> None:
    """Reject bundle contracts this profile cannot interpret.

    Call this before consuming any result produced under a model bundle, so
    an incompatible decoder or offset contract fails before its results are
    accepted.

    Args:
        offset_contract_version: Offset-contract version the bundle declares.
        decoder_contract_version: Span decoder contract version the bundle
            declares.

    Raises:
        KitParityError: ``offset_contract_version_unsupported`` or
            ``decoder_contract_version_unsupported``.
    """
    if not _is_int(offset_contract_version) or (
        offset_contract_version not in OFFSET_CONTRACT_VERSIONS
    ):
        raise KitParityError(
            "offset_contract_version_unsupported", "$.offset_contract_version"
        )
    if not _is_int(decoder_contract_version) or (
        decoder_contract_version not in DECODER_CONTRACT_VERSIONS
    ):
        raise KitParityError(
            "decoder_contract_version_unsupported", "$.decoder_contract_version"
        )


def validate_result(result: KitParityResult) -> None:
    """Validate a result against the profile vocabulary and ordering rules.

    Args:
        result: Result to validate.

    Raises:
        KitParityError: On the first violation, in canonical field order.
    """
    if not isinstance(result.deidentified_text, str):
        raise KitParityError("field_type_invalid", "$.deidentified_text")
    previous: KitParitySpan | None = None
    for index, span in enumerate(result.spans):
        path = f"$.spans[{index}]"
        _validate_span(span, path)
        _check_order(span, previous, path)
        previous = span


def serialize_result(result: KitParityResult) -> str:
    """Return the canonical JSON serialization of a result.

    Args:
        result: Result to serialize.

    Returns:
        Canonical JSON text; encode it as UTF-8 for the canonical bytes.

    Raises:
        KitParityError: When the result violates the profile.
    """
    validate_result(result)
    spans = []
    for span in result.spans:
        record: dict[str, Any] = {
            "action": span.action,
            "canonical_label": span.canonical_label,
            "end": span.end,
            "label": span.label,
            "start": span.start,
        }
        if span.confidence_ppm is not None:
            record["confidence_ppm"] = span.confidence_ppm
        if span.replacement is not None:
            record["replacement"] = span.replacement
        spans.append(record)
    payload = {
        "deidentified_text": result.deidentified_text,
        "offset_unit": OFFSET_UNIT,
        "profile": PROFILE_NAME,
        "profile_version": PROFILE_VERSION,
        "spans": spans,
    }
    return json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def parse_result(payload: str | bytes) -> KitParityResult:
    """Parse and strictly validate canonical kit-parity result JSON.

    Checks run in a fixed order shared by every surface: JSON syntax, object
    shape, profile name, profile version, unknown top-level keys, offset unit,
    de-identified text, then each span field in canonical key order, span
    ordering, and finally that the input bytes are the canonical bytes.

    Args:
        payload: Canonical JSON as text or UTF-8 bytes.

    Returns:
        The parsed result.

    Raises:
        KitParityError: On the first violation.
    """
    if isinstance(payload, bytes):
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise KitParityError("json_invalid") from error
    else:
        text = payload
    try:
        document = json.loads(text, parse_constant=_reject_constant)
    except ValueError as error:
        raise KitParityError("json_invalid") from error
    if not isinstance(document, dict):
        raise KitParityError("result_not_object")

    profile = _required(document, "profile", "$", str)
    if profile != PROFILE_NAME:
        raise KitParityError("profile_unknown", "$.profile")
    version = _required(document, "profile_version", "$", int)
    if version not in SUPPORTED_PROFILE_VERSIONS:
        raise KitParityError("profile_version_unsupported", "$.profile_version")
    if set(document) - _TOP_LEVEL_KEYS:
        raise KitParityError("field_unknown", "$")
    if _required(document, "offset_unit", "$", str) != OFFSET_UNIT:
        raise KitParityError("offset_unit_unsupported", "$.offset_unit")
    deidentified_text = _required(document, "deidentified_text", "$", str)
    raw_spans = _required(document, "spans", "$", list)

    spans: list[KitParitySpan] = []
    previous: KitParitySpan | None = None
    for index, raw in enumerate(raw_spans):
        path = f"$.spans[{index}]"
        span = _parse_span(raw, path)
        _check_order(span, previous, path)
        spans.append(span)
        previous = span

    result = KitParityResult(deidentified_text=deidentified_text, spans=tuple(spans))
    if serialize_result(result) != text:
        raise KitParityError("result_not_canonical")
    return result


def accept_result(
    payload: str | bytes,
    *,
    offset_contract_version: int,
    decoder_contract_version: int,
) -> KitParityResult:
    """Gate a bundle's contracts, then parse a result produced under it.

    Args:
        payload: Canonical result JSON.
        offset_contract_version: Offset-contract version the bundle declares.
        decoder_contract_version: Decoder contract version the bundle declares.

    Returns:
        The parsed result.

    Raises:
        KitParityError: A compatibility reason before the payload is read, or
            the first parse violation.
    """
    check_compatibility(
        offset_contract_version=offset_contract_version,
        decoder_contract_version=decoder_contract_version,
    )
    return parse_result(payload)


def apply_spans(source_text: str, spans: Sequence[KitParitySpan]) -> str:
    """Apply span replacements right to left at scalar offsets.

    Args:
        source_text: Original text.
        spans: Ordered, non-overlapping spans. ``keep`` spans are left as is.

    Returns:
        The de-identified text implied by the spans.

    Raises:
        KitParityError: When the spans violate the profile, or
            ``offset_out_of_bounds`` when a span ends past the source text.
    """
    validate_result(KitParityResult(deidentified_text="", spans=tuple(spans)))
    length = scalar_length(source_text)
    for index, span in enumerate(spans):
        if span.end > length:
            raise KitParityError("offset_out_of_bounds", f"$.spans[{index}].end")
    result = source_text
    for span in reversed(spans):
        if span.replacement is not None:
            result = result[: span.start] + span.replacement + result[span.end :]
    return result


def verify_result(result: KitParityResult, source_text: str) -> None:
    """Check that a result's offsets and bytes agree with its source text.

    Args:
        result: Parsed or projected result.
        source_text: The original text the result was produced from.

    Raises:
        KitParityError: ``offset_out_of_bounds`` or
            ``deidentified_text_mismatch``.
    """
    validate_result(result)
    expected = apply_spans(source_text, result.spans)
    if expected.encode("utf-8") != result.deidentified_text.encode("utf-8"):
        raise KitParityError("deidentified_text_mismatch", "$.deidentified_text")


def from_deidentification_result(result: DeidentificationResult) -> KitParityResult:
    """Project a Python de-identification result into the kit-parity profile.

    ``canonical_label`` falls back to :func:`openmed.core.labels.normalize_label`
    of the entity label when the entity carries none. Methods outside
    :data:`KIT_ACTIONS` (for example ``shift_dates`` or ``format_preserve``)
    are rejected rather than mapped.

    Args:
        result: Result returned by ``openmed.deidentify``.

    Returns:
        The validated projection.

    Raises:
        KitParityError: When an entity cannot be represented in the profile.
    """
    from .labels import normalize_label

    located: list[tuple[int, int, PIIEntity]] = []
    for index, entity in enumerate(result.pii_entities):
        start, end = entity.start, entity.end
        if start is None or not _is_int(start):
            raise KitParityError("field_missing", f"$.spans[{index}].start")
        if end is None or not _is_int(end):
            raise KitParityError("field_missing", f"$.spans[{index}].end")
        located.append((start, end, entity))
    located.sort(key=lambda item: (item[0], item[1]))
    spans: list[KitParitySpan] = []
    for index, (start, end, entity) in enumerate(located):
        path = f"$.spans[{index}]"
        action = entity.action
        if not action:
            raise KitParityError("field_missing", f"{path}.action")
        canonical = entity.canonical_label or normalize_label(entity.label)
        replacement: str | None = None
        if action != "keep":
            if entity.redacted_text is None:
                raise KitParityError("field_missing", f"{path}.replacement")
            replacement = entity.redacted_text
        try:
            confidence = confidence_to_ppm(entity.confidence)
        except KitParityError as error:
            raise KitParityError(error.code, f"{path}.confidence_ppm") from None
        spans.append(
            KitParitySpan(
                label=entity.label,
                canonical_label=canonical,
                action=action,
                start=start,
                end=end,
                confidence_ppm=confidence,
                replacement=replacement,
            )
        )
    projected = KitParityResult(
        deidentified_text=result.deidentified_text, spans=tuple(spans)
    )
    validate_result(projected)
    return projected


def _parse_span(raw: object, path: str) -> KitParitySpan:
    if not isinstance(raw, dict):
        raise KitParityError("field_type_invalid", path)
    if set(raw) - _SPAN_KEYS:
        raise KitParityError("field_unknown", path)
    label = _required(raw, "label", path, str)
    _check_label(label, path)
    canonical_label = _required(raw, "canonical_label", path, str)
    _check_canonical_label(canonical_label, path)
    action = _required(raw, "action", path, str)
    _check_action(action, path)
    start = _required(raw, "start", path, int)
    _check_start(start, path)
    end = _required(raw, "end", path, int)
    _check_end(start, end, path)
    confidence = _optional(raw, "confidence_ppm", path, int)
    _check_confidence(confidence, path)
    replacement = _optional(raw, "replacement", path, str)
    _check_replacement(action, replacement, path)
    return KitParitySpan(
        label=label,
        canonical_label=canonical_label,
        action=action,
        start=start,
        end=end,
        confidence_ppm=confidence,
        replacement=replacement,
    )


def _validate_span(span: KitParitySpan, path: str) -> None:
    _typed(span.label, f"{path}.label", str)
    _check_label(span.label, path)
    _typed(span.canonical_label, f"{path}.canonical_label", str)
    _check_canonical_label(span.canonical_label, path)
    _typed(span.action, f"{path}.action", str)
    _check_action(span.action, path)
    _typed(span.start, f"{path}.start", int)
    _check_start(span.start, path)
    _typed(span.end, f"{path}.end", int)
    _check_end(span.start, span.end, path)
    if span.confidence_ppm is not None:
        _typed(span.confidence_ppm, f"{path}.confidence_ppm", int)
    _check_confidence(span.confidence_ppm, path)
    if span.replacement is not None:
        _typed(span.replacement, f"{path}.replacement", str)
    _check_replacement(span.action, span.replacement, path)


def _check_label(label: str, path: str) -> None:
    if not LABEL_PATTERN.match(label):
        raise KitParityError("label_invalid", f"{path}.label")


def _check_canonical_label(label: str, path: str) -> None:
    if label not in _LABELS:
        raise KitParityError("label_unknown", f"{path}.canonical_label")


def _check_action(action: str, path: str) -> None:
    if action not in _ACTIONS:
        raise KitParityError("action_unknown", f"{path}.action")


def _check_start(start: int, path: str) -> None:
    if start < 0:
        raise KitParityError("offset_invalid", f"{path}.start")


def _check_end(start: int, end: int, path: str) -> None:
    if end <= start:
        raise KitParityError("offset_invalid", f"{path}.end")


def _check_confidence(confidence: int | None, path: str) -> None:
    if confidence is not None and not 0 <= confidence <= CONFIDENCE_SCALE:
        raise KitParityError("confidence_invalid", f"{path}.confidence_ppm")


def _check_replacement(action: str, replacement: str | None, path: str) -> None:
    if action == "keep":
        if replacement is not None:
            raise KitParityError("replacement_unexpected", f"{path}.replacement")
    elif replacement is None:
        raise KitParityError("field_missing", f"{path}.replacement")


def _check_order(
    span: KitParitySpan, previous: KitParitySpan | None, path: str
) -> None:
    if previous is None:
        return
    if (span.start, span.end) <= (previous.start, previous.end):
        raise KitParityError("span_order_invalid", path)
    if span.start < previous.end:
        raise KitParityError("span_overlap", path)


def _required(document: Mapping[str, Any], key: str, path: str, kind: type) -> Any:
    value = document.get(key, _MISSING)
    if value is _MISSING:
        raise KitParityError("field_missing", f"{path}.{key}")
    return _typed(value, f"{path}.{key}", kind)


def _optional(document: Mapping[str, Any], key: str, path: str, kind: type) -> Any:
    value = document.get(key, _MISSING)
    if value is _MISSING:
        return None
    return _typed(value, f"{path}.{key}", kind)


def _typed(value: Any, path: str, kind: type) -> Any:
    if value is None:
        raise KitParityError("field_null", path)
    if kind is int:
        if not _is_int(value):
            raise KitParityError("field_type_invalid", path)
    elif not isinstance(value, kind):
        raise KitParityError("field_type_invalid", path)
    return value


def _is_int(value: object) -> bool:
    return type(value) is int


def _reject_constant(_constant: str) -> float:
    raise ValueError("non-finite JSON number")


def _check_scalar_offset(text: str, offset: int) -> None:
    if not _is_int(offset) or not 0 <= offset <= len(text):
        raise KitParityError("offset_out_of_bounds", "$.offset")


def _native_to_scalar(text: str, offset: int, width: Callable[[str], int]) -> int:
    if not _is_int(offset) or offset < 0:
        raise KitParityError("offset_out_of_bounds", "$.offset")
    position = 0
    for index, char in enumerate(text):
        if position == offset:
            return index
        position += width(char)
        if position > offset:
            raise KitParityError("offset_not_scalar_boundary", "$.offset")
    if position == offset:
        return len(text)
    raise KitParityError("offset_out_of_bounds", "$.offset")
