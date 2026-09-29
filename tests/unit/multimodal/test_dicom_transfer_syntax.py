"""Focused tests for bounded DICOM transfer-syntax preflight.

Every payload is a small synthetic byte string built in this module: no fixture
file, no imaging library, and no network access are involved.
"""

from __future__ import annotations

import json
import struct
from dataclasses import FrozenInstanceError, replace

import pytest

from openmed.multimodal.dicom_transfer_syntax import (
    DEFAULT_MAX_FILE_META_BYTES,
    DEFAULT_MAX_FILE_META_ELEMENTS,
    DEFAULT_MAX_TRANSFER_SYNTAX_UID_BYTES,
    TRANSFER_SYNTAX_CATALOG,
    TRANSFER_SYNTAX_OUTCOMES,
    TRANSFER_SYNTAX_REASON_CODES,
    TRANSFER_SYNTAX_SCHEMA_VERSION,
    TransferSyntaxCapability,
    TransferSyntaxError,
    TransferSyntaxOutcome,
    TransferSyntaxReport,
    describe_transfer_syntax,
    read_dicom_transfer_syntax,
)

_PREAMBLE = b"\x00" * 128
_MAGIC = b"DICM"
_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.7"
_SOP_INSTANCE_UID = "1.2.3.4.5.6.7.8.9"
_LOCAL_PATH = "C:\\data\\scan.dcm"
_PATIENT_NAME = b"DOE^JOHN"
_PRIVATE_UID = "1.3.6.1.4.1.5962.1"
_UNKNOWN_STANDARD_UID = "1.2.840.10008.1.2.4.201"
_JPEG_BASELINE_UID = "1.2.840.10008.1.2.4.50"

_LONG_VRS = frozenset({b"OB", b"OD", b"OF", b"OL", b"OV", b"OW", b"SQ", b"SV"})
_LONG_VRS |= frozenset({b"UC", b"UN", b"UR", b"UT", b"UV"})

_EXPECTED_CATALOG = {
    "1.2.840.10008.1.2": (TransferSyntaxOutcome.NATIVE, False, False),
    "1.2.840.10008.1.2.1": (TransferSyntaxOutcome.NATIVE, False, False),
    "1.2.840.10008.1.2.1.99": (TransferSyntaxOutcome.REVIEW, False, False),
    "1.2.840.10008.1.2.2": (TransferSyntaxOutcome.REVIEW, False, True),
    "1.2.840.10008.1.2.4.50": (TransferSyntaxOutcome.OPTIONAL_DECODER, True, False),
    "1.2.840.10008.1.2.4.51": (TransferSyntaxOutcome.OPTIONAL_DECODER, True, False),
    "1.2.840.10008.1.2.4.57": (TransferSyntaxOutcome.OPTIONAL_DECODER, True, False),
    "1.2.840.10008.1.2.4.70": (TransferSyntaxOutcome.OPTIONAL_DECODER, True, False),
    "1.2.840.10008.1.2.4.80": (TransferSyntaxOutcome.OPTIONAL_DECODER, True, False),
    "1.2.840.10008.1.2.4.81": (TransferSyntaxOutcome.OPTIONAL_DECODER, True, False),
    "1.2.840.10008.1.2.4.90": (TransferSyntaxOutcome.OPTIONAL_DECODER, True, False),
    "1.2.840.10008.1.2.4.91": (TransferSyntaxOutcome.OPTIONAL_DECODER, True, False),
    "1.2.840.10008.1.2.4.92": (TransferSyntaxOutcome.OPTIONAL_DECODER, True, False),
    "1.2.840.10008.1.2.4.93": (TransferSyntaxOutcome.OPTIONAL_DECODER, True, False),
    "1.2.840.10008.1.2.5": (TransferSyntaxOutcome.OPTIONAL_DECODER, True, False),
}

_REPORT_KEYS = (
    "decoder_required",
    "element_count",
    "file_meta_bytes",
    "outcome",
    "reason",
    "reason_codes",
    "schema_version",
    "transfer_syntax_name",
    "transfer_syntax_uid",
)

_GOLDEN_REASON = "transfer_syntax_jpeg_baseline"
_GOLDEN_JSON = '{"decoder_required":true,"element_count":6,"file_meta_bytes":272,"outcome":"optional_decoder","reason":"transfer_syntax_jpeg_baseline","reason_codes":["transfer_syntax_jpeg_baseline"],"schema_version":"openmed.multimodal.dicom_transfer_syntax.v1","transfer_syntax_name":"JPEG Baseline (Process 1)","transfer_syntax_uid":"1.2.840.10008.1.2.4.50"}'


def _element(tag: tuple[int, int], vr: bytes, value: bytes) -> bytes:
    assert len(value) % 2 == 0, (vr, value)
    head = struct.pack("<HH", tag[0], tag[1]) + vr
    if vr in _LONG_VRS:
        return head + b"\x00\x00" + struct.pack("<I", len(value)) + value
    return head + struct.pack("<H", len(value)) + value


def _padded_uid(value: str) -> bytes:
    raw = value.encode("ascii")
    return raw + (b"\x00" if len(raw) % 2 else b"")


def _file_meta(
    ts_uid: str | None = None,
    *,
    group_length: int | None = None,
    extra_body: bytes = b"",
) -> bytes:
    body = b""
    body += _element((0x0002, 0x0001), b"OB", b"\x00\x01")
    body += _element((0x0002, 0x0002), b"UI", _padded_uid(_SOP_CLASS_UID))
    body += _element((0x0002, 0x0003), b"UI", _padded_uid(_SOP_INSTANCE_UID))
    body += _element((0x0002, 0x0016), b"AE", _LOCAL_PATH.encode("ascii"))
    body += extra_body
    if ts_uid is not None:
        body += _element((0x0002, 0x0010), b"UI", _padded_uid(ts_uid))
    declared = len(body) if group_length is None else group_length
    head = _element((0x0002, 0x0000), b"UL", struct.pack("<I", declared))
    tail = _element((0x0008, 0x0005), b"CS", b"ISO_IR 100")
    tail += _element((0x0010, 0x0010), b"PN", _PATIENT_NAME)
    tail += b"\x00\xff" * 16
    return _PREAMBLE + _MAGIC + head + body + tail


def _read(payload: bytes | bytearray | memoryview, **kwargs: object):
    return read_dicom_transfer_syntax(payload, **kwargs)  # type: ignore[arg-type]


def test_default_bounds_are_stable() -> None:
    assert DEFAULT_MAX_FILE_META_BYTES == 64 * 1024
    assert DEFAULT_MAX_FILE_META_ELEMENTS == 64
    assert DEFAULT_MAX_TRANSFER_SYNTAX_UID_BYTES == 128


def test_schema_version_is_stable() -> None:
    assert (
        TRANSFER_SYNTAX_SCHEMA_VERSION == "openmed.multimodal.dicom_transfer_syntax.v1"
    )


def test_outcomes_and_reason_codes_are_closed_sets() -> None:
    assert TRANSFER_SYNTAX_OUTCOMES == {
        "native",
        "optional_decoder",
        "review",
        "unsupported",
    }
    assert len(TRANSFER_SYNTAX_REASON_CODES) == 24
    assert len(set(TRANSFER_SYNTAX_REASON_CODES)) == 24
    assert list(TRANSFER_SYNTAX_REASON_CODES) == sorted(TRANSFER_SYNTAX_REASON_CODES)
    assert set(TRANSFER_SYNTAX_OUTCOMES) == {
        outcome.value for outcome in TransferSyntaxOutcome
    }


def test_catalog_matches_the_expected_transfer_syntaxes() -> None:
    assert set(TRANSFER_SYNTAX_CATALOG) == set(_EXPECTED_CATALOG)
    for uid, capability in TRANSFER_SYNTAX_CATALOG.items():
        assert capability == describe_transfer_syntax(uid)
        assert capability.uid == uid
        assert capability.name
        expected_outcome, decoder_required, retired = _EXPECTED_CATALOG[uid]
        assert capability.outcome is expected_outcome
        assert capability.decoder_required is decoder_required
        assert capability.retired is retired
        assert capability.reason in TRANSFER_SYNTAX_REASON_CODES
        assert (capability.outcome is TransferSyntaxOutcome.OPTIONAL_DECODER) is (
            capability.decoder_required
        )


@pytest.mark.parametrize("uid", sorted(_EXPECTED_CATALOG))
def test_declared_catalog_syntax_is_routed(uid: str) -> None:
    report = _read(_file_meta(uid))
    capability = TRANSFER_SYNTAX_CATALOG[uid]
    assert report.outcome is capability.outcome
    assert report.reason == capability.reason
    assert report.reason_codes == (capability.reason,)
    assert report.transfer_syntax_uid == uid
    assert report.transfer_syntax_name == capability.name
    assert report.decoder_required is capability.decoder_required
    assert report.element_count == 6
    assert report.schema_version == TRANSFER_SYNTAX_SCHEMA_VERSION


def test_only_native_syntaxes_report_ok() -> None:
    native = _read(_file_meta("1.2.840.10008.1.2.1"))
    compressed = _read(_file_meta(_JPEG_BASELINE_UID))
    assert native.ok is True
    assert compressed.ok is False
    assert native.outcome is TransferSyntaxOutcome.NATIVE


def test_catalog_report_json_is_byte_stable() -> None:
    report = _read(_file_meta(_JPEG_BASELINE_UID))
    assert report.to_json() == _GOLDEN_JSON
    payload = json.loads(report.to_json())
    assert list(payload) == sorted(_REPORT_KEYS)
    assert payload["reason_codes"] == [_GOLDEN_REASON]


def test_report_mapping_is_limited_to_transfer_syntax_metadata() -> None:
    report = _read(_file_meta("1.2.840.10008.1.2.4.50"))
    mapping = report.to_dict()
    assert tuple(mapping) == (
        "schema_version",
        "outcome",
        "reason",
        "reason_codes",
        "transfer_syntax_uid",
        "transfer_syntax_name",
        "decoder_required",
        "file_meta_bytes",
        "element_count",
    )
    blob = report.to_json()
    for secret in (
        _SOP_CLASS_UID,
        _SOP_INSTANCE_UID,
        _LOCAL_PATH,
        "DOE",
        "ISO_IR",
    ):
        assert secret not in blob
    assert "ok" not in blob


def test_catalog_capability_mapping_is_deterministic() -> None:
    capability = TRANSFER_SYNTAX_CATALOG["1.2.840.10008.1.2.2"]
    assert capability.to_dict() == {
        "decoder_required": False,
        "name": "Explicit VR Big Endian",
        "outcome": "review",
        "reason": "transfer_syntax_big_endian_retired",
        "retired": True,
        "uid": "1.2.840.10008.1.2.2",
    }


def test_report_and_capability_are_immutable() -> None:
    report = _read(_file_meta("1.2.840.10008.1.2.1"))
    with pytest.raises(FrozenInstanceError):
        report.outcome = TransferSyntaxOutcome.UNSUPPORTED  # type: ignore[misc]
    capability = TRANSFER_SYNTAX_CATALOG["1.2.840.10008.1.2.1"]
    with pytest.raises(FrozenInstanceError):
        capability.retired = True  # type: ignore[misc]


def _group_end(payload: bytes) -> int:
    return payload.index(b"ISO_IR") - 8


def test_report_counts_describe_the_parsed_group() -> None:
    payload = _file_meta("1.2.840.10008.1.2.1")
    report = _read(payload)
    assert report.element_count == 6
    assert report.file_meta_bytes == _group_end(payload)


def test_unknown_and_private_uids_are_unsupported() -> None:
    unknown = _read(_file_meta(_UNKNOWN_STANDARD_UID))
    assert unknown.outcome is TransferSyntaxOutcome.UNSUPPORTED
    assert unknown.reason == "transfer_syntax_unknown"
    assert unknown.transfer_syntax_uid == _UNKNOWN_STANDARD_UID
    assert unknown.transfer_syntax_name is None
    private = _read(_file_meta(_PRIVATE_UID))
    assert private.reason == "transfer_syntax_private"
    assert private.transfer_syntax_uid == _PRIVATE_UID
    assert private.reason_codes == ("transfer_syntax_private",)


def test_missing_transfer_syntax_element_is_reported() -> None:
    report = _read(_file_meta())
    assert report.reason == "transfer_syntax_missing"
    assert report.transfer_syntax_uid is None
    assert report.element_count == 5


@pytest.mark.parametrize(
    ("name", "reason"),
    (
        ("no-magic", "file_meta_missing"),
        ("short", "file_meta_missing"),
        ("malformed-uid", "transfer_syntax_malformed"),
        ("odd-length", "file_meta_malformed"),
        ("unknown-vr", "file_meta_malformed"),
        ("wrong-vr", "file_meta_malformed"),
        ("group-length", "file_meta_group_length_mismatch"),
        ("truncated-value", "file_meta_truncated"),
        ("truncated-header", "file_meta_truncated"),
        ("element-limit", "file_meta_element_limit"),
    ),
)
def test_malformed_declarations_return_stable_reasons(name: str, reason: str) -> None:
    payloads = {
        "no-magic": b"\x00" * 200,
        "short": b"\x00" * 64,
        "malformed-uid": _PREAMBLE
        + _MAGIC
        + _element((0x0002, 0x0010), b"UI", b"1.2.abc\x00"),
        "odd-length": _PREAMBLE
        + _MAGIC
        + struct.pack("<HH", 0x0002, 0x0010)
        + b"UI"
        + struct.pack("<H", 5)
        + b"1.2.4",
        "unknown-vr": _PREAMBLE
        + _MAGIC
        + struct.pack("<HH", 0x0002, 0x0010)
        + b"ZZ"
        + struct.pack("<H", 4)
        + b"1.2.",
        "wrong-vr": _PREAMBLE
        + _MAGIC
        + _element((0x0002, 0x0010), b"LO", b"1.2.4\x00"),
        "group-length": _file_meta("1.2.840.10008.1.2.1", group_length=4),
        "truncated-value": _file_meta("1.2.840.10008.1.2.1")[:160],
        "truncated-header": _file_meta("1.2.840.10008.1.2.1")[:137],
        "element-limit": _file_meta("1.2.840.10008.1.2.1"),
    }
    kwargs = {"max_elements": 1} if name == "element-limit" else {}
    report = _read(payloads[name], **kwargs)
    assert report.outcome is TransferSyntaxOutcome.UNSUPPORTED
    assert report.reason == reason
    assert report.reason_codes == (reason,)
    assert report.ok is False


def test_group_length_mismatch_keeps_the_declared_uid() -> None:
    report = _read(_file_meta("1.2.840.10008.1.2.1", group_length=4))
    assert report.reason == "file_meta_group_length_mismatch"
    assert report.transfer_syntax_uid == "1.2.840.10008.1.2.1"
    assert report.transfer_syntax_name == "Explicit VR Little Endian"


def test_truncation_before_the_uid_is_reported_not_raised() -> None:
    payload = _file_meta("1.2.840.10008.1.2.1")
    report = _read(payload, max_bytes=140)
    assert report.reason == "file_meta_truncated"
    assert report.transfer_syntax_uid is None


def test_oversized_uid_value_is_malformed() -> None:
    report = _read(_file_meta("1." + "2" * 70), max_uid_bytes=16)
    assert report.reason == "transfer_syntax_malformed"


def test_truncated_value_before_the_uid_is_counted() -> None:
    payload = _file_meta("1.2.840.10008.1.2.1")
    report = _read(payload[:160])
    assert report.reason == "file_meta_truncated"
    assert report.element_count == 2


@pytest.mark.parametrize(
    "kwargs",
    (
        {"max_bytes": 0},
        {"max_bytes": -1},
        {"max_bytes": True},
        {"max_elements": 0},
        {"max_elements": True},
        {"max_uid_bytes": -1},
        {"max_uid_bytes": True},
    ),
)
def test_invalid_bounds_raise_categorized_errors(kwargs: dict[str, object]) -> None:
    with pytest.raises(TransferSyntaxError) as excinfo:
        _read(b"", **kwargs)
    name = next(iter(kwargs))
    assert excinfo.value.category == f"{name}_invalid"
    assert str(excinfo.value) == f"{name}_invalid"


@pytest.mark.parametrize("source", ("text", 3, None, ["bytes"]))
def test_invalid_source_raises_categorized_error(source: object) -> None:
    with pytest.raises(TransferSyntaxError) as excinfo:
        read_dicom_transfer_syntax(source)  # type: ignore[arg-type]
    assert excinfo.value.category == "source_invalid"


def test_bytes_like_sources_are_accepted_without_mutation() -> None:
    payload = _file_meta("1.2.840.10008.1.2.1")
    mutable = bytearray(payload)
    view = memoryview(payload)
    for source in (payload, mutable, view):
        report = _read(source)
        assert report.transfer_syntax_uid == "1.2.840.10008.1.2.1"
    assert bytes(mutable) == payload
    assert view.obj is payload


@pytest.mark.parametrize(
    "uid",
    ("", "1.2.", ".1.2", "01.2.3", "1.2.a", "1.2.3.4.", " 1.2.3", "1.2.3.4.5.6" * 12),
)
def test_describe_rejects_malformed_identifiers(uid: str) -> None:
    with pytest.raises(TransferSyntaxError) as excinfo:
        describe_transfer_syntax(uid)
    assert excinfo.value.category == "uid_invalid"


@pytest.mark.parametrize("uid", (b"1.2.840", 4, None))
def test_describe_rejects_non_string_identifiers(uid: object) -> None:
    with pytest.raises(TransferSyntaxError) as excinfo:
        describe_transfer_syntax(uid)  # type: ignore[arg-type]
    assert excinfo.value.category == "uid_invalid"


def test_describe_returns_none_for_unknown_valid_identifiers() -> None:
    assert describe_transfer_syntax(_UNKNOWN_STANDARD_UID) is None
    assert describe_transfer_syntax("1.2.840.10008.1.2.1.98") is None


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"reason": "unknown_reason"}, "reason"),
        ({"outcome": "native"}, "outcome"),
        ({"decoder_required": 1}, "booleans"),
        ({"retired": "yes"}, "booleans"),
        ({"uid": "1.2.a"}, "uid"),
        ({"name": ""}, "name"),
    ),
)
def test_capability_validation_rejects_bad_values(
    override: dict[str, object], message: str
) -> None:
    fields = {
        "uid": "1.2.840.10008.1.2.1",
        "name": "Explicit VR Little Endian",
        "outcome": TransferSyntaxOutcome.NATIVE,
        "reason": "transfer_syntax_explicit_little_endian",
        "decoder_required": False,
    }
    fields.update(override)
    with pytest.raises(ValueError, match=message):
        TransferSyntaxCapability(**fields)  # type: ignore[arg-type]


def _report(**overrides: object) -> TransferSyntaxReport:
    fields = {
        "outcome": TransferSyntaxOutcome.NATIVE,
        "reason": "transfer_syntax_explicit_little_endian",
        "reason_codes": ("transfer_syntax_explicit_little_endian",),
        "transfer_syntax_uid": "1.2.840.10008.1.2.1",
        "transfer_syntax_name": "Explicit VR Little Endian",
        "decoder_required": False,
        "file_meta_bytes": 270,
        "element_count": 6,
    }
    fields.update(overrides)
    return TransferSyntaxReport(**fields)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"reason_codes": ()}, "reason_codes must not be empty"),
        ({"reason_codes": ("unknown",)}, "known reason codes"),
        ({"reason": "transfer_syntax_deflated"}, "first reason code"),
        ({"transfer_syntax_uid": "1.2.a"}, "DICOM UID"),
        ({"decoder_required": "no"}, "boolean"),
        ({"file_meta_bytes": -1}, "non-negative"),
        ({"element_count": True}, "non-negative"),
        ({"outcome": "native"}, "TransferSyntaxOutcome"),
        ({"schema_version": "v2"}, "schema_version is unsupported"),
    ),
)
def test_report_validation_rejects_bad_values(
    override: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _report(**override)


def test_report_round_trips_through_replace() -> None:
    report = _report()
    assert report.to_dict()["reason_codes"] == [
        "transfer_syntax_explicit_little_endian"
    ]
    assert replace(report, element_count=7).element_count == 7
    assert json.loads(report.to_json())["outcome"] == "native"


def test_reads_are_deterministic() -> None:
    payload = _file_meta(_UNKNOWN_STANDARD_UID)
    first = _read(payload)
    second = _read(payload)
    assert first.to_dict() == second.to_dict()
    assert first.to_json() == second.to_json()


def test_reads_run_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access is not allowed")

    monkeypatch.setattr("socket.socket.connect", _fail)
    monkeypatch.setattr("socket.create_connection", _fail)
    for uid in (_JPEG_BASELINE_UID, _UNKNOWN_STANDARD_UID, _PRIVATE_UID):
        report = _read(_file_meta(uid))
        assert report.reason
    assert _read(_file_meta()).reason == "transfer_syntax_missing"
