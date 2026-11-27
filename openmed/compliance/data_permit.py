"""Offline checks of export metadata against a host-supplied data permit."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from openmed.risk.minimum_necessary import _normalize_fields


class OutputForm(StrEnum):
    """Supported declarations of the form of a prepared export."""

    ANONYMISED = "anonymised"
    PSEUDONYMISED = "pseudonymised"
    AGGREGATE = "aggregate"


def _codes(values: tuple[str, ...], name: str) -> tuple[str, ...]:
    if type(values) not in (list, tuple) or not values:
        raise ValueError(f"invalid_{name}")
    return _normalize_fields(values, field_name=name)


def _form(value: str) -> OutputForm:
    if type(value) not in (str, OutputForm) or value not in tuple(OutputForm):
        raise ValueError("invalid_output_form")
    return OutputForm(value)


@dataclass(frozen=True)
class DataPermit:
    """Purpose codes, field allowlist and output forms from an issued permit."""

    purposes: tuple[str, ...]
    allowed_fields: tuple[str, ...]
    output_forms: tuple[OutputForm, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "purposes", _codes(self.purposes, "purposes"))
        object.__setattr__(
            self, "allowed_fields", _codes(self.allowed_fields, "allowed_fields")
        )
        if type(self.output_forms) not in (list, tuple) or not 1 <= len(
            self.output_forms
        ) <= len(OutputForm):
            raise ValueError("invalid_output_forms")
        forms = tuple(_form(code) for code in self.output_forms)
        if len(set(forms)) != len(forms):
            raise ValueError("duplicate_output_form")
        object.__setattr__(self, "output_forms", forms)


@dataclass(frozen=True)
class PermitCheck:
    """Value-free decision with controlled reasons and disallowed field codes."""

    violations: tuple[str, ...]
    disallowed_fields: tuple[str, ...]

    @property
    def allowed(self) -> bool:
        """Return whether all declared dimensions are permitted."""
        return not self.violations

    def require_allowed(self) -> None:
        """Raise before export when any dimension is outside the permit."""
        if not self.allowed:
            raise PermissionError("export_outside_data_permit")


def check_export_manifest(
    permit: DataPermit,
    *,
    purpose: str,
    fields: tuple[str, ...],
    output_form: OutputForm | str,
) -> PermitCheck:
    """Compare schema codes and a declared output form before exporting.

    Args:
        permit: Terms supplied by the host from its approved permit.
        purpose: Purpose code for this export.
        fields: Field codes from the final export schema.
        output_form: Host's declaration of the prepared output form.
    """
    (purpose_code,) = _codes((purpose,), "purpose")
    extra = tuple(
        sorted(set(_codes(fields, "fields")).difference(permit.allowed_fields))
    )
    reasons = []
    if purpose_code not in permit.purposes:
        reasons.append("purpose_not_permitted")
    if _form(output_form) not in permit.output_forms:
        reasons.append("output_form_not_permitted")
    if extra:
        reasons.append("fields_not_permitted")
    return PermitCheck(tuple(reasons), extra)
