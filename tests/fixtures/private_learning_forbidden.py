"""Synthetic forbidden-field catalog shared by private-learning validators.

Every value is a placeholder: there is no real institutional, client, or
clinical data here. Tensor-like values are integer lists so that strict JSON
number parsing does not change which rejection a validator reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final


@dataclass(frozen=True, slots=True)
class ForbiddenFieldCase:
    """One identifying or value-bearing field that metadata must never carry.

    Attributes:
        reason_code: Stable machine-readable identifier for the rejection.
        field: Field name injected into an otherwise valid payload.
        value: Placeholder value for that field.
        marker: Text that must never be echoed in an error, or ``None`` when
            the placeholder is not text.
    """

    reason_code: str
    field: str
    value: Any
    marker: str | None


FORBIDDEN_FIELD_CASES: Final[tuple[ForbiddenFieldCase, ...]] = (
    ForbiddenFieldCase(
        "forbidden_site_id", "site_id", "placeholder-site-id", "placeholder-site-id"
    ),
    ForbiddenFieldCase(
        "forbidden_client_id",
        "client_id",
        "placeholder-client-id",
        "placeholder-client-id",
    ),
    ForbiddenFieldCase(
        "forbidden_path", "path", "/placeholder/path", "/placeholder/path"
    ),
    ForbiddenFieldCase(
        "forbidden_endpoint",
        "endpoint",
        "https://placeholder.invalid/endpoint",
        "placeholder.invalid",
    ),
    ForbiddenFieldCase(
        "forbidden_example",
        "examples",
        ["placeholder example text"],
        "placeholder example text",
    ),
    ForbiddenFieldCase("forbidden_tensor", "tensors", [[1, 2], [3, 4]], None),
    ForbiddenFieldCase("forbidden_gradient", "gradients", [1, 2, 3], None),
    ForbiddenFieldCase(
        "forbidden_message",
        "message",
        "placeholder free-form message",
        "placeholder free-form message",
    ),
    ForbiddenFieldCase(
        "forbidden_local_metric", "local_metrics", {"placeholder_metric": 1}, None
    ),
    ForbiddenFieldCase("forbidden_patient_count", "patient_count", 7, None),
)

__all__ = ["FORBIDDEN_FIELD_CASES", "ForbiddenFieldCase"]
