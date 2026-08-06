"""License-quarantined subprocess bridge for sdcMicro disclosure risk.

The GPL-2.0-licensed R package is never imported into the OpenMed process. This
module writes a private temporary CSV, invokes an explicitly selected
``Rscript`` executable, and accepts only aggregate JSON risk measures back.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

__all__ = [
    "GPL_NOTICE",
    "LICENSE_ACKNOWLEDGEMENT_ENV",
    "SDCMicroBridgeError",
    "SDCMicroLicenseError",
    "SDCMicroUnavailableError",
    "run_sdcmicro",
    "sdcmicro_risk_report",
]

LICENSE_ACKNOWLEDGEMENT_ENV: Final = "OPENMED_ACCEPT_SDCMICRO_LICENSE"
GPL_NOTICE: Final = (
    "sdcMicro is GPL-2.0 licensed and is executed only as a separate Rscript "
    "process. Set OPENMED_ACCEPT_SDCMICRO_LICENSE=1 only after reviewing and "
    "accepting the sdcMicro license."
)
_ACCEPTED_ACKNOWLEDGEMENTS: Final = frozenset({"1", "true", "yes", "accept"})
_SDCMICRO_UNAVAILABLE_SENTINEL: Final = "OPENMED_SDCMICRO_UNAVAILABLE"
_MAX_RESULT_BYTES: Final = 1_000_000

_R_BRIDGE_SCRIPT: Final = r"""
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("internal bridge error: expected input, output, and config paths")
}

input_path <- args[[1]]
output_path <- args[[2]]
config_path <- args[[3]]

if (!requireNamespace("sdcMicro", quietly = TRUE)) {
  cat("OPENMED_SDCMICRO_UNAVAILABLE\n", file = stderr())
  quit(save = "no", status = 69L)
}
if (!requireNamespace("jsonlite", quietly = TRUE)) {
  cat("OPENMED_SDCMICRO_UNAVAILABLE\n", file = stderr())
  quit(save = "no", status = 69L)
}

config <- jsonlite::read_json(config_path, simplifyVector = TRUE)
key_vars <- unlist(config$quasi_identifiers, use.names = FALSE)
sensitive_vars <- unlist(config$sensitive_attributes, use.names = FALSE)
if (is.null(sensitive_vars)) {
  sensitive_vars <- character(0)
}
weight_var <- config$weight_column
if (is.null(weight_var)) {
  weight_var <- NULL
}

table_data <- utils::read.csv(
  input_path,
  check.names = FALSE,
  stringsAsFactors = FALSE,
  na.strings = config$missing_sentinel
)
risk <- sdcMicro::measure_risk(
  table_data,
  keyVars = key_vars,
  w = weight_var
)

individual_risk <- as.numeric(risk$Res[, "risk"])
sample_frequency <- as.numeric(risk$Res[, "fk"])
valid_risk <- individual_risk[is.finite(individual_risk)]
valid_frequency <- sample_frequency[
  is.finite(sample_frequency) & sample_frequency > 0
]

safe_stat <- function(values, statistic) {
  if (length(values) == 0) {
    return(NA_real_)
  }
  as.numeric(statistic(values))
}

class_sizes <- sort(unique(valid_frequency))
class_size_distribution <- lapply(class_sizes, function(size) {
  list(
    size = as.integer(size),
    class_count = as.integer(round(sum(valid_frequency == size) / size))
  )
})
class_counts <- vapply(
  class_size_distribution,
  function(item) item$class_count,
  integer(1)
)
target_k <- as.integer(config$target_k)
k_violating <- sample_frequency < target_k

l_diversity <- list()
if (length(sensitive_vars) > 0) {
  l_values <- sdcMicro::ldiversity(
    table_data,
    keyVars = key_vars,
    ldiv_index = sensitive_vars,
    l_recurs_c = as.numeric(config$recursive_l_constant)
  )
  target_l <- as.integer(config$target_l)

  l_diversity <- lapply(sensitive_vars, function(attribute) {
    distinct_values <- as.numeric(
      l_values[, paste0(attribute, "_Distinct_Ldiversity")]
    )
    entropy_values <- as.numeric(
      l_values[, paste0(attribute, "_Entropy_Ldiversity")]
    )
    recursive_values <- as.numeric(
      l_values[, paste0(attribute, "_Recursive_Ldiversity")]
    )
    violating <- is.finite(distinct_values) & distinct_values < target_l
    countable <- violating & is.finite(sample_frequency) & sample_frequency > 0

    list(
      attribute = attribute,
      achieved_distinct = safe_stat(
        distinct_values[is.finite(distinct_values)], min
      ),
      achieved_entropy = safe_stat(
        entropy_values[is.finite(entropy_values)], min
      ),
      achieved_recursive = safe_stat(
        recursive_values[is.finite(recursive_values)], min
      ),
      target = target_l,
      violating_class_count = as.integer(round(sum(1 / sample_frequency[countable])))
    )
  })
}

payload <- list(
  package_version = as.character(utils::packageVersion("sdcMicro")),
  row_count = as.integer(nrow(table_data)),
  global_risk = as.numeric(risk$global_risk),
  global_risk_pct = as.numeric(risk$global_risk_pct),
  expected_reidentifications = as.numeric(risk$global_risk_ER),
  individual_risk = list(
    max = safe_stat(valid_risk, max),
    mean = safe_stat(valid_risk, mean),
    median = safe_stat(valid_risk, stats::median),
    p95 = if (length(valid_risk) == 0) {
      NA_real_
    } else {
      as.numeric(stats::quantile(valid_risk, 0.95, names = FALSE))
    }
  ),
  k_anonymity = list(
    achieved_k = if (length(valid_frequency) == 0) {
      0L
    } else {
      as.integer(min(valid_frequency))
    },
    target_k = target_k,
    class_count = as.integer(sum(class_counts)),
    class_size_distribution = class_size_distribution,
    singleton_class_count = as.integer(sum(class_counts[class_sizes == 1L])),
    singleton_record_count = as.integer(sum(sample_frequency == 1L, na.rm = TRUE)),
    violating_class_count = as.integer(sum(
      class_counts[class_sizes < target_k]
    )),
    violating_record_count = as.integer(sum(k_violating, na.rm = TRUE))
  ),
  l_diversity = l_diversity
)

jsonlite::write_json(
  payload,
  output_path,
  auto_unbox = TRUE,
  digits = NA,
  na = "null",
  null = "null"
)
Sys.chmod(output_path, mode = "0600")
"""


class SDCMicroBridgeError(RuntimeError):
    """Base exception for fail-closed sdcMicro bridge failures."""


class SDCMicroLicenseError(SDCMicroBridgeError):
    """Raised when the required GPL-2.0 acknowledgement is absent."""


class SDCMicroUnavailableError(SDCMicroBridgeError):
    """Raised when Rscript or the sdcMicro package is unavailable."""


def run_sdcmicro(
    records: Sequence[Mapping[str, object]],
    *,
    quasi_identifiers: Sequence[str],
    sensitive_attributes: Sequence[str] = (),
    rscript: str | os.PathLike[str] = "Rscript",
    weight_column: str | None = None,
    target_k: int = 2,
    target_l: int = 2,
    recursive_l_constant: float = 2.0,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Return aggregate sdcMicro disclosure-risk measures for a table.

    Args:
        records: Non-empty table represented as row mappings with string column
            names. Values must be scalar or ``None``.
        quasi_identifiers: Categorical key columns used for risk and
            k-anonymity calculations.
        sensitive_attributes: Optional columns used for l-diversity measures.
            sdcMicro supports at most five in one calculation.
        rscript: User-selected ``Rscript`` executable name or path.
        weight_column: Optional sampling-weight column.
        target_k: k-anonymity threshold used to count violating classes.
        target_l: distinct l-diversity threshold used to count violations.
        recursive_l_constant: sdcMicro recursive l-diversity constant.
        timeout: Maximum subprocess runtime in seconds.

    Returns:
        A PHI-safe aggregate mapping aligned with OpenMed risk-report fields.

    Raises:
        SDCMicroLicenseError: If the GPL-2.0 acknowledgement is absent.
        SDCMicroUnavailableError: If Rscript or sdcMicro cannot be used.
        SDCMicroBridgeError: If the subprocess or result contract fails.
        TypeError: If the table contains unsupported rows or values.
        ValueError: If columns or thresholds are invalid.

    Note:
        Set ``OPENMED_ACCEPT_SDCMICRO_LICENSE=1`` only after reviewing the
        sdcMicro GPL-2.0 license. The package runs strictly out of process; this
        module never imports ``rpy2`` or sdcMicro code.
    """

    _require_license_acknowledgement()
    resolved_rscript = _resolve_rscript(rscript)
    options = _validated_options(
        quasi_identifiers=quasi_identifiers,
        sensitive_attributes=sensitive_attributes,
        weight_column=weight_column,
        target_k=target_k,
        target_l=target_l,
        recursive_l_constant=recursive_l_constant,
        timeout=timeout,
    )
    rows, fieldnames = _materialize_rows(records)
    _validate_selected_columns(fieldnames, options)
    missing_sentinel = _missing_sentinel(rows)

    with tempfile.TemporaryDirectory(prefix="openmed-sdcmicro-") as directory:
        workspace = Path(directory)
        input_path = workspace / "input.csv"
        output_path = workspace / "aggregate.json"
        config_path = workspace / "config.json"
        script_path = workspace / "bridge.R"

        _write_csv(input_path, rows, fieldnames, missing_sentinel)
        _write_private_text(script_path, _R_BRIDGE_SCRIPT)
        _write_private_text(
            config_path,
            json.dumps(
                {
                    **options,
                    "missing_sentinel": missing_sentinel,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        _create_private_file(output_path)

        command = [
            resolved_rscript,
            "--vanilla",
            os.fspath(script_path),
            os.fspath(input_path),
            os.fspath(output_path),
            os.fspath(config_path),
        ]
        try:
            completed = subprocess.run(  # noqa: S603 - explicit executable/argv
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=options["timeout"],
                cwd=directory,
            )
        except subprocess.TimeoutExpired as exc:
            raise SDCMicroBridgeError(
                f"sdcMicro Rscript timed out after {options['timeout']:g} seconds"
            ) from exc
        except FileNotFoundError as exc:
            raise SDCMicroUnavailableError(
                f"Rscript executable became unavailable: {resolved_rscript}"
            ) from exc
        except OSError as exc:
            raise SDCMicroUnavailableError(
                f"Rscript could not be started: {resolved_rscript}"
            ) from exc

        if completed.returncode != 0:
            if _SDCMICRO_UNAVAILABLE_SENTINEL in completed.stderr:
                raise SDCMicroUnavailableError(
                    "sdcMicro is unavailable to the selected Rscript; install it "
                    "separately after reviewing its GPL-2.0 license"
                )
            raise SDCMicroBridgeError(
                "sdcMicro Rscript failed with exit status "
                f"{completed.returncode}; no result was accepted"
            )

        raw_result = _read_result(output_path)

    return _map_result(
        raw_result,
        expected_row_count=len(rows),
        sensitive_attributes=options["sensitive_attributes"],
        weight_column=options["weight_column"],
    )


sdcmicro_risk_report = run_sdcmicro


def _require_license_acknowledgement() -> None:
    print(GPL_NOTICE, file=sys.stderr)
    acknowledgement = os.environ.get(LICENSE_ACKNOWLEDGEMENT_ENV, "")
    if acknowledgement.strip().lower() not in _ACCEPTED_ACKNOWLEDGEMENTS:
        raise SDCMicroLicenseError(
            f"sdcMicro invocation blocked: set {LICENSE_ACKNOWLEDGEMENT_ENV}=1 "
            "only after accepting the GPL-2.0 license"
        )


def _resolve_rscript(rscript: str | os.PathLike[str]) -> str:
    candidate = os.fspath(rscript)
    if not candidate or "\x00" in candidate:
        raise ValueError("rscript must be a non-empty executable name or path")
    resolved = shutil.which(candidate)
    if resolved is None:
        raise SDCMicroUnavailableError(
            f"Rscript executable is unavailable: {candidate}"
        )
    return os.path.abspath(resolved)


def _validated_options(
    *,
    quasi_identifiers: Sequence[str],
    sensitive_attributes: Sequence[str],
    weight_column: str | None,
    target_k: int,
    target_l: int,
    recursive_l_constant: float,
    timeout: float,
) -> dict[str, Any]:
    quasi_columns = _validated_column_selection(
        quasi_identifiers,
        label="quasi_identifiers",
        allow_empty=False,
    )
    sensitive_columns = _validated_column_selection(
        sensitive_attributes,
        label="sensitive_attributes",
        allow_empty=True,
    )
    if len(sensitive_columns) > 5:
        raise ValueError("sensitive_attributes supports at most five columns")
    overlap = sorted(set(quasi_columns) & set(sensitive_columns))
    if overlap:
        raise ValueError(
            "sensitive_attributes must not also be quasi_identifiers: "
            + ", ".join(overlap)
        )
    if weight_column is not None:
        if not isinstance(weight_column, str) or not weight_column:
            raise ValueError("weight_column must be a non-empty string or None")
        if weight_column in quasi_columns or weight_column in sensitive_columns:
            raise ValueError("weight_column must not also be a selected risk column")
    _validate_positive_integer(target_k, "target_k")
    _validate_positive_integer(target_l, "target_l")
    if isinstance(recursive_l_constant, bool) or not isinstance(
        recursive_l_constant, (int, float)
    ):
        raise TypeError("recursive_l_constant must be a finite number greater than 1")
    recursive_value = float(recursive_l_constant)
    if not math.isfinite(recursive_value) or recursive_value <= 1:
        raise ValueError("recursive_l_constant must be a finite number greater than 1")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise TypeError("timeout must be a positive finite number")
    timeout_value = float(timeout)
    if not math.isfinite(timeout_value) or timeout_value <= 0:
        raise ValueError("timeout must be a positive finite number")
    return {
        "quasi_identifiers": list(quasi_columns),
        "sensitive_attributes": list(sensitive_columns),
        "weight_column": weight_column,
        "target_k": target_k,
        "target_l": target_l,
        "recursive_l_constant": recursive_value,
        "timeout": timeout_value,
    }


def _validated_column_selection(
    columns: Sequence[str],
    *,
    label: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    if isinstance(columns, (str, bytes)) or not isinstance(columns, Sequence):
        raise TypeError(f"{label} must be a sequence of column names")
    normalized: list[str] = []
    for column in columns:
        if not isinstance(column, str):
            raise TypeError(f"{label} entries must be strings")
        if not column or "\x00" in column:
            raise ValueError(f"{label} entries must be non-empty column names")
        normalized.append(column)
    if not normalized and not allow_empty:
        raise ValueError(f"{label} must contain at least one column")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} must not contain duplicate columns")
    return tuple(normalized)


def _validate_positive_integer(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be a positive integer")
    if value < 1:
        raise ValueError(f"{label} must be a positive integer")


def _materialize_rows(
    records: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[str]]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("records must be a sequence of row mappings")
    if not records:
        raise ValueError("records must contain at least one row")

    rows: list[dict[str, object]] = []
    fieldnames: list[str] = []
    known_fields: set[str] = set()
    for row_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"records[{row_index}] must be a row mapping")
        row: dict[str, object] = {}
        for field, value in record.items():
            if not isinstance(field, str) or not field or "\x00" in field:
                raise TypeError("record column names must be non-empty strings")
            _validate_scalar(value, row_index=row_index, field=field)
            row[field] = value
            if field not in known_fields:
                known_fields.add(field)
                fieldnames.append(field)
        rows.append(row)
    if not fieldnames:
        raise ValueError("records must contain at least one column")
    return rows, fieldnames


def _validate_scalar(value: object, *, row_index: int, field: str) -> None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    raise TypeError(
        f"records[{row_index}][{field!r}] must be a string, number, boolean, or None"
    )


def _validate_selected_columns(
    fieldnames: Sequence[str],
    options: Mapping[str, Any],
) -> None:
    available = set(fieldnames)
    selected = [
        *options["quasi_identifiers"],
        *options["sensitive_attributes"],
    ]
    if options["weight_column"] is not None:
        selected.append(options["weight_column"])
    missing = sorted(set(selected) - available)
    if missing:
        raise ValueError(
            "selected columns are absent from records: " + ", ".join(missing)
        )


def _missing_sentinel(rows: Sequence[Mapping[str, object]]) -> str:
    values = {str(value) for row in rows for value in row.values() if value is not None}
    while True:
        candidate = f"__OPENMED_MISSING_{uuid.uuid4().hex}__"
        if candidate not in values:
            return candidate


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
    missing_sentinel: str,
) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: missing_sentinel
                    if row.get(field) is None
                    else row.get(field)
                    for field in fieldnames
                }
            )


def _write_private_text(path: Path, content: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
        handle.write(content)


def _create_private_file(path: Path) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)


def _read_result(path: Path) -> Mapping[str, Any]:
    try:
        size = path.stat().st_size
        if size <= 0 or size > _MAX_RESULT_BYTES:
            raise SDCMicroBridgeError(
                "sdcMicro aggregate JSON result has an invalid size"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
    except SDCMicroBridgeError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SDCMicroBridgeError(
            "sdcMicro did not produce a valid aggregate JSON result"
        ) from exc
    if not isinstance(payload, Mapping):
        raise SDCMicroBridgeError("sdcMicro aggregate result must be a JSON object")
    return payload


def _map_result(
    raw: Mapping[str, Any],
    *,
    expected_row_count: int,
    sensitive_attributes: Sequence[str],
    weight_column: str | None,
) -> dict[str, Any]:
    row_count = _required_int(raw, "row_count", minimum=1)
    if row_count != expected_row_count:
        raise SDCMicroBridgeError("sdcMicro result row_count does not match the input")
    global_risk = _required_float(raw, "global_risk", minimum=0.0, maximum=1.0)
    global_risk_pct = _required_float(
        raw, "global_risk_pct", minimum=0.0, maximum=100.0
    )
    expected = _required_float(raw, "expected_reidentifications", minimum=0.0)
    version = raw.get("package_version")
    if (
        not isinstance(version, str)
        or not version
        or len(version) > 64
        or not all(character.isalnum() or character in ".+-" for character in version)
    ):
        raise SDCMicroBridgeError("sdcMicro result has an invalid package_version")

    raw_k = raw.get("k_anonymity")
    if not isinstance(raw_k, Mapping):
        raise SDCMicroBridgeError("sdcMicro result is missing k_anonymity measures")
    k_anonymity = {
        "achieved_k": _required_int(raw_k, "achieved_k", minimum=0),
        "target_k": _required_int(raw_k, "target_k", minimum=1),
        "class_count": _required_int(raw_k, "class_count", minimum=0),
        "class_size_distribution": _class_size_distribution(raw_k),
        "singleton_class_count": _required_int(
            raw_k, "singleton_class_count", minimum=0
        ),
        "singleton_record_count": _required_int(
            raw_k, "singleton_record_count", minimum=0
        ),
        "k_violating_class_count": _required_int(
            raw_k, "violating_class_count", minimum=0
        ),
        "violating_record_count": _required_int(
            raw_k, "violating_record_count", minimum=0
        ),
    }
    k_anonymity["meets_target"] = k_anonymity["achieved_k"] >= k_anonymity["target_k"]

    raw_individual = raw.get("individual_risk")
    if not isinstance(raw_individual, Mapping):
        raise SDCMicroBridgeError("sdcMicro result is missing individual_risk measures")
    individual_risk = {
        name: _required_float(raw_individual, name, minimum=0.0, maximum=1.0)
        for name in ("max", "mean", "median", "p95")
    }

    attributes = _attribute_disclosure(
        raw.get("l_diversity", []),
        expected_attributes=sensitive_attributes,
    )
    return {
        "schema_version": "1.0",
        "artifact": "sdcmicro_disclosure_risk_report",
        "detail_level": "aggregate_phi_safe",
        "not_an_expert_determination": True,
        "qualified_expert_review_required": True,
        "engine": {
            "name": "sdcMicro",
            "version": version,
            "execution": "subprocess",
            "license": "GPL-2.0",
        },
        "row_count": row_count,
        "reid_rate": global_risk,
        "k_min": k_anonymity["achieved_k"],
        "k_anonymity": k_anonymity,
        "sample_identity_risk": {
            "attacker_model": "sdcmicro_super_population",
            "global": global_risk,
            "global_percent": global_risk_pct,
            "expected_reidentifications": expected,
            **individual_risk,
            "population_risk_estimated": weight_column is not None,
        },
        "attribute_disclosure": attributes,
        "warnings": [
            "sdcMicro GPL-2.0 measures were produced by a separate Rscript process."
        ],
    }


def _class_size_distribution(raw_k: Mapping[str, Any]) -> list[dict[str, int]]:
    raw_distribution = raw_k.get("class_size_distribution")
    if not isinstance(raw_distribution, list):
        raise SDCMicroBridgeError(
            "sdcMicro result has an invalid class_size_distribution"
        )
    distribution: list[dict[str, int]] = []
    for item in raw_distribution:
        if not isinstance(item, Mapping):
            raise SDCMicroBridgeError(
                "sdcMicro result has an invalid class_size_distribution entry"
            )
        distribution.append(
            {
                "size": _required_int(item, "size", minimum=1),
                "class_count": _required_int(item, "class_count", minimum=1),
            }
        )
    return distribution


def _attribute_disclosure(
    raw_attributes: Any,
    *,
    expected_attributes: Sequence[str],
) -> list[dict[str, Any]]:
    if not isinstance(raw_attributes, list):
        raise SDCMicroBridgeError("sdcMicro result has invalid l_diversity measures")
    attributes: list[dict[str, Any]] = []
    for raw_attribute in raw_attributes:
        if not isinstance(raw_attribute, Mapping):
            raise SDCMicroBridgeError(
                "sdcMicro result has an invalid l_diversity entry"
            )
        attribute = raw_attribute.get("attribute")
        if not isinstance(attribute, str) or not attribute:
            raise SDCMicroBridgeError(
                "sdcMicro result has an invalid l_diversity attribute"
            )
        achieved = _required_float(raw_attribute, "achieved_distinct", minimum=0.0)
        target = _required_int(raw_attribute, "target", minimum=1)
        violating_classes = _required_int(
            raw_attribute, "violating_class_count", minimum=0
        )
        attributes.append(
            {
                "attribute": attribute,
                "l_diversity": {
                    "metric": "distinct",
                    "achieved": achieved,
                    "threshold": target,
                    "violating_classes": violating_classes,
                    "meets_target": achieved >= target and violating_classes == 0,
                    "entropy_achieved": _optional_float(
                        raw_attribute, "achieved_entropy", minimum=0.0
                    ),
                    "recursive_achieved": _optional_float(
                        raw_attribute, "achieved_recursive", minimum=0.0
                    ),
                },
            }
        )
    if [item["attribute"] for item in attributes] != list(expected_attributes):
        raise SDCMicroBridgeError(
            "sdcMicro result l_diversity attributes do not match the request"
        )
    return attributes


def _required_int(
    mapping: Mapping[str, Any],
    key: str,
    *,
    minimum: int,
) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SDCMicroBridgeError(f"sdcMicro result has an invalid {key}")
    return value


def _required_float(
    mapping: Mapping[str, Any],
    key: str,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SDCMicroBridgeError(f"sdcMicro result has an invalid {key}")
    normalized = float(value)
    if (
        not math.isfinite(normalized)
        or normalized < minimum
        or (maximum is not None and normalized > maximum)
    ):
        raise SDCMicroBridgeError(f"sdcMicro result has an invalid {key}")
    return normalized


def _optional_float(
    mapping: Mapping[str, Any],
    key: str,
    *,
    minimum: float,
) -> float | None:
    if mapping.get(key) is None:
        return None
    return _required_float(mapping, key, minimum=minimum)
