"""Teacher ensemble configuration loader and registry for weak supervision."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import yaml

from openmed.core.anonymizer.providers.clinical_ids import (
    validate_iban,
    validate_luhn,
    validate_mrn,
    validate_npi,
    validate_phone_us,
    validate_ssn,
    validate_uk_nhs_number,
)
from openmed.core.labels import normalize_label
from openmed.core.model_registry import load_manifest_rows
from openmed.training.weak_labeling import SpanValidator, WeakLabelSpan


class EnsembleError(Exception):
    """Base exception class for all teacher ensemble operational errors."""


class EnsembleConfigError(EnsembleError, ValueError):
    """Raised when an ensemble configuration schema or value is invalid."""


class EnsembleManifestError(EnsembleError, KeyError):
    """Raised when a declared ensemble member model ID is absent from the manifest."""


class EnsembleValidatorError(EnsembleError, KeyError):
    """Raised when a declared validator function name is unknown."""


VALIDATOR_REGISTRY: Dict[str, Callable[[str], bool]] = {
    "validate_ssn": validate_ssn,
    "validate_phone_us": validate_phone_us,
    "validate_npi": validate_npi,
    "validate_mrn": validate_mrn,
    "validate_luhn": validate_luhn,
    "validate_iban": validate_iban,
    "validate_uk_nhs_number": validate_uk_nhs_number,
}

LABEL_TO_VALIDATOR_KEY: Dict[str, str] = {
    "SSN": "validate_ssn",
    "PHONE": "validate_phone_us",
    "NPI": "validate_npi",
    "MRN": "validate_mrn",
    "LUHN": "validate_luhn",
    "CREDIT_CARD": "validate_luhn",
    "SIN": "validate_luhn",
    "IBAN": "validate_iban",
    "NHS": "validate_uk_nhs_number",
    "NHS_NUMBER": "validate_uk_nhs_number",
}


@dataclass(frozen=True)
class EnsembleMember:
    """Configuration metadata for an individual teacher ensemble member.

    Attributes:
        id: Unique identifier for the member (model repo ID or filter name).
        member_type: Type category ('model', 'filter', or 'validator').
        weight: Strictly positive float weight assigned to this member.
        target_entities: Optional sequence of target entity labels.
        metadata: Arbitrary metadata dictionary for description or provenance.
    """

    id: str
    member_type: str
    weight: float
    target_entities: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate member field invariants upon construction."""
        if not isinstance(self.id, str) or not self.id.strip():
            raise EnsembleConfigError("Member ID cannot be empty.")
        if self.id != self.id.strip() or len(self.id) > 200:
            raise EnsembleConfigError(
                "Member ID must be trimmed and at most 200 characters."
            )
        if self.member_type not in {"model", "filter", "validator"}:
            raise EnsembleConfigError(
                f"Invalid member type '{self.member_type}' for member '{self.id}'. "
                "Must be one of: 'model', 'filter', 'validator'."
            )
        if (
            isinstance(self.weight, bool)
            or not isinstance(self.weight, (int, float))
            or not math.isfinite(self.weight)
            or self.weight <= 0.0
            or self.weight > 1.0
        ):
            raise EnsembleConfigError(
                f"Member weight for '{self.id}' must be finite and in (0.0, 1.0], "
                f"got {self.weight}."
            )
        if any(
            not isinstance(label, str) or not label.strip()
            for label in self.target_entities
        ):
            raise EnsembleConfigError(
                f"Target entities for member '{self.id}' must be non-empty strings."
            )
        if len(set(self.target_entities)) != len(self.target_entities):
            raise EnsembleConfigError(
                f"Target entities for member '{self.id}' must be unique."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert ensemble member configuration to a plain dictionary."""
        return {
            "id": self.id,
            "member_type": self.member_type,
            "weight": self.weight,
            "target_entities": list(self.target_entities),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class FamilyEnsembleConfig:
    """Ensemble configuration for a target entity family.

    Attributes:
        family: Name of the entity family (e.g. ClinicalPrivacy, DirectID).
        agreement_threshold: Continuous threshold in range (0.0, 1.0].
        members: Tuple of declared ensemble member configurations.
        validators: Tuple of registered validator function names.
        metadata: Arbitrary metadata dictionary for description or provenance.
    """

    family: str
    agreement_threshold: float
    members: Tuple[EnsembleMember, ...]
    validators: Tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate family ensemble configuration invariants."""
        if not self.family:
            raise EnsembleConfigError("Family name cannot be empty.")
        if (
            isinstance(self.agreement_threshold, bool)
            or not isinstance(self.agreement_threshold, (int, float))
            or not math.isfinite(self.agreement_threshold)
            or self.agreement_threshold <= 0.0
            or self.agreement_threshold > 1.0
        ):
            raise EnsembleConfigError(
                f"Agreement threshold for family '{self.family}' must be in "
                f"(0.0, 1.0], got {self.agreement_threshold}."
            )
        if not self.members:
            raise EnsembleConfigError(
                f"Family '{self.family}' must declare at least one ensemble member."
            )
        member_ids = [member.id for member in self.members]
        if len(set(member_ids)) != len(member_ids):
            raise EnsembleConfigError(
                f"Family '{self.family}' must not declare duplicate member IDs."
            )
        if len(set(self.validators)) != len(self.validators):
            raise EnsembleConfigError(
                f"Family '{self.family}' must not declare duplicate validators."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert family ensemble configuration to a plain dictionary."""
        return {
            "family": self.family,
            "agreement_threshold": self.agreement_threshold,
            "members": [m.to_dict() for m in self.members],
            "validators": list(self.validators),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TeacherEnsembleConfig:
    """Root configuration holding multi-family teacher ensembles.

    Attributes:
        schema_version: Semantic schema version string.
        families: Mapping from family name to family ensemble configuration.
    """

    schema_version: str
    families: Mapping[str, FamilyEnsembleConfig]

    def __post_init__(self) -> None:
        """Validate teacher ensemble root configuration invariants."""
        if self.schema_version != "openmed.training.teacher_ensemble.v1":
            raise EnsembleConfigError(
                f"Unsupported schema_version '{self.schema_version}'. "
                "Expected 'openmed.training.teacher_ensemble.v1'."
            )
        if not self.families:
            raise EnsembleConfigError(
                "TeacherEnsembleConfig must contain a non-empty families mapping."
            )
        for key, family in self.families.items():
            if key != family.family:
                raise EnsembleConfigError(
                    f"Family mapping key '{key}' does not match family '{family.family}'."
                )

    def to_dict(self) -> Dict[str, Any]:
        """Convert root configuration to a plain dictionary."""
        return {
            "schema_version": self.schema_version,
            "families": {k: v.to_dict() for k, v in self.families.items()},
        }


@dataclass(frozen=True)
class AgreementPolicy:
    """Consensus agreement policy consumed by the weak labeling adjudication engine.

    Attributes:
        family: Name of the target entity family.
        agreement_threshold: Continuous threshold configured for this family.
        min_agreeing_models: Integer count of agreeing generator models required.
        member_weights: Mapping from member ID to configured weight.
        validators: Tuple of compiled SpanValidator callables.
    """

    family: str
    agreement_threshold: float
    min_agreeing_models: int
    member_weights: Mapping[str, float]
    validators: Tuple[SpanValidator, ...]

    def validate_detector_sources(self, sources: Iterable[str]) -> None:
        """Require runtime detector sources to match the configured teacher set.

        Args:
            sources: Detector source IDs supplied to weak labeling.

        Raises:
            EnsembleConfigError: If a source is missing or was not declared.
        """
        configured = set(self.member_weights)
        supplied = set(sources)
        if supplied != configured:
            missing = sorted(configured - supplied)
            unexpected = sorted(supplied - configured)
            raise EnsembleConfigError(
                f"Detector sources do not match family '{self.family}': "
                f"missing={missing}, unexpected={unexpected}."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert agreement policy metadata to a plain dictionary."""
        return {
            "family": self.family,
            "agreement_threshold": self.agreement_threshold,
            "min_agreeing_models": self.min_agreeing_models,
            "member_weights": dict(self.member_weights),
            "num_validators": len(self.validators),
        }


def _build_single_span_validator(
    val_name: str, fn: Callable[[str], bool]
) -> SpanValidator:
    """Wrap a string checksum validator into a label-aware SpanValidator callable.

    Args:
        val_name: Name of the validator registered in VALIDATOR_REGISTRY.
        fn: String validator function expecting text as input.

    Returns:
        A callable taking a WeakLabelSpan and returning True if valid or irrelevant.
    """

    def validator(span: WeakLabelSpan) -> bool:
        target_key = LABEL_TO_VALIDATOR_KEY.get(
            span.label.upper()
        ) or LABEL_TO_VALIDATOR_KEY.get(normalize_label(span.label))
        if target_key == val_name:
            return fn(span.text)
        return True

    return validator


def build_span_validators(
    validator_names: Sequence[str],
) -> Tuple[SpanValidator, ...]:
    """Resolve validator names against VALIDATOR_REGISTRY and build SpanValidator adapters.

    Args:
        validator_names: Sequence of registered validator function names.

    Returns:
        Tuple of compiled SpanValidator callables.

    Raises:
        EnsembleValidatorError: If any validator name is unknown.
    """
    adapters: List[SpanValidator] = []
    for val_name in validator_names:
        if val_name not in VALIDATOR_REGISTRY:
            raise EnsembleValidatorError(
                f"Validator '{val_name}' is not registered in VALIDATOR_REGISTRY."
            )
        fn = VALIDATOR_REGISTRY[val_name]
        adapters.append(_build_single_span_validator(val_name, fn))
    return tuple(adapters)


def load_teacher_ensemble_config(
    path_or_preset: str | Path | None = None,
) -> TeacherEnsembleConfig:
    """Load teacher ensemble configuration from YAML file or default repository preset.

    Args:
        path_or_preset: Optional filesystem path to YAML config. Defaults to committed
            openmed/training/configs/teacher_ensemble.yaml.

    Returns:
        Validated TeacherEnsembleConfig instance.

    Raises:
        EnsembleConfigError: If file is missing, unparseable, or contains schema errors.
        EnsembleValidatorError: If any family declares an unknown validator name.
    """
    if path_or_preset is None:
        path = Path(__file__).resolve().parent / "configs" / "teacher_ensemble.yaml"
    else:
        path = Path(path_or_preset)

    if not path.is_file():
        raise EnsembleConfigError(f"Configuration file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as exc:
        raise EnsembleConfigError(f"Failed to parse YAML configuration: {exc}") from exc

    if not isinstance(data, dict):
        raise EnsembleConfigError(
            "Invalid configuration root; must be a dictionary mapping."
        )

    schema_ver = data.get("schema_version", "")
    if not isinstance(schema_ver, str):
        raise EnsembleConfigError("schema_version must be a string.")
    families_raw = data.get("families", {})
    if not isinstance(families_raw, dict):
        raise EnsembleConfigError("Families key must be a dictionary mapping.")

    parsed_families: Dict[str, FamilyEnsembleConfig] = {}
    for fam_key, fam_data in families_raw.items():
        if not isinstance(fam_key, str) or not fam_key.strip():
            raise EnsembleConfigError("Family keys must be non-empty strings.")
        if not isinstance(fam_data, dict):
            raise EnsembleConfigError(f"Family entry '{fam_key}' must be a dictionary.")

        members_raw = fam_data.get("members", [])
        if not isinstance(members_raw, list):
            raise EnsembleConfigError(f"Members for family '{fam_key}' must be a list.")

        parsed_members: List[EnsembleMember] = []
        for m_data in members_raw:
            if not isinstance(m_data, dict):
                raise EnsembleConfigError(
                    f"Member in family '{fam_key}' must be a dictionary."
                )

            m_id = m_data.get("id", "")
            m_type = m_data.get("type", "")
            if not isinstance(m_id, str) or not isinstance(m_type, str):
                raise EnsembleConfigError(
                    f"Member id and type in family '{fam_key}' must be strings."
                )
            raw_weight = m_data.get("weight", 1.0)
            if isinstance(raw_weight, bool):
                raise EnsembleConfigError(
                    f"Invalid weight in member '{m_id}' of family '{fam_key}'."
                )
            try:
                m_weight = float(raw_weight)
            except (TypeError, ValueError) as exc:
                raise EnsembleConfigError(
                    f"Invalid weight in member '{m_id}' of family '{fam_key}': {exc}"
                ) from exc
            targets_raw = m_data.get("target_entities", [])
            if not isinstance(targets_raw, list):
                raise EnsembleConfigError(
                    f"Target entities for member '{m_id}' must be a list."
                )
            m_targets = tuple(targets_raw)
            m_desc = m_data.get("description", "")
            if not isinstance(m_desc, str):
                raise EnsembleConfigError(
                    f"Description for member '{m_id}' must be a string."
                )

            parsed_members.append(
                EnsembleMember(
                    id=m_id,
                    member_type=m_type,
                    weight=m_weight,
                    target_entities=m_targets,
                    metadata={"description": m_desc},
                )
            )

        validators_raw = fam_data.get("validators", [])
        if not isinstance(validators_raw, list):
            raise EnsembleConfigError(
                f"Validators for family '{fam_key}' must be a list."
            )

        for val_name in validators_raw:
            if not isinstance(val_name, str):
                raise EnsembleConfigError(
                    f"Validators for family '{fam_key}' must be strings."
                )
            if val_name not in VALIDATOR_REGISTRY:
                raise EnsembleValidatorError(
                    f"Validator '{val_name}' declared in family '{fam_key}' "
                    "is not registered in VALIDATOR_REGISTRY."
                )

        raw_threshold = fam_data.get("agreement_threshold", 0.5)
        if isinstance(raw_threshold, bool):
            raise EnsembleConfigError(
                f"Invalid agreement_threshold in family '{fam_key}'."
            )
        try:
            agreement_threshold = float(raw_threshold)
        except (TypeError, ValueError) as exc:
            raise EnsembleConfigError(
                f"Invalid agreement_threshold in family '{fam_key}': {exc}"
            ) from exc

        family_name = fam_data.get("family", fam_key)
        if not isinstance(family_name, str):
            raise EnsembleConfigError(f"Family name for '{fam_key}' must be a string.")
        description = fam_data.get("description", "")
        if not isinstance(description, str):
            raise EnsembleConfigError(
                f"Description for family '{fam_key}' must be a string."
            )

        parsed_families[fam_key] = FamilyEnsembleConfig(
            family=family_name,
            agreement_threshold=agreement_threshold,
            members=tuple(parsed_members),
            validators=tuple(validators_raw),
            metadata={"description": description},
        )

    return TeacherEnsembleConfig(
        schema_version=schema_ver,
        families=parsed_families,
    )


def validate_ensemble_against_manifest(
    config: TeacherEnsembleConfig,
    manifest_rows_or_path: Sequence[Mapping[str, Any]] | str | Path | None = None,
) -> None:
    """Validate that every declared ensemble member exists in the manifest.

    Args:
        config: Loaded TeacherEnsembleConfig instance.
        manifest_rows_or_path: Optional manifest rows sequence, Path, or str pointing
            to models.jsonl. If None, resolves default models.jsonl via load_manifest_rows.

    Raises:
        EnsembleManifestError: If a declared member ID is absent from the manifest.
    """
    manifest_ids: Set[str] = set()

    if isinstance(manifest_rows_or_path, (str, Path)):
        manifest_path = Path(manifest_rows_or_path)
        if manifest_path.is_file() and manifest_path.stat().st_size > 0:
            rows = load_manifest_rows(manifest_path)
            manifest_ids = {
                r.get("repo_id") or r.get("model_id")
                for r in rows
                if isinstance(r, Mapping) and (r.get("repo_id") or r.get("model_id"))
            }
        else:
            manifest_ids = set()
    elif isinstance(manifest_rows_or_path, Sequence):
        manifest_ids = {
            r.get("repo_id") or r.get("model_id")
            for r in manifest_rows_or_path
            if isinstance(r, Mapping) and (r.get("repo_id") or r.get("model_id"))
        }
    else:
        rows = load_manifest_rows()
        manifest_ids = {
            r.get("repo_id") or r.get("model_id")
            for r in rows
            if isinstance(r, Mapping) and (r.get("repo_id") or r.get("model_id"))
        }

    for fam_key, family in config.families.items():
        for member in family.members:
            if member.id not in manifest_ids:
                raise EnsembleManifestError(
                    f"Ensemble member '{member.id}' in family '{fam_key}' "
                    "was not found in the model manifest (models.jsonl)."
                )


def resolve_family_agreement_policy(
    config: TeacherEnsembleConfig,
    family: str,
) -> AgreementPolicy:
    """Resolve and construct the AgreementPolicy for a target entity family.

    Calculates the discrete minimum agreeing models count using the formula:
        k = max(2, ceil(agreement_threshold * total_generators))

    Args:
        config: Loaded TeacherEnsembleConfig instance.
        family: Name of the target entity family (e.g. ClinicalPrivacy, DirectID).

    Returns:
        AgreementPolicy ready to be consumed by weak_label_document().

    Raises:
        EnsembleConfigError: If family is not configured, has fewer than 2 generator
            members, or if min_agreeing_models exceeds total generators.
    """
    if family not in config.families:
        raise EnsembleConfigError(
            f"Family '{family}' is not configured in the ensemble configuration."
        )

    fam_config = config.families[family]
    generators = [m for m in fam_config.members if m.member_type in {"model", "filter"}]
    total_generators = len(generators)
    if total_generators < 2:
        raise EnsembleConfigError(
            f"Family '{family}' has {total_generators} generator member(s) "
            "('model' or 'filter'). Inter-model teacher ensembling requires at "
            "least 2 generator members."
        )

    raw_min = math.ceil(fam_config.agreement_threshold * total_generators)
    min_agreeing = max(2, raw_min)
    if min_agreeing > total_generators:
        raise EnsembleConfigError(
            f"Family '{family}' configuration requires {min_agreeing} agreeing "
            f"models, which exceeds the total number of generator members ({total_generators})."
        )

    weights = {m.id: m.weight for m in fam_config.members}
    validators = build_span_validators(fam_config.validators)

    return AgreementPolicy(
        family=family,
        agreement_threshold=fam_config.agreement_threshold,
        min_agreeing_models=min_agreeing,
        member_weights=weights,
        validators=validators,
    )


__all__ = [
    "AgreementPolicy",
    "EnsembleConfigError",
    "EnsembleError",
    "EnsembleManifestError",
    "EnsembleMember",
    "EnsembleValidatorError",
    "FamilyEnsembleConfig",
    "TeacherEnsembleConfig",
    "build_span_validators",
    "load_teacher_ensemble_config",
    "resolve_family_agreement_policy",
    "validate_ensemble_against_manifest",
]
