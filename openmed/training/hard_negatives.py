"""Hard-negative generation and batch sampling for training recipes."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from openmed.core.anonymizer.providers import clinical_ids

PROVIDER_FACILITY_NAME = "provider_facility_name"
CLINICAL_ABBREVIATION_OR_DRUG_NAME = "clinical_abbreviation_or_drug_name"
LAB_DOSAGE_ACCOUNT_LIKE_VALUE = "lab_dosage_account_like_value"
STRUCTURALLY_VALID_FAKE_ID = "structurally_valid_fake_id"

HARD_NEGATIVE_CATEGORIES = (
    PROVIDER_FACILITY_NAME,
    CLINICAL_ABBREVIATION_OR_DRUG_NAME,
    LAB_DOSAGE_ACCOUNT_LIKE_VALUE,
    STRUCTURALLY_VALID_FAKE_ID,
)

_PROVIDER_FACILITY_TEMPLATES = (
    "Dr. {provider} reviewed the discharge plan at {facility}.",
    "{facility} cardiology service signed the training note.",
    "The consult was routed to {provider}, PharmD, at {facility}.",
)
_PROVIDER_NAMES = (
    "Mira Patel",
    "Elias Chen",
    "Nora Williams",
    "Samir Haddad",
    "Helen Okafor",
)
_FACILITY_NAMES = (
    "Riverside Teaching Clinic",
    "North Valley Simulation Center",
    "Summit General Training Ward",
    "Harbor View Demo Hospital",
)
_ABBREVIATION_DRUG_VALUES = (
    "ASA 81 mg daily",
    "KCl 20 mEq by mouth",
    "D5W at 75 mL/hr",
    "HCTZ 25 mg every morning",
    "IVIG 2 g/kg protocol",
    "CD4 420 cells/uL",
)
_LAB_DOSAGE_TEMPLATES = (
    "amoxicillin-clavulanate {dose_a}-{dose_b} mg twice daily",
    "blood pressure {systolic}/{diastolic} mmHg after ambulation",
    "platelet count {platelets},000/uL on repeat CBC",
    "heparin {units} units every 8 hours",
)


@dataclass(frozen=True)
class HardNegativeExample:
    """A generated negative example that intentionally resembles PHI text."""

    text: str
    category: str
    subtype: str
    value: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_training_item(self) -> dict[str, Any]:
        item = {
            "text": self.text,
            "labels": [],
            "is_hard_negative": True,
            "hard_negative_category": self.category,
            "hard_negative_subtype": self.subtype,
        }
        if self.value is not None:
            item["hard_negative_value"] = self.value
        if self.metadata:
            item["metadata"] = dict(self.metadata)
        return item


class HardNegativeGenerator:
    """Emit deterministic hard negatives for the four section 6.3 categories."""

    def __init__(self, *, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._category_index = 0
        self._fake_id_index = 0

    def generate(self, category: str | None = None) -> HardNegativeExample:
        category_name = category or self._next_category()
        if category_name == PROVIDER_FACILITY_NAME:
            return self.provider_facility_name()
        if category_name == CLINICAL_ABBREVIATION_OR_DRUG_NAME:
            return self.clinical_abbreviation_or_drug_name()
        if category_name == LAB_DOSAGE_ACCOUNT_LIKE_VALUE:
            return self.lab_dosage_account_like_value()
        if category_name == STRUCTURALLY_VALID_FAKE_ID:
            return self.structurally_valid_fake_id()
        raise ValueError(f"unknown hard-negative category: {category_name}")

    def generate_all_categories(self) -> tuple[HardNegativeExample, ...]:
        return tuple(self.generate(category) for category in HARD_NEGATIVE_CATEGORIES)

    def provider_facility_name(self) -> HardNegativeExample:
        provider = self._choice(_PROVIDER_NAMES)
        facility = self._choice(_FACILITY_NAMES)
        template = self._choice(_PROVIDER_FACILITY_TEMPLATES)
        return HardNegativeExample(
            text=template.format(provider=provider, facility=facility),
            category=PROVIDER_FACILITY_NAME,
            subtype="synthetic_provider_facility",
            metadata={"contains_real_patient_identifier": False},
        )

    def clinical_abbreviation_or_drug_name(self) -> HardNegativeExample:
        value = self._choice(_ABBREVIATION_DRUG_VALUES)
        return HardNegativeExample(
            text=f"Medication and shorthand mention: {value}.",
            category=CLINICAL_ABBREVIATION_OR_DRUG_NAME,
            subtype="clinical_shorthand",
            value=value,
        )

    def lab_dosage_account_like_value(self) -> HardNegativeExample:
        template = self._choice(_LAB_DOSAGE_TEMPLATES)
        text = template.format(
            dose_a=self._rng.choice((250, 500, 875)),
            dose_b=self._rng.choice((62, 125)),
            systolic=self._rng.randint(100, 149),
            diastolic=self._rng.randint(60, 95),
            platelets=self._rng.randint(120, 450),
            units=self._rng.choice((5000, 7500, 10000)),
        )
        return HardNegativeExample(
            text=text,
            category=LAB_DOSAGE_ACCOUNT_LIKE_VALUE,
            subtype="lab_or_dosage_numeric_pattern",
        )

    def structurally_valid_fake_id(self) -> HardNegativeExample:
        generators = (
            ("npi", clinical_ids.generate_npi, clinical_ids.validate_npi),
            ("luhn", clinical_ids.generate_luhn_identifier, clinical_ids.validate_luhn),
            ("ssn", clinical_ids.generate_ssn, clinical_ids.validate_ssn),
        )
        subtype, generate_value, validator = generators[
            self._fake_id_index % len(generators)
        ]
        self._fake_id_index += 1
        value = generate_value(rng=self._rng)
        if not validator(value):
            raise RuntimeError(f"generated invalid {subtype} hard-negative identifier")
        return HardNegativeExample(
            text=f"Synthetic checksum fixture {subtype.upper()}: {value}",
            category=STRUCTURALLY_VALID_FAKE_ID,
            subtype=subtype,
            value=value,
            metadata={"validator": f"clinical_ids.validate_{subtype}"},
        )

    def _next_category(self) -> str:
        category = HARD_NEGATIVE_CATEGORIES[
            self._category_index % len(HARD_NEGATIVE_CATEGORIES)
        ]
        self._category_index += 1
        return category

    def _choice(self, values: Sequence[str]) -> str:
        return values[self._rng.randrange(len(values))]


@dataclass
class HardNegativeSampler:
    """Guarantee at least one hard negative in each sampled training batch."""

    min_hard_negatives_per_batch: int = 1
    generator: HardNegativeGenerator = field(default_factory=HardNegativeGenerator)

    def __post_init__(self) -> None:
        if self.min_hard_negatives_per_batch <= 0:
            raise ValueError("min_hard_negatives_per_batch must be positive")

    def sample_batch(
        self,
        batch: Sequence[Mapping[str, Any]],
        *,
        recipe_config: Any | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Return a batch with the configured hard-negative minimum satisfied."""

        items = [_copy_item(item) for item in batch]
        if recipe_config is not None and not requires_hard_negative_sampler(
            recipe_config
        ):
            return tuple(items)

        existing = count_hard_negatives(items)
        for _ in range(self.min_hard_negatives_per_batch - existing):
            items.append(self.generator.generate().to_training_item())
        return tuple(items)


def sampler_for_recipe(
    recipe_config: Any,
    *,
    seed: int | None = None,
    min_hard_negatives_per_batch: int = 1,
) -> HardNegativeSampler:
    """Create a sampler for a recipe config that requires hard negatives."""

    if not requires_hard_negative_sampler(recipe_config):
        raise ValueError("recipe config does not require hard negatives")
    config_seed = seed if seed is not None else getattr(recipe_config, "seed", None)
    return HardNegativeSampler(
        min_hard_negatives_per_batch=min_hard_negatives_per_batch,
        generator=HardNegativeGenerator(seed=config_seed),
    )


def sample_hard_negatives(
    batch: Sequence[Mapping[str, Any]],
    *,
    recipe_config: Any | None = None,
    seed: int | None = None,
    min_hard_negatives_per_batch: int = 1,
) -> tuple[dict[str, Any], ...]:
    """Convenience wrapper for adding hard negatives to a batch."""

    if recipe_config is not None:
        if not requires_hard_negative_sampler(recipe_config):
            return tuple(_copy_item(item) for item in batch)
        sampler = sampler_for_recipe(
            recipe_config,
            seed=seed,
            min_hard_negatives_per_batch=min_hard_negatives_per_batch,
        )
    else:
        sampler = HardNegativeSampler(
            min_hard_negatives_per_batch=min_hard_negatives_per_batch,
            generator=HardNegativeGenerator(seed=seed),
        )
    return sampler.sample_batch(batch, recipe_config=recipe_config)


def requires_hard_negative_sampler(recipe_config: Any) -> bool:
    if isinstance(recipe_config, Mapping):
        return bool(recipe_config.get("hard_negatives_required"))
    return bool(getattr(recipe_config, "hard_negatives_required", False))


def count_hard_negatives(batch: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for item in batch if bool(item.get("is_hard_negative")))


def _copy_item(item: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(item))


__all__ = [
    "CLINICAL_ABBREVIATION_OR_DRUG_NAME",
    "HARD_NEGATIVE_CATEGORIES",
    "LAB_DOSAGE_ACCOUNT_LIKE_VALUE",
    "PROVIDER_FACILITY_NAME",
    "STRUCTURALLY_VALID_FAKE_ID",
    "HardNegativeExample",
    "HardNegativeGenerator",
    "HardNegativeSampler",
    "count_hard_negatives",
    "requires_hard_negative_sampler",
    "sample_hard_negatives",
    "sampler_for_recipe",
]
