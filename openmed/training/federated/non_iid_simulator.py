"""Exact-quota, seeded non-IID assignment of metadata-only synthetic fixtures."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from openmed.core.clinical_language import normalize_clinical_language

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_CODE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\Z")


@dataclass(frozen=True)
class SyntheticPartitionExample:
    """Synthetic fixture identity and controlled language/label/specialty metadata."""

    example_digest: str
    language: str
    labels: tuple[str, ...]
    specialty: str
    synthetic: bool = True

    def __post_init__(self) -> None:
        language, _locale = normalize_clinical_language(self.language)
        if (
            language == "auto"
            or self.synthetic is not True
            or not isinstance(self.example_digest, str)
            or not _DIGEST.fullmatch(self.example_digest)
        ):
            raise ValueError(
                "example requires an explicit synthetic identity and language"
            )
        if (
            not isinstance(self.labels, Sequence)
            or isinstance(self.labels, str)
            or not 1 <= len(self.labels) <= 64
            or any(not isinstance(code, str) for code in self.labels)
            or len(set(self.labels)) != len(self.labels)
        ):
            raise ValueError("provide distinct controlled synthetic labels")
        if any(
            not isinstance(code, str) or not _CODE.fullmatch(code)
            for code in (*self.labels, self.specialty)
        ):
            raise ValueError("invalid controlled partition category")
        object.__setattr__(self, "language", language)
        object.__setattr__(self, "labels", tuple(sorted(self.labels)))


@dataclass(frozen=True)
class InstitutionArchetype:
    """Synthetic institution profile with an exact quota and configurable skews."""

    code: str
    sample_count: int
    language_weights: Mapping[str, int]
    label_weights: Mapping[str, int]
    specialty_weights: Mapping[str, int]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.code, str)
            or not _CODE.fullmatch(self.code)
            or isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or not 0 <= self.sample_count <= 10_000
        ):
            raise ValueError("invalid synthetic institution archetype")
        if any(
            not isinstance(mapping, Mapping) or len(mapping) > 128
            for mapping in (
                self.language_weights,
                self.label_weights,
                self.specialty_weights,
            )
        ):
            raise ValueError("partition weights must be bounded mappings")
        language_weights = {}
        for language, weight in self.language_weights.items():
            code, _locale = normalize_clinical_language(language)
            if code == "auto" or code in language_weights:
                raise ValueError("duplicate or implicit language weight")
            language_weights[code] = weight
        for mapping in (language_weights, self.label_weights, self.specialty_weights):
            if not isinstance(mapping, Mapping) or len(mapping) > 128:
                raise ValueError("partition weights must be bounded mappings")
            for category, weight in mapping.items():
                if (
                    not isinstance(category, str)
                    or not _CODE.fullmatch(category)
                    or isinstance(weight, bool)
                    or not isinstance(weight, int)
                    or not 0 <= weight <= 1_000_000
                ):
                    raise ValueError(
                        "partition weights require controlled codes and bounded integers"
                    )
        object.__setattr__(self, "language_weights", MappingProxyType(language_weights))
        object.__setattr__(
            self, "label_weights", MappingProxyType(dict(self.label_weights))
        )
        object.__setattr__(
            self, "specialty_weights", MappingProxyType(dict(self.specialty_weights))
        )


def simulate_non_iid_partitions(
    examples: Sequence[SyntheticPartitionExample],
    archetypes: Sequence[InstitutionArchetype],
    *,
    seed: int,
) -> dict[str, object]:
    """Assign exact quotas using skew-weighted seeded ranks and augmenting paths.

    Missing language/specialty weights default to one; missing label weights do
    likewise. Zero weights forbid assignment. Reassignment searches preserve
    earlier quotas, preventing a greedy client from consuming another client's
    only eligible fixture. Actual achieved distributions are reported exactly;
    requested weights are preferences, not promised prevalence proportions.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**53:
        raise ValueError("simulation seed must be a bounded non-negative integer")
    if (
        len(examples) > 10_000
        or len(archetypes) > 64
        or not all(
            isinstance(example, SyntheticPartitionExample) for example in examples
        )
        or not all(isinstance(profile, InstitutionArchetype) for profile in archetypes)
    ):
        raise ValueError("provide bounded validated fixtures and archetypes")
    rows = {example.example_digest: example for example in examples}
    profiles = {profile.code: profile for profile in archetypes}
    if len(rows) != len(examples) or len(profiles) != len(archetypes):
        raise ValueError("duplicate synthetic fixture or archetype identity")
    if sum(profile.sample_count for profile in profiles.values()) > len(rows):
        raise ValueError("institution quotas exceed available synthetic examples")
    rankings = {}
    for code, profile in sorted(profiles.items()):
        ranked = []
        for digest, example in sorted(rows.items()):
            weights = [
                profile.language_weights.get(example.language, 1),
                profile.specialty_weights.get(example.specialty, 1),
                *[profile.label_weights.get(label, 1) for label in example.labels],
            ]
            if not all(weights):
                continue
            hashed = hashlib.sha256(f"{seed}:{code}:{digest}".encode()).digest()
            uniform = (int.from_bytes(hashed[:6], "big") + 1) / (2**48 + 1)
            # Log-domain weighted exponential ranks avoid overflow/underflow
            # from multiplying skew weights for multi-label examples.
            priority = math.log(-math.log(uniform)) - math.fsum(
                math.log(weight) for weight in weights
            )
            ranked.append((priority, digest))
        rankings[code] = [digest for _priority, digest in sorted(ranked)]
    ownership = {}
    work = 0

    def assign(code: str, visited: set[str]) -> bool:
        nonlocal work
        if code in visited:
            return False
        visited = visited | {code}
        for digest in rankings[code]:
            work += 1
            if work > 2_000_000:
                raise ValueError("synthetic partition search work limit exceeded")
            owner = ownership.get(digest)
            if owner == code:
                continue
            if owner is None or assign(owner, visited):
                ownership[digest] = code
                return True
        return False

    order = sorted(profiles, key=lambda code: (len(rankings[code]), code))
    for code in order:
        for _index in range(profiles[code].sample_count):
            if not assign(code, set()):
                raise ValueError(
                    "synthetic institution quotas have insufficient eligible fixtures"
                )
    partitions = {}
    for code, profile in sorted(profiles.items()):
        selected = sorted(
            digest for digest, owner in ownership.items() if owner == code
        )
        languages = Counter(rows[digest].language for digest in selected)
        labels = Counter(label for digest in selected for label in rows[digest].labels)
        specialties = Counter(rows[digest].specialty for digest in selected)
        partitions[code] = {
            "example_digests": selected,
            "sample_count": len(selected),
            "language_counts": dict(sorted(languages.items())),
            "label_counts": dict(sorted(labels.items())),
            "specialty_counts": dict(sorted(specialties.items())),
            "declared_skews": {
                "language_weights": dict(sorted(profile.language_weights.items())),
                "label_weights": dict(sorted(profile.label_weights.items())),
                "specialty_weights": dict(sorted(profile.specialty_weights.items())),
            },
        }
    fixture_rows = [
        {
            "digest": example.example_digest,
            "language": example.language,
            "labels": list(example.labels),
            "specialty": example.specialty,
        }
        for example in sorted(examples, key=lambda row: row.example_digest)
    ]
    fixture_digest = hashlib.sha256(
        json.dumps(fixture_rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "synthetic": True,
        "seed": seed,
        "fixture_manifest_digest": fixture_digest,
        "partitions": partitions,
        "unassigned_example_digests": sorted(set(rows) - set(ownership)),
        "assigned_count": len(ownership),
        "total_count": len(rows),
    }
