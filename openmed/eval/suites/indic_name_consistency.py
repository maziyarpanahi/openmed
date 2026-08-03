"""Synthetic consistency gate for Indic personal-name surrogates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openmed.core.anonymizer import Anonymizer
from openmed.core.indic_name_match import detect_name_script
from openmed.core.surrogate_vault import SurrogateVault

INDIC_NAME_CONSISTENCY = "indic-name-consistency"
INDIC_NAME_FIXTURE_PATH = (
    Path(__file__).parents[1] / "golden" / "fixtures" / "indic_name_variants.json"
)
_EVAL_SECRET = "openmed-synthetic-indic-name-consistency"


@dataclass(frozen=True)
class IndicNameVariantGroup:
    """One synthetic identity, its spellings, and collision negatives."""

    fixture_id: str
    language: str
    note: str
    variants: tuple[str, ...]
    negative_surfaces: tuple[str, ...]


@dataclass(frozen=True)
class IndicNameConsistencyResult:
    """Aggregate privacy-safe outcome for the Indic name consistency gate."""

    passed: bool
    group_count: int
    variant_count: int
    surrogate_identity_count: int
    collision_count: int
    leakage_count: int
    script_mismatch_count: int
    deterministic: bool
    failures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result without raw name surfaces."""

        return {
            "collision_count": self.collision_count,
            "deterministic": self.deterministic,
            "failures": list(self.failures),
            "group_count": self.group_count,
            "leakage_count": self.leakage_count,
            "passed": self.passed,
            "script_mismatch_count": self.script_mismatch_count,
            "surrogate_identity_count": self.surrogate_identity_count,
            "variant_count": self.variant_count,
        }


@dataclass(frozen=True)
class _RunEvidence:
    identity_hashes: tuple[str, ...]
    rendered_surrogates: tuple[str, ...]
    collision_count: int
    leakage_count: int
    script_mismatch_count: int
    failures: tuple[str, ...]


def load_indic_name_fixtures(
    path: str | Path | None = None,
) -> list[IndicNameVariantGroup]:
    """Load and validate the bundled synthetic Indic name fixture set."""

    fixture_path = Path(path) if path is not None else INDIC_NAME_FIXTURE_PATH
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Indic name fixture schema_version must be 1")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("synthetic") is not True:
        raise ValueError("Indic name fixtures must be explicitly synthetic")
    if metadata.get("contains_real_phi") is not False:
        raise ValueError("Indic name fixtures must declare contains_real_phi=false")

    groups_payload = payload.get("groups")
    if not isinstance(groups_payload, list) or not groups_payload:
        raise ValueError("Indic name fixtures must contain at least one group")
    groups: list[IndicNameVariantGroup] = []
    seen_ids: set[str] = set()
    for item in groups_payload:
        if not isinstance(item, dict):
            raise ValueError("Indic name fixture groups must be objects")
        fixture_id = str(item.get("id") or "")
        variants = tuple(str(value) for value in item.get("variants") or ())
        negatives = tuple(str(value) for value in item.get("negative_surfaces") or ())
        if not fixture_id or fixture_id in seen_ids:
            raise ValueError("Indic name fixture ids must be non-empty and unique")
        if len(variants) < 3:
            raise ValueError(f"{fixture_id} must contain at least three variants")
        if not any(detect_name_script(value) != "latin" for value in variants):
            raise ValueError(f"{fixture_id} must contain an Indic-script variant")
        if sum(detect_name_script(value) == "latin" for value in variants) < 2:
            raise ValueError(f"{fixture_id} must contain two Latin romanizations")
        if not negatives:
            raise ValueError(f"{fixture_id} must contain collision negatives")
        seen_ids.add(fixture_id)
        groups.append(
            IndicNameVariantGroup(
                fixture_id=fixture_id,
                language=str(item.get("language") or "hi"),
                note=str(item.get("note") or ""),
                variants=variants,
                negative_surfaces=negatives,
            )
        )
    return groups


def indic_name_consistency_metadata(
    *,
    fixture_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return metadata for the synthetic Indic name consistency suite."""

    resolved = Path(fixture_path) if fixture_path else INDIC_NAME_FIXTURE_PATH
    return {
        "collision_gate": "zero",
        "fixture_path": str(resolved),
        "leakage_gate": "zero",
        "stdlib_fallback": True,
        "suite": INDIC_NAME_CONSISTENCY,
        "synthetic": True,
    }


def evaluate_indic_name_consistency(
    fixtures: list[IndicNameVariantGroup] | None = None,
    *,
    similarity_threshold: float = 0.80,
    seed: int = 668,
) -> IndicNameConsistencyResult:
    """Evaluate identity reuse, collisions, leakage, scripts, and determinism."""

    resolved = fixtures if fixtures is not None else load_indic_name_fixtures()
    first = _run_once(
        resolved,
        similarity_threshold=similarity_threshold,
        seed=seed,
    )
    second = _run_once(
        resolved,
        similarity_threshold=similarity_threshold,
        seed=seed,
    )
    deterministic = (
        first.identity_hashes == second.identity_hashes
        and first.rendered_surrogates == second.rendered_surrogates
    )
    failures = list(first.failures)
    if not deterministic:
        failures.append("suite:nondeterministic")
    group_count = len(resolved)
    return IndicNameConsistencyResult(
        passed=(
            not failures
            and first.collision_count == 0
            and first.leakage_count == 0
            and first.script_mismatch_count == 0
            and deterministic
        ),
        group_count=group_count,
        variant_count=sum(len(group.variants) for group in resolved),
        surrogate_identity_count=len(set(first.identity_hashes)),
        collision_count=first.collision_count,
        leakage_count=first.leakage_count,
        script_mismatch_count=first.script_mismatch_count,
        deterministic=deterministic,
        failures=tuple(failures),
    )


def _run_once(
    fixtures: list[IndicNameVariantGroup],
    *,
    similarity_threshold: float,
    seed: int,
) -> _RunEvidence:
    vault = SurrogateVault.in_memory(
        _EVAL_SECRET,
        transliteration_aware_name_matching=True,
        indic_name_similarity_threshold=similarity_threshold,
    )
    anonymizer = Anonymizer(
        lang="hi",
        consistent=True,
        seed=seed,
        transliteration_aware_name_matching=True,
        indic_name_normalizer=vault.indic_name_normalizer,
    )
    identity_hashes: list[str] = []
    rendered_surrogates: list[str] = []
    failures: list[str] = []
    collision_count = 0
    leakage_count = 0
    script_mismatch_count = 0

    for group in fixtures:
        group_keys = [
            vault.key_for(surface, label="PERSON", lang=group.language)
            for surface in group.variants
        ]
        if len({key.text_hash for key in group_keys}) != 1:
            failures.append(f"{group.fixture_id}:identity_mismatch")
        identity_hashes.append(group_keys[0].text_hash)

        replacements: dict[str, str] = {}
        for surface in group.variants:

            def create(attempt: int, source: str = surface) -> str:
                return anonymizer.surrogate_identity(
                    source,
                    "PERSON",
                    lang=group.language,
                    attempt=attempt,
                )

            def render(identity: str, source: str = surface) -> str:
                return anonymizer.render_name_surrogate(
                    identity,
                    source_surface=source,
                )

            replacement = vault.get_or_create(
                surface,
                label="PERSON",
                lang=group.language,
                create_surrogate=create,
                render_surrogate=render,
            )
            replacements[surface] = replacement
            rendered_surrogates.append(replacement)
            source_script = detect_name_script(surface)
            if (
                source_script != "other"
                and detect_name_script(replacement) != source_script
            ):
                script_mismatch_count += 1
                failures.append(f"{group.fixture_id}:script_mismatch")

        output = group.note
        for surface in sorted(group.variants, key=len, reverse=True):
            output = output.replace(surface, replacements[surface])
        folded_output = output.casefold()
        if any(surface.casefold() in folded_output for surface in group.variants):
            leakage_count += 1
            failures.append(f"{group.fixture_id}:variant_leakage")

        group_key = group_keys[0]
        for negative in group.negative_surfaces:
            negative_key = vault.key_for(
                negative,
                label="PERSON",
                lang=group.language,
            )
            if negative_key == group_key:
                collision_count += 1
                failures.append(f"{group.fixture_id}:negative_collision")

    return _RunEvidence(
        identity_hashes=tuple(identity_hashes),
        rendered_surrogates=tuple(rendered_surrogates),
        collision_count=collision_count,
        leakage_count=leakage_count,
        script_mismatch_count=script_mismatch_count,
        failures=tuple(failures),
    )


__all__ = [
    "INDIC_NAME_CONSISTENCY",
    "INDIC_NAME_FIXTURE_PATH",
    "IndicNameConsistencyResult",
    "IndicNameVariantGroup",
    "evaluate_indic_name_consistency",
    "indic_name_consistency_metadata",
    "load_indic_name_fixtures",
]
