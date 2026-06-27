"""Search helpers for the committed OpenMed model manifest."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .model_registry import MANIFEST_PATH, load_manifest_rows


@dataclass(frozen=True)
class ModelQuery:
    """Filters used to search the committed OpenMed model manifest."""

    task: str | None = None
    language: str | None = None
    tier: str | None = None
    max_params: int | None = None
    min_params: int | None = None
    format: str | None = None
    license: str | None = None
    query: str | None = None
    require_params: bool = False


@dataclass(frozen=True)
class ModelSearchResult:
    """Typed model row returned by :func:`search_models`."""

    repo_id: str
    family: str | None
    task: str | None
    languages: tuple[str, ...] = ()
    tier: str | None = None
    param_count: int | None = None
    architecture: str | None = None
    base_model: str | None = None
    formats: tuple[str, ...] = ()
    canonical_labels: tuple[str, ...] = ()
    benchmark: Mapping[str, Any] | list[Mapping[str, Any]] = field(default_factory=dict)
    arxiv: str | None = None
    license: str | None = None
    reproducibility_hash: str | None = None
    released: str | None = None
    manifest_row: Mapping[str, Any] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )


def search_models(
    *,
    task: str | None = None,
    language: str | None = None,
    tier: str | None = None,
    max_params: int | None = None,
    min_params: int | None = None,
    format: str | None = None,
    license: str | None = None,
    query: str | None = None,
    require_params: bool = False,
    manifest_path: str | Path | None = None,
) -> list[ModelSearchResult]:
    """Search the local model manifest and return matching rows.

    The search reads the committed JSONL manifest through
    :func:`openmed.core.model_registry.load_manifest_rows`; it does not perform
    live model discovery or contact the Hugging Face Hub.
    """

    model_query = ModelQuery(
        task=task,
        language=language,
        tier=tier,
        max_params=max_params,
        min_params=min_params,
        format=format,
        license=license,
        query=query,
        require_params=require_params,
    )
    rows = load_manifest_rows(Path(manifest_path) if manifest_path else MANIFEST_PATH)
    results = [
        _result_from_row(row) for row in rows if _matches_query(row, model_query)
    ]
    return sorted(results, key=_result_sort_key)


# Device tiers ordered from the most constrained to the least constrained.
DEVICE_TIERS: tuple[str, ...] = ("phone", "laptop", "workstation", "server")
_DEVICE_TIER_RANK = {name: index for index, name in enumerate(DEVICE_TIERS)}

# Approximate usable RAM budgets (in MB) per device tier. These are only
# consulted when a manifest row carries ``peak_ram_mb`` enrichment; rows without
# it are retained and ranked by parameter count so the command degrades
# gracefully before the enrichment task lands.
_DEVICE_RAM_BUDGET_MB = {
    "phone": 6_000,
    "laptop": 16_000,
    "workstation": 64_000,
    "server": 512_000,
}


def recommend_models(
    *,
    device_tier: str,
    task: str | None = None,
    language: str | None = None,
    manifest_path: str | Path | None = None,
) -> list[ModelSearchResult]:
    """Rank on-device models that fit ``device_tier`` for a task and language.

    Task/language filtering reuses :func:`search_models`; this adds device-tier
    fit and ranking on top. A row fits when its ``recommended_tier`` is no larger
    than the requested tier and, when present, its ``peak_ram_mb`` is within the
    tier budget. Rows lacking that enrichment are retained and ranked by
    parameter count, so the command still returns a shortlist today.
    """

    tier = device_tier.casefold()
    if tier not in _DEVICE_TIER_RANK:
        raise ValueError(
            f"Unknown device tier: {device_tier!r}. "
            f"Expected one of: {', '.join(DEVICE_TIERS)}."
        )

    candidates = search_models(
        task=task,
        language=language,
        manifest_path=manifest_path,
    )
    fitted = [result for result in candidates if _fits_device_tier(result, tier)]
    return sorted(fitted, key=lambda result: _recommendation_sort_key(result, tier))


def _fits_device_tier(result: ModelSearchResult, tier: str) -> bool:
    recommended = _recommended_tier(result)
    if recommended is not None and recommended in _DEVICE_TIER_RANK:
        if _DEVICE_TIER_RANK[recommended] > _DEVICE_TIER_RANK[tier]:
            return False

    peak_ram = _peak_ram_mb(result, tier)
    if peak_ram is not None and peak_ram > _DEVICE_RAM_BUDGET_MB[tier]:
        return False

    return True


def _recommendation_sort_key(
    result: ModelSearchResult, tier: str
) -> tuple[int, float, float, str]:
    return (
        0 if _has_tier_enrichment(result, tier) else 1,
        _size_metric(result, tier),
        -_benchmark_score(result),
        result.repo_id.casefold(),
    )


def _has_tier_enrichment(result: ModelSearchResult, tier: str) -> bool:
    return (
        _recommended_tier(result) is not None or _peak_ram_mb(result, tier) is not None
    )


def _size_metric(result: ModelSearchResult, tier: str) -> float:
    peak_ram = _peak_ram_mb(result, tier)
    if peak_ram is not None:
        return peak_ram
    if result.param_count is not None:
        return float(result.param_count)
    return float("inf")


def _benchmark_score(result: ModelSearchResult) -> float:
    benchmark = result.benchmark
    entries = benchmark if isinstance(benchmark, list) else [benchmark]
    scores: list[float] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        for key, value in entry.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            lowered = str(key).casefold()
            if "recall" in lowered or "f1" in lowered:
                scores.append(float(value))
    return max(scores) if scores else 0.0


def _recommended_tier(result: ModelSearchResult) -> str | None:
    value = result.manifest_row.get("recommended_tier")
    if isinstance(value, str) and value.strip():
        return value.strip().casefold()
    return None


def _peak_ram_mb(result: ModelSearchResult, tier: str) -> float | None:
    value = result.manifest_row.get("peak_ram_mb")
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Mapping):
        candidate = value.get(tier)
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            return float(candidate)
    return None


def _matches_query(row: Mapping[str, Any], query: ModelQuery) -> bool:
    if query.task and not _matches_text(row.get("task"), query.task):
        return False

    if query.language and not _matches_language(row, query.language):
        return False

    if query.tier and not _matches_text(row.get("tier"), query.tier):
        return False

    if query.license and not _matches_text(row.get("license"), query.license):
        return False

    if query.format and not _matches_format(row, query.format):
        return False

    if not _matches_param_budget(row, query):
        return False

    if query.query and not _matches_free_text(row, query.query):
        return False

    return True


def _matches_text(value: Any, expected: str) -> bool:
    if value is None:
        return False
    return str(value).casefold() == expected.casefold()


def _matches_language(row: Mapping[str, Any], expected: str) -> bool:
    expected = expected.casefold()
    return expected in {
        str(language).casefold() for language in _sequence(row, "languages")
    }


def _matches_format(row: Mapping[str, Any], expected: str) -> bool:
    expected = _normalize_format(expected)
    for value in _sequence(row, "formats"):
        candidate = _normalize_format(value)
        if candidate == expected or candidate.startswith(f"{expected}-"):
            return True

    legacy_format = row.get("format") or row.get("model_format")
    if legacy_format is not None:
        candidate = _normalize_format(legacy_format)
        return candidate == expected or candidate.startswith(f"{expected}-")
    return False


def _matches_param_budget(row: Mapping[str, Any], query: ModelQuery) -> bool:
    param_count = row.get("param_count")
    if not isinstance(param_count, int):
        if query.require_params:
            return False
        if query.min_params is not None:
            return False
        return True

    if query.min_params is not None and param_count < query.min_params:
        return False
    if query.max_params is not None and param_count > query.max_params:
        return False
    return True


def _matches_free_text(row: Mapping[str, Any], query: str) -> bool:
    needle = query.casefold()
    for key in ("repo_id", "family"):
        value = row.get(key)
        if value is not None and needle in str(value).casefold():
            return True
    return False


def _result_from_row(row: Mapping[str, Any]) -> ModelSearchResult:
    return ModelSearchResult(
        repo_id=str(row["repo_id"]),
        family=_optional_str(row.get("family")),
        task=_optional_str(row.get("task")),
        languages=tuple(str(value) for value in _sequence(row, "languages")),
        tier=_optional_str(row.get("tier")),
        param_count=row.get("param_count")
        if isinstance(row.get("param_count"), int)
        else None,
        architecture=_optional_str(row.get("architecture")),
        base_model=_optional_str(row.get("base_model")),
        formats=tuple(str(value) for value in _sequence(row, "formats")),
        canonical_labels=tuple(
            str(value) for value in _sequence(row, "canonical_labels")
        ),
        benchmark=_benchmark_from_row(row),
        arxiv=_optional_str(row.get("arxiv")),
        license=_optional_str(row.get("license")),
        reproducibility_hash=_optional_str(row.get("reproducibility_hash")),
        released=_optional_str(row.get("released")),
        manifest_row=dict(row),
    )


def _result_sort_key(result: ModelSearchResult) -> tuple[str, str, str]:
    return (
        result.repo_id.casefold(),
        (result.family or "").casefold(),
        (result.released or "").casefold(),
    )


def _sequence(row: Mapping[str, Any], key: str) -> tuple[Any, ...]:
    value = row.get(key)
    if isinstance(value, list | tuple):
        return tuple(value)
    return ()


def _benchmark_from_row(
    row: Mapping[str, Any],
) -> Mapping[str, Any] | list[Mapping[str, Any]]:
    value = row.get("benchmark")
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, list):
        return [dict(entry) for entry in value if isinstance(entry, Mapping)]
    return {}


def _normalize_format(value: Any) -> str:
    return str(value).strip().casefold().replace("_", "-")


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
