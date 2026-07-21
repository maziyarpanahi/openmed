"""Coverage reporting for committed golden de-identification fixtures."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from openmed.core.decoding.spans import iter_grapheme_clusters
from openmed.core.labels import CANONICAL_LABELS
from openmed.core.manifest_schema import (
    LANGUAGE_SCRIPT_TARGETS,
    SCRIPT_COVERAGE_TARGETS,
    SCRIPT_COVERAGE_UNK_THRESHOLD,
)
from openmed.core.pii_i18n import (
    SUPPORTED_LANGUAGES,
    TOKENIZER_SCRIPT_FAKE_DATA,
)
from openmed.eval.golden import (
    GOLDEN_CATEGORIES,
    HARD_NEGATIVE_CATEGORY,
    GoldenFixture,
    load_golden_fixtures,
)

GOLDEN_EDGE_CASE_CATEGORIES: tuple[str, ...] = GOLDEN_CATEGORIES


@dataclass(frozen=True)
class FixtureCoverageReport:
    """Golden fixture coverage across labels, languages, and edge cases."""

    fixture_count: int
    covered_labels: tuple[str, ...]
    missing_labels: tuple[str, ...]
    covered_languages: tuple[str, ...]
    missing_languages: tuple[str, ...]
    covered_categories: tuple[str, ...]
    missing_categories: tuple[str, ...]
    category_counts: Mapping[str, int]
    hard_negative_fixture_count: int
    hard_negative_candidate_count: int
    hard_negative_labels: tuple[str, ...]
    hard_negative_languages: tuple[str, ...]
    hard_negative_difficulty_buckets: Mapping[str, int]

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-ready mapping."""
        category_counts = {
            category: int(self.category_counts.get(category, 0))
            for category in GOLDEN_EDGE_CASE_CATEGORIES
        }
        difficulty_buckets = {
            bucket: int(self.hard_negative_difficulty_buckets.get(bucket, 0))
            for bucket in _HARD_NEGATIVE_DIFFICULTY_BUCKETS
        }
        return {
            "fixture_count": self.fixture_count,
            "labels": {
                "covered": list(self.covered_labels),
                "missing": list(self.missing_labels),
            },
            "languages": {
                "covered": list(self.covered_languages),
                "missing": list(self.missing_languages),
            },
            "categories": {
                "covered": list(self.covered_categories),
                "missing": list(self.missing_categories),
            },
            "category_counts": category_counts,
            "hard_negatives": {
                "candidate_count": self.hard_negative_candidate_count,
                "difficulty_buckets": difficulty_buckets,
                "fixture_count": self.hard_negative_fixture_count,
                "labels": list(self.hard_negative_labels),
                "languages": list(self.hard_negative_languages),
            },
        }

    def to_markdown(self) -> str:
        """Render a byte-stable Markdown coverage report."""
        label_status = _status_by_value(
            supported=sorted(CANONICAL_LABELS),
            covered=self.covered_labels,
        )
        language_status = _status_by_value(
            supported=sorted(SUPPORTED_LANGUAGES),
            covered=self.covered_languages,
        )

        lines = [
            "# Golden Fixture Coverage",
            "",
            "## Summary",
            "",
            "| Scope | Covered | Missing | Total |",
            "|---|---:|---:|---:|",
            (
                f"| Labels | {len(self.covered_labels)} | "
                f"{len(self.missing_labels)} | {len(CANONICAL_LABELS)} |"
            ),
            (
                f"| Languages | {len(self.covered_languages)} | "
                f"{len(self.missing_languages)} | {len(SUPPORTED_LANGUAGES)} |"
            ),
            (
                f"| Categories | {len(self.covered_categories)} | "
                f"{len(self.missing_categories)} | "
                f"{len(GOLDEN_EDGE_CASE_CATEGORIES)} |"
            ),
            f"| Fixtures | {self.fixture_count} | 0 | {self.fixture_count} |",
            "",
            "## Labels",
            "",
            "| Label | Status |",
            "|---|---|",
        ]
        for label, status in label_status:
            lines.append(f"| `{label}` | {status} |")

        lines.extend(
            [
                "",
                "## Languages",
                "",
                "| Language | Status |",
                "|---|---|",
            ]
        )
        for language, status in language_status:
            lines.append(f"| `{language}` | {status} |")

        lines.extend(
            [
                "",
                "## Categories",
                "",
                "| Category | Fixture Count | Status |",
                "|---|---:|---|",
            ]
        )
        covered_categories = set(self.covered_categories)
        for category in GOLDEN_EDGE_CASE_CATEGORIES:
            status = "covered" if category in covered_categories else "missing"
            count = int(self.category_counts.get(category, 0))
            lines.append(f"| `{category}` | {count} | {status} |")

        lines.extend(
            [
                "",
                "## Hard Negatives",
                "",
                "| Scope | Count |",
                "|---|---:|",
                f"| Fixtures | {self.hard_negative_fixture_count} |",
                f"| Candidates | {self.hard_negative_candidate_count} |",
                f"| Labels | {len(self.hard_negative_labels)} |",
                f"| Languages | {len(self.hard_negative_languages)} |",
                "",
                "| Difficulty Bucket | Candidate Count |",
                "|---|---:|",
            ]
        )
        for bucket in _HARD_NEGATIVE_DIFFICULTY_BUCKETS:
            count = int(self.hard_negative_difficulty_buckets.get(bucket, 0))
            lines.append(f"| `{bucket}` | {count} |")

        return "\n".join(lines) + "\n"

    def write_markdown(self, path: str | Path) -> Path:
        """Write the Markdown report to *path*."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
        return output_path

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]


def fixture_coverage_report(
    path: str | Path | None = None,
    *,
    fixtures: Sequence[GoldenFixture] | None = None,
) -> FixtureCoverageReport:
    """Compute coverage for golden fixtures.

    Args:
        path: Optional fixture JSON file or directory. Defaults to the committed
            golden fixture directory.
        fixtures: Optional preloaded fixtures. Pass this in tests or callers
            that need to report a filtered fixture set.

    Returns:
        A report listing covered and missing canonical labels, supported
        languages, and golden edge-case categories.
    """
    if path is not None and fixtures is not None:
        raise ValueError("pass either path or fixtures, not both")

    source = list(fixtures) if fixtures is not None else load_golden_fixtures(path)

    observed_labels = {span.label for fixture in source for span in fixture.gold_spans}
    observed_languages = {fixture.language for fixture in source}
    category_counts = Counter(fixture.category for fixture in source)
    ordered_category_counts = {
        category: int(category_counts.get(category, 0))
        for category in GOLDEN_EDGE_CASE_CATEGORIES
    }
    hard_negative_candidates = _hard_negative_candidates(source)
    hard_negative_difficulty_buckets = _difficulty_buckets(hard_negative_candidates)

    covered_labels = tuple(
        label for label in sorted(CANONICAL_LABELS) if label in observed_labels
    )
    missing_labels = tuple(
        label for label in sorted(CANONICAL_LABELS) if label not in observed_labels
    )
    covered_languages = tuple(
        language
        for language in sorted(SUPPORTED_LANGUAGES)
        if language in observed_languages
    )
    missing_languages = tuple(
        language
        for language in sorted(SUPPORTED_LANGUAGES)
        if language not in observed_languages
    )
    covered_categories = tuple(
        category
        for category in GOLDEN_EDGE_CASE_CATEGORIES
        if ordered_category_counts[category] > 0
    )
    missing_categories = tuple(
        category
        for category in GOLDEN_EDGE_CASE_CATEGORIES
        if ordered_category_counts[category] == 0
    )

    return FixtureCoverageReport(
        fixture_count=len(source),
        covered_labels=covered_labels,
        missing_labels=missing_labels,
        covered_languages=covered_languages,
        missing_languages=missing_languages,
        covered_categories=covered_categories,
        missing_categories=missing_categories,
        category_counts=ordered_category_counts,
        hard_negative_fixture_count=ordered_category_counts.get(
            HARD_NEGATIVE_CATEGORY,
            0,
        ),
        hard_negative_candidate_count=len(hard_negative_candidates),
        hard_negative_labels=tuple(
            sorted(
                {
                    str(candidate.get("label"))
                    for candidate in hard_negative_candidates
                    if candidate.get("label")
                }
            )
        ),
        hard_negative_languages=tuple(
            sorted(
                {
                    fixture.language
                    for fixture in source
                    if fixture.category == HARD_NEGATIVE_CATEGORY
                }
            )
        ),
        hard_negative_difficulty_buckets=hard_negative_difficulty_buckets,
    )


def golden_fixture_coverage_report(
    path: str | Path | None = None,
    *,
    fixtures: Sequence[GoldenFixture] | None = None,
) -> FixtureCoverageReport:
    """Alias for :func:`fixture_coverage_report` with explicit golden naming."""
    return fixture_coverage_report(path, fixtures=fixtures)


def _status_by_value(
    *,
    supported: Sequence[str],
    covered: Sequence[str],
) -> list[tuple[str, str]]:
    covered_values = set(covered)
    return [
        (value, "covered" if value in covered_values else "missing")
        for value in supported
    ]


_HARD_NEGATIVE_DIFFICULTY_BUCKETS = (
    "0.00-0.25",
    "0.25-0.50",
    "0.50-0.75",
    "0.75-1.00",
)


def _hard_negative_candidates(
    fixtures: Sequence[GoldenFixture],
) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    for fixture in fixtures:
        if fixture.category != HARD_NEGATIVE_CATEGORY:
            continue
        candidates = fixture.metadata.get("hard_negative_candidates")
        if not isinstance(candidates, list):
            continue
        rows.extend(
            candidate for candidate in candidates if isinstance(candidate, Mapping)
        )
    return rows


def _difficulty_buckets(
    candidates: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    buckets = {bucket: 0 for bucket in _HARD_NEGATIVE_DIFFICULTY_BUCKETS}
    for candidate in candidates:
        try:
            score = float(candidate.get("difficulty_score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        if score < 0.25:
            buckets["0.00-0.25"] += 1
        elif score < 0.50:
            buckets["0.25-0.50"] += 1
        elif score < 0.75:
            buckets["0.50-0.75"] += 1
        else:
            buckets["0.75-1.00"] += 1
    return buckets


_BYTE_FALLBACK_TOKEN_RE = re.compile(
    r"^(?:<0x[0-9a-f]{2}>|\[0x[0-9a-f]{2}\])$",
    re.IGNORECASE,
)


def _gpt2_byte_decoder() -> dict[str, int]:
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values.extend(range(ord("¡"), ord("¬") + 1))
    byte_values.extend(range(ord("®"), ord("ÿ") + 1))
    unicode_values = list(byte_values)
    offset = 0
    for byte_value in range(256):
        if byte_value in byte_values:
            continue
        byte_values.append(byte_value)
        unicode_values.append(256 + offset)
        offset += 1
    return {
        chr(unicode_value): byte_value
        for byte_value, unicode_value in zip(byte_values, unicode_values)
    }


_GPT2_BYTE_DECODER = _gpt2_byte_decoder()


@dataclass(frozen=True)
class TokenizerCoverageReport:
    """Tokenizer representation metrics for every PII manifest entry."""

    models: Mapping[str, Mapping[str, object]]
    model_count: int
    script_count: int = len(SCRIPT_COVERAGE_TARGETS)
    schema_version: int = 1
    unk_threshold: float = SCRIPT_COVERAGE_UNK_THRESHOLD

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-ready report."""
        return {
            "schema_version": self.schema_version,
            "unk_threshold": self.unk_threshold,
            "model_count": self.model_count,
            "script_count": self.script_count,
            "scripts": list(SCRIPT_COVERAGE_TARGETS),
            "models": {
                model_id: dict(value) for model_id, value in self.models.items()
            },
        }

    def to_markdown(self) -> str:
        """Render the model-by-script audit as a Markdown table."""
        flagged = 0
        unsupported = 0
        rows: list[str] = []
        for model_id, result in self.models.items():
            languages = ", ".join(str(item) for item in result.get("languages", []))
            scripts = result.get("scripts", {})
            if not isinstance(scripts, Mapping):
                continue
            for script in SCRIPT_COVERAGE_TARGETS:
                metrics = scripts.get(script, {})
                if not isinstance(metrics, Mapping):
                    continue
                unk_rate = float(metrics.get("unk_rate", 0.0))
                byte_rate = float(metrics.get("byte_fallback_rate", 0.0))
                tokens_per_grapheme = float(metrics.get("tokens_per_grapheme", 0.0))
                verdict = str(metrics.get("verdict", "unclaimed"))
                threshold_flag = "FLAG" if unk_rate > self.unk_threshold else ""
                flagged += bool(threshold_flag)
                unsupported += verdict == "unsupported"
                rows.append(
                    f"| `{model_id}` | {languages} | `{script}` | "
                    f"{unk_rate * 100:.2f}% | {byte_rate * 100:.2f}% | "
                    f"{tokens_per_grapheme:.3f} | {verdict} | {threshold_flag} |"
                )

        lines = [
            "# PII Tokenizer Script Coverage Audit",
            "",
            (
                f"This committed audit covers {self.model_count} PII-family models "
                f"across {self.script_count} script targets. The unsupported threshold "
                f"is strictly greater than {self.unk_threshold * 100:.0f}% UNK tokens "
                "on a script claimed by the model's declared language."
            ),
            "",
            f"- Model-script pairs above the UNK threshold: {flagged}",
            f"- Claimed model-script pairs marked unsupported: {unsupported}",
            "",
            "| Model | Languages | Script | UNK | Byte fallback | Tokens/grapheme | Verdict | Threshold flag |",
            "|---|---|---|---:|---:|---:|---|---|",
            *rows,
            "",
        ]
        return "\n".join(lines)


def audit_pii_tokenizers(
    manifest_rows: Sequence[Mapping[str, object]],
    *,
    tokenizer_loader: Callable[[str], object],
    existing_models: Mapping[str, Mapping[str, object]] | None = None,
    on_model: Callable[[str, Mapping[str, object]], None] | None = None,
    max_workers: int = 1,
) -> TokenizerCoverageReport:
    """Audit every PII manifest tokenizer against all configured scripts.

    The tokenizer loader is injected so default unit tests do not import or
    download optional model dependencies.
    """
    pii_rows = [row for row in manifest_rows if row.get("family") == "PII"]
    models: dict[str, Mapping[str, object]] = dict(existing_models or {})
    expected_ids = {
        str(row.get("repo_id"))
        for row in pii_rows
        if isinstance(row.get("repo_id"), str)
    }
    models = {
        model_id: value
        for model_id, value in models.items()
        if model_id in expected_ids
    }

    pending_rows: list[Mapping[str, object]] = []
    for row in pii_rows:
        model_id = row.get("repo_id")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("every PII manifest row must have a non-empty repo_id")
        existing = models.get(model_id)
        if _complete_model_audit(existing):
            models[model_id] = _refresh_model_audit_claims(
                existing,
                languages=row.get("languages"),
            )
            continue
        pending_rows.append(row)

    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if max_workers == 1:
        audited = (
            _audit_pii_row(row, tokenizer_loader=tokenizer_loader)
            for row in pending_rows
        )
        for model_id, result in audited:
            models[model_id] = result
            if on_model is not None:
                on_model(model_id, result)
    else:
        executor = ThreadPoolExecutor(max_workers=max_workers)
        futures = {}
        try:
            futures = {
                executor.submit(
                    _audit_pii_row,
                    row,
                    tokenizer_loader=tokenizer_loader,
                ): row
                for row in pending_rows
            }
            for future in as_completed(futures):
                model_id, result = future.result()
                models[model_id] = result
                if on_model is not None:
                    on_model(model_id, result)
        except BaseException:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)

    ordered_models = {
        str(row["repo_id"]): models[str(row["repo_id"])]
        for row in pii_rows
        if str(row["repo_id"]) in models
    }
    if len(ordered_models) != len(pii_rows):
        raise RuntimeError(
            f"audit covered {len(ordered_models)} of {len(pii_rows)} PII entries"
        )
    return TokenizerCoverageReport(
        models=ordered_models,
        model_count=len(ordered_models),
    )


def _audit_pii_row(
    row: Mapping[str, object],
    *,
    tokenizer_loader: Callable[[str], object],
) -> tuple[str, Mapping[str, object]]:
    model_id = str(row["repo_id"])
    try:
        tokenizer = tokenizer_loader(model_id)
        scripts = audit_tokenizer_scripts(
            tokenizer,
            languages=row.get("languages"),
        )
    except Exception as exc:
        raise RuntimeError(f"tokenizer audit failed for {model_id}: {exc}") from exc
    return model_id, {
        "languages": list(row.get("languages") or []),
        "tokenizer_source": model_id,
        "scripts": scripts,
    }


def audit_tokenizer_scripts(
    tokenizer: object,
    *,
    languages: object = None,
    probes: Mapping[str, Mapping[str, Sequence[str]]] = TOKENIZER_SCRIPT_FAKE_DATA,
) -> dict[str, dict[str, float | str]]:
    """Measure one tokenizer over all 11 synthetic script probe corpora."""
    claimed_scripts = _claimed_script_targets(languages)
    results: dict[str, dict[str, float | str]] = {}
    for script in SCRIPT_COVERAGE_TARGETS:
        categories = probes.get(script)
        if not isinstance(categories, Mapping):
            raise ValueError(f"missing tokenizer probes for script: {script}")
        texts = [text for values in categories.values() for text in values]
        metrics = _tokenizer_metrics(tokenizer, texts)
        verdict = "unclaimed"
        if script in claimed_scripts:
            verdict = (
                "unsupported"
                if metrics["unk_rate"] > SCRIPT_COVERAGE_UNK_THRESHOLD
                else "supported"
            )
        results[script] = {**metrics, "verdict": verdict}
    return results


def update_manifest_script_coverage(
    manifest_rows: Sequence[Mapping[str, object]],
    report: TokenizerCoverageReport,
) -> list[dict[str, object]]:
    """Return manifest rows populated from a complete tokenizer audit."""
    updated: list[dict[str, object]] = []
    covered_ids: set[str] = set()
    for source in manifest_rows:
        row = dict(source)
        if row.get("family") != "PII":
            updated.append(row)
            continue
        model_id = row.get("repo_id")
        result = report.models.get(str(model_id))
        if not isinstance(result, Mapping):
            raise ValueError(f"audit report is missing PII model: {model_id}")
        scripts = result.get("scripts")
        if not isinstance(scripts, Mapping) or set(scripts) != set(
            SCRIPT_COVERAGE_TARGETS
        ):
            raise ValueError(f"audit report is incomplete for PII model: {model_id}")
        row["script_coverage"] = {
            script: dict(scripts[script]) for script in SCRIPT_COVERAGE_TARGETS
        }
        covered_ids.add(str(model_id))
        updated.append(row)

    expected_ids = {
        str(row.get("repo_id")) for row in manifest_rows if row.get("family") == "PII"
    }
    if covered_ids != expected_ids:
        raise ValueError("audit report does not cover 100% of PII manifest entries")
    return updated


def load_transformers_tokenizer(model_id: str) -> object:
    """Load a tokenizer through the optional ``hf`` dependency group."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            'tokenizer audit requires the "hf" extra: uv pip install -e ".[hf]"'
        ) from exc
    return AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        trust_remote_code=False,
    )


def _tokenizer_metrics(
    tokenizer: object,
    texts: Sequence[str],
) -> dict[str, float]:
    token_count = 0
    unknown_count = 0
    byte_fallback_count = 0
    grapheme_count = 0
    byte_level = _uses_byte_level_tokenization(tokenizer)
    for text in texts:
        token_ids, tokens = _encode_tokens(tokenizer, text)
        token_count += len(token_ids)
        unknown_count += _unknown_token_count(tokenizer, token_ids, tokens)
        byte_fallback_count += sum(
            _is_byte_fallback_token(token, byte_level=byte_level) for token in tokens
        )
        grapheme_count += _grapheme_count(text)

    return {
        "unk_rate": round(unknown_count / token_count if token_count else 0.0, 6),
        "byte_fallback_rate": round(
            byte_fallback_count / token_count if token_count else 0.0,
            6,
        ),
        "tokens_per_grapheme": round(
            token_count / grapheme_count if grapheme_count else 0.0,
            6,
        ),
    }


def _encode_tokens(tokenizer: object, text: str) -> tuple[list[int], list[str]]:
    if not callable(tokenizer):
        raise TypeError("tokenizer must be callable")
    encoded = tokenizer(text, add_special_tokens=False)
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise TypeError("tokenizer output must contain input_ids")
    raw_ids = encoded["input_ids"]
    if hasattr(raw_ids, "tolist"):
        raw_ids = raw_ids.tolist()
    if isinstance(raw_ids, list) and raw_ids and isinstance(raw_ids[0], list):
        raw_ids = raw_ids[0]
    if not isinstance(raw_ids, list):
        raise TypeError("tokenizer input_ids must be a list")
    token_ids = [int(token_id) for token_id in raw_ids]
    converter = getattr(tokenizer, "convert_ids_to_tokens", None)
    if not callable(converter):
        raise TypeError("tokenizer must implement convert_ids_to_tokens")
    raw_tokens = converter(token_ids)
    if isinstance(raw_tokens, str):
        raw_tokens = [raw_tokens]
    tokens = [str(token) for token in raw_tokens]
    if len(tokens) != len(token_ids):
        raise ValueError("tokenizer returned a different number of tokens and IDs")
    return token_ids, tokens


def _unknown_token_count(
    tokenizer: object,
    token_ids: Sequence[int],
    tokens: Sequence[str],
) -> int:
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if isinstance(unk_id, int):
        return sum(token_id == unk_id for token_id in token_ids)
    unk_token = getattr(tokenizer, "unk_token", None)
    if isinstance(unk_token, str):
        return sum(token == unk_token for token in tokens)
    return 0


def _uses_byte_level_tokenization(tokenizer: object) -> bool:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    pre_tokenizer = getattr(backend, "pre_tokenizer", None)
    decoder = getattr(backend, "decoder", None)
    return "ByteLevel" in f"{pre_tokenizer!r} {decoder!r}"


def _is_byte_fallback_token(token: str, *, byte_level: bool) -> bool:
    if _BYTE_FALLBACK_TOKEN_RE.fullmatch(token):
        return True
    if not byte_level:
        return False
    try:
        raw_bytes = bytes(_GPT2_BYTE_DECODER[character] for character in token)
    except KeyError:
        return False
    try:
        raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


def _grapheme_count(text: str) -> int:
    return sum(
        not text[start:end].isspace() for start, end in iter_grapheme_clusters(text)
    )


def _claimed_script_targets(languages: object) -> set[str]:
    if not isinstance(languages, Sequence) or isinstance(languages, (str, bytes)):
        return set()
    claimed: set[str] = set()
    for language in languages:
        if not isinstance(language, str):
            continue
        normalized = language.lower().replace("_", "-")
        claimed.update(LANGUAGE_SCRIPT_TARGETS.get(normalized, ()))
    return claimed


def _complete_model_audit(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    scripts = value.get("scripts")
    if not isinstance(scripts, Mapping) or set(scripts) != set(SCRIPT_COVERAGE_TARGETS):
        return False
    return all(
        isinstance(scripts[script], Mapping)
        and {
            "unk_rate",
            "byte_fallback_rate",
            "tokens_per_grapheme",
            "verdict",
        }
        <= set(scripts[script])
        for script in SCRIPT_COVERAGE_TARGETS
    )


def _refresh_model_audit_claims(
    value: Mapping[str, object],
    *,
    languages: object,
) -> dict[str, object]:
    """Reconcile resumed audit verdicts with current manifest language claims."""
    claimed_scripts = _claimed_script_targets(languages)
    current_languages = (
        list(languages)
        if isinstance(languages, Sequence) and not isinstance(languages, (str, bytes))
        else []
    )
    scripts = value["scripts"]
    assert isinstance(scripts, Mapping)  # guarded by _complete_model_audit
    refreshed_scripts: dict[str, dict[str, object]] = {}
    for script in SCRIPT_COVERAGE_TARGETS:
        metrics = dict(scripts[script])
        verdict = "unclaimed"
        if script in claimed_scripts:
            verdict = (
                "unsupported"
                if float(metrics["unk_rate"]) > SCRIPT_COVERAGE_UNK_THRESHOLD
                else "supported"
            )
        metrics["verdict"] = verdict
        refreshed_scripts[script] = metrics
    return {
        **value,
        "languages": current_languages,
        "scripts": refreshed_scripts,
    }


__all__ = [
    "FixtureCoverageReport",
    "GOLDEN_EDGE_CASE_CATEGORIES",
    "TokenizerCoverageReport",
    "audit_pii_tokenizers",
    "audit_tokenizer_scripts",
    "fixture_coverage_report",
    "golden_fixture_coverage_report",
    "load_transformers_tokenizer",
    "update_manifest_script_coverage",
]
