"""Machine-readable support information for OpenMed integrations.

The matrix in this module is deliberately declarative and standard-library
only. Reading or serializing it does not import an integration dependency,
inspect installed packages, or contact a documentation or service URL. The
optional requirements are copied from ``pyproject.toml`` so callers can review
support boundaries before installing an adapter.

``validate_capability_matrix`` provides the offline coherence check used by
the unit tests. When a repository checkout is available it also verifies that
the referenced source modules, documentation pages, tests, and optional extra
names exist locally.
"""

from __future__ import annotations

import json
import posixpath
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final, overload
from urllib.parse import urlsplit

SCHEMA_VERSION: Final[int] = 1
SUPPORTED_PYTHON: Final[str] = ">=3.10"
MATRIX_DOCUMENTATION_PATH: Final[str] = "docs/integrations/matrix.md"

_ALLOWED_POLICIES: Final[frozenset[str]] = frozenset(
    {"local-only", "configured-network", "user-supplied-resource"}
)
_ALLOWED_TEST_GUARANTEES: Final[frozenset[str]] = frozenset(
    {"offline-unit", "offline-contract"}
)
_CAPABILITY_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_REQUIREMENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True)
class IntegrationCapability:
    """Describe one supported integration surface.

    Attributes:
        name: Stable machine-readable capability key.
        surface: Human-readable integration surface name.
        module: Lazy adapter or integration module that implements the surface.
        extra: Optional ``openmed`` extra that provides the integration.
        optional_dependencies: PEP 508 requirement strings declared by
            ``extra``. Core-only surfaces use an empty tuple.
        documentation: Repository-relative Markdown pages describing the
            surface. Links are local paths and are never fetched.
        tests: Repository-relative offline test files covering the surface.
        policy: Network or resource policy for the integration itself.
        test_guarantee: The kind of offline test evidence recorded for the
            surface.
        description: PHI-free summary suitable for a status page or report.
    """

    name: str
    surface: str
    module: str
    extra: str | None
    optional_dependencies: tuple[str, ...]
    documentation: tuple[str, ...]
    tests: tuple[str, ...]
    policy: str
    test_guarantee: str
    description: str
    supported_python: str = SUPPORTED_PYTHON

    def __post_init__(self) -> None:
        """Normalize sequence fields without importing any adapter."""

        object.__setattr__(
            self,
            "optional_dependencies",
            tuple(str(item) for item in self.optional_dependencies),
        )
        object.__setattr__(
            self,
            "documentation",
            tuple(str(item) for item in self.documentation),
        )
        object.__setattr__(self, "tests", tuple(str(item) for item in self.tests))

    @property
    def dependency_names(self) -> tuple[str, ...]:
        """Return normalized distribution names from the requirements."""

        names: list[str] = []
        for requirement in self.optional_dependencies:
            match = _REQUIREMENT_NAME.match(requirement.strip())
            if match is not None:
                names.append(_normalize_distribution_name(match.group(0)))
        return tuple(names)

    @property
    def supported_versions(self) -> tuple[str, ...]:
        """Return the version-pinned requirement strings for this surface."""

        return self.optional_dependencies

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible, deterministic capability record."""

        return {
            "description": self.description,
            "documentation": list(self.documentation),
            "extra": self.extra,
            "module": self.module,
            "name": self.name,
            "optional_dependencies": list(self.optional_dependencies),
            "policy": self.policy,
            "supported_python": self.supported_python,
            "supported_versions": list(self.supported_versions),
            "surface": self.surface,
            "test_guarantee": self.test_guarantee,
            "tests": list(self.tests),
        }


@dataclass(frozen=True)
class CapabilityMatrix:
    """Versioned collection of :class:`IntegrationCapability` records."""

    capabilities: tuple[IntegrationCapability, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Freeze the supplied capability sequence for stable iteration."""

        object.__setattr__(self, "capabilities", tuple(self.capabilities))

    def __iter__(self) -> Iterator[IntegrationCapability]:
        """Iterate over capabilities in their declared stable order."""

        return iter(self.capabilities)

    def __len__(self) -> int:
        """Return the number of capabilities in the matrix."""

        return len(self.capabilities)

    @overload
    def __getitem__(self, index: int) -> IntegrationCapability: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[IntegrationCapability, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> IntegrationCapability | tuple[IntegrationCapability, ...]:
        """Return one capability or a tuple slice."""

        return self.capabilities[index]

    def get(self, name: str) -> IntegrationCapability:
        """Return a capability by normalized name.

        Raises:
            KeyError: If *name* is not in the matrix.
        """

        key = _normalize_capability_name(name)
        for capability in self.capabilities:
            compact_name = capability.name.replace("_", "")
            if capability.name == key or compact_name == key.replace("_", ""):
                return capability
        known = ", ".join(capability.name for capability in self.capabilities)
        raise KeyError(f"unknown integration capability {name!r}; available: {known}")

    def to_dict(self) -> dict[str, Any]:
        """Return the complete matrix as JSON-compatible data."""

        return {
            "capabilities": [capability.to_dict() for capability in self.capabilities],
            "schema_version": self.schema_version,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the matrix deterministically as JSON."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    def to_markdown(self) -> str:
        """Render a compact deterministic Markdown view of the matrix."""

        lines = [
            "# Integration Capability Matrix",
            "",
            "| Capability | Surface | Optional requirements | Policy | Offline evidence | Test files | Documentation |",
            "|---|---|---|---|---|---|---|",
        ]
        for capability in self.capabilities:
            requirements = (
                ", ".join(
                    f"`{_markdown_cell(requirement)}`"
                    for requirement in capability.optional_dependencies
                )
                or "core"
            )
            tests = ", ".join(f"`{_markdown_cell(test)}`" for test in capability.tests)
            documentation = ", ".join(
                _markdown_doc_link(path) for path in capability.documentation
            )
            lines.append(
                "| "
                f"`{_markdown_cell(capability.name)}` | "
                f"{_markdown_cell(capability.surface)} | "
                f"{requirements} | "
                f"`{_markdown_cell(capability.policy)}` | "
                f"`{_markdown_cell(capability.test_guarantee)}` "
                f"({len(capability.tests)} test file(s)) | "
                f"{tests} | "
                f"{documentation} |"
            )
        return "\n".join(lines) + "\n"


def _capability(
    name: str,
    surface: str,
    module: str,
    *,
    extra: str | None,
    dependencies: tuple[str, ...] = (),
    documentation: tuple[str, ...],
    tests: tuple[str, ...],
    policy: str = "local-only",
    test_guarantee: str = "offline-unit",
    description: str,
) -> IntegrationCapability:
    """Build one concise entry for the canonical matrix."""

    return IntegrationCapability(
        name=name,
        surface=surface,
        module=module,
        extra=extra,
        optional_dependencies=dependencies,
        documentation=documentation,
        tests=tests,
        policy=policy,
        test_guarantee=test_guarantee,
        description=description,
    )


# Keep this tuple sorted by ``name``. It is the source of truth for the
# machine-readable report and the documentation table in
# ``docs/integrations/matrix.md``.
INTEGRATION_CAPABILITIES: Final[tuple[IntegrationCapability, ...]] = (
    _capability(
        "beam",
        "Apache Beam",
        "openmed.interop.beam_transform",
        extra="beam",
        dependencies=("apache-beam>=2.73,<3",),
        documentation=("docs/feature-map.md",),
        tests=("tests/unit/interop/test_beam_transform.py",),
        description="Worker-local Beam transform for de-identification.",
    ),
    _capability(
        "cda",
        "CDA/C-CDA",
        "openmed.interop.cda",
        extra=None,
        documentation=("docs/fhir-interop.md",),
        tests=("tests/unit/interop/test_cda.py",),
        description="Local XML de-identification for CDA and C-CDA documents.",
    ),
    _capability(
        "columnar",
        "Columnar redaction",
        "openmed.integrations.columnar_redactor",
        extra="columnar",
        dependencies=("pyarrow>=16",),
        documentation=("docs/integrations/columnar-redactor.md",),
        tests=("tests/unit/integrations/test_columnar_redactor.py",),
        description="Parquet, ORC, and Arrow column redaction with safe manifests.",
    ),
    _capability(
        "dask",
        "Dask DataFrame",
        "openmed.integrations.dask_accessor",
        extra="dask",
        dependencies=("dask[dataframe]>=2024.8",),
        documentation=("docs/integrations/dask.md",),
        tests=("tests/unit/integrations/test_dask_accessor.py",),
        description="Partition-local DataFrame and Series de-identification.",
    ),
    _capability(
        "duckdb",
        "DuckDB UDFs",
        "openmed.interop.duckdb_udf",
        extra="duckdb",
        dependencies=("duckdb>=1.0,<2", "numpy>=1.26"),
        documentation=("docs/duckdb-deidentification.md",),
        tests=("tests/unit/interop/test_duckdb_udf.py",),
        description="Local SQL functions for clinical extraction and redaction.",
    ),
    _capability(
        "fhir",
        "FHIR operations and bulk NDJSON",
        "openmed.interop.fhir_operations",
        extra=None,
        documentation=("docs/fhir-interop.md",),
        tests=(
            "tests/unit/interop/test_fhir_bulk_ndjson.py",
            "tests/unit/interop/test_fhir_deidentify_operation.py",
        ),
        description="Deterministic FHIR resource and bulk export transformations.",
    ),
    _capability(
        "gliner_biomed",
        "GLiNER-BioMed",
        "openmed.interop.gliner_biomed",
        extra="gliner",
        dependencies=("gliner[tokenizers]>=0.2.0", "torch>=2.0"),
        documentation=("docs/zero-shot-ner.md",),
        tests=("tests/unit/interop/test_gliner_biomed_adapter.py",),
        description="Optional zero-shot biomedical entity adapter.",
    ),
    _capability(
        "haystack",
        "Haystack document redaction",
        "openmed.interop.haystack",
        extra="haystack",
        dependencies=("haystack-ai>=2,<3",),
        documentation=("docs/integrations-haystack.md",),
        tests=("tests/unit/interop/test_haystack_redaction.py",),
        description="Haystack 2.x component that redacts documents locally.",
    ),
    _capability(
        "hl7v2",
        "HL7 v2",
        "openmed.interop.hl7v2",
        extra=None,
        documentation=("docs/hl7v2-deidentification.md",),
        tests=("tests/unit/interop/test_hl7v2.py",),
        description="Segment-aware local de-identification for HL7 v2 messages.",
    ),
    _capability(
        "indic",
        "Indic language helpers",
        "openmed.interop.indic",
        extra="indic",
        dependencies=("indic-nlp-library>=0.92",),
        documentation=("docs/feature-map.md",),
        tests=("tests/unit/interop/test_language_adapters.py",),
        description="Optional Indic segmentation and transliteration helpers.",
    ),
    _capability(
        "lakehouse",
        "Lakehouse table redaction",
        "openmed.integrations.lakehouse_redact",
        extra="columnar",
        dependencies=("pyarrow>=16",),
        documentation=("docs/integrations/lakehouse-redaction.md",),
        tests=("tests/unit/integrations/test_lakehouse_redact.py",),
        description="Snapshot-based redaction for local Parquet table roots.",
    ),
    _capability(
        "langchain",
        "LangChain",
        "openmed.interop.langchain",
        extra="langchain",
        dependencies=("langchain-core>=0.2,<2",),
        documentation=("docs/integrations-langchain.md",),
        tests=("tests/unit/interop/test_langchain_redaction.py",),
        description="Runnable and transform adapters for local context redaction.",
    ),
    _capability(
        "langgraph",
        "LangGraph",
        "openmed.interop.graph_orchestration",
        extra="langgraph",
        dependencies=("langgraph>=0.2,<2",),
        documentation=("docs/feature-map.md",),
        tests=("tests/unit/interop/test_graph_orchestration.py",),
        description="State-graph nodes for local de-identification workflows.",
    ),
    _capability(
        "llamaindex",
        "LlamaIndex",
        "openmed.interop.llamaindex",
        extra="llamaindex",
        dependencies=("llama-index-core>=0.10,<1",),
        documentation=("docs/integrations-llamaindex.md",),
        tests=("tests/unit/interop/test_llamaindex_redaction.py",),
        description="Node and metadata redaction before local synthesis or storage.",
    ),
    _capability(
        "omop",
        "OMOP/CDM",
        "openmed.interop.omop.cdm_loader",
        extra=None,
        documentation=("docs/feature-map.md",),
        tests=("tests/unit/interop/test_omop_cdm_loader.py",),
        description="Deterministic loader foundations for OMOP clinical notes.",
    ),
    _capability(
        "openmrs",
        "OpenMRS REST and FHIR2",
        "openmed.interop.openmrs",
        extra="openmrs",
        dependencies=("httpx>=0.27",),
        documentation=("docs/openmrs-adapter.md",),
        tests=("tests/unit/interop/test_openmrs_adapter.py",),
        policy="configured-network",
        test_guarantee="offline-unit",
        description="Facility-configured REST/FHIR2 adapter with local redaction.",
    ),
    _capability(
        "pandas",
        "pandas DataFrame",
        "openmed.interop.pandas_accessor",
        extra="pandas",
        dependencies=("pandas>=2.0",),
        documentation=("docs/feature-map.md",),
        tests=("tests/unit/interop/test_dataframe_accessor.py",),
        description="DataFrame de-identification and safe release helpers.",
    ),
    _capability(
        "philter",
        "PHILTER",
        "openmed.interop.philter",
        extra="philter",
        dependencies=("philter-ucsf>=1.0.3,<2",),
        documentation=("docs/feature-map.md",),
        tests=("tests/unit/interop/test_philter_adapter.py",),
        description="Canonical span conversion for PHILTER output.",
    ),
    _capability(
        "polars",
        "Polars DataFrame",
        "openmed.interop.polars_accessor",
        extra="polars",
        dependencies=("polars>=0.20",),
        documentation=("docs/feature-map.md",),
        tests=("tests/unit/interop/test_dataframe_accessor.py",),
        description="Polars de-identification and safe release helpers.",
    ),
    _capability(
        "prefect",
        "Prefect",
        "openmed.interop.prefect_tasks",
        extra="prefect",
        dependencies=("prefect>=3.7,<4",),
        documentation=("docs/prefect-integration.md",),
        tests=("tests/unit/interop/test_prefect_tasks.py",),
        description="Optional task and flow wrappers for local batch jobs.",
    ),
    _capability(
        "presidio",
        "Microsoft Presidio",
        "openmed.interop.presidio",
        extra="presidio",
        dependencies=("presidio-analyzer>=2.2.354,<3",),
        documentation=("docs/feature-map.md",),
        tests=("tests/unit/interop/test_presidio_adapter.py",),
        description="RecognizerResult conversion to canonical OpenMed spans.",
    ),
    _capability(
        "pydeid",
        "pyDeid",
        "openmed.interop.pydeid",
        extra="pydeid",
        dependencies=("pyDeid>=0.0.1",),
        documentation=("docs/feature-map.md",),
        tests=("tests/unit/interop/test_pydeid_adapter.py",),
        description="Canonical span conversion for pyDeid results.",
    ),
    _capability(
        "quickumls",
        "QuickUMLS",
        "openmed.interop.quickumls",
        extra="quickumls",
        dependencies=("quickumls>=1.4,<2",),
        documentation=("docs/feature-map.md",),
        tests=("tests/unit/interop/test_scispacy_linker_adapter.py",),
        policy="user-supplied-resource",
        description="Offline conversion of caller-supplied licensed UMLS matches.",
    ),
    _capability(
        "ray",
        "Ray Data",
        "openmed.interop.ray_data",
        extra="ray",
        dependencies=("ray[data]>=2.30,<3",),
        documentation=("docs/feature-map.md",),
        tests=("tests/unit/interop/test_ray_data.py",),
        description="Actor-based batch column de-identification for Ray Data.",
    ),
    _capability(
        "scispacy",
        "scispaCy UMLS linker",
        "openmed.interop.scispacy_linker",
        extra="scispacy",
        dependencies=("scispacy>=0.5.4,<1",),
        documentation=("docs/feature-map.md",),
        tests=("tests/unit/interop/test_scispacy_linker_adapter.py",),
        policy="user-supplied-resource",
        description="Offline conversion of a caller-configured licensed linker.",
    ),
    _capability(
        "scrubadub",
        "scrubadub",
        "openmed.interop.scrubadub",
        extra="scrubadub",
        dependencies=("scrubadub>=2.0,<3",),
        documentation=("docs/feature-map.md",),
        tests=("tests/unit/interop/test_scrubadub_adapter.py",),
        description="Filth span conversion to canonical OpenMed entities.",
    ),
    _capability(
        "search_pipeline",
        "Search pipeline redaction",
        "openmed.interop.search_pipeline",
        extra="haystack",
        dependencies=("haystack-ai>=2,<3",),
        documentation=("docs/integrations-haystack.md",),
        tests=("tests/unit/interop/test_search_pipeline.py",),
        description="Framework-neutral redaction before indexing or generation.",
    ),
    _capability(
        "spacy",
        "spaCy",
        "openmed.interop.spacy_component",
        extra="spacy",
        dependencies=("click>=8.0", "spacy>=3.8.9"),
        documentation=("docs/spacy-component.md",),
        tests=("tests/unit/interop/test_spacy_component.py",),
        description="spaCy pipeline component for canonical OpenMed spans.",
    ),
    _capability(
        "spark",
        "PySpark",
        "openmed.interop.spark_udf",
        extra="spark",
        dependencies=("pyspark>=3.5,<4", "pandas>=2.0", "pyarrow>=16"),
        documentation=("docs/spark-deidentification.md",),
        tests=(
            "tests/unit/interop/test_spark_udf.py",
            "tests/unit/integrations/test_spark_streaming.py",
        ),
        description="Local pandas UDF and streaming de-identification surfaces.",
    ),
    _capability(
        "zh",
        "Chinese language helpers",
        "openmed.interop.zh",
        extra="zh",
        dependencies=("jieba>=0.42", "opencc>=1.4.1,<2", "pypinyin>=0.51"),
        documentation=("docs/chinese-segmentation-operations.md",),
        tests=("tests/unit/interop/test_language_adapters.py",),
        description="Optional Chinese segmentation, conversion, and pinyin tools.",
    ),
)

INTEGRATION_CAPABILITY_MATRIX: Final[CapabilityMatrix] = CapabilityMatrix(
    capabilities=INTEGRATION_CAPABILITIES,
)

# Short aliases make the matrix discoverable without duplicating the data.
CAPABILITY_MATRIX: Final[CapabilityMatrix] = INTEGRATION_CAPABILITY_MATRIX


def capabilities() -> tuple[IntegrationCapability, ...]:
    """Return the canonical integration records in stable order."""

    return INTEGRATION_CAPABILITIES


def capability(name: str) -> IntegrationCapability:
    """Return one canonical integration record by name."""

    return INTEGRATION_CAPABILITY_MATRIX.get(name)


def validate_capability_matrix(
    matrix: CapabilityMatrix | Iterable[IntegrationCapability] = (
        INTEGRATION_CAPABILITY_MATRIX
    ),
    *,
    repository_root: str | Path | None = None,
) -> None:
    """Validate matrix structure and local repository references.

    The validation is intentionally offline. If *repository_root* is omitted,
    the source checkout containing this module is used when it has a
    ``pyproject.toml``; installed wheels simply receive the schema checks.

    Raises:
        ValueError: If one or more records are incomplete or inconsistent.
    """

    if isinstance(matrix, CapabilityMatrix):
        entries = matrix.capabilities
        schema_version = matrix.schema_version
    else:
        entries = tuple(matrix)
        schema_version = SCHEMA_VERSION

    errors: list[str] = []
    if schema_version != SCHEMA_VERSION:
        errors.append(f"unsupported schema_version {schema_version!r}")
    if not entries:
        errors.append("matrix must contain at least one capability")

    names: set[str] = set()
    for index, entry in enumerate(entries):
        prefix = f"capability[{index}]"
        if not isinstance(entry, IntegrationCapability):
            errors.append(f"{prefix} is not an IntegrationCapability")
            continue
        if not _CAPABILITY_NAME.fullmatch(entry.name):
            errors.append(f"{prefix}.name is not a stable lowercase key")
        if entry.name in names:
            errors.append(f"duplicate capability name: {entry.name!r}")
        names.add(entry.name)
        if not entry.surface.strip():
            errors.append(f"{prefix}.surface is empty")
        if not entry.module.startswith("openmed."):
            errors.append(f"{prefix}.module must be an openmed module")
        if entry.extra == "":
            errors.append(f"{prefix}.extra must be None or a non-empty extra")
        if entry.extra is None and entry.optional_dependencies:
            errors.append(f"{prefix} has dependencies but no optional extra")
        if entry.extra is not None and not _CAPABILITY_NAME.fullmatch(
            entry.extra.replace("-", "_")
        ):
            errors.append(f"{prefix}.extra is not a stable extra name")
        if entry.policy not in _ALLOWED_POLICIES:
            errors.append(f"{prefix}.policy is unsupported: {entry.policy!r}")
        if entry.test_guarantee not in _ALLOWED_TEST_GUARANTEES:
            errors.append(
                f"{prefix}.test_guarantee is unsupported: {entry.test_guarantee!r}"
            )
        if entry.supported_python != SUPPORTED_PYTHON:
            errors.append(f"{prefix}.supported_python must be {SUPPORTED_PYTHON!r}")
        if not entry.documentation:
            errors.append(f"{prefix}.documentation is empty")
        if not entry.tests:
            errors.append(f"{prefix}.tests is empty")
        for requirement in entry.optional_dependencies:
            if not _valid_requirement(requirement):
                errors.append(f"{prefix} has invalid requirement {requirement!r}")

    entry_names = tuple(
        entry.name for entry in entries if isinstance(entry, IntegrationCapability)
    )
    if entry_names != tuple(sorted(entry_names)):
        errors.append("capabilities must be sorted by name")

    root = _resolve_repository_root(repository_root)
    if root is not None:
        project_extras = _project_optional_dependencies(root / "pyproject.toml")
        for index, entry in enumerate(entries):
            if not isinstance(entry, IntegrationCapability):
                continue
            prefix = f"capability[{index}]"
            _validate_local_path_fields(entry, root, prefix, errors)
            if entry.extra is not None and project_extras is not None:
                declared = {
                    _normalize_distribution_name(_requirement_name(requirement))
                    for requirement in project_extras.get(entry.extra, ())
                    if _requirement_name(requirement)
                }
                if entry.extra not in project_extras:
                    errors.append(
                        f"{prefix}.extra {entry.extra!r} is not declared in pyproject.toml"
                    )
                for requirement in entry.optional_dependencies:
                    dependency_name = _requirement_name(requirement)
                    if (
                        dependency_name
                        and _normalize_distribution_name(dependency_name)
                        not in declared
                    ):
                        errors.append(
                            f"{prefix} requirement {requirement!r} is not declared "
                            f"by openmed[{entry.extra}]"
                        )
            elif entry.extra is not None and project_extras == {}:
                errors.append(
                    f"{prefix} optional extra validation could not read pyproject.toml"
                )
        _validate_matrix_documentation(root, entries, errors)

    if errors:
        detail = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"invalid integration capability matrix:\n{detail}")


def _validate_local_path_fields(
    entry: IntegrationCapability,
    root: Path,
    prefix: str,
    errors: list[str],
) -> None:
    module_relative = Path("openmed", *entry.module.split(".")[1:])
    module_candidates = (
        root / module_relative.with_suffix(".py"),
        root / module_relative / "__init__.py",
    )
    if not any(candidate.is_file() for candidate in module_candidates):
        errors.append(f"{prefix}.module does not resolve locally: {entry.module}")

    for field_name, values, expected_root, suffix in (
        ("documentation", entry.documentation, "docs", ".md"),
        ("tests", entry.tests, "tests", ".py"),
    ):
        for value in values:
            relative = _safe_relative_path(value)
            if relative is None or not relative.startswith(f"{expected_root}/"):
                errors.append(f"{prefix}.{field_name} has unsafe path: {value!r}")
                continue
            path_without_anchor = relative.split("#", 1)[0]
            candidate = root / path_without_anchor
            if not candidate.is_file() or not path_without_anchor.endswith(suffix):
                errors.append(
                    f"{prefix}.{field_name} does not resolve locally: {value!r}"
                )


def _resolve_repository_root(repository_root: str | Path | None) -> Path | None:
    if repository_root is not None:
        return Path(repository_root).resolve()
    candidate = Path(__file__).resolve().parents[2]
    return candidate if (candidate / "pyproject.toml").is_file() else None


def _validate_matrix_documentation(
    root: Path,
    entries: Sequence[IntegrationCapability],
    errors: list[str],
) -> None:
    """Check the matrix page's local links and row coverage without fetching."""

    path = root / MATRIX_DOCUMENTATION_PATH
    if not path.is_file():
        return
    try:
        markdown = path.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(
            f"cannot read {MATRIX_DOCUMENTATION_PATH}: {exc.__class__.__name__}"
        )
        return

    for entry in entries:
        if f"`{entry.name}`" not in markdown:
            errors.append(f"{MATRIX_DOCUMENTATION_PATH} does not list {entry.name!r}")
        for requirement in entry.optional_dependencies:
            if f"`{requirement}`" not in markdown:
                errors.append(
                    f"{MATRIX_DOCUMENTATION_PATH} omits requirement {requirement!r}"
                )
        for test_path in entry.tests:
            if f"`{test_path}`" not in markdown:
                errors.append(
                    f"{MATRIX_DOCUMENTATION_PATH} omits test path {test_path!r}"
                )

    link_pattern = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)")
    for target in link_pattern.findall(markdown):
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        relative = _safe_relative_path(
            posixpath.normpath(posixpath.join("docs/integrations", parsed.path))
        )
        if relative is None or not relative.startswith("docs/"):
            errors.append(f"{MATRIX_DOCUMENTATION_PATH} has unsafe link: {target!r}")
            continue
        if not (root / relative.split("#", 1)[0]).is_file():
            errors.append(
                f"{MATRIX_DOCUMENTATION_PATH} link does not resolve: {target!r}"
            )


def _project_optional_dependencies(path: Path) -> Mapping[str, Sequence[str]] | None:
    if not path.is_file():
        return None
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            return None
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    project = payload.get("project", {})
    extras = project.get("optional-dependencies", {})
    if not isinstance(extras, Mapping):
        return None
    return {
        str(name): tuple(value) if isinstance(value, list) else ()
        for name, value in extras.items()
    }


def _valid_requirement(requirement: str) -> bool:
    value = str(requirement).strip()
    if not value or _requirement_name(value) == "":
        return False
    return not any(character in value for character in "/\\\n\r")


def _requirement_name(requirement: str) -> str:
    match = _REQUIREMENT_NAME.match(str(requirement).strip())
    return match.group(0) if match is not None else ""


def _normalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _normalize_capability_name(name: str) -> str:
    return str(name or "").strip().lower().replace("-", "_")


def _safe_relative_path(value: str) -> str | None:
    raw = str(value).replace("\\", "/")
    path = PurePosixPath(raw.split("#", 1)[0])
    if path.is_absolute() or ".." in path.parts:
        return None
    normalized = posixpath.normpath(raw)
    return None if normalized in {".", ""} else normalized


def _markdown_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _markdown_doc_link(path: str) -> str:
    raw_path, _, anchor = str(path).partition("#")
    relative = posixpath.relpath(raw_path, "docs/integrations")
    href = f"{relative}#{anchor}" if anchor else relative
    return f"[{raw_path}]({href})"


validate_matrix = validate_capability_matrix

__all__ = [
    "CAPABILITY_MATRIX",
    "MATRIX_DOCUMENTATION_PATH",
    "SCHEMA_VERSION",
    "SUPPORTED_PYTHON",
    "CapabilityMatrix",
    "INTEGRATION_CAPABILITIES",
    "INTEGRATION_CAPABILITY_MATRIX",
    "IntegrationCapability",
    "capabilities",
    "capability",
    "validate_capability_matrix",
    "validate_matrix",
]
