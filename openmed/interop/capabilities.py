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
from typing import Any, Final, cast, overload
from urllib.parse import urlsplit

SCHEMA_VERSION: Final[int] = 1
SUPPORTED_PYTHON: Final[str] = ">=3.10"
MATRIX_DOCUMENTATION_PATH: Final[str] = "docs/integrations/matrix.md"
MAX_CAPABILITIES: Final[int] = 256

_MAX_SEQUENCE_ITEMS: Final[int] = 64
_MAX_TEXT_LENGTH: Final[int] = 4_096
_MAX_NAME_INPUT_LENGTH: Final[int] = 256

_ALLOWED_POLICIES: Final[frozenset[str]] = frozenset(
    {"local-only", "configured-network", "user-supplied-resource"}
)
_ALLOWED_TEST_GUARANTEES: Final[frozenset[str]] = frozenset(
    {"offline-unit", "offline-contract"}
)
_CAPABILITY_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_REQUIREMENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


def _bounded_tuple(value: Any, *, label: str, maximum: int) -> tuple[Any, ...]:
    """Snapshot an iterable without unbounded materialization or hook leakage."""

    try:
        iterator = iter(value)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise TypeError(f"{label} must be a bounded iterable") from None

    collected: list[Any] = []
    for _ in range(maximum + 1):
        try:
            item = next(iterator)
        except StopIteration:
            return tuple(collected)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise ValueError(f"{label} iteration failed") from None
        if len(collected) == maximum:
            raise ValueError(f"{label} exceed the limit of {maximum}")
        collected.append(item)
    raise AssertionError("bounded iteration must return or raise")


def _validate_bounded_text(value: Any, label: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{label} must be a string")
    if len(value) > _MAX_TEXT_LENGTH:
        raise ValueError(f"{label} exceeds the safe length limit")


def _bounded_text_tuple(value: Any, *, label: str) -> tuple[str, ...]:
    items = _bounded_tuple(value, label=label, maximum=_MAX_SEQUENCE_ITEMS)
    for item in items:
        _validate_bounded_text(item, f"{label} item")
    return cast(tuple[str, ...], items)


@dataclass(frozen=True, slots=True, repr=False)
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

        for value, field_name in (
            (self.name, "name"),
            (self.surface, "surface"),
            (self.module, "module"),
            (self.policy, "policy"),
            (self.test_guarantee, "test_guarantee"),
            (self.description, "description"),
            (self.supported_python, "supported_python"),
        ):
            _validate_bounded_text(value, field_name)
        if self.extra is not None:
            _validate_bounded_text(self.extra, "extra")
        object.__setattr__(
            self,
            "optional_dependencies",
            _bounded_text_tuple(
                self.optional_dependencies,
                label="optional_dependencies",
            ),
        )
        object.__setattr__(
            self,
            "documentation",
            _bounded_text_tuple(self.documentation, label="documentation"),
        )
        object.__setattr__(
            self,
            "tests",
            _bounded_text_tuple(self.tests, label="tests"),
        )

    def __repr__(self) -> str:
        """Render the declaration without retaining caller-supplied text."""

        return "IntegrationCapability(<validated metadata>)"

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

        validated = _validated_capability_copy(self)
        _assert_capability_report_safe(validated)
        return {
            "description": validated.description,
            "documentation": list(validated.documentation),
            "extra": validated.extra,
            "module": validated.module,
            "name": validated.name,
            "optional_dependencies": list(validated.optional_dependencies),
            "policy": validated.policy,
            "supported_python": validated.supported_python,
            "supported_versions": list(validated.supported_versions),
            "surface": validated.surface,
            "test_guarantee": validated.test_guarantee,
            "tests": list(validated.tests),
        }


def _validated_capability_copy(
    capability: IntegrationCapability,
) -> IntegrationCapability:
    """Revalidate frozen public state before it reaches a report surface."""

    if type(capability) is not IntegrationCapability:
        raise TypeError("capability must be an IntegrationCapability record")
    try:
        return IntegrationCapability(
            name=capability.name,
            surface=capability.surface,
            module=capability.module,
            extra=capability.extra,
            optional_dependencies=capability.optional_dependencies,
            documentation=capability.documentation,
            tests=capability.tests,
            policy=capability.policy,
            test_guarantee=capability.test_guarantee,
            description=capability.description,
            supported_python=capability.supported_python,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise ValueError("capability contains invalid metadata") from None


def _assert_capability_report_safe(capability: IntegrationCapability) -> None:
    """Reject metadata that is unsafe or ambiguous on public report surfaces."""

    valid = (
        _CAPABILITY_NAME.fullmatch(capability.name) is not None
        and bool(capability.surface.strip())
        and capability.module.startswith("openmed.")
        and capability.extra != ""
        and (capability.extra is not None or not capability.optional_dependencies)
        and (
            capability.extra is None
            or _CAPABILITY_NAME.fullmatch(capability.extra.replace("-", "_"))
            is not None
        )
        and capability.policy in _ALLOWED_POLICIES
        and capability.test_guarantee in _ALLOWED_TEST_GUARANTEES
        and capability.supported_python == SUPPORTED_PYTHON
        and bool(capability.description.strip())
        and bool(capability.documentation)
        and bool(capability.tests)
        and all(_valid_requirement(value) for value in capability.optional_dependencies)
        and all(
            (relative := _safe_relative_path(value)) is not None
            and relative.startswith("docs/")
            and relative.split("#", 1)[0].endswith(".md")
            for value in capability.documentation
        )
        and all(
            (relative := _safe_relative_path(value)) is not None
            and relative.startswith("tests/")
            and relative.split("#", 1)[0].endswith(".py")
            for value in capability.tests
        )
    )
    if not valid:
        raise ValueError("capability cannot be serialized safely")


@dataclass(frozen=True, slots=True)
class CapabilityMatrix:
    """Versioned collection of :class:`IntegrationCapability` records."""

    capabilities: tuple[IntegrationCapability, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Freeze the supplied capability sequence for stable iteration."""

        if type(self.schema_version) is not int:
            raise TypeError("schema_version must be an integer")
        capabilities = _bounded_tuple(
            self.capabilities,
            label="capabilities",
            maximum=MAX_CAPABILITIES,
        )
        if any(type(entry) is not IntegrationCapability for entry in capabilities):
            raise TypeError("capabilities must contain IntegrationCapability records")
        validated = tuple(
            _validated_capability_copy(entry)
            for entry in cast(tuple[IntegrationCapability, ...], capabilities)
        )
        object.__setattr__(self, "capabilities", validated)

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
            _assert_capability_report_safe(capability)
        for capability in self.capabilities:
            compact_name = capability.name.replace("_", "")
            if capability.name == key or compact_name == key.replace("_", ""):
                return capability
        known = ", ".join(capability.name for capability in self.capabilities)
        raise KeyError(f"unknown integration capability; available: {known}")

    def to_dict(self) -> dict[str, Any]:
        """Return the complete matrix as JSON-compatible data."""

        validated = CapabilityMatrix(
            capabilities=self.capabilities,
            schema_version=self.schema_version,
        )
        return {
            "capabilities": [
                capability.to_dict() for capability in validated.capabilities
            ],
            "schema_version": validated.schema_version,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the matrix deterministically as JSON."""

        if type(indent) is not int or not 0 <= indent <= 8:
            raise ValueError("indent must be an integer between 0 and 8")
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    def to_markdown(self) -> str:
        """Render a compact deterministic Markdown view of the matrix."""

        validated = CapabilityMatrix(
            capabilities=self.capabilities,
            schema_version=self.schema_version,
        )
        for capability in validated.capabilities:
            _assert_capability_report_safe(capability)
        lines = [
            "# Integration Capability Matrix",
            "",
            "| Capability | Surface | Optional requirements | Policy | Offline evidence | Test files | Documentation |",
            "|---|---|---|---|---|---|---|",
        ]
        for capability in validated.capabilities:
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
        "airflow",
        "Apache Airflow",
        "openmed.interop.airflow",
        extra="airflow",
        dependencies=("apache-airflow>=3.2.2,<4",),
        documentation=("docs/integrations/airflow.md",),
        tests=("tests/unit/interop/test_airflow.py",),
        description="Bounded local redaction operator for files and record batches.",
    ),
    _capability(
        "arrow_flight",
        "Arrow Flight",
        "openmed.integrations.arrow_flight",
        extra="columnar",
        dependencies=("pyarrow>=16",),
        documentation=("docs/integrations/arrow-flight.md",),
        tests=("tests/unit/integrations/test_arrow_flight.py",),
        policy="configured-network",
        description="Authenticated record-batch redaction over caller-hosted Flight.",
    ),
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
        "dagster",
        "Dagster",
        "openmed.integrations.dagster_assets",
        extra="dagster",
        dependencies=("dagster>=1.8,<2",),
        documentation=("docs/feature-map.md",),
        tests=("tests/unit/integrations/test_dagster_assets.py",),
        description="Partitioned local ops and assets with counts-only metadata.",
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
        "dataflow",
        "Apache Beam Dataflow",
        "openmed.integrations.dataflow_processor",
        extra="beam",
        dependencies=("apache-beam>=2.73,<3",),
        documentation=("docs/feature-map.md",),
        tests=("tests/unit/integrations/test_dataflow_processor.py",),
        description="Bundle-scoped record redaction for Beam and Dataflow workers.",
    ),
    _capability(
        "dataflow_tool",
        "Embedded dataflow tools",
        "openmed.integrations.dataflow_tool_processor",
        extra=None,
        documentation=("docs/feature-map.md",),
        tests=("tests/unit/integrations/test_dataflow_tool_processor.py",),
        description="Bounded callable and JSON-lines processor for local flows.",
    ),
    _capability(
        "distributed_sql",
        "Distributed SQL UDF",
        "openmed.integrations.distributed_sql_udf",
        extra=None,
        documentation=("docs/integrations/distributed-sql-udf.md",),
        tests=("tests/unit/integrations/test_distributed_sql_udf.py",),
        description="Vectorized worker-local UDF with deterministic registration.",
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
        "executable_udf",
        "Executable UDF",
        "openmed.integrations.executable_udf",
        extra=None,
        documentation=("docs/feature-map.md",),
        tests=("tests/unit/integrations/test_executable_udf.py",),
        description="Streaming stdin and stdout column redaction for local engines.",
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
        documentation=("docs/integrations/langchain.md",),
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
        documentation=("docs/integrations/llamaindex.md",),
        tests=("tests/unit/interop/test_llamaindex_redaction.py",),
        description="Node and metadata redaction before local synthesis or storage.",
    ),
    _capability(
        "log_redactor",
        "Structured log redaction",
        "openmed.integrations.log_redactor",
        extra=None,
        documentation=("docs/feature-map.md",),
        tests=("tests/unit/integrations/test_log_redactor.py",),
        description="Local structured-event redaction before log emission.",
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
        "pandas_on_spark",
        "Pandas API on Spark",
        "openmed.integrations.pandas_on_spark",
        extra="spark",
        dependencies=("pyspark>=3.5,<4", "pandas>=2.0", "pyarrow>=16"),
        documentation=("docs/integrations/pandas-on-spark.md",),
        tests=("tests/unit/integrations/test_pandas_on_spark.py",),
        description="Distributed pandas-style accessors with worker-local models.",
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
        "postgres",
        "PostgreSQL transaction adapter",
        "openmed.interop.postgres",
        extra=None,
        documentation=("docs/integrations/postgres.md",),
        tests=("tests/unit/interop/test_postgres.py",),
        description="Caller-owned transaction redaction with parameterized writes.",
    ),
    _capability(
        "postgres_plpython",
        "PostgreSQL PL/Python",
        "openmed.integrations.postgres_plpython",
        extra=None,
        documentation=("docs/integrations/postgres.md",),
        tests=("tests/unit/integrations/test_postgres_plpython.py",),
        description="Generated local PL/Python bodies for in-database redaction.",
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
        "ray_map_batches",
        "Ray Data map_batches",
        "openmed.integrations.ray_map_batches",
        extra="ray",
        dependencies=("ray[data]>=2.30,<3",),
        documentation=("docs/integrations/ray-map-batches.md",),
        tests=("tests/unit/integrations/test_ray_map_batches.py",),
        description="Stateful model actors for deterministic Ray batch redaction.",
    ),
    _capability(
        "remote_function",
        "Warehouse remote function",
        "openmed.integrations.remote_function",
        extra="service",
        dependencies=("fastapi>=0.110",),
        documentation=("docs/integrations/warehouse-remote-function.md",),
        tests=("tests/unit/integrations/test_remote_function.py",),
        policy="configured-network",
        description="Caller-hosted batch handler for warehouse text redaction.",
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
        "search_ingest",
        "Search ingest sidecar",
        "openmed.integrations.search_ingest_processor",
        extra="service",
        dependencies=("fastapi>=0.110",),
        documentation=("docs/feature-map.md",),
        tests=("tests/unit/integrations/test_search_ingest_processor.py",),
        policy="configured-network",
        description="Caller-hosted redaction sidecar for search document envelopes.",
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
        "sqlalchemy",
        "SQLAlchemy",
        "openmed.integrations.sqlalchemy_redact",
        extra="sqlalchemy",
        dependencies=("sqlalchemy>=2.0,<3",),
        documentation=("docs/integrations/sqlalchemy.md",),
        tests=("tests/unit/integrations/test_sqlalchemy_redact.py",),
        description="Write-time local de-identification for selected ORM columns.",
    ),
    _capability(
        "stream_processor",
        "Stream processor",
        "openmed.integrations.stream_processor",
        extra=None,
        documentation=("docs/feature-map.md",),
        tests=("tests/unit/integrations/test_stream_processor.py",),
        description="Framework-neutral record redaction for caller-owned streams.",
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

    if type(matrix) is CapabilityMatrix:
        validated_matrix = CapabilityMatrix(
            capabilities=matrix.capabilities,
            schema_version=matrix.schema_version,
        )
        entries = validated_matrix.capabilities
        schema_version = validated_matrix.schema_version
    else:
        raw_entries = _bounded_tuple(
            matrix,
            label="capability matrix",
            maximum=MAX_CAPABILITIES,
        )
        entries = tuple(
            _validated_capability_copy(entry)
            if type(entry) is IntegrationCapability
            else entry
            for entry in raw_entries
        )
        schema_version = SCHEMA_VERSION

    errors: list[str] = []
    if schema_version != SCHEMA_VERSION:
        errors.append("unsupported schema_version")
    if not entries:
        errors.append("matrix must contain at least one capability")

    names: set[str] = set()
    for index, entry in enumerate(entries):
        prefix = f"capability[{index}]"
        if type(entry) is not IntegrationCapability:
            errors.append(f"{prefix} is not an IntegrationCapability")
            continue
        if not _CAPABILITY_NAME.fullmatch(entry.name):
            errors.append(f"{prefix}.name is not a stable lowercase key")
        if entry.name in names:
            errors.append(f"duplicate capability name at {prefix}")
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
            errors.append(f"{prefix}.policy is unsupported")
        if entry.test_guarantee not in _ALLOWED_TEST_GUARANTEES:
            errors.append(f"{prefix}.test_guarantee is unsupported")
        if entry.supported_python != SUPPORTED_PYTHON:
            errors.append(f"{prefix}.supported_python is unsupported")
        if not entry.documentation:
            errors.append(f"{prefix}.documentation is empty")
        if not entry.tests:
            errors.append(f"{prefix}.tests is empty")
        for requirement_index, requirement in enumerate(entry.optional_dependencies):
            if not _valid_requirement(requirement):
                errors.append(
                    f"{prefix}.optional_dependencies[{requirement_index}] is invalid"
                )

    entry_names = tuple(
        entry.name for entry in entries if type(entry) is IntegrationCapability
    )
    if entry_names != tuple(sorted(entry_names)):
        errors.append("capabilities must be sorted by name")

    root = _resolve_repository_root(repository_root)
    if root is not None:
        project_extras = _project_optional_dependencies(root / "pyproject.toml")
        for index, entry in enumerate(entries):
            if type(entry) is not IntegrationCapability:
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
                    errors.append(f"{prefix}.extra is not declared in pyproject.toml")
                for requirement_index, requirement in enumerate(
                    entry.optional_dependencies
                ):
                    dependency_name = _requirement_name(requirement)
                    if (
                        dependency_name
                        and _normalize_distribution_name(dependency_name)
                        not in declared
                    ):
                        errors.append(
                            f"{prefix}.optional_dependencies[{requirement_index}] "
                            "is not declared by its OpenMed extra"
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
        errors.append(f"{prefix}.module does not resolve locally")

    for field_name, values, expected_root, suffix in (
        ("documentation", entry.documentation, "docs", ".md"),
        ("tests", entry.tests, "tests", ".py"),
    ):
        for value_index, value in enumerate(values):
            relative = _safe_relative_path(value)
            if relative is None or not relative.startswith(f"{expected_root}/"):
                errors.append(
                    f"{prefix}.{field_name}[{value_index}] has an unsafe path"
                )
                continue
            path_without_anchor = relative.split("#", 1)[0]
            candidate = root / path_without_anchor
            if not candidate.is_file() or not path_without_anchor.endswith(suffix):
                errors.append(
                    f"{prefix}.{field_name}[{value_index}] does not resolve locally"
                )


def _resolve_repository_root(repository_root: str | Path | None) -> Path | None:
    if repository_root is not None:
        if type(repository_root) is str:
            if len(repository_root) > _MAX_TEXT_LENGTH:
                raise ValueError("repository_root exceeds the safe length limit")
        elif not isinstance(repository_root, Path):
            raise TypeError("repository_root must be a string or Path")
        try:
            return Path(repository_root).resolve()
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise ValueError("repository_root could not be resolved") from None
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

    for entry_index, entry in enumerate(entries):
        if f"`{entry.name}`" not in markdown:
            errors.append(
                f"{MATRIX_DOCUMENTATION_PATH} omits capability[{entry_index}]"
            )
        for requirement_index, requirement in enumerate(entry.optional_dependencies):
            if f"`{requirement}`" not in markdown:
                errors.append(
                    f"{MATRIX_DOCUMENTATION_PATH} omits capability[{entry_index}] "
                    f"requirement[{requirement_index}]"
                )
        for test_index, test_path in enumerate(entry.tests):
            if f"`{test_path}`" not in markdown:
                errors.append(
                    f"{MATRIX_DOCUMENTATION_PATH} omits capability[{entry_index}] "
                    f"test[{test_index}]"
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
            errors.append(f"{MATRIX_DOCUMENTATION_PATH} has an unsafe link")
            continue
        if not (root / relative.split("#", 1)[0]).is_file():
            errors.append(f"{MATRIX_DOCUMENTATION_PATH} has an unresolved link")


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
    if type(requirement) is not str or len(requirement) > _MAX_TEXT_LENGTH:
        return False
    value = requirement.strip()
    if not value or _requirement_name(value) == "":
        return False
    return not any(character in value for character in "/\\\n\r")


def _requirement_name(requirement: str) -> str:
    if type(requirement) is not str or len(requirement) > _MAX_TEXT_LENGTH:
        return ""
    match = _REQUIREMENT_NAME.match(requirement.strip())
    return match.group(0) if match is not None else ""


def _normalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _normalize_capability_name(name: str) -> str:
    if type(name) is not str:
        raise TypeError("capability name must be a string")
    if len(name) > _MAX_NAME_INPUT_LENGTH:
        raise ValueError("capability name exceeds the safe length limit")
    return name.strip().lower().replace("-", "_")


def _safe_relative_path(value: str) -> str | None:
    if type(value) is not str or len(value) > _MAX_TEXT_LENGTH:
        return None
    if "\x00" in value or any(ord(character) < 32 for character in value):
        return None
    raw = value.replace("\\", "/")
    path = PurePosixPath(raw.split("#", 1)[0])
    if path.is_absolute() or ".." in path.parts:
        return None
    normalized = posixpath.normpath(raw)
    return None if normalized in {".", ""} else normalized


def _markdown_cell(value: str) -> str:
    _validate_bounded_text(value, "Markdown cell")
    return value.replace("|", "\\|").replace("\n", " ").strip()


def _markdown_doc_link(path: str) -> str:
    _validate_bounded_text(path, "documentation path")
    raw_path, _, anchor = path.partition("#")
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
    "MAX_CAPABILITIES",
    "capabilities",
    "capability",
    "validate_capability_matrix",
    "validate_matrix",
]
