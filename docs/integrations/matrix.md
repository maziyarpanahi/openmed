# Integration Capability Matrix

This page is the human-readable view of the machine-readable
`openmed.interop.capabilities.INTEGRATION_CAPABILITY_MATRIX`. Each row records
the integration module, optional requirements, network or resource policy, and
the offline test files that provide its current support evidence.

All rows support Python `>=3.10`. A `core` requirement means the surface uses
OpenMed's base dependencies and does not require an optional extra. Optional
requirements are copied from the matching `pyproject.toml` extra; they are
install-time metadata, not runtime imports performed by the matrix.

Custom matrices are bounded to 256 capability records and 64 values per
dependency, documentation, or test-path sequence. Public JSON and Markdown
renderers revalidate frozen records before emitting them, so tampered or
unbounded metadata fails closed without retaining caller-supplied values in
errors.

## Policy meanings

- `local-only`: the adapter performs its transformation locally after its
  dependencies and any caller-provided artifacts are present.
- `configured-network`: the surface can call a user-configured endpoint, but
  importing or inspecting the capability matrix never makes that call.
- `user-supplied-resource`: the caller is responsible for supplying any
  licensed resource; OpenMed does not download or bundle it.

Every listed test is an offline unit or contract test. Test fixtures are
synthetic, and the matrix contains only module names, paths, version
constraints, and PHI-free descriptions.

## Matrix

| Capability | Surface | Optional requirements | Policy | Offline evidence | Test files | Documentation |
|---|---|---|---|---|---|---|
| `airflow` | Apache Airflow | `apache-airflow>=3.2.2,<4` | `local-only` | `offline-unit` (1 test file) | `tests/unit/interop/test_airflow.py` | [Airflow Redaction Operator](airflow.md) |
| `arrow_flight` | Arrow Flight | `pyarrow>=16` | `configured-network` | `offline-unit` (1 test file) | `tests/unit/integrations/test_arrow_flight.py` | [Arrow Flight De-identification](arrow-flight.md) |
| `beam` | Apache Beam | `apache-beam>=2.73,<3` | `local-only` | `offline-unit` (1 test file) | `tests/unit/interop/test_beam_transform.py` | [Feature Map](../feature-map.md) |
| `cda` | CDA/C-CDA | core | `local-only` | `offline-unit` (1 test file) | `tests/unit/interop/test_cda.py` | [FHIR Interop Helpers](../fhir-interop.md) |
| `columnar` | Columnar redaction | `pyarrow>=16` | `local-only` | `offline-unit` (1 test file) | `tests/unit/integrations/test_columnar_redactor.py` | [Columnar Redactor](columnar-redactor.md) |
| `dagster` | Dagster | `dagster>=1.8,<2` | `local-only` | `offline-unit` (1 test file) | `tests/unit/integrations/test_dagster_assets.py` | [Feature Map](../feature-map.md) |
| `dask` | Dask DataFrame | `dask[dataframe]>=2024.8` | `local-only` | `offline-unit` (1 test file) | `tests/unit/integrations/test_dask_accessor.py` | [Dask DataFrame De-identification](dask.md) |
| `dataflow` | Apache Beam Dataflow | `apache-beam>=2.73,<3` | `local-only` | `offline-unit` (1 test file) | `tests/unit/integrations/test_dataflow_processor.py` | [Feature Map](../feature-map.md) |
| `dataflow_tool` | Embedded dataflow tools | core | `local-only` | `offline-unit` (1 test file) | `tests/unit/integrations/test_dataflow_tool_processor.py` | [Feature Map](../feature-map.md) |
| `distributed_sql` | Distributed SQL UDF | core | `local-only` | `offline-unit` (1 test file) | `tests/unit/integrations/test_distributed_sql_udf.py` | [Distributed SQL De-identification UDF](distributed-sql-udf.md) |
| `duckdb` | DuckDB UDFs | `duckdb>=1.0,<2`, `numpy>=1.26` | `local-only` | `offline-unit` (1 test file) | `tests/unit/interop/test_duckdb_udf.py` | [DuckDB De-identification UDFs](../duckdb-deidentification.md) |
| `executable_udf` | Executable UDF | core | `local-only` | `offline-unit` (1 test file) | `tests/unit/integrations/test_executable_udf.py` | [Feature Map](../feature-map.md) |
| `fhir` | FHIR operations and bulk NDJSON | core | `local-only` | `offline-unit` (2 test files) | `tests/unit/interop/test_fhir_bulk_ndjson.py`, `tests/unit/interop/test_fhir_deidentify_operation.py` | [FHIR Interop Helpers](../fhir-interop.md) |
| `gliner_biomed` | GLiNER-BioMed | `gliner[tokenizers]>=0.2.0`, `torch>=2.0` | `local-only` | `offline-unit` (1 test file) | `tests/unit/interop/test_gliner_biomed_adapter.py` | [Zero-shot NER](../zero-shot-ner.md) |
| `haystack` | Haystack document redaction | `haystack-ai>=2,<3` | `local-only` | `offline-unit` (1 test file) | `tests/unit/interop/test_haystack_redaction.py` | [Haystack Redaction Component](../integrations-haystack.md) |
| `hl7v2` | HL7 v2 | core | `local-only` | `offline-unit` (1 test file) | `tests/unit/interop/test_hl7v2.py` | [HL7 v2 De-identification](../hl7v2-deidentification.md) |
| `indic` | Indic language helpers | `indic-nlp-library>=0.92` | `local-only` | `offline-unit` (1 test file) | `tests/unit/interop/test_language_adapters.py` | [Feature Map](../feature-map.md) |
| `lakehouse` | Lakehouse table redaction | `pyarrow>=16` | `local-only` | `offline-unit` (1 test file) | `tests/unit/integrations/test_lakehouse_redact.py` | [Lakehouse Table Redaction](lakehouse-redaction.md) |
| `langchain` | LangChain | `langchain-core>=0.2,<2` | `local-only` | `offline-unit` (1 test file) | `tests/unit/interop/test_langchain_redaction.py` | [LangChain Redaction Node](langchain.md) |
| `langgraph` | LangGraph | `langgraph>=0.2,<2` | `local-only` | `offline-unit` (1 test file) | `tests/unit/interop/test_graph_orchestration.py` | [Feature Map](../feature-map.md) |
| `llamaindex` | LlamaIndex | `llama-index-core>=0.10,<1` | `local-only` | `offline-unit` (1 test file) | `tests/unit/interop/test_llamaindex_redaction.py` | [LlamaIndex Redaction Transform](llamaindex.md) |
| `log_redactor` | Structured log redaction | core | `local-only` | `offline-unit` (1 test file) | `tests/unit/integrations/test_log_redactor.py` | [Feature Map](../feature-map.md) |
| `omop` | OMOP/CDM | core | `local-only` | `offline-unit` (1 test file) | `tests/unit/interop/test_omop_cdm_loader.py` | [Feature Map](../feature-map.md) |
| `openmrs` | OpenMRS REST and FHIR2 | `httpx>=0.27` | `configured-network` | `offline-unit` (1 test file) | `tests/unit/interop/test_openmrs_adapter.py` | [OpenMRS Adapter](../openmrs-adapter.md) |
| `pandas` | pandas DataFrame | `pandas>=2.0` | `local-only` | `offline-unit` (1 test file) | `tests/unit/interop/test_dataframe_accessor.py` | [Feature Map](../feature-map.md) |
| `pandas_on_spark` | Pandas API on Spark | `pyspark>=3.5,<4`, `pandas>=2.0`, `pyarrow>=16` | `local-only` | `offline-unit` (1 test file) | `tests/unit/integrations/test_pandas_on_spark.py` | [Pandas-on-Spark De-identification](pandas-on-spark.md) |
| `philter` | PHILTER | `philter-ucsf>=1.0.3,<2` | `local-only` | `offline-unit` (1 test file) | `tests/unit/interop/test_philter_adapter.py` | [Feature Map](../feature-map.md) |
| `polars` | Polars DataFrame | `polars>=0.20` | `local-only` | `offline-unit` (1 test file) | `tests/unit/interop/test_dataframe_accessor.py` | [Feature Map](../feature-map.md) |
| `postgres` | PostgreSQL transaction adapter | core | `local-only` | `offline-unit` (1 test file) | `tests/unit/interop/test_postgres.py` | [PostgreSQL Redaction Adapter](postgres.md) |
| `postgres_plpython` | PostgreSQL PL/Python | core | `local-only` | `offline-unit` (1 test file) | `tests/unit/integrations/test_postgres_plpython.py` | [PostgreSQL Redaction Adapter](postgres.md) |
| `prefect` | Prefect | `prefect>=3.7,<4` | `local-only` | `offline-unit` (1 test file) | `tests/unit/interop/test_prefect_tasks.py` | [Prefect Batch De-identification](../prefect-integration.md) |
| `presidio` | Microsoft Presidio | `presidio-analyzer>=2.2.354,<3` | `local-only` | `offline-unit` (1 test file) | `tests/unit/interop/test_presidio_adapter.py` | [Feature Map](../feature-map.md) |
| `pydeid` | pyDeid | `pyDeid>=0.0.1` | `local-only` | `offline-unit` (1 test file) | `tests/unit/interop/test_pydeid_adapter.py` | [Feature Map](../feature-map.md) |
| `quickumls` | QuickUMLS | `quickumls>=1.4,<2` | `user-supplied-resource` | `offline-unit` (1 test file) | `tests/unit/interop/test_scispacy_linker_adapter.py` | [Feature Map](../feature-map.md) |
| `ray` | Ray Data | `ray[data]>=2.30,<3` | `local-only` | `offline-unit` (1 test file) | `tests/unit/interop/test_ray_data.py` | [Feature Map](../feature-map.md) |
| `ray_map_batches` | Ray Data map_batches | `ray[data]>=2.30,<3` | `local-only` | `offline-unit` (1 test file) | `tests/unit/integrations/test_ray_map_batches.py` | [Ray Data map-batches](ray-map-batches.md) |
| `remote_function` | Warehouse remote function | `fastapi>=0.110` | `configured-network` | `offline-unit` (1 test file) | `tests/unit/integrations/test_remote_function.py` | [Warehouse Remote-Function Handler](warehouse-remote-function.md) |
| `scispacy` | scispaCy UMLS linker | `scispacy>=0.5.4,<1` | `user-supplied-resource` | `offline-unit` (1 test file) | `tests/unit/interop/test_scispacy_linker_adapter.py` | [Feature Map](../feature-map.md) |
| `scrubadub` | scrubadub | `scrubadub>=2.0,<3` | `local-only` | `offline-unit` (1 test file) | `tests/unit/interop/test_scrubadub_adapter.py` | [Feature Map](../feature-map.md) |
| `search_ingest` | Search ingest sidecar | `fastapi>=0.110` | `configured-network` | `offline-unit` (1 test file) | `tests/unit/integrations/test_search_ingest_processor.py` | [Feature Map](../feature-map.md) |
| `search_pipeline` | Search pipeline redaction | `haystack-ai>=2,<3` | `local-only` | `offline-unit` (1 test file) | `tests/unit/interop/test_search_pipeline.py` | [Haystack Redaction Component](../integrations-haystack.md) |
| `spacy` | spaCy | `click>=8.0`, `spacy>=3.8.9` | `local-only` | `offline-unit` (1 test file) | `tests/unit/interop/test_spacy_component.py` | [spaCy Pipeline Component](../spacy-component.md) |
| `spark` | PySpark | `pyspark>=3.5,<4`, `pandas>=2.0`, `pyarrow>=16` | `local-only` | `offline-unit` (2 test files) | `tests/unit/interop/test_spark_udf.py`, `tests/unit/integrations/test_spark_streaming.py` | [PySpark De-identification UDFs](../spark-deidentification.md) |
| `sqlalchemy` | SQLAlchemy | `sqlalchemy>=2.0,<3` | `local-only` | `offline-unit` (1 test file) | `tests/unit/integrations/test_sqlalchemy_redact.py` | [SQLAlchemy Write-time Redaction](sqlalchemy.md) |
| `stream_processor` | Stream processor | core | `local-only` | `offline-unit` (1 test file) | `tests/unit/integrations/test_stream_processor.py` | [Feature Map](../feature-map.md) |
| `zh` | Chinese language helpers | `jieba>=0.42`, `opencc>=1.4.1,<2`, `pypinyin>=0.51` | `local-only` | `offline-unit` (1 test file) | `tests/unit/interop/test_language_adapters.py` | [Chinese Segmentation Operations](../chinese-segmentation-operations.md) |

The machine-readable record can be inspected without installing any optional
dependency:

```python
from openmed.interop.capabilities import CAPABILITY_MATRIX

print(CAPABILITY_MATRIX.to_json())
```

The coherence test validates every local documentation and test path and
checks that each optional requirement name is declared by its matching
`pyproject.toml` extra:
`tests/unit/interop/test_capabilities.py`.
