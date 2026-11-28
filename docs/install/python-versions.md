# Optional extras on Python 3.14

This page records which optional extras of the `openmed` package resolve on
Python 3.14. It was produced on 2026-11-28 from the repository state at commit
`5b79cae`.

## Method

For each extra defined under `[project.optional-dependencies]` in
`pyproject.toml`, the following command was run from the repository root in a
fresh Python 3.14.10 virtual environment on Linux x86_64 with uv 0.13.8:

```console
uv venv --python 3.14 /tmp/v314
uv pip install --dry-run --python /tmp/v314/bin/python ".[EXTRA]"
```

The command was run against the local checkout (`.[EXTRA]`) so that it reflects
the `pyproject.toml` in this repository. A published release can differ.

## What the results mean

- **Resolves** means the dry run exited with status 0: uv found a set of
  packages that satisfy the extra on Python 3.14. A dry run does not download
  or build every package, and it does not import anything. It does not prove
  that an install will succeed for a package that has to be built from source.
- **Not tested** means the extra was not run on that platform. macOS was not
  tested, so Apple-specific extras such as `mlx` and `coreml` have no macOS
  result here.
- No extra failed to resolve in this run, so no row links an upstream issue.

## Results

| Extra | Command target | Linux x86_64 | macOS |
| --- | --- | --- | --- |
| `dev` | `openmed[dev]` | Resolves | Not tested |
| `cli` | `openmed[cli]` | Resolves | Not tested |
| `integrity` | `openmed[integrity]` | Resolves | Not tested |
| `mcp` | `openmed[mcp]` | Resolves | Not tested |
| `lid` | `openmed[lid]` | Resolves | Not tested |
| `openmrs` | `openmed[openmrs]` | Resolves | Not tested |
| `fhir` | `openmed[fhir]` | Resolves | Not tested |
| `presidio` | `openmed[presidio]` | Resolves | Not tested |
| `medspacy` | `openmed[medspacy]` | Resolves | Not tested |
| `philter` | `openmed[philter]` | Resolves | Not tested |
| `pydeid` | `openmed[pydeid]` | Resolves | Not tested |
| `scrubadub` | `openmed[scrubadub]` | Resolves | Not tested |
| `pandas` | `openmed[pandas]` | Resolves | Not tested |
| `polars` | `openmed[polars]` | Resolves | Not tested |
| `columnar` | `openmed[columnar]` | Resolves | Not tested |
| `dask` | `openmed[dask]` | Resolves | Not tested |
| `dagster` | `openmed[dagster]` | Resolves | Not tested |
| `sqlalchemy` | `openmed[sqlalchemy]` | Resolves | Not tested |
| `spacy` | `openmed[spacy]` | Resolves | Not tested |
| `zh` | `openmed[zh]` | Resolves | Not tested |
| `indic` | `openmed[indic]` | Resolves | Not tested |
| `scispacy` | `openmed[scispacy]` | Resolves | Not tested |
| `quickumls` | `openmed[quickumls]` | Resolves | Not tested |
| `interop-gpl` | `openmed[interop-gpl]` | Resolves | Not tested |
| `grounding-validate` | `openmed[grounding-validate]` | Resolves | Not tested |
| `email-msg-gpl` | `openmed[email-msg-gpl]` | Resolves | Not tested |
| `multimodal` | `openmed[multimodal]` | Resolves | Not tested |
| `notebook` | `openmed[notebook]` | Resolves | Not tested |
| `ocr-paddle` | `openmed[ocr-paddle]` | Resolves | Not tested |
| `zh-pkuseg` | `openmed[zh-pkuseg]` | Resolves | Not tested |
| `zh-hanlp` | `openmed[zh-hanlp]` | Resolves | Not tested |
| `awq` | `openmed[awq]` | Resolves | Not tested |
| `gptq` | `openmed[gptq]` | Resolves | Not tested |
| `grounding` | `openmed[grounding]` | Resolves | Not tested |
| `langchain` | `openmed[langchain]` | Resolves | Not tested |
| `langgraph` | `openmed[langgraph]` | Resolves | Not tested |
| `haystack` | `openmed[haystack]` | Resolves | Not tested |
| `llamaindex` | `openmed[llamaindex]` | Resolves | Not tested |
| `agents` | `openmed[agents]` | Resolves | Not tested |
| `beam` | `openmed[beam]` | Resolves | Not tested |
| `airflow` | `openmed[airflow]` | Resolves | Not tested |
| `duckdb` | `openmed[duckdb]` | Resolves | Not tested |
| `kafka` | `openmed[kafka]` | Resolves | Not tested |
| `prefect` | `openmed[prefect]` | Resolves | Not tested |
| `cloud` | `openmed[cloud]` | Resolves | Not tested |
| `journey` | `openmed[journey]` | Resolves | Not tested |
| `ray` | `openmed[ray]` | Resolves | Not tested |
| `spark` | `openmed[spark]` | Resolves | Not tested |
| `snowflake` | `openmed[snowflake]` | Resolves | Not tested |
| `hf` | `openmed[hf]` | Resolves | Not tested |
| `otel` | `openmed[otel]` | Resolves | Not tested |
| `service` | `openmed[service]` | Resolves | Not tested |
| `triton` | `openmed[triton]` | Resolves | Not tested |
| `operator` | `openmed[operator]` | Resolves | Not tested |
| `gliner` | `openmed[gliner]` | Resolves | Not tested |
| `gliclass` | `openmed[gliclass]` | Resolves | Not tested |
| `gliformer` | `openmed[gliformer]` | Resolves | Not tested |
| `gliner2` | `openmed[gliner2]` | Resolves | Not tested |
| `edge-asr` | `openmed[edge-asr]` | Resolves | Not tested |
| `edge-vlm` | `openmed[edge-vlm]` | Resolves | Not tested |
| `mlx` | `openmed[mlx]` | Resolves | Not tested |
| `coreml` | `openmed[coreml]` | Resolves | Not tested |
| `onnx` | `openmed[onnx]` | Resolves | Not tested |
| `edge-diarization` | `openmed[edge-diarization]` | Resolves | Not tested |
| `onnx-runtime` | `openmed[onnx-runtime]` | Resolves | Not tested |
| `edge-sbc` | `openmed[edge-sbc]` | Resolves | Not tested |
| `openvino` | `openmed[openvino]` | Resolves | Not tested |
| `docs` | `openmed[docs]` | Resolves | Not tested |
| `yasbd` | `openmed[yasbd]` | Resolves | Not tested |

## Re-running

Re-run the command above for any extra after changing `pyproject.toml`. The
test `tests/unit/install/test_python_versions_doc.py` checks that every extra in
`pyproject.toml` appears on this page.
