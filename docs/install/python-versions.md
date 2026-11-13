# Python 3.14 extras compatibility

OpenMed's CI tests Python 3.11 and 3.12 and installs only a few optional
extras. This page records which extras in `pyproject.toml` install on
Python 3.14, so you can check before you pick an interpreter.

Checked on **2026-11-14** against `master` (OpenMed 3.3.0) with uv 0.13.8
and CPython 3.14.10. Dependency releases change quickly, so re-run the
check below before relying on a row.

## How it was checked

Each extra was resolved from the repository root with a dry run, once for the
host Linux platform and once for macOS on Apple silicon:

```bash
# Linux: Ubuntu 26.04, x86_64, glibc 2.43
uv pip install --dry-run --python 3.14 ".[<extra>]"

# macOS 14+ on Apple silicon
MACOSX_DEPLOYMENT_TARGET=14.0 uv pip install --dry-run --python 3.14 \
  --python-platform aarch64-apple-darwin ".[<extra>]"
```

A dry run only resolves dependencies, and uv accepts a source distribution
(sdist) when no compatible wheel exists. To find the dependencies that have no
Python 3.14 wheel, each extra was also resolved with `--only-binary :all:`.
Every sdist that this check reported was then built on Linux with
`uv pip install --no-binary <package>` (GCC 15). The macOS builds were not run.

## Status values

- **install**: resolves, and every dependency either has a Python 3.14 wheel or
  builds from source on Linux. The notes name any native builds.
- **fail**: a required dependency has no Python 3.14 wheel and its sdist does
  not build.
- **platform-limited**: the extra's dependencies are limited to another
  platform by environment markers, so installing it adds nothing here.

Every extra needs `jieba`, a core dependency published only as a pure-Python
sdist. It builds without a compiler. Rows with no note need no other
source builds.

## Results

| Extra | Linux x86_64 | macOS arm64 | Notes |
| --- | --- | --- | --- |
| `dev` | install | install |  |
| `cli` | install | install |  |
| `integrity` | install | install |  |
| `mcp` | install | install |  |
| `lid` | install | install | Builds `pycld2` from source: needs a C++ compiler. |
| `openmrs` | install | install |  |
| `fhir` | install | install |  |
| `presidio` | install | install |  |
| `medspacy` | install | install | Builds `pyrush`, `pysimstring`, `medspacy-unqlite` and `quicksectx` from source: needs a C/C++ compiler. |
| `philter` | install | install |  |
| `pydeid` | install | install |  |
| `scrubadub` | install | install |  |
| `pandas` | install | install |  |
| `polars` | install | install |  |
| `columnar` | install | install |  |
| `dask` | install | install |  |
| `dagster` | install | install |  |
| `sqlalchemy` | install | install |  |
| `spacy` | install | install |  |
| `zh` | install | install |  |
| `indic` | install | install |  |
| `scispacy` | install | install | scispacy 0.6 caps NumPy below 2, so NumPy 1.26.4 and `nmslib-metabrainz` build from source: needs a C/C++ compiler. |
| `quickumls` | fail | fail | no wheel published for `leveldb` 0.201, and its sdist does not compile on 3.14 (uses `PyUnicode_AS_UNICODE`, removed from the C API). |
| `interop-gpl` | install | install |  |
| `grounding-validate` | install | install |  |
| `email-msg-gpl` | install | install |  |
| `multimodal` | install | install |  |
| `notebook` | install | install |  |
| `ocr-paddle` | install | install |  |
| `zh-pkuseg` | fail | fail | no wheel published for `pkuseg` 0.0.25, and its sdist imports NumPy at build time without declaring it, so isolated builds fail. |
| `zh-hanlp` | install | install |  |
| `awq` | install | platform-limited | Dependencies are Linux-only; on macOS the extra installs nothing. |
| `gptq` | fail | fail | no wheel published for `auto-gptq` 0.7.1, and its sdist needs PyTorch installed at build time, so isolated builds fail. |
| `grounding` | install | install | Builds `hnswlib` from source: needs a C++ compiler. |
| `langchain` | install | install |  |
| `langgraph` | install | install |  |
| `haystack` | install | install |  |
| `llamaindex` | install | install |  |
| `agents` | install | install |  |
| `beam` | install | install |  |
| `airflow` | install | install |  |
| `duckdb` | install | install |  |
| `kafka` | install | install |  |
| `prefect` | install | install |  |
| `cloud` | install | install |  |
| `journey` | install | install |  |
| `ray` | install | install |  |
| `spark` | install | install |  |
| `snowflake` | install | install |  |
| `hf` | install | install |  |
| `otel` | install | install |  |
| `service` | install | install |  |
| `triton` | install | install |  |
| `operator` | install | install |  |
| `gliner` | install | install |  |
| `gliclass` | install | install |  |
| `gliformer` | install | install |  |
| `gliner2` | install | install |  |
| `edge-asr` | platform-limited | install | Dependencies are macOS arm64-only; on Linux the extra installs nothing. |
| `edge-vlm` | platform-limited | install | Dependencies are macOS arm64-only; on Linux the extra installs nothing. |
| `mlx` | install | install |  |
| `coreml` | install | install |  |
| `onnx` | install | install |  |
| `edge-diarization` | install | install |  |
| `onnx-runtime` | install | install |  |
| `edge-sbc` | install | install |  |
| `openvino` | install | install |  |
| `docs` | install | install |  |
| `yasbd` | install | install |  |

Rows marked as needing a compiler were built on Linux only. On macOS, install
the Xcode Command Line Tools before trying them.
