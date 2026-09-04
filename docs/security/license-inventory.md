# Dependency License Inventory

This is OpenMed's checked-in review inventory for the dependency declarations
in [`pyproject.toml`](../../pyproject.toml). It covers the base dependency set
and every optional integration except the `dev` extra. Version constraints stay
in `pyproject.toml`; this file records the locally reviewed SPDX-like license
expression and the extras that use each distribution.

The inventory gate is intentionally offline. Run it from the repository root:

```bash
python scripts/licenses/inventory.py
```

The gate reads this table and `pyproject.toml`, then fails closed if a declared
distribution is missing here or has an unknown or restricted license. Add a
reviewed row before adding a dependency. Do not paste package metadata, URLs,
credentials, or application data into this inventory.

| Dependency | Scope | License expression |
| --- | --- | --- |
| `accelerate` | `hf` | `Apache-2.0` |
| `adlfs` | `cloud` | `BSD-3-Clause` |
| `apache-beam` | `beam` | `Apache-2.0` |
| `auto-gptq` | `gptq` | `MIT` |
| `autoawq` | `awq` | `MIT` |
| `click` | `spacy` | `BSD-3-Clause` |
| `confluent-kafka` | `kafka` | `Apache-2.0` |
| `coremltools` | `coreml` | `BSD-3-Clause` |
| `cryptography` | `integrity` | `Apache-2.0 OR BSD-3-Clause` |
| `dask` | `dask` | `BSD-3-Clause` |
| `duckdb` | `duckdb` | `MIT` |
| `easyocr` | `multimodal` | `Apache-2.0` |
| `faker` | `default` | `MIT` |
| `fastapi` | `service` | `MIT` |
| `fsspec` | `cloud` | `BSD-3-Clause` |
| `gcsfs` | `cloud` | `BSD-3-Clause` |
| `gliner` | `gliner` | `Apache-2.0` |
| `griffe` | `docs` | `ISC` |
| `grpcio` | `service` | `Apache-2.0` |
| `hanlp` | `zh-hanlp` | `Apache-2.0` |
| `haystack-ai` | `haystack` | `Apache-2.0` |
| `hnswlib` | `grounding` | `Apache-2.0` |
| `httpx` | `openmrs, service` | `BSD-3-Clause` |
| `huggingface-hub` | `coreml, hf, mlx, onnx-runtime` | `Apache-2.0` |
| `indic-nlp-library` | `indic` | `MIT` |
| `jieba` | `default, zh` | `MIT` |
| `langchain-core` | `agents, langchain` | `MIT` |
| `langgraph` | `agents, langgraph` | `MIT` |
| `llama-index-core` | `agents, llamaindex` | `MIT` |
| `markdown-it-py` | `multimodal` | `MIT` |
| `mcp` | `mcp` | `MIT` |
| `mkdocs` | `docs` | `BSD-2-Clause` |
| `mkdocs-git-revision-date-localized-plugin` | `docs` | `MIT` |
| `mkdocs-llmstxt` | `docs` | `ISC` |
| `mkdocs-material` | `docs` | `MIT` |
| `mkdocs-minify-plugin` | `docs` | `MIT` |
| `mkdocs-static-i18n` | `docs` | `MIT` |
| `mkdocstrings` | `docs` | `ISC` |
| `mlx` | `mlx` | `MIT` |
| `mlx-lm` | `mlx` | `MIT` |
| `nncf` | `openvino` | `Apache-2.0` |
| `numpy` | `duckdb, grounding, multimodal, onnx-runtime` | `BSD-3-Clause` |
| `onnx` | `multimodal, onnx` | `Apache-2.0` |
| `onnxruntime` | `onnx, onnx-runtime, openvino` | `MIT` |
| `onnxscript` | `onnx` | `MIT` |
| `opencc` | `zh` | `Apache-2.0` |
| `opentelemetry-api` | `service` | `Apache-2.0` |
| `opentelemetry-exporter-otlp-proto-http` | `service` | `Apache-2.0` |
| `opentelemetry-sdk` | `service` | `Apache-2.0` |
| `openvino` | `openvino` | `Apache-2.0` |
| `paddleocr` | `ocr-paddle` | `Apache-2.0` |
| `pandas` | `pandas, spark` | `BSD-3-Clause` |
| `pdfplumber` | `multimodal` | `MIT` |
| `philter-ucsf` | `philter` | `BSD-3-Clause` |
| `piexif` | `multimodal` | `MIT` |
| `pikepdf` | `multimodal` | `MPL-2.0` |
| `pillow` | `multimodal` | `HPND` |
| `pkuseg` | `zh-pkuseg` | `MIT` |
| `polars` | `polars` | `MIT` |
| `prefect` | `prefect` | `Apache-2.0` |
| `presidio-analyzer` | `presidio` | `MIT` |
| `protobuf` | `service` | `BSD-3-Clause` |
| `pyarrow` | `columnar, spark` | `Apache-2.0` |
| `pycld2` | `lid` | `Apache-2.0` |
| `pydeid` | `pydeid` | `MIT` |
| `pydicom` | `multimodal` | `MIT` |
| `pymdown-extensions` | `docs` | `MIT` |
| `pypinyin` | `zh` | `MIT` |
| `pysbd` | `default` | `MIT` |
| `pyspark` | `spark` | `Apache-2.0` |
| `pytesseract` | `multimodal` | `Apache-2.0` |
| `python-doctr` | `multimodal` | `Apache-2.0` |
| `python-docx` | `multimodal` | `MIT` |
| `pyyaml` | `default` | `MIT` |
| `quickumls` | `quickumls` | `MIT` |
| `rapidfuzz` | `grounding` | `MIT` |
| `ray` | `ray` | `Apache-2.0` |
| `rich` | `cli` | `MIT` |
| `s3fs` | `cloud` | `BSD-3-Clause` |
| `safetensors` | `mlx` | `Apache-2.0` |
| `scispacy` | `scispacy` | `Apache-2.0` |
| `scrubadub` | `scrubadub` | `Apache-2.0` |
| `spacy` | `spacy` | `MIT` |
| `tiktoken` | `mlx` | `MIT` |
| `tokenizers` | `hf, mlx, onnx-runtime` | `Apache-2.0` |
| `torch` | `coreml, gliner, onnx` | `BSD-3-Clause` |
| `transformers` | `awq, coreml, hf, mlx, onnx, openvino` | `Apache-2.0` |
| `typer` | `cli` | `MIT` |
| `uvicorn` | `service` | `BSD-3-Clause` |
| `yasbd-lib` | `yasbd` | `MPL-2.0` |
