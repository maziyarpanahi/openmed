# Troubleshooting & Common Errors

This page maps the concrete failures you are most likely to hit — **symptom → cause → fix** — with the
exact error strings OpenMed raises and the commands that resolve them. If your problem is conceptual
rather than an error, start with the [FAQ](faq.md); if you are setting up for the first time, see the
[Quick Start](getting-started.md) and [Configuration & Validation](configuration.md).

!!! tip "Run the built-in doctor first"
    Before digging in, run the offline environment check. It reports your Python version and architecture,
    which selected runtime extras are importable, whether an `HF_TOKEN` is present, and whether offline mode
    is active — with a remediation `Hint:` on anything that is not ready.

    ```bash
    openmed doctor          # human-readable check list
    openmed doctor --json   # machine-readable; non-zero exit on any FAIL
    ```

    Sample output:

    ```text
    PASS python_version: 3.11.9
    PASS python_arch: arm64
    WARN hf: transformers not installed
          Hint: Install with: pip install transformers
    PASS openmed_offline: 0
    ```

---

## Install / Extras

OpenMed ships a small permissive core and keeps heavy or platform-specific stacks behind
[optional extras declared in `pyproject.toml`](https://github.com/maziyarpanahi/openmed/blob/master/pyproject.toml).
If you call a feature whose extra is not installed, you get an `ImportError`/`ModuleNotFoundError` or a
`MissingDependencyError` (a subclass of `ImportError`) whose message names the fix.

### `ModuleNotFoundError: No module named 'transformers'` — the model runtime is missing

**Symptom.** Loading a model or calling `analyze_text`, `extract_pii`, or `deidentify` on a Hub model fails
with a missing `transformers` import, or you see:

```text
ImportError: HuggingFace transformers is required. Install with: pip install transformers
```

**Cause.** The core install does not pull in the Hugging Face runtime. The transformers/tokenizers/accelerate
stack lives in the `hf` extra.

**Fix.**

```bash
pip install "openmed[hf]"
```

### `RuntimeError: No inference backend available` — neither the HF nor MLX backend is installed

**Symptom.** Backend auto-detection cannot find a usable runtime:

```text
RuntimeError: No inference backend available. Install at least one: pip install openmed[hf] or pip install openmed[mlx]
```

Related backend errors:

- `ValueError: Unknown backend 'xyz'. Available: ['hf', 'mlx']` — you passed a `backend=` value that is not
  `"hf"` or `"mlx"`.
- `RuntimeError: Backend 'mlx' is not available. Install its dependencies first.` — you selected a backend
  explicitly but its dependencies are not importable on this machine.

**Cause.** No inference backend is importable. `hf` is the portable default; `mlx` is the Apple-Silicon path.

**Fix.** Install at least one backend (`hf` works everywhere):

```bash
pip install "openmed[hf]"     # portable PyTorch/Transformers backend
pip install "openmed[mlx]"    # Apple Silicon MLX backend
```

### `MissingDependencyError: Optional dependency '…' is required` — a feature extra is missing

**Symptom.** A capability-specific call fails with a message shaped like:

```text
Optional dependency 'pdfplumber' is required for this operation. Install with: pip install "openmed[multimodal]".
```

**Cause.** OpenMed defers document/image intake, zero-shot NER, framework bridges, exporters, and connectors
to purpose-built extras. The message always names the exact install command.

**Fix.** Install the extra the message names. Common ones:

| Feature | Extra | Install command |
|---|---|---|
| Document / image intake + OCR (`pdfplumber`, `python-docx`, `Pillow`, DICOM, docTR/Tesseract/EasyOCR) | `multimodal` | `pip install "openmed[multimodal]"` |
| Heavier PaddleOCR backend | `ocr-paddle` | `pip install "openmed[ocr-paddle]"` |
| Zero-shot GLiNER NER | `gliner` | `pip install "openmed[gliner]"` |
| REST service | `service` | `pip install "openmed[service]"` |
| MCP server | `mcp` | `pip install "openmed[mcp]"` |
| Optional Typer/Rich CLI | `cli` | `pip install "openmed[cli]"` |
| Apple Silicon MLX runtime | `mlx` | `pip install "openmed[mlx]"` |
| Core ML packaging | `coreml` | `pip install "openmed[coreml]"` |
| ONNX Runtime | `onnx` | `pip install "openmed[onnx]"` |
| OpenVINO runtime | `openvino` | `pip install "openmed[openvino]"` |
| pandas / polars accessors | `pandas` / `polars` | `pip install "openmed[pandas]"` / `pip install "openmed[polars]"` |
| Dask / DuckDB | `dask` / `duckdb` | `pip install "openmed[dask]"` / `pip install "openmed[duckdb]"` |
| spaCy component | `spacy` | `pip install "openmed[spacy]"` |
| Presidio bridge | `presidio` | `pip install "openmed[presidio]"` |
| Grounding (rapidfuzz) | `grounding` | `pip install "openmed[grounding]"` |
| LangChain / LlamaIndex | `langchain` / `llamaindex` | `pip install "openmed[langchain]"` / `pip install "openmed[llamaindex]"` |
| Kafka / cloud object storage | `kafka` / `cloud` | `pip install "openmed[kafka]"` / `pip install "openmed[cloud]"` |
| AWQ / GPTQ export | `awq` / `gptq` | `pip install "openmed[awq]"` / `pip install "openmed[gptq]"` |

You can stack extras: `pip install "openmed[hf,service]"` or `pip install "openmed[hf,mlx,docs]"`.

### `TesseractNotFoundError` / OCR reports no engine — the system binary is missing

**Symptom.** OCR fails even though `openmed[multimodal]` is installed, with a message like:

```text
Optional dependency 'tesseract' is required for this operation. Install with: pip install "openmed[multimodal]" and install the system Tesseract binary (e.g. `brew install tesseract` or `apt-get install tesseract-ocr`).
```

**Cause.** Tesseract is a **system** binary that `pip` does not install. The Python extra provides only the
`pytesseract` wrapper.

**Fix.** Install the OS package alongside the extra:

```bash
pip install "openmed[multimodal]"
brew install tesseract              # macOS
sudo apt-get install tesseract-ocr  # Debian/Ubuntu
```

docTR and EasyOCR ship inside `openmed[multimodal]` and need no separate system binary; PaddleOCR lives in
the separate `openmed[ocr-paddle]` extra.

---

## Model Download & Offline

### `ValueError: Could not load model …` — download failed or the model was not found

**Symptom.** The first call that needs a model fails while resolving files from the Hugging Face Hub:

```text
ValueError: Could not load model OpenMed/…: <underlying Hub / network error>
```

The wrapped error is whatever `transformers`/`huggingface_hub` raised — a connection timeout, an HTTP 401 on
a gated repo, an HTTP 404 for a wrong id, or a proxy/TLS failure.

**Cause.** OpenMed downloads model artifacts on first use and caches them under `~/.cache/openmed` (override
with `OpenMedConfig(cache_dir="...")`). Failures are almost always network reachability, a proxy, or a
private/gated repo needing a token. Service container deployments also expose `OPENMED_CACHE_DIR` for their
mounted model cache.

**Fix.**

- **Behind a proxy / corporate firewall.** Export standard proxy variables before the first run:

  ```bash
  export HTTPS_PROXY=http://proxy.internal:8080
  export HTTP_PROXY=http://proxy.internal:8080
  ```

- **Using a Hub mirror.** OpenMed honors the standard Hugging Face endpoint variable; point it at your mirror:

  ```bash
  export HF_ENDPOINT=https://hf-mirror.example.com
  ```

- **Private or gated model.** Provide a token so the Hub authorizes the download:

  ```bash
  export HF_TOKEN=hf_xxx
  ```

  `openmed doctor` reports `WARN hf_token: present=False` when no token is set.

- **Wrong identifier.** Confirm the id (`org/model`) or registry alias exists — see the
  [Model Registry](model-registry.md) and `openmed models list`.

### Cold start on the first request is slow

**Symptom.** The first `analyze_text` / `extract_pii` / `deidentify` call takes noticeably longer than later
calls.

**Cause.** First use downloads model artifacts, then loads weights and the tokenizer into memory. Subsequent
calls reuse the warm, cached pipeline.

**Fix.** Warm the cache and reuse loaders rather than paying the cost per call:

- Pre-download once so later runs hit the cache: instantiate a `ModelLoader` (or run any single request) on a
  machine with network before going offline.
- Reuse one `ModelLoader`/pipeline instead of constructing a new one per document — see
  [ModelLoader & Pipelines](model-loader.md).
- In the REST service, pre-load models at startup with `OPENMED_SERVICE_PRELOAD_MODELS` and keep them resident
  with `OPENMED_SERVICE_KEEP_ALIVE` — see [REST Service](rest-service.md).

### Running offline / air-gapped, or `OfflineModeError` on inference

**Symptom.** You want no network calls after startup, or an operation raises:

```text
OfflineModeError: OPENMED_OFFLINE/local_only=True blocks outbound network access after model loading. Pre-download required model files into the configured cache, pass a local model path, or disable offline mode before remote fetches. Blocked action: socket connection.
```

**Cause.** Local-only mode is active (`OPENMED_OFFLINE=1` or `OpenMedConfig(local_only=True)`). It passes
`local_files_only=True` to Hub loaders, sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and
`HF_DATASETS_OFFLINE=1`, and blocks outbound sockets during inference. The error means an operation tried to
reach the network after the model was loaded — usually because the required files were not cached first.

**Fix.** Warm the cache **before** enabling offline mode, then turn it on:

```bash
# 1. Download once with network available (populates ~/.cache/openmed)
python -c "from openmed.core import ModelLoader; ModelLoader().load_model('disease_detection_superclinical')"

# 2. Then run strictly local
export OPENMED_OFFLINE=1
```

```python
from openmed.core import OpenMedConfig

config = OpenMedConfig(local_only=True)  # defaults to ~/.cache/openmed
```

You can also skip the Hub entirely by passing a **local directory** as `model_name`/`model_id`; OpenMed then
loads with `local_files_only=True` so a missing tokenizer, config, or weight file fails locally instead of
downloading. See [Configuration & Validation](configuration.md#local-only-offline-mode) and the
[FAQ](faq.md#can-openmed-run-fully-locally-or-in-an-air-gapped-environment).

---

## Performance / Cold-start & Memory

### High or growing memory use on long documents or batches

**Symptom.** Memory climbs when processing many documents, or a single very long document uses more memory
than expected.

**Cause.** Each pipeline holds a model in memory, and long inputs are tokenized into large tensors. Creating a
new pipeline per document multiplies resident memory.

**Fix.**

- **Reuse one loader/pipeline** across documents; do not build a fresh pipeline per call. See
  [ModelLoader & Pipelines](model-loader.md).
- **Batch** repeated work through the batch processor rather than looping one call at a time — see
  [Batch Processing](batch-processing.md).
- **Long inputs use sentence windows and tokenizer limits by default.** `analyze_text` enables sentence
  detection and groups sentences into bounded windows, while the tokenizer uses the model's maximum length
  (`truncation=True`). A single oversized sentence can still be truncated. Pass an explicit `max_length` to
  bound per-request memory, or set `truncation=False` only when you have verified the input fits.
- **In the REST service**, bound the working set: `OPENMED_SERVICE_MAX_RESIDENT_MODELS` (LRU eviction),
  `OPENMED_SERVICE_MODEL_MEMORY_BUDGET_BYTES`, and `OPENMED_SERVICE_MAX_TEXT_LENGTH` (defaults to 1,000,000
  characters; requests above it are rejected). Use the model-lifecycle endpoints to unload models. See
  [REST Service](rest-service.md).

### `StreamingBufferError` on streaming de-identification

**Symptom.** A streaming de-identification run raises:

```text
StreamingBufferError: carry-over buffer exceeded max_buffer before any prefix could be finalized; increase max_buffer for this stream
```

**Cause.** The carry-over buffer that keeps entity spans intact across chunk boundaries grew past `max_buffer`
before a safe prefix could be emitted — typically a very long unbroken run of text.

**Fix.** Increase `max_buffer` above the longest identifier or unsplittable text run the stream must retain.
Changing the source chunk size alone does not make an overlong unsafe tail safe to emit.

---

## Device (CPU / GPU / MLX)

### Choosing CPU, CUDA, or Apple Silicon

**Symptom.** You want to control which device inference runs on, or a GPU is not being used.

**Cause.** The PyTorch device default is `"auto"`: OpenMed probes MPS, then CUDA, then falls back to CPU.
Explicit config or environment values override that choice; MLX is a separate backend and requires Apple
Silicon plus the matching dependencies.

**Fix.** Select the device through `OpenMedConfig` or environment variables:

```python
from openmed.core import OpenMedConfig, ModelLoader

config = OpenMedConfig(device="cuda")        # or "cuda:1", "mps", "cpu"
loader = ModelLoader(config=config)
```

```bash
export OPENMED_DEVICE=cuda:1                  # legacy alias
export OPENMED_TORCH_DEVICE=cuda:1           # PyTorch backend device
```

Set only one of those variables when possible; `OPENMED_TORCH_DEVICE` takes precedence over the legacy
`OPENMED_DEVICE`. The auto-detection order is **MPS → CUDA → CPU**, and `"gpu"` is normalized to `"cuda"`.
If you request `device="cuda"` on a host without CUDA, PyTorch raises its standard device error (for example
`AssertionError: Torch not compiled with CUDA enabled` or `RuntimeError: No CUDA GPUs are available`) — fall
back to `device="cpu"` or install a CUDA-enabled PyTorch build. See
[Configuration & Validation](configuration.md#cache-device-tips).

### Apple Silicon: MLX vs. Torch MPS

**Symptom.** You are on an Apple-Silicon Mac and want acceleration, or an MLX-only artifact fails on a
non-Apple host.

**Cause.** MLX runs only on Apple Silicon. On other hosts, supported MLX token-classification artifacts fall
back to their upstream PyTorch model. MLX language-model features do not have that fallback and require
`mlx-lm`:

```text
ImportError: mlx-lm is required for OpenMed MLX language models. Install with: pip install openmed[mlx]
```

**Fix.** On Apple Silicon, install the MLX extra and verify the backend resolves:

```bash
pip install "openmed[mlx]"
python -c "from openmed.core.backends import get_backend; print(type(get_backend()).__name__)"
```

For the PyTorch MPS path and its tuning defaults, see [MLX Backend](mlx-backend.md) and
[Torch MPS Performance](performance-mps.md). On non-Apple hosts, keep `device="cpu"` or `device="cuda"`.

---

## Input & Language Errors

### `ValueError: Input text cannot be empty` / `cannot be None`

**Symptom.** A call to the public API raises one of:

```text
ValueError: Input text cannot be None
ValueError: Input text cannot be empty
ValueError: Input text too long. Maximum length: <n>
```

**Cause.** OpenMed validates inputs before inference. Empty, `None`, or over-length text is rejected up front
so API clients get a clear error instead of a downstream crash.

**Fix.** Pass non-empty text; strip/guard user input before calling. If you need a length ceiling, set it
explicitly with the validation helper:

```python
from openmed.utils.validation import validate_input

# validate_input strips surrounding whitespace automatically.
text = validate_input(user_supplied_text, max_length=2000, allow_empty=False)
```

### `ValueError: Unsupported language '…'`

**Symptom.** A PII call with an unrecognized `lang` raises:

```text
ValueError: Unsupported language 'xx'. Supported: ['ar', 'de', 'en', 'es', 'fr', 'he', 'hi', 'id', 'it', 'ja', 'ko', 'nl', 'pt', 'ro', 'te', 'th', 'tr']
```

**Cause.** PII extraction and de-identification support **17 supported PII language codes: ar, de, en, es,
fr, he, hi, id, it, ja, ko, nl, pt, ro, te, th, and tr**. Passing anything outside that set (or a mistyped
code) raises this error.

**Fix.** Use one of the supported codes with `extract_pii(..., lang="<code>")`. Clinical NER coverage depends
on the selected registry model — check each model's `languages` in the
[Model Registry](model-registry.md). See the [FAQ](faq.md#which-languages-are-supported).

### `ValueError: method must be one of …` (de-identification)

**Symptom.** `deidentify(..., method="…")` with an unknown method raises:

```text
ValueError: method must be one of ('mask', 'remove', 'replace', 'hash', 'shift_dates', 'format_preserve')
```

**Cause.** `deidentify()` accepts a fixed set of methods (default `method="mask"`). A typo or unsupported
value is rejected.

**Fix.** Pick a valid method for your reversibility needs (see the
[FAQ](faq.md#is-de-identification-reversible) and [PII Anonymization](anonymization.md)):

```python
from openmed import deidentify

# Synthetic example only; do not paste real patient text into documentation or issues.
result = deidentify("Patient John Doe, DOB 01/15/1970", method="mask")
print(result.deidentified_text)   # read output from .deidentified_text (not .text)
```

!!! warning "Common de-identification gotchas"
    - Read results from `result.deidentified_text`, **not** `result.text`.
    - Redacted output is not reversible by itself. For an authorized reversible workflow, pass
      `keep_mapping=True` and restore with `reidentify(result.deidentified_text, result.mapping)`. The mapping
      contains raw identifiers and must be protected as PHI. Date shifts can also be reversed by someone who
      knows the offset.
    - `date_shift_days` requires `method="shift_dates"` (otherwise
      `ValueError: date_shift_days requires method='shift_dates'`), and `patient_key` requires
      `date_shift_secret` (`ValueError: patient_key requires date_shift_secret`).
    - Detection is an assistive control, not a substitute for a privacy review. Always inspect output before
      release.

---

## REST Service & MCP Setup

### REST service fails to import or start

**Symptom.** Starting the API fails with `ModuleNotFoundError: No module named 'fastapi'` (or `uvicorn`).

**Cause.** The REST surface (`fastapi`, `uvicorn`, tracing/observability deps) lives in the `service` extra.

**Fix.** Install the extra and launch the ASGI app with uvicorn:

```bash
pip install "openmed[service]"
uvicorn openmed.service.app:app --host 127.0.0.1 --port 8080
```

Tune behavior with `OPENMED_SERVICE_*` variables, e.g. `OPENMED_PROFILE=dev`,
`OPENMED_SERVICE_PRELOAD_MODELS`, `OPENMED_SERVICE_MAX_TEXT_LENGTH`, and
`OPENMED_SERVICE_TRUSTED_HOSTS` (defaults to loopback only). See [REST Service](rest-service.md),
[REST Authentication](serving/authentication.md), and [REST Tracing](serving/tracing.md).
Keep the loopback bind for local troubleshooting. Before exposing the service on a network, configure
authentication, TLS at the ingress or reverse proxy, and an exact trusted-host allowlist.

### MCP server fails to start

**Symptom.** Launching the MCP server raises:

```text
RuntimeError: The MCP SDK is not installed. Install OpenMed with the MCP extra: pip install "openmed[mcp]"
```

**Cause.** The Model Context Protocol SDK lives in the `mcp` extra.

**Fix.** Install the extra and run the server module. It defaults to `stdio` transport; use
`streamable-http` to serve over HTTP:

```bash
pip install "openmed[mcp]"
python -m openmed.mcp.server                                             # stdio (default)
python -m openmed.mcp.server --transport streamable-http --host 127.0.0.1 --port 8081
```

The HTTP transport reads `OPENMED_MCP_HOST` (default `127.0.0.1`), `OPENMED_MCP_PORT` (default `8081`),
`OPENMED_MCP_PATH` (default `/mcp`), and `OPENMED_MCP_TRANSPORT` (default `stdio`).

### The optional Typer CLI reports `Typer/Rich not installed`

**Symptom.** Starting the optional Typer interface with `python -m openmed.cli.typer_app` raises:

```text
RuntimeError: Typer/Rich not installed. Install with `pip install .[cli]` or `pip install typer rich`.
```

**Cause.** The optional rich terminal UI depends on `rich` and `typer`, which live in the `cli` extra. The
standard `openmed` command uses `argparse` and does not need this extra.

**Fix.**

```bash
pip install "openmed[cli]"
python -m openmed.cli.typer_app --help
```

Use `openmed --help` for the standard CLI (`analyze`, `batch`, `deid`, `pii`, `audit`, `risk`, `models`,
`config`, `doctor`, and more).

---

## Still stuck?

- Run `openmed doctor` and include its output when reporting a problem.
- Check the [FAQ](faq.md) for conceptual questions (offline use, model selection, reversibility).
- Review [Configuration & Validation](configuration.md) for cache paths, device selection, profiles, and
  environment overrides.
- If you believe you found a bug, open an issue at
  [maziyarpanahi/openmed](https://github.com/maziyarpanahi/openmed/issues) with the error text, the command
  you ran, and your OpenMed version (`openmed doctor` reports it). Use only synthetic text in reproductions,
  and remove patient content, reversible mappings, access tokens, credentials, and private model URLs from
  commands, tracebacks, logs, and screenshots before sharing them.
