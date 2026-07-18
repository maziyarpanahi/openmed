# Quick Start

This guide gets you from a blank workstation to copying results from the docs within minutes. It uses
[uv](https://github.com/astral-sh/uv) for dependency management, but any Python 3.11+ environment works.

Working with intermittent connectivity, offline clinics, OpenMRS, or DHIS2?
Use the [African developer onboarding guide](africa-onboarding.md) for a
low-bandwidth model setup, local-only inference, privacy-profile pointers, and
FHIR-based integration recipes.

## 1. Bootstrap the environment

=== "macOS/Linux"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh  # install uv (skip if already installed)
    uv venv --python 3.11                           # create a dedicated virtualenv
    source .venv/bin/activate                       # or use `uv python` directly

    # install OpenMed with Hugging Face extras and doc tooling
    uv pip install ".[hf]"
    ```

=== "Windows PowerShell"

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    uv venv --python 3.11
    .venv\Scripts\Activate.ps1

    # install OpenMed with Hugging Face extras and doc tooling
    uv pip install ".[hf]"
    ```

Need the zero-shot GLiNER stack or dev tools? Stack extras as needed:

```bash
uv pip install ".[hf,gliner]"      # add GLiNER + transformers
uv pip install ".[dev]"            # pytest + coverage + linting
```

Installing behind an institutional proxy or package mirror, or preparing for
a metered/offline deployment? Follow the
[low-bandwidth, mirror, and proxy installation guide](low-bandwidth-install.md)
to configure pip, `HF_ENDPOINT`, the shared model cache, and diagnostics.

For scanned images and document OCR, install the multimodal extra plus the
system Tesseract binary:

```bash
uv pip install ".[multimodal]"
brew install tesseract             # macOS
sudo apt-get install tesseract-ocr # Debian/Ubuntu
```

PaddleOCR is available as a heavier opt-in OCR backend:

```bash
uv pip install ".[ocr-paddle]"
```

CDA/C-CDA XML de-identification is available in the core install. It redacts
structured header PHI, sweeps CDA section narrative text, keeps XML parseable,
and only routes `.xml` files that look like CDA documents:

```python
from openmed.interop.cda import redact_cda

redacted_xml = redact_cda("synthetic_ccda.xml", date_shift_days=30)
```

On an Apple Silicon Mac, you can start directly on the new MLX path:

```bash
uv pip install ".[mlx]"            # Python MLX runtime + tokenizer/artifact deps
uv run python -c "from openmed.core.backends import get_backend; print(type(get_backend()).__name__)"
```

If you want the full launch surface on one machine, combine them:

```bash
uv pip install ".[hf,mlx,docs]"
```

## 2. Run `analyze_text`

```python
from openmed import analyze_text

text = "Metastatic breast cancer treated with paclitaxel and trastuzumab."

resp = analyze_text(text, model_name="disease_detection_superclinical")
print(resp.entities[0])

# Want ready-to-embed HTML instead? Ask for the "html" output format:
html = analyze_text(text, model_name="disease_detection_superclinical", output_format="html")
print(html)  # ready for dashboards or docs
```

Prefer a quick script entrypoint? Run a one-file smoke script:

```bash
uv run python examples/pii_model_comparison.py
```

## 3. De-identify PII

```python
from openmed import deidentify

result = deidentify("Patient John Doe, DOB 01/15/1970", method="mask")
print(result.deidentified_text)
# Patient [first_name] [last_name], DOB [date]
```

`deidentify()` supports five methods (`mask`, `remove`, `replace`, `hash`,
`shift_dates`) — see the [Anonymization quickstart](anonymization.md#quickstart-choosing-a-method)
for a runnable example of each, plus how to reverse one with `reidentify()`.

## 4. Pull a model reliably for offline use

Use the model pull command to warm the Hugging Face cache before working
offline. Downloads resume after interrupted transfers, retry transient network
failures, and verify every file against Hub metadata:

```bash
openmed models pull disease_detection_superclinical
```

On a metered or unstable connection, pin the revision, limit transfer speed,
and tune the retry count explicitly:

```bash
openmed models pull disease_detection_superclinical \
  --revision main \
  --max-bandwidth 524288 \
  --retries 5
```

Progress contains only repository filenames and byte/file totals. After the
pull completes, set `OPENMED_OFFLINE=1`; the same command then performs a
cache-only lookup and never attempts a network connection.

## 5. Copy code snippets from the docs

All code blocks ship with Material for MkDocs copy buttons. Invoking the command palette (`/` or `cmd/ctrl + K`) lets you
search for “GLiNER,” “OpenMedConfig,” or “token classification,” then copy the snippet that appears in the preview pane.
If you rely on AI copilots (ChatGPT, Copilot, etc.), point them at the published docs URL so they crawl the same
structured Markdown and surface canonical answers.

## 6. Optional: pin configuration

```python
from openmed.core import OpenMedConfig, ModelLoader

config = OpenMedConfig.from_env_fallback(
    cache_dir="~/.cache/openmed",
    device="cuda",
    default_org="OpenMed",
)
loader = ModelLoader(config=config)
ner = loader.create_pipeline("disease_detection_superclinical")
entities = ner("Hydroxyurea dose reduced after platelet drop.")
```

Continue to the **Configuration** section for the full YAML/ENV schema,
PHI-aware validation helpers, and logging setup.
