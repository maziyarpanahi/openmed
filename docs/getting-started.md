# Quick Start

This guide gets you from a blank workstation to copying results from the docs within minutes. It uses
[uv](https://github.com/astral-sh/uv) for dependency management, but any Python 3.11+ environment works.

## 1. Bootstrap the environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh  # install uv (skip if already installed)
uv venv --python 3.11                           # create a dedicated virtualenv
source .venv/bin/activate                       # or use `uv python` directly

# install OpenMed with Hugging Face extras and doc tooling
uv pip install ".[hf]"
```

Need the zero-shot GLiNER stack or dev tools? Stack extras as needed:

```bash
uv pip install ".[hf,gliner]"      # add GLiNER + transformers
uv pip install ".[dev]"            # pytest + coverage + linting
```

## 2. Run `analyze_text`

```python
from openmed import analyze_text

resp = analyze_text(
    "Metastatic breast cancer treated with paclitaxel and trastuzumab.",
    model_name="disease_detection_superclinical",
    include_html=True,
)

print(resp.entities[0])
print(resp.html_snippet)  # ready for dashboards or docs
```

Prefer the CLI? Reuse the example script:

```bash
uv run python examples/analyze_cli.py \
  --text "Imatinib inhibits BCR-ABL in chronic myeloid leukemia."
```

## 3. Copy code snippets from the docs

All code blocks ship with Material for MkDocs copy buttons. Invoking the command palette (`/` or `cmd/ctrl + K`) lets you
search for “GLiNER,” “OpenMedConfig,” or “token classification,” then copy the snippet that appears in the preview pane.
If you rely on AI copilots (ChatGPT, Copilot, etc.), point them at the published docs URL so they crawl the same
structured Markdown and surface canonical answers.

## 4. Optional: pin configuration

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

Continue to the **Configuration** section for the full YAML/ENV schema, PHI-aware validation helpers, and logging setup.
