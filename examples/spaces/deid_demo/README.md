---
title: OpenMed Multilingual De-identification
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.20.0
python_version: "3.11"
app_file: app.py
pinned: false
license: apache-2.0
models:
  - OpenMed/OpenMed-PII-Chinese-BigMed-Large-278M-v1
  - OpenMed/OpenMed-PII-Hindi-SuperClinical-Small-44M-v1
  - OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1
tags:
  - de-identification
  - healthcare
  - multilingual
---

# OpenMed multilingual de-identification demo

Try OpenMed de-identification with fabricated Simplified Chinese, Hindi, English,
or Hinglish notes. Unicode script detection selects the matching checkpoint and
language-aware pattern pack, and a manual override is available for mixed-script
examples.

> **Synthetic data only. Never paste real patient data or protected health
> information (PHI).** The demo does not log input text, enable analytics, cache
> de-identification results, or write notes to disk.

The first request for a language downloads its model checkpoint. Later requests
reuse the runtime's model cache; user text is never added to that cache.

## Run locally

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Open the local URL printed by Gradio. The app starts with an English fabricated
note; switch the UI language to preload the corresponding Chinese or Hindi sample.

## Deploy as a Space

Use the contents of this directory as the root of a Gradio Space repository. The
YAML block above pins the SDK and points directly to `app.py`; no file changes are
required.
