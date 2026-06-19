# OpenMed — Ecosystem Overview

> A synthesis of OpenMed across all its surfaces: this codebase, the website
> ([openmed.life](https://openmed.life/)), the Hugging Face organization
> ([huggingface.co/OpenMed](https://huggingface.co/OpenMed)), the research paper
> ([arXiv:2508.01630](https://arxiv.org/abs/2508.01630)), and the OpenMed Agent
> preview ([agent.openmed.life](https://agent.openmed.life/)).
>
> Compiled 2026-06-19. This is a landscape document, not API docs — for usage start
> with [Welcome](index.md) or [openmed.life/docs](https://openmed.life/docs).

## What OpenMed is

OpenMed is a **local-first, open-source healthcare AI project** — clinical entity
extraction and HIPAA PII de-identification that runs **100% on-device** — created by
Maziyar Panahi and distributed as a Python library, a Swift/iOS kit, a Hugging Face
model hub, and a research paper. Everything is **Apache-2.0**, with full training
recipes published.

The thesis is **data sovereignty**: download models once, then all inference runs in
your own Python process, Docker container, or Swift app — no telemetry, no license
check-in, no outbound calls at runtime, so PHI never leaves your network.

> "Clinical AI that never leaves the device. Turn clinical text into structured insight
> with one line of code."

Headline figures (README / website): **1,000+ models · 12 languages · 247 PII
checkpoints · 100% on-device · 6M+ PyPI downloads · state-of-the-art on 10 of 12 NER
benchmarks**.

## The three-tier stack

OpenMed is layered, and every layer reinforces the "PHI stays on your side" posture:

1. **Models (Hugging Face)** — open encoder NER + multilingual PII / privacy-filter
   models, *plus* new medical-reasoning SFT datasets that seed a generative tier.
2. **Toolkit (Python `openmed` + Swift `OpenMedKit`)** — one-liner `analyze_text` /
   `deidentify` / `reidentify`, FHIR R4 bundle export, batch + REST service; runs from a
   notebook down to an offline iPhone via Apple MLX.
3. **Agent ([agent.openmed.life](https://agent.openmed.life/))** — an inspectable,
   terminal-native runtime that orchestrates those models plus standard medical
   vocabularies into auditable clinical workflows. Invite-only preview today.

The trajectory: began as the best open **biomedical NER** models (the paper), became an
**on-device clinical NLP + de-identification toolkit** (this repo, PyPI), and is now
extending upward into **generative medical reasoning** and a **workflow agent** — while
keeping everything Apache-2.0 and on the user's own hardware.

## 1. This codebase

Python 3.10+ package (v1.5.x line) plus native Swift, built on Hugging Face
`transformers` + `huggingface-hub`.

- **`openmed/` (Python core)** — one-liner API: `analyze_text()`, `extract_pii()`,
  `deidentify()`, `reidentify()`. Submodules: `core/` (model loading, PII, registry,
  audit), `clinical/` (FHIR/OMOP exporters, ConText temporality/certainty axes),
  `ner/`, `processing/` (batch), `service/` (FastAPI REST), `eval/` (benchmarks,
  confidence intervals, re-identification attacks), plus `mlx/`, `coreml/`, `torch/`
  backends and an `mcp/` integration.
- **`swift/OpenMedKit`** — on-device inference for iOS / iPadOS / macOS via Apple's
  `mlx-swift` and `swift-transformers`. The README leads with an iPhone demo that scans
  and de-identifies a clinical note fully offline.
- **`models.jsonl`** — curated catalog of **1,518 model entries** (the "1,000+"
  headline counts unique models; the catalog also includes format variants).

Recent development themes (git history) map to concrete features:

- **Reversible de-identification** — de-identify → re-identify round-trip, with a fix
  for repeated entity types using unique placeholders.
- **FHIR R4 transaction Bundle assembler** — deterministic `urn:uuid` references with
  in-bundle reference rewriting; handles de-identified resources gracefully.
- **Bootstrap confidence intervals** — non-parametric CIs for benchmark / PHI-leakage
  metrics, used in adversarial re-identification evaluation.
- **Clinical context axes** — ConText-style temporality (recent / historical /
  hypothetical) and certainty (certain / uncertain) for condition status.

## 2. The models (Hugging Face org)

- **~1,091 models** live on the Hub. Top downloads are domain NER models — BloodCancer,
  Chemical, Oncology, Species, Pharma, Disease, Genomic, DNA, Anatomy — each in size
  variants (TinyMed 65M → MultiMed 568M). The most-downloaded single model has ~195K
  downloads; the most-*liked* are the SuperClinical DeBERTa pharma/oncology models.
- **18 collections** — a heavy multilingual PII / de-identification push (English,
  French, German, Italian, Spanish, Dutch, Hindi, Telugu, Portuguese), plus a
  **privacy-filter** family, **Medical MLX Models**, **ZeroShot Medical NER**, and
  **Medical Leaderboards**.
- **26 datasets** — clinical corpora (MultiCARE, MedDialog, DDI, DrugProt) *and* the
  new direction below.
- **2 demo Spaces** — `openmed-ner-models`, `openmed-clinical-ner`.

## 3. The research (arXiv:2508.01630)

*"OpenMed NER: Open-Source, Domain-Adapted State-of-the-Art Transformers for Biomedical
NER Across 12 Public Datasets"* (Maziyar Panahi).

The SOTA comes from a cheap, reproducible recipe rather than scale:

- **Domain-adaptive pre-training (DAPT)** on a 350k-passage corpus from *ethically
  sourced* public data: PubMed, arXiv, and de-identified MIMIC-III clinical notes.
- Backbones: **DeBERTa-v3, PubMedBERT, BioELECTRA**.
- Task fine-tuning with **LoRA, updating < 1.5% of parameters** — which is why 109M–568M
  models beat far larger closed systems.
- Result: **new SOTA micro-F1 on 10 of 12 benchmarks**, F1 up to **0.998** (Gellus),
  beating closed-source SOTA by **up to 36%** on some datasets.

The models are **encoder** transformers (extract/classify structured entities), not
generative chat models — small enough to run on a laptop or phone, with a tiny training
footprint (<1.2 kg CO₂e).

## 4. The reasoning-LLM pivot

The newest Hugging Face datasets are not a side project — they are training data for a
**generative** medical tier:

- **`Medical-Reasoning-SFT-Mega`** — **1.79M deduplicated samples / 3.78B tokens** of
  medical chain-of-thought reasoning, distilled from **seven frontier models**
  (GPT-OSS-120B, Qwen3-Next-80B, Trinity-Mini, Nemotron-Nano-30B, GLM-4.5-Air,
  MiniMax-M2.1, Baichuan-M3-235B) using a "fair distribution" dedup so no single teacher
  dominates.
- A **SynthVision** synthetic medical-imaging pipeline (seeds → annotated →
  cross-validated) signals a medical-vision direction as well.

This is a categorical step beyond the encoder-NER thesis of the paper, and it
corroborates the Agent product below.

## 5. OpenMed Agent (preview)

A **terminal-native, inspectable agent runtime** for clinical workflows. Currently
**Preview 0.1.2, invite-only** (distributed to approved evaluators).

- **Hybrid architecture** — a local operator runtime separated from protected
  medical-service endpoints.
- **Inspectable workflows** — describe a task in natural language, then inspect the
  plan, tool traces, and outputs; draft preview, approval tokens, and visible provenance
  before anything finalizes. No background telemetry.
- **Built-in medical tooling** — PubMed, ICD-10, CPT, SNOMED, LOINC, RxNorm, HCC
  mapping.
- **Four modes** — clinical, consumer, coordination, plan.
- **Workflows** — prior authorization & appeals, care coordination, consumer
  health-record import, clinical documentation, FHIR work, medical coding, claims
  explanation, literature search.
- **Extensible via MCP** — attach organization-specific tools as JSON-RPC servers.

## Quick orientation

```python
from openmed import analyze_text

result = analyze_text(
    "Patient started on imatinib for chronic myeloid leukemia.",
    model_name="disease_detection_superclinical",
    confidence_threshold=0.55,
)
for entity in result.entities:
    print(entity.label, entity.text, entity.confidence)
```

Install: `uv pip install "openmed[hf]"` (standard) or `openmed[mlx]` (Apple Silicon).

## A note on the model counts

"1,000+" (site / README), **1,091** (live Hugging Face model count), and **1,518**
(`models.jsonl` catalog with format variants) all describe the same model family at
different granularities — they are consistent, not contradictory.

## Sources

- Website — https://openmed.life/ and https://openmed.life/docs
- GitHub — https://github.com/maziyarpanahi/openmed
- Hugging Face org — https://huggingface.co/OpenMed
- Paper — https://arxiv.org/abs/2508.01630
- OpenMed Agent — https://agent.openmed.life/
- Founder blog — https://huggingface.co/blog/MaziyarPanahi/open-health-ai
- PyPI — https://pypi.org/project/openmed/
