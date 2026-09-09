# Health Universe PHI Replacement Agent

OpenMed includes a root-level Health Universe A2A agent that converts platform-
extracted documents into draft Markdown with detected identifiers replaced by
synthetic values.

The workflow is:

1. list source documents with the Health Universe SDK;
2. wait for Health Universe platform OCR and extraction;
3. download the platform-extracted Markdown, not the raw document;
4. run the pinned OpenMed PHI model inside the agent container;
5. upload opaque draft `.md` files and a count-only safety report.

Every result is a review draft. Replacement does not establish HIPAA Safe
Harbor and cannot protect an identifier the detector misses.

## Privacy boundary

OpenMed model inference is forced offline. The agent does not call OpenAI, the
Hugging Face API, or another model service with document text. The model is
downloaded while the container image is built and loaded from `/app/model` at
runtime.

The deployed workflow is not an on-device workflow: source documents and
extracted Markdown are handled within the Health Universe platform. Use it only
in an appropriately authorized Health Universe environment. The local runner,
when supplied with already-extracted Markdown, does not use platform OCR or
network document APIs.

The saved safety report contains counts, labels, hashed platform document IDs,
and error types. It does not contain source text, identifier values, surrogate
values, source filenames, or source paths. The replacement Markdown itself may
still contain missed PHI and must remain protected.

## Install

Install the optional Health Universe and Hugging Face dependencies:

```bash
uv pip install -e ".[hf,health-universe-agent]"
```

No OpenAI API key is used. Health Universe supplies document authorization to
the deployed agent through the A2A request context.

## Run local Markdown

The local SDK client does not perform OCR. Create this layout using synthetic
or otherwise authorized already-extracted Markdown:

```text
test/patient/
└── source/
    └── record.md
```

Run:

```bash
python local.py test/patient
```

Outputs are written to `test/patient/artifact/` by default:

- `Deidentified_Document_0001.md`
- `Deidentification_Safety_Report.json`

Only `.md` source files are accepted by the local runner. Production accepts
the document types supported by Health Universe platform extraction.

## Deploy

The root-level deployment entry point is `deployment:app`. Build the dedicated
agent image from the repository root:

```bash
docker build -f Dockerfile.health-universe-agent -t openmed-phi-agent .
docker run --rm -p 8000:8000 openmed-phi-agent
```

The container downloads the pinned model snapshot during the build. Runtime
model access and common telemetry paths are disabled by default. The model
revision pin makes the build repeatable, but OpenMed does not currently have a
trusted publisher artifact hash for this model.

## Controls and limitations

- `method="replace"`, threshold `0.5`, and the deterministic safety sweep are
  enabled.
- OpenMed result caching and saved re-identification mappings are disabled.
- Replacement seeds are stable within a Health Universe thread.
- Opaque output filenames prevent source filenames from being copied into
  artifacts.
- Individual documents fail closed when extraction, inference, or output
  writing fails.
- Replacement collisions and invalid generated IP addresses are counted in the
  safety report.
- Fake names are not guaranteed to match patient gender, and generated values
  may not preserve the source field's semantic format.

The bundled `hipaa_safe_harbor` policy is not passed because it forces masking
instead of replacement. Safety-sweep detection remains enabled, but human
review is still required.
