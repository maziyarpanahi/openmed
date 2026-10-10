# Synthetic clinical brief walkthrough

This is a recording script, not a clinical result or a finished video. Use only
the embedded synthetic note. Keep the fixture-provider disclaimer visible.

## Run locally

From a source checkout with development dependencies installed:

```bash
python examples/v30_clinical_brief.py
python -m pytest tests/integration/test_clinical_pipeline_e2e.py -q
```

The default CPU path has no downloads. It uses the actual public de-identification,
extraction, grounding and composition APIs, with explicitly labelled fixture
NER/NLI providers. The structured-identifier sweep runs normally. The recorded
synthetic review transitions are golden-test state, not real clinician approval.
The example accepts no user note argument and must not be adapted for real
patients by retaining its fixture providers.

`--model mlx` invokes the real cached local summarizer. Its output still has to
pass exact evidence alignment, NLI and privacy checks. A refusal is an expected
possible outcome, not permission to disable those checks. The fixture NLI
provider is still not a trained model even when generation uses MLX.

## Recording beats

1. Show a page labelled **SYNTHETIC** beside the terminal. If demonstrating OCR,
   scan that page in the local app; do not claim the CLI example itself runs OCR.
2. Show the local model/cache status and the human-review disclaimer. Do not
   imply absent NLI or review configuration has been supplied automatically.
3. Run de-identification. Keep only the redacted note visible after this step.
4. Show extraction offsets and the local grounding candidate count.
5. Run the CPU brief example. Show three source-linked claims and verdicts.
6. Open the value-free review packet beside the protected synthetic summary.
   Point to evidence offsets, input/output digests and the review-required state.
7. End with the release-gate status: functional fixtures do not establish
   clinical quality. Record any refusal honestly instead of substituting output.

## Golden regression

`tests/fixtures/clinical/e2e/brief_golden.json` pins the aggregate hand-offs and
protected synthetic result. The test regenerates the pipeline output, compares
the complete mapping with pytest's readable diff, blocks socket connections and
enforces a one-minute runtime ceiling. Existing extraction/assertion/grounding/
FHIR golden tests remain in `tests/integration/test_pipeline_e2e.py`.
