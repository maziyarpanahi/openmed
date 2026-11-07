# OpenPHI-Vision development gates

OpenPHI-Vision executes local candidates on public synthetic inputs and inspects their actual PNG exports. Critical burned-in leakage and document-any-PHI leakage are independent failure gates: good quality, utility or speed cannot compensate for either. A passing report is development evidence, not permission to deploy clinically.

## Tracks and what they measure

| Track | Executed control | Boundary |
| --- | --- | --- |
| Document | Two rendered pages with private carriers and metadata | PNG pages, not a PDF parser benchmark |
| Image | Raster redaction and fresh export | Known synthetic carrier, not detector recall |
| DICOM preview | Native RGB DICOM headers and pixels to fresh PNG | Not a complete DICOM rewrite or modality qualification |
| WSI label and overview | Two associated rasters and the slide tile planner | Not whole-slide decoding or diagnostic pathology |
| OCR conflict | Actual OCR/VLM routing of disagreeing observations | Supplied synthetic readings, not OCR recognition |
| Mixed language | English, French and Spanish synthetic text | Public controls, not population-level language performance |
| Cross document | Actual packet-consistency checks and all-page inspection | Two synthetic patient identities, not clinical linkage validation |

`SyntheticVisionControl` knows the fixture layout. It masks the upper region, calls the pixel-coverage gate and verifies residual known white carriers. The evaluator scans the whole output, including moved carriers, and checks declared private metadata. This intentionally privileged control verifies software wiring. It does not establish clinical identifier recall, model accuracy, contamination resistance or independent validation.

## Run the controls

From a source checkout with development dependencies installed (`uv sync --extra dev`):

```python
from openmed.eval.suites.openphi_vision import (
    SyntheticVisionControl,
    VisionGatePolicy,
    run_openphi_vision,
    synthetic_vision_cases,
)

cases = synthetic_vision_cases(cases_per_track=3)
policy = VisionGatePolicy(
    minimum_cases_per_track=3,
    minimum_utility_rate=1.0,
    minimum_quality_rate=1.0,
    maximum_abstention_rate=0.0,
)
report = run_openphi_vision(SyntheticVisionControl(), cases, policy)
assert report["passed"]
assert report["executed_cases"] == 21
assert report["requires_review"] and not report["clinical_readiness"]
```

The corpus is generated in memory. No weights, clinical datasets or network access are needed. PNG generation requires Pillow; the DICOM control also requires pydicom. These are existing development dependencies. No raw input or output assets are returned in the report.

## Read the evidence

Each required track has a case count, emitted/assessed/failed counts, and separate rates with descriptive 95% Wilson intervals. English, French and Spanish slices overlap; do not add their counts. A two-page case contributes one event when either page leaks. Unknown metadata or an invalid export is an assessment failure, not a measured clean result. Missing tracks, duplicate cases, incomplete runs and all-abstention candidates cannot produce a pass.

- **Critical burned-in:** residual known white carrier anywhere in any output page.
- **Document-any-PHI:** residual carrier, declared private metadata or a copied declared private marker.
- **Quality:** the candidate's structural review-hold decision matches the expected decision.
- **Utility:** all original public control pixels remain exactly unchanged.
- **Abstention:** an explicit no-export result, reported separately from failed processing.

The privacy denominator is assessed cases; quality, utility and abstention use executed cases. Failed or cancelled work cannot be hidden by those denominators. Zero observed leaks still have an interval with a positive upper bound. Public development fixtures do not support a claim about clinical deployment risk.

## Candidate and provenance contract

A `VisionCandidate` receives `VisionInput`, executes locally and returns fresh RGB PNG pages in `VisionOutput`, or explicitly abstains. Its immutable `ReleaseProvenance` binds model, preprocessing, hardware, runtime and model/dataset license snapshot digests. The suite adds hashes of the actual case bytes and oracle, harness and policy. The output digest binds the actual export hashes and decisions; the report digest also binds aggregate results and provenance.

Independent adapter code and weights retain their separate licenses. Snapshot hashes record which terms accompanied the evaluated assets. They are not an allowlist, runtime licensing decision or acceptance of terms. A changed checkpoint or license snapshot yields different provenance. Users supply and assess their own model assets; the suite does not bundle restricted weights or datasets.

The default control records its source artifact, installed dependency versions, operating system and machine architecture. This is a development declaration, not device attestation. For a real device evaluation, supply a candidate with a hardware manifest that identifies the actual CPU/accelerator, memory, OS and execution configuration, plus immutable weight and preprocessing manifests. Keep those manifests under the caller's governance; reports contain their digests only.

Trusted callbacks are not sandboxed by this API. They must not download, log, retain or remotely forward inputs. Cancellation is checked between bounded calls; it cannot interrupt arbitrary provider code. Python immutable byte objects and external copies are not securely erased by this evaluator.

See [EdgeHealthBench profiles](edge-health-bench.md) for separately measured device evidence.
