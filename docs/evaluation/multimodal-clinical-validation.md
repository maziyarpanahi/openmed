# Multimodal validation packs and blinded review

OpenMed provides a local review workflow for six separate tasks: ECG acquisition quality, ECG interval candidates, form grounding, DICOM frame coverage, burned-in identifier coverage, and whole-slide tiling/provenance. Each pack declares its intended use, reviewer qualifications, sampling construction, license snapshot, and limitations. These are non-diagnostic tasks. Passing a rubric does not establish clinical efficacy, diagnostic adequacy, certification, or readiness for patient care.

The bundled fixtures execute real component APIs against generated sources. They contain no patient records, external model weights, restricted datasets, or completed clinician reviews. Reviews constructed in tests are explicitly `development_simulation`. Teams must obtain their own qualified review and independent validation before making clinical claims.

## Protected inputs and artifact lineage

Import the API from `openmed.eval.suites.multimodal_clinical`. A `ValidationPack` holds one task, one to 256 distinct source units, and exactly two candidate configurations. Each case contains original in-memory assets, protected review context, two executed observations, and one closed slice: nominal, adverse acquisition, or ambiguity/conflict. Assets are bounded to 4 MiB each, eight per case, and 64 MiB per pack. The current source carriers are PNG, TIFF, native DICOM, and JSON waveform evidence; larger studies require independently designed bounded sampling.

`CandidateLineage` binds model or algorithm, preprocessing, runtime, hardware, harness, and license snapshot digests. The pack additionally binds the source bytes, actual outputs, observed abstentions and safety failures, sample construction manifest, and dataset license snapshot. Supply immutable artifact digests, not a mutable repository name or URL. Store the corresponding protected manifests so a digest can be resolved during audit. Changing any bound input invalidates previous review evidence.

License metadata records provenance. There is no model-license allowlist or runtime license gate. Independently written adapter code and third-party dependencies have their own licenses; callers are responsible for the terms of the exact model revision and data they select. A later terms change needs a new snapshot for newly evaluated artifacts, not a silent replacement of the old evidence.

Source assets, review packets, mappings, ratings, and the private assessment remain protected in-memory objects. The package makes no network request, downloads no model, creates no review files, and sends nothing to reviewers. The evaluator controls storage, access, retention, and delivery. Hashes are identifiers, not anonymization; do not publish protected manifests or predictable-content hashes merely because they contain no plaintext.

## Qualified blinded review

Create a `ReviewerQualification` for each reviewer with task scope, qualification-evidence digest, conflict-screening digest, stable reviewer identity digest, and review origin. Actual reviews use `ReviewOrigin.ATTESTED`; synthetic exercises use `ReviewOrigin.SIMULATED`. These declarations are caller governed. The library does not verify credentials, identify people, detect two identities belonging to one person, or independently certify conflict clearance. Recused and out-of-scope reviewers cannot receive a prepared review.

Call `prepare_review(pack, reviewer, key=evaluator_key)` separately for each reviewer. Generate and retain a secret key of at least 32 bytes outside reviewer-visible materials. The returned object contains two separately handled parts:

- `packets`: reviewer-visible context, source asset references, blinded outputs, and the four binary rubric criteria: task quality, retained utility, privacy, and provenance. Resolve source references against the exact pack's assets through your protected review interface.
- `mapping`: evaluator-only, authenticated alias-to-candidate mapping. Never send it, the key, or the candidate lineage manifest to reviewers.

Aliases are balanced and reviewer-specific. Packet identity and rubric are bound to the exact artifact set. Metadata blinding cannot hide a model identity that a candidate prints in its own output or stylistic clues; evaluators must assess that limitation before review. No web application or remote review service is installed.

For each candidate in every packet, collect a `PacketRating` with `packet_digest(packet)`, its alias, and four boolean scores. `CompletedReview` must cover both candidates in every source unit. `assess_validation` requires two to eight distinct, eligible complete reviews and verifies the original packets and sealed mappings before resolving aliases. Ratings are caller-attested evidence, not cryptographically signed testimony from a clinician.

## Disagreement and safety decisions

Assess one candidate at a time. A disagreement holds the result until a third qualified reviewer, distinct from every original reviewer, adjudicates it. An `Adjudication` contains that reviewer's independently prepared packet, a rating, the exact set of conflicting rating digests, and a typed `DisagreementReason`. Stale conflicts, changed credentials, wrong keys, swapped candidate aliases, missing ratings, and duplicate reviews are refused.

Privacy and provenance failures are non-compensable. A failure recorded by any original reviewer, the adjudicator, or the executed observation blocks the candidate even if a majority or adjudicator otherwise prefers it. Quality and utility must both satisfy the rubric across all cases. An all-abstention candidate stays on hold; abstention is reported separately and cannot count as useful emission. The private `ValidationAssessment.decision` is `blocked`, `hold`, or `rubric_satisfied`. None of these values authorizes clinical use or automatic deployment.

The private assessment also retains exact pack and candidate digests. Keep the complete evidence chain under evaluator access controls. The serializable public report deliberately excludes the decision and lineage identifiers, because a pass/fail flag can disclose a small sensitive failure cell.

## Aggregate reporting and uncertainty

`assessment.to_json()` emits only closed task/slice labels, review origin, fixed limitations, and suppressed aggregate metrics. It excludes source text, outputs, identifiers, reviewer identities, packet hashes, and decision reasons tied to a source unit. If any review or adjudication is simulated, the report remains labelled development simulation.

Each source unit contributes once to each binary metric regardless of the number of reviewers. Published metrics include quality, utility, privacy, provenance, emission, reviewer agreement, and adjudication. Wilson 95% intervals describe case-level binary counts; convenience samples and repeated acquisitions do not justify population inference or clinical claims. A matching source digest cannot be counted twice in a pack. Different byte encodings or related acquisitions can still be correlated; sampling governance must handle those relationships.

The minimum publication cell size is five and can be raised. A nonzero numerator or complement below that threshold suppresses the entire slice, including all counts, rates, intervals, and disagreement metrics. Any suppressed slice also suppresses the overall totals to prevent subtraction. Empty declared slices are reported in the same suppressed shape as small slices. This protects one report; it is not differential privacy and does not prevent differencing across repeated, overlapping releases. Apply an external publication policy before releasing multiple reports.

## Executable development controls

The source checkout includes `tests/fixtures/eval/multimodal_clinical.py`; it is test material, not a dependency of the installed library. With development and imaging dependencies installed, run:

```python
from openmed.eval.suites.multimodal_clinical import ValidationTask
from tests.fixtures.eval.multimodal_clinical import development_pack

for task in ValidationTask:
    pack = development_pack(task)
    assert pack.describe()["non_diagnostic"]
    assert not pack.describe()["independent_validation"]
    assert len(pack.cases) == 1
```

The controls construct original sources and execute two explicit configurations:

| Task | Executed control | Boundary |
| --- | --- | --- |
| ECG quality | Two-lead triangles at 100 Hz, 1-second versus 5-second duration gates | Synthetic acquisition evidence only |
| ECG intervals | Actual fiducial/interval extraction and acquisition-quality refusal | Measurement confirmation is simulated; no clinician review |
| Forms | Rendered label/value pixels, source boxes, two association-gap policies | Grounding proposals remain review-required |
| DICOM | Six native CT frames, two selection budgets, coverage manifests | No diagnostic interpretation or compressed-codec claim |
| Pixel identifiers | Actual bright-marker detection, full versus missing mask, residual checks | Known-carrier wiring, not clinical PHI sensitivity |
| Whole-slide provenance | Actual generated TIFF, two tile sizes, RGB means from original crops | No pathology model or tissue diagnosis |

Fixture lineage fingerprints the fixture and component source, actual runtime package versions, platform class, configurations, and generated bytes. Each pack has only one source unit, intentionally too few for published rates. Regenerating the same fixture does not produce independent samples. Qualification and adversarial tests exercise changed evidence, unresolved disagreements, minority safety failures, abstention, and complementary suppression separately from these component controls.

Run the focused regression suite with:

```bash
python -m pytest tests/unit/eval/test_multimodal_clinical_validation.py -q
```
