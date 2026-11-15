# ECG waveform evidence windows

`openmed.multimodal.ecg.evidence_windows` attaches [ECG timing candidates](ecg-interval-candidates.md) to the original lead samples and fiducial brackets used to produce them. Each window carries its candidate digest, source context, quality state and transform provenance. It supports deterministic metadata serialization and restoration against the original local waveform. No signal samples or numeric measurement values enter the serialized metadata or audit report.

These remain non-diagnostic measurement candidates for human review. A window does not approve a result, resolve an abstention or qualify a detector. This layer performs no network access, file access, filtering, clinical action or automatic export to clinical reasoning. It preserves the interval extractor's limitations and the requirement for an independently evaluated acquisition and detector policy. This issue scopes the implementation to Python.

## Complete evidence for each candidate

`build_evidence_windows` calls the actual interval extractor on the supplied waveform. That extractor freshly reassesses [signal quality](ecg-signal-quality.md), source/calibration binding and explicit reviewer confirmation. The window builder accepts no previously supplied pass report, invented landmarks or caller-created candidate DTO as a substitute.

| Candidate | Window contents | Missing evidence |
| --- | --- | --- |
| Rate | Both consecutive ventricular complexes, with each beat's onset, peak and offset brackets | No pair means no rate window; a single peak cannot support rate |
| PR | The observed P wave and its corresponding ventricular complex | An unresolved P wave has no PR window |
| QRS | The ventricular complex and its onset, peak and offset brackets | An unresolved ventricular complex has no window |
| QT | The ventricular complex and its observed T wave | An unresolved T wave has no QT window |

Every window includes the detector's quiet guards, followed by the explicit `WindowPolicy.padding_samples` on each side. The builder refuses the entire set if requested padding crosses the source boundary, any window exceeds `maximum_window_samples`, or the sum of all window lengths exceeds `maximum_total_samples`. Overlap counts toward the total. It never silently clips context, truncates windows or drops inconvenient observations to fit a limit.

An aggregate candidate can abstain while individual lead windows remain available. For example, if one lead lacks a P wave, PR remains abstained for the whole required-lead set; the other lead's observed P/QRS window is still useful for review and retains the abstained state and reason. Missing windows are not fabricated. If the detector finds no usable fiducials, the set contains zero windows and the original abstentions. No window availability can upgrade an aggregate result.

Window indices use the original normalized source grid. `start_sample` is inclusive and `end_sample` exclusive. Each `WindowFiducial` retains an inclusive uncertainty bracket and a lead-local `beat_index`; those indices are not rebased when displaying a slice. The source-relative time for index `i` is `start_offset_seconds + i / sample_rate_hz`. It is not a wall-clock timestamp. Rate windows explicitly retain the two consecutive beat identities.

## Provenance and transforms

The result retains the protected `IntervalCandidates` object, including the fresh quality report. Its metadata binds the original source digest, full normalized waveform fingerprint, detector result and policy, acquisition digest, quality algorithm/policy, window policy and ordered upstream transform digests. Quality metadata includes controlled global/per-lead states and findings; numeric quality metrics remain in the protected in-memory object.

`upstream_transform_digests` is required. Supply the ordered SHA-256 digests of preprocessing manifests associated with the normalized source. An explicit empty tuple declares no upstream transforms. These digests are host declarations; this library cannot prove that an importer disclosed every filter or used the declared calibration. Keep the actual manifests in the host's controlled provenance store. Reordering the chain changes every dependent context/window digest.

The locally computed `transform_digest` binds that upstream chain and source/acquisition fingerprints to the actual detector algorithm and policy, its record-median centering, and the original-grid window operation and policy. Median centering is used only inside detection. `samples_for` returns the original normalized millivolt samples, with no additional filter or centering. The window and bundle digests bind descriptors to their candidate and complete source context. Hashes detect inconsistent evidence; they do not authenticate the source, a caller or clinical approval.

## Protected data and serialization

Use `to_metadata_json()` for controlled persistence or transfer of evidence descriptors. It includes source grid timing, window offsets and fiducials, so it remains protected review metadata and must not be used as an operational log. It omits waveform samples, rate/interval values, agreement scores, raw quality metrics, paths and free-form source headers. Array order is part of the evidence identity. Use `to_report_json()` for operational output: only hashes, controlled candidate states/reasons and window counts are included, with the non-diagnostic notice.

`restore_evidence_windows` requires the original local waveform plus the expected acquisition evidence, quality/detector/window policies and upstream transform chain. It parses bounded metadata with duplicate-key and nonfinite-number rejection, reruns source quality and measurement, rebuilds all windows, and compares the complete typed metadata to the saved representation. Whitespace and object-key order may change; omitted/reordered/extra windows, changed fiducials, boolean or floating sample indices, changed policies, stale source data, unknown fields or altered abstentions are rejected. Recomputing the serialized hashes cannot substitute for matching fresh source evidence. Restoring without the original waveform is deliberately unsupported.

For a local viewer, call `samples_for` with the exact member window digest, original immutable waveform and explicit reviewer confirmation. It rechecks the full waveform fingerprint, including optional channels, and slices only the referenced lead and source interval. `WindowSamples` holds protected values and the source-relative grid, omits them from `repr`, and has no serialization method. The library performs no logging or persistence; host code must control its viewer, memory copies and storage. Confirmation is an application assertion, not identity authentication or a sandbox.

The existing clinical evidence packet boundary accepts only synthetic, reviewed references. This layer does not label real ECG evidence synthetic, invent approved review histories or reinterpret sample offsets as text offsets to cross that boundary. The window set supports measurement review and preserves all abstentions. Clinical evidence admission remains a separate explicit workflow.

## Synthetic example

First run the complete synthetic example in the [interval-candidate guide](ecg-interval-candidates.md#runnable-synthetic-example). It defines `wave`, `quality`, `acquisition` and `policy` using generated triangle pulses and complete original acquisition evidence. Continue with the following code. All limits are software fixtures, not clinical recommendations; no sample or measurement values are printed.

```python
from openmed.multimodal.ecg.evidence_windows import (
    WindowPolicy, build_evidence_windows, restore_evidence_windows,
)
from openmed.multimodal.ecg.interval_candidates import TimingKind

window_policy = WindowPolicy(
    version="synthetic-windows-v1", evidence_digest="e" * 64,
    padding_samples=3, maximum_window_samples=200,
    maximum_total_samples=10000,
)
evidence = build_evidence_windows(
    wave, quality, acquisition, policy, window_policy,
    upstream_transform_digests=(), reviewer_confirmed=True,
)
metadata = evidence.to_metadata_json()  # Protected offsets/fiducials; not a log.
restored = restore_evidence_windows(
    metadata, wave, quality, acquisition, policy, window_policy,
    upstream_transform_digests=(), reviewer_confirmed=True,
)
assert restored.digest == evidence.digest
rate_window = next(
    window for window in restored.windows if window.kind is TimingKind.RATE
)
view = restored.samples_for(
    rate_window.digest, wave, reviewer_confirmed=True,
)
assert len(view.samples_mv) == rate_window.end_sample - rate_window.start_sample
assert view.source_start_sample == rate_window.start_sample
assert view.start_offset_seconds == (
    wave.start_offset_seconds + rate_window.start_sample / wave.sample_rate_hz
)
audit_json = restored.to_report_json()  # Counts, controlled decisions and digests.
```

Run `.venv/bin/python -m pytest tests/unit/multimodal/ecg/test_evidence_windows.py -q` for synthetic sample-slice, fiducial, replay, stale-source, tampering, resource-bound, consent and value-free reporting controls. These validate software evidence linkage, not clinical measurement accuracy.
