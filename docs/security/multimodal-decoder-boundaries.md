# Multimodal decoder containment

`probe_media` runs a bounded, closed-choice decoder in a disposable Linux worker and returns a fixed categorical result. It separates malformed or resource-exhausting native parsing from the parent process. It accepts bytes, never a filename, URL, executable path or provider callback. It returns no decoded media, text or metadata.

## Example

```python
import io
import wave
from openmed.multimodal.decoder_guard import (
    DecoderKind, DecoderLimits, DecodeStatus, probe_media,
)

encoded = io.BytesIO()
with wave.open(encoded, "wb") as stream:
    stream.setnchannels(1)
    stream.setsampwidth(2)
    stream.setframerate(8000)
    stream.writeframes(b"\0" * 32)
report = probe_media(
    encoded.getvalue(),
    DecoderKind.AUDIO,
    limits=DecoderLimits(input_bytes=1024, decoded_bytes=1024),
)
assert report.status in (DecodeStatus.ACCEPTED, DecodeStatus.ISOLATION)
assert not report.to_dict()["clinical_readiness"]
```

`probe_completed` means this bounded probe finished under the controls. It does not mean the file is sanitized, all content is understood, privacy is verified or clinical use is approved. Existing ingestion providers are not automatically routed through this API. Keep the shared [preflight checks](../multimodal/preflight-cli.md), modality-specific safety checks and privacy gates in the calling workflow.

## Enforced worker controls

The supported ABIs are native Linux x86-64 and AArch64. The child must successfully install and verify its resource limits before reading/parsing media. Unsupported platforms or unavailable controls return `isolation_unavailable`; there is no in-process fallback.

- Hard address-space ceilings include the interpreter, native libraries and decoded allocations. They are not a measurement of model-exclusive RSS or GPU memory.
- A CPU soft limit terminates excess compute; the hard limit is one second later. The parent also enforces the declared wall deadline, cancellation and process-group termination, and reaps its worker.
- Regular-file output size and core-file limits are zero. Dumpability is disabled so a piped system core handler cannot receive the worker's media memory.
- A native-ABI syscall filter denies everything except a small set needed for reading installed code/data, memory management, clocks, signals and pipe results. Writable opens, file mutations, new sockets, process creation, execution, resource-limit changes and alternative syscall ABIs are denied.
- Input travels through a pipe; the only worker handshake is a fixed byte after controls are installed. Decoder stdout/stderr go to the null device, and native exceptions become fixed exit categories. No temporary media file or decoded output is created by the API.
- The child starts with isolated Python settings, no bytecode writes, an empty credential environment, disabled user-cache locations and one-thread native-library settings. The installed interpreter and libraries remain trusted software.

The wall deadline starts after OS process creation and covers worker initialization and input feeding. OS process creation itself is not interruptible through this interface; scheduling and kernel teardown are not real-time guarantees. Keep the process inside a separately governed deployment sandbox/container with appropriate host limits.

Linux describes syscall filtering as a building block rather than a complete sandbox. This API does not prevent every information flow or replace OS isolation, filesystem namespaces, an LSM, dependency patching or malware scanning. It can read installed dependencies and accessible host files; use a deployment sandbox to restrict that read surface. See the [kernel seccomp documentation](https://docs.kernel.org/userspace-api/seccomp_filter.html) and [Python resource-limit documentation](https://docs.python.org/3/library/resource.html).

## Closed decoder choices

| Kind | Local implementation | Scope |
| --- | --- | --- |
| PDF | Optional pikepdf | Parse page/object structure, validate syntax and read bounded streams; no rendering or text extraction |
| TIFF | Optional Pillow | Validate directory graph, bound dimensions/frame count and actually load raster pixels |
| WSI TIFF | Optional Pillow | Same bounded TIFF probe; not a full-slide clinical reader or replacement for the tile planner |
| DICOM | Optional pydicom and NumPy | Native implicit/explicit little-endian pixels; compressed transfer syntaxes are refused |
| WAV | Python standard library | Uncompressed PCM header/data consistency and decoded-byte ceiling |
| Video | Optional PyAV | One video stream, bounded geometry/frame count and actual RGB decoding |

Backends are never downloaded or installed by the API. Callers supply compatible optional packages in a trusted local interpreter, review their dependency/codec licenses and maintain their security updates. They are not added to OpenMed's core dependencies. The video backend runs out of process; no codec libraries or media are bundled in this change.

The TIFF directory walk rejects cycles, repeated directory references, out-of-range tables and excessive metadata nodes before native decoding. These conservative refusals may reject unusual but valid shared-directory files. Native parser recovery, codec variations and library versions can affect other categories; qualify and pin the environment used for release evidence. Arbitrary compressed DICOM, proprietary WSI formats and remote video playlists are not qualified by the included controls.

## Outcomes and evidence

Reports contain the declared kind, a fixed status, input byte count, input/limit digests, whether controls were applied, and whether the owned worker was reaped. Oversized rejected inputs are not hashed in full and have a null source digest. Cancellation errors cancel safely. A crash, timeout, resource limit, missing backend or unavailable isolation never becomes acceptance. Reports do not echo filenames, metadata, bytes or parser exception messages.

The API does not erase caller-owned immutable bytes, external copies, swap or host snapshots. It owns and closes only its pipes and worker. A completed probe is not an all-data destruction receipt.

## Reproducible development controls

`tests/fixtures/multimodal/hostile_media` generates minimal synthetic controls for all six kinds. The integration suite exercises truncation, seeded malformed headers/metadata, cyclic containers, oversized geometry, compressed expansion, real crashes, timeouts, CPU/memory exhaustion and recovery of subsequent parent calls. No patient data or executable exploit corpus is committed.

Run `python -m pytest tests/integration/multimodal/test_hostile_media.py -q`. Linux qualification should set `OPENMED_REQUIRE_DECODER_BACKENDS=1` after separately installing all optional backends. macOS/Windows refuse worker execution and skip the Linux-only cases; those skips are not evidence that native containment passed. The tests verify categorical outcomes and actual process recovery without adding fragile wall-clock performance assertions.
