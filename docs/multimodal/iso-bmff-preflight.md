# HEIF and M4A header preflight

`openmed.multimodal.iso_bmff.preflight_iso_bmff` and OpenMedKit's
`ISOBMFFPreflight.read` inspect caller-owned complete local container bytes.
They return `decoder_required`, never decoded pixels, audio samples, EXIF values,
item names, paths or codec configuration. No decoder, provider, network call,
model asset or new dependency is involved.

```python
from openmed.multimodal.iso_bmff import preflight_iso_bmff

# source is caller-owned bytes from a local HEIF or M4A capture.
report = preflight_iso_bmff(source, max_header_bytes=1_048_576)
assert report.status == "decoder_required"
```

```swift
let report = try ISOBMFFPreflight.read(localContainerData)
```

## Declared metadata

HEIF geometry comes from the primary `pitm` item's associated `ispe` property,
not the first thumbnail or grid tile. `item_count` includes all `infe` entries;
thumbnail and auxiliary counts are unique source image items with `thmb` and
`auxl` references. `burst_count` counts unique image items explicitly listed in
`brst` entity groups. Roles may overlap. EXIF presence comes from `Exif` item
types; XMP presence from `mime` items with `application/rdf+xml` content type.
Names and metadata values are not decoded or returned. These are presence
declarations, not evidence that payloads exist or are safe.

HEIF supports `pitm`/`iref` versions 0–1, `iinf` 0–1, `infe` 2–3 and `ipma` 0–1,
including wide identifiers and property indices. Timed image sequence tracks
are detected by brand but refused by the item reader; counts never pretend
to inventory unexamined tracks. Image transforms, grid decoding, EXIF scrubbing,
encryption and codec licensing are outside this header-only API.

M4A requires exactly one sound track and one version-0 audio sample description.
`mp4a` reports `mpeg4_audio`, without claiming AAC qualification; `alac` reports
`apple_lossless`. Channels and the 16.16 sample rate are declared sample-entry
values, not measured decoder output. Duration is `mdhd` media duration divided
by its timescale (versions 0–1), not presentation time. Edit lists, fragmented
movies, encryption, unknown codecs and multiple audio tracks are refused.
Audio results carry `non_diagnostic` and `reviewer_confirmation_required=True`;
hosts must require confirmation before consequential use. Preflight does not
interpret audio or trigger clinical decisions.

Detection checks the complete first `ftyp` box within the existing 132-byte
prefix bound, including extended-size boxes. HEIC, generic HEIF and their
sequence brands return their registered image MIME types; M4A returns
`audio/mp4`. Ambiguous image/audio brands, AVIF and generic `isom` alone return
unknown. No filename or unbounded brand search is used.

## Bounds and refusals

Defaults on both platforms: absolute header offset 1 MiB, depth 16, 4096 boxes,
declared box size 4 GiB and primary-image area 100 million pixels. Limits are
positive integers; maximum configurable depth is 64. Memory does not scale with
declared box sizes. Payloads of media and unknown boxes are skipped by offset.
Every visited box envelope is checked against its parent and complete source
length. Terminal size-zero and extended-size boxes are supported.

The byte bound includes skipped space. A `moov` after a large `mdat`, or `meta`
after padding beyond the bound, raises `bmff_metadata_out_of_bounds`; the reader
does not seek beyond the bound looking for metadata. A terminal media payload
may extend beyond the bound because it is not read. The caller supplies complete
container data; this API does not load files or stream media itself. Caller-owned
source buffers remain the caller's privacy responsibility.

`BmffPreflightError.category` and its string are identical controlled codes.
Swift's `BMFFPreflightError.rawValue` uses the same vocabulary:

- `bmff_truncated`, `bmff_box_size_invalid`, `bmff_size_overflow`
- `bmff_metadata_out_of_bounds`, `bmff_declared_size_limit`
- `bmff_depth_limit`, `bmff_box_count_limit`, `bmff_pixel_limit`
- `bmff_metadata_missing_or_duplicate`, `bmff_metadata_invalid`
- `bmff_signature_invalid`, `bmff_layout_unsupported`
- `bmff_item_cycle`, `bmff_codec_unsupported`, `bmff_audio_track_ambiguous`

Item references are validated against the item inventory and checked for cycles
without recursive graph traversal. Unsupported layouts fail closed. Python
invalid arguments raise constant-message `TypeError`/`ValueError`; Swift invalid
limits raise `bmff_limits_invalid`. Reports never certify payload integrity,
metadata removal, decoder availability, provider qualification or clinical safety.

## Synthetic verification

The shared offline fixtures contain hand-derived primary dimensions 4032×3024,
six items (one thumbnail, one auxiliary, two burst-group members), and two-channel
48 kHz audio with media duration 2.5 seconds. Python and Swift consume the same
success and refusal fixtures, including multilingual synthetic private names.
Unit tests also sweep truncated prefixes and test exact byte boundaries.

```bash
.venv/bin/python -m pytest tests/unit/multimodal/test_iso_bmff.py tests/integration/test_iso_bmff_preflight.py -q
swift test --package-path swift/OpenMedKit --filter ISOBMFFPreflightTests
```

Format references: [HEIF technical information](https://nokiatech.github.io/heif/technical.html),
[registered burst entity groups](https://mp4ra.org/registered-types/entity-groups),
and [Apple sound sample description](https://developer.apple.com/documentation/quicktime-file-format/sound_sample_description_version_0).
