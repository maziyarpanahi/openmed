# Ordered multimodal batch preflight

`preflight_asset_batch` composes the existing `preflight_asset` contract over
ordered validated manifests and caller-owned local sources. It preserves input
positions rather than applying the inventory `AssetBatch` canonical-ID sort.
It does not decode media, load a model, choose a provider or delete an asset.

```python
import hashlib
from openmed.multimodal.asset_manifest import AssetManifest
from openmed.multimodal.batch_preflight import (
    BatchPreflightAsset, BatchPreflightLimits, preflight_asset_batch,
)

header = b'\x89PNG\r\n\x1a\n'
manifest = AssetManifest(
    'synthetic_image', 'image/png', hashlib.sha256(header).hexdigest(),
    len(header), width=8, height=8,
)
limits = BatchPreflightLimits(
    bytes=1024, pages=10, pixels=10000, frames=10, duration=10, rejections=0,
)
report = preflight_asset_batch([BatchPreflightAsset(manifest, header)], limits=limits)
assert report.outcome == 'accept'
print(report.to_json())
```

Aggregate bytes, pages, pixels, frames, duration and rejected positions are
checked against explicit inclusive ceilings. Integer totals have the existing
manifest/pixel-product bounds; duration uses deterministic `math.fsum`. Overflow
or missing applicable geometry/duration produces a finding and a `None` total,
never an assumed zero. PDF raster geometry is not represented by the existing
preflight contract: PDFs retain its abstention and hold the batch. This does
not newly claim that arbitrary signature fragments are decodable media.

Each asset has exactly one ordered result: `accept`, `abstain`, `error` or
`not_evaluated`. Accepted and abstaining assets retain their actual shared
preflight report without rewriting it. Duplicate IDs and content digests,
aggregate failures and source-read errors produce independent ordered findings.
Source bytes, stream representations and underlying exception messages are
excluded from results; sources stay caller-owned and are never closed.

- `continue` examines subsequent assets admitted by the aggregate byte ceiling,
  retaining all findings and metadata. It does **not** override rejection.
- `hold` stops source reads at the first rejected position and holds the packet.
- `fail_fast` stops source reads at the first rejected position and returns the
  explicit fail-fast outcome.

Aggregate and duplicate metadata checks continue across all submitted manifests
in every mode. Exceeding or overflowing the aggregate byte ceiling prevents
source reads even in continue mode. Any finding prevents `accept`; a continue
report with a rejected asset therefore has outcome `hold`. Later stopped assets
remain `not_evaluated`. At most 10,000 assets are consumed, with one additional
probe to detect an over-limit iterable.

Reports are deterministic JSON containing checked resource counts, policy
ceilings and existing privacy-safe manifest/finding metadata only. Input text,
transcripts, DICOM values, credentials, paths and raw source bytes are excluded.
Preflight acceptance is resource/metadata admission, not clinical validity,
packet completeness or proof that subsequent redaction succeeds.
