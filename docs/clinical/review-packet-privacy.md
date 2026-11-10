# Final privacy scan for clinical review packets

`openmed.clinical.review_packet_privacy` applies a configured leakage detector
to the exact rendered packet immediately before export or local persistence.
The detector is dependency-injected, so the gate performs no model download or
network call.

```python
from openmed.clinical.review_packet_privacy import persist_review_packet

report = persist_review_packet(
    "review-packet.json",
    rendered_packet,
    configured_local_detector,
)
```

Detector results can expose `findings`, `entities`, or `pii_entities`. Each item
must provide character `start` and `end` offsets plus an entity class using
`entity_class`, `canonical_label`, `label`, or `entity_type`. A detector can
mark a finding critical with `critical=True` or a `severity`/`risk_level` of
`critical`, `high`, or `blocking`. Otherwise OpenMed's direct-identifier and
high-risk label set determines whether the finding blocks output; callers can
provide an explicit `critical_entity_classes` set.

Reports contain only schema metadata, counts, content hashes, and finding entity
classes, offsets, and hashes. Detector-provided text is ignored. On a critical
finding, `ReviewPacketPrivacyBlocked` is raised before the exporter callback is
called or the destination path is opened. Detector failures and malformed
offsets also fail closed with value-free errors.

This gate is a final local safety check for a human-review artifact. It is not a
compliance certification or an autonomous clinical decision mechanism.
