# Clinical review packet migrations

`openmed.clinical.review_packet_migrations` upgrades persisted review packet
mappings through explicit local schema steps. The current packet schema is
version 2, and versions 1 and 2 are supported.

```python
from openmed.clinical.review_packet_migrations import migrate_review_packet

result = migrate_review_packet(stored_packet)
migrated_packet = result.packet
safe_report = result.report.to_dict()
```

The version 1 to version 2 step adds
`safety.privacy_scan_required = true`. It preserves every existing field and
deep-copies the packet. If adding the requirement would overwrite an existing
incompatible field, migration stops with `LossyReviewPacketMigrationError`.
Backward and unsupported-version migrations are also rejected.

Migration reports contain only source and target versions plus changed field
paths and operations. They contain no packet values, and packet content is
excluded from the result's `repr()`. The implementation is deterministic and
uses no external services.

Applications should migrate in protected local memory, run the final review
packet privacy scan, and persist only after all required safety gates pass.
