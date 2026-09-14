# Schema snapshot compatibility

`openmed.structured.schema_snapshot` provides a local, deterministic check for
structured schema changes. A snapshot contains field metadata only: a stable
field path, its type, and whether the field is optional. It never needs an
example record and never copies example, default, or description values into a
snapshot or report.

## Create and compare snapshots

```python
from openmed.structured import SchemaSnapshot, compare_schema_snapshots

before = SchemaSnapshot(
    version="1.0.0",
    fields={
        "subject.identifier": {"type": "string", "optional": False},
        "encounter.note": {"type": "string", "optional": True},
    },
)
after = SchemaSnapshot(
    version="1.1.0",
    fields={
        "subject.identifier": {"type": "string", "optional": False},
        "encounter.note": {"type": "string", "optional": True},
        "encounter.count": {"type": "integer", "optional": True},
    },
)

report = compare_schema_snapshots(before, after)
assert report.compatible
assert [change.path for change in report.additions] == ["encounter.count"]
```

Field mappings are normalized in path order, so equivalent snapshots produce
the same `to_dict()` and `to_json()` output regardless of input mapping order.
Common type aliases (`int`, `float`, `bool`, `list`, and `dict`) are normalized
to `integer`, `number`, `boolean`, `array`, and `object`. A type union may be a
pipe-delimited string or a sequence; including `null` marks the field optional.
An explicit `optional=False`, `nullable=False`, or `required=True` therefore
cannot be combined with a nullable type.

Snapshot inputs are bounded before metadata is inspected: a snapshot accepts
at most 10,000 fields, a type union accepts at most 128 members, field paths
are limited to 1,024 characters, and versions are limited to 64 characters.
Unreadable caller mappings and sequences fail with value-free errors rather
than forwarding their exception text.

## Compatibility rules

Rules version `1` accepts these changes without a major release:

- adding an optional field;
- widening a field type, such as `integer` to `number`;
- making a required field optional.

Required additions, field removals, narrowing or otherwise changing a type,
and making an optional field required are reported as breaking changes. The
transition is `compatible` only when the version does not regress and any
breaking change is accompanied by a strictly greater major version. A major
release therefore satisfies the version gate, while
`report.has_breaking_changes` remains true for release notes and review.

The report separates `additions`, `removals`, and `changes`, and exposes
`incompatible_changes` with only the breaking field records. Every record
contains schema metadata and rule reasons; example payload values are not
serialized. `report.to_json()` emits canonical JSON for a local release
artifact and performs no network access. Public change and report objects
accept only the fixed rule-reason and violation vocabulary, so caller-provided
strings cannot be injected into release evidence through those fields.
