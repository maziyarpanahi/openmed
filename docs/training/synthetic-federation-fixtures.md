# Synthetic federation metadata fixtures

`tests.fixtures.private_learning.generate_private_learning_fixture(seed)` returns
fresh bounded capability, schedule, update, dropout and aggregate metadata for
offline test doubles. `render_private_learning_fixture(seed)` provides deterministic
ASCII JSON. The fixture schema is `openmed.synthetic.federation_fixture.v1`.

Seeds are unsigned 64-bit integers and cohorts contain 1–128 entries. SHA-256 domain
separation produces update/model references and a seed commitment; these are
synthetic references, not signatures or hashes of real model artifacts. No wall
clock, model library, network call, training data or institutional identity is used.
Capabilities are coarse synthetic declarations, not a production enrollment
protocol or evidence of measured device support. Schedule, update and aggregate
metadata reuse OpenMed's existing builders and validators. Aggregate values are
suppressed by the existing minimum-group policy for small synthetic cohorts.

Pass exactly one named `variant`: `capability`, `schedule`, `update`, `dropout`, or
`aggregate`. Each changes only its corresponding component, and remains explicitly
marked as synthetic and corrupted. Valid fixtures do not claim successful training,
privacy accounting, secure aggregation or clinical qualification. Test consumers
must reject corrupted fixtures for normal-run inputs.

Run `pytest tests/unit/training/test_private_learning_fixtures.py -q` offline.
