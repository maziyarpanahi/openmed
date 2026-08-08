# Encrypted surrogate mappings

Reversible redaction mappings are re-identification material. Treat them as
secrets and keep them separate from the de-identified text. OpenMed provides a
small local storage boundary in
`openmed.core.surrogate_vault_crypto.SurrogateVaultCrypto` for applications
that need to retain a mapping.

## Key handling

The caller must provide exactly 32 bytes of key material. The helper does not
generate, persist, print, or recover a key. Generate and load that material
from the application's approved secret-management path, for example:

```python
import os

from openmed.core.surrogate_vault_crypto import SurrogateVaultCrypto

key = bytes.fromhex(os.environ["OPENMED_SURROGATE_VAULT_KEY_HEX"])
vault = SurrogateVaultCrypto(key)
```

The key is never written into the encrypted envelope. Do not put it in source
control, a report, a log message, a filename, or the same location as the
encrypted mapping. A missing, non-bytes, or incorrectly sized key fails before
any mapping operation.

The implementation uses deterministic AES-256-SIV authenticated encryption.
Canonical serialization makes equivalent mappings produce the same envelope,
which is useful for local reproducibility. Determinism reveals only whether two
complete encrypted mappings are equal; it does not reveal mapping keys or
values. Use a fresh key when this equality signal is not acceptable.

Install the optional cryptography dependency before using this feature:

```bash
pip install "openmed[integrity]"
```

## Encrypt and restore a mapping

The mapping remains in process memory while it is encrypted. Only the JSON
envelope is written to disk, and file writes use an owner-only temporary file
followed by an atomic replacement:

```python
from pathlib import Path

mapping = {
    "<SURROGATE_A>": "<SYNTHETIC_SOURCE_A>",
    "<SURROGATE_B>": "<SYNTHETIC_SOURCE_B>",
}

path = Path("private/surrogate-mapping.json")
vault.write(path, mapping)
restored = vault.read(path)
assert restored == mapping
```

For one-shot operations, use `save_mapping(path, mapping, key)` and
`load_mapping(path, key)`. The serialized envelope contains a schema version,
the algorithm name, and base64 ciphertext. It contains no plaintext mapping
values, source text, key material, or network metadata.

## Operational boundaries

- Keep the encrypted mapping in a separate access-controlled location from
  redacted output.
- Do not log the mapping, key, encrypted payload, or exception objects from an
  untrusted serialization boundary.
- Re-identification remains an application authorization decision; encryption
  does not provide compliance certification or make a reversible workflow
  anonymous.
- The helper performs no network access and has no telemetry or remote key
  lookup. Missing or tampered files fail closed with high-level errors that do
  not include mapping contents.
