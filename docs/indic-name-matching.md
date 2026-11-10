# Transliteration-aware Indic name matching

OpenMed can link Devanagari or other supported Brahmic-script personal names
to their Latin spelling variants before a surrogate-vault key is created. The
feature is opt-in so existing vault behavior remains unchanged.

```python
from openmed import OpenMedConfig, SurrogateVault, deidentify

config = OpenMedConfig(
    transliteration_aware_name_matching=True,
    indic_name_similarity_threshold=0.80,
)
vault = SurrogateVault.in_memory("replace-with-a-local-secret", config=config)

result = deidentify(
    "संजय is also recorded as Sanjay and Sanjai.",
    method="replace",
    lang="hi",
    config=config,
    surrogate_vault=vault,
    consistent=True,
    seed=668,
)
```

The vault stores one encrypted Latin surrogate identity for the canonical
name key. Latin mentions receive the Latin surface; Indic-script mentions
receive a deterministic surrogate rendered in the source script. The
persisted key contains an HMAC-SHA256 digest, the canonical `PERSON` label,
and the `indic` language bucket. Raw names and raw canonical keys are never
written to the vault. The matching backend, version, and threshold are stored
as non-sensitive metadata and authenticated with the vault key so an edited
file cannot silently change identity-linking behavior.

## Stdlib fallback and collision threshold

The default normalizer has no runtime dependency beyond Python. It folds the
major Brahmic scripts to conservative Latin phonetic forms and handles common
romanization differences such as `Sanjay`/`Sanjai`,
`Krishna`/`Krishnaa`, and `Lakshmi`/`Laxmi`.

`indic_name_similarity_threshold` accepts values from `0.5` through `1.0`.
The default `0.80` admits the supported spelling folds while keeping nearby
names such as `Sanjay` and `Sanjana` separate. Raise the threshold when false
joins are more costly than missed joins. The equivalent environment settings
are:

```bash
export OPENMED_TRANSLITERATION_AWARE_NAME_MATCHING=1
export OPENMED_INDIC_NAME_SIMILARITY_THRESHOLD=0.80
```

Surfaces longer than 512 characters bypass phonetic folding and receive a
distinct bounded digest key. This fail-closed path prevents pathological input
from causing expensive fuzzy work or collapsing oversized spans together.

## Optional local transliteration weights

OpenMed does not download or bundle IndicXlit or Aksharantar weights. If an
application already has a local IndicXlit-style model, pass an adapter when
the vault is created:

```python
class LocalTransliterator:
    def to_latin(self, text: str) -> str:
        return local_model.to_latin(text)

    def from_latin(self, text: str, target_script: str) -> str:
        return local_model.from_latin(text, target_script)


vault = SurrogateVault.in_memory(
    "replace-with-a-local-secret",
    config=config,
    transliterator=LocalTransliterator(),
)
```

A callable that accepts a source string and returns Latin text is also valid
for canonicalization; output rendering then uses the stdlib fallback. Keep the
model local for PHI workflows. A file-backed vault created with a user-supplied
transliterator must be reopened with the same adapter and weights so canonical
keys remain reproducible.

The bundled synthetic `indic-name-consistency` evaluation suite checks name
identity reuse, negative-name collisions, code-mixed leakage, script rendering,
and deterministic fallback behavior.
