# ICD-10-CM ↔ SNOMED CT code crosswalk

`openmed.clinical.grounding.crosswalk` provides a deterministic, offline
crosswalk over a UMLS mapping that the caller supplies from a licensed local
environment. OpenMed does not bundle UMLS or SNOMED CT data, download either
terminology, or fall back to a network service.

The source must contain local `MRCONSO` and `MRMAP` files. A release directory
with the conventional `MRCONSO.RRF` and `MRMAP.RRF` names is the simplest
configuration:

```python
from openmed.clinical.grounding.crosswalk import UMLSCrosswalk

crosswalk = UMLSCrosswalk("/licensed/local/umls-release")
candidates = crosswalk.crosswalk("E11.9", "ICD-10-CM", "SNOMED CT")

for candidate in candidates:
    print(candidate.target_code, candidate.provenance["map_rule"])
```

Separate files can be supplied when a local projection keeps them in different
directories:

```python
from openmed.clinical.grounding.crosswalk import crosswalk

candidates = crosswalk(
    "44054006",
    "SNOMEDCT",
    "ICD10CM",
    mrconso_path="/licensed/local/projection/MRCONSO.RRF",
    mrmap_path="/licensed/local/projection/MRMAP.RRF",
)
```

Both directions are supported. The result is a tuple of
`CrosswalkCandidate` objects. Each candidate exposes `target_code`, `code`,
`source_code`, `source_system`, `target_system`, and `map_rule`. Its
`provenance` contains the map rule, optional map advice, optional numeric map
priority, source and target CUIs when available, and the marker
`data_source="user-supplied-local"`.

One-to-many mappings are ordered deterministically. Lower numeric
`map_priority` values come first; ties are resolved by target code, map rule,
map advice, and source row. Duplicate target codes are returned once. A source
that does not provide a priority remains deterministic through those tie-break
fields.

Small headered CSV, TSV, and JSONL projections are also accepted. A mapping
projection can use these fields:

```text
source_system,source_code,target_system,target_code,map_rule,map_priority,map_advice
ICD10CM,E11.9,SNOMEDCT_US,44054006,RULE-EQUIVALENT,10,USE
```

The parser also accepts the standard pipe-delimited UMLS RRF positions. A
projection service may therefore expose only the needed rows and fields while
keeping the licensed release outside the OpenMed package. All file reads occur
locally; the crosswalk does not create sockets or make HTTP requests.

If no source is configured, the engine raises a clear restricted-vocabulary
configuration error. Callers remain responsible for UMLS and SNOMED CT
licensing and for validating the returned mappings. Crosswalk results are
assistive interoperability candidates, not autonomous diagnosis, treatment,
billing, or clinical coding decisions.
