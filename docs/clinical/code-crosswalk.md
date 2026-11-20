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

## Exporting redistributable crosswalks as FHIR ConceptMaps

The bundled redistributable crosswalks (ICD-10-CN, CHPO, and Indic HPO
aliases) and caller-supplied `CrosswalkResource` tables can be handed to a
FHIR server or terminology tool as R4 `ConceptMap` JSON:

```python
import json

from openmed.clinical.grounding.crosswalk import (
    crosswalk_resource_to_conceptmap,
    load_default_crosswalks,
)

resource = load_default_crosswalks()[0]  # the ICD-10-CN starter table
concept_map = crosswalk_resource_to_conceptmap(resource)
print(json.dumps(concept_map, ensure_ascii=False, indent=2))
```

The ConceptMap carries one `group` per source/target system pair, one
`element` per distinct source code, and one `target` per mapped code with
`relatedto` equivalence. The resource's `name` and `version` are copied, its
`license_id` becomes `copyright`, and its `content_hash` becomes the
`identifier` value. Groups, elements, and targets are sorted, so exporting
the same resource twice yields byte-identical JSON.

Group `source` and `target` keep the resource's system labels unless the
registry (`openmed.clinical.grounding.systems.system_uri`) has a URI for
them. Of the systems in the bundled crosswalks only `HPO` has one, so it
becomes `http://human-phenotype-ontology.org`, while `ICD-10-CN`, `CHPO`,
and `OPENMED-INDIC-ALIAS` are exported as labels. WHO `ICD10`/`ICD-10`
labels also stay labels: the registry has no plain ICD-10 URI and maps
`icd10` to the ICD-10-CM URI, which would mislabel those groups. An RxNorm
label or URI stays `http://www.nlm.nih.gov/research/umls/rxnorm`.

Only resources declared redistributable can be exported.
`CrosswalkLicenseError` is raised for everything else: the licensed
`UMLSCrosswalk` engine, a resource not declared redistributable, and any
resource that maps UMLS, SNOMED CT, or CPT spelled as a registry label
(`UMLS`, `SNOMED CT`, `cpt4`), a known system URI (`http://snomed.info/sct`,
a SNOMED edition URI such as `http://snomed.info/sct/900000000000207008`, or
the AMA CPT URI `http://www.ama-assn.org/go/cpt`), or a UMLS-style source
abbreviation (`SNOMEDCT_US`, `SNOMEDCT_UK`, `SNOMEDCT_VET`).
