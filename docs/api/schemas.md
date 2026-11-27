# Published JSON Schemas

OpenMed ships its JSON Schemas inside the `openmed` package. The documentation
site also publishes a copy of each packaged schema whose `$id` is on
`openmed.life`, at the URL in its `$id`. That lets external validators and
client generators resolve a schema from its identifier.

## URL layout

A schema whose `$id` is `https://openmed.life/schemas/<path>` is served at
exactly that URL. The file there is byte-identical to the packaged file. For
example:

| `$id` | Packaged file |
| --- | --- |
| `https://openmed.life/schemas/config.schema.json` | `openmed/core/config.schema.json` |
| `https://openmed.life/schemas/compliance/control_evidence.schema.json` | `openmed/compliance/schemas/control_evidence.schema.json` |
| `https://openmed.life/schemas/clinical_evidence_packet.schema.json` | `openmed/core/schemas/json/clinical_evidence_packet.schema.json` |

`https://openmed.life/schemas/index.json` lists every packaged schema exactly
once:

| Field | Meaning |
| --- | --- |
| `$id` | The schema's declared identifier. |
| `schema_version` | The declared version: a top-level `schema_version`, else a `const` or the highest `enum` value of the `schema_version` property, else a trailing semver segment in `$id`; `null` when the schema declares none. |
| `sha256` | SHA-256 of the packaged file's bytes. |
| `source` | The schema's path inside the package. |
| `published` | `true` when a copy is served at the `$id` URL. |
| `path` | Path of the served copy below the site root, or `null`. |
| `reason` | `id_host_not_served` for an unpublished schema, otherwise `null`. |

The index also records the served `host`, its own format version
(`schema_version: 1`) and a `summary` with the published, unpublished and total
counts. To check a downloaded copy against your installed package, compare its
SHA-256 with the `sha256` field of its entry.

## Schemas that are not published yet

Many packaged schemas still declare an `$id` on `https://openmed.dev/`, a host
the documentation site does not serve. They appear in the index with
`published: false` and `reason: "id_host_not_served"`, and no copy is staged.
They need an `$id` on `openmed.life` before they can be published.

## Staging rules

`scripts/docs/stage_pages.py`, run by `make docs-build` and by the Pages
workflow, copies the schemas after the marketing site and writes the index.
Staging fails when:

- a schema has no `$id`, or its `$id` is not an `https` URL;
- the `$id` host is neither `openmed.life` nor a host listed as not served yet
  (`openmed.dev`);
- the `$id` carries credentials, a port, a query or a fragment, or does not
  name a `.json` file below `/schemas/`;
- two schemas would publish to the same path, or a schema would replace a file
  that is already staged; or
- after staging, a published copy is missing, differs from the packaged bytes,
  or the index does not list every packaged schema once.

Error messages name only the schema's package path, never the `$id` value or
the schema content.

## OpenMed never fetches `$id` URLs

The published copies are for external tools. OpenMed does not download them:

- `openmed.core.schemas.load_schema()` and `load_schema_bundle()` read the
  packaged files through `importlib.resources`;
- configuration validation (`OpenMedConfig.validate()`,
  `OpenMedConfig.from_dict()` and `load_config_from_file()`) reads the bundled
  `config.schema.json` next to `openmed/core/config.py`; and
- packaged schemas contain only local `$ref` pointers (`#/...`), so a validator
  never has to resolve another schema's `$id`.

`tests/unit/core/test_schema_loading_offline.py` blocks sockets, DNS lookups and
`urllib` while it loads every bundled schema, checks each one against its
metaschema, and validates a span record and the default configuration.
