# Topical Agent Skill packs

The full `skills/` catalog is useful when an agent needs broad coverage, but
loading every skill can add unnecessary context. The topical pack manifest
provides small, versioned selections for common OpenMed workflows:

| Pack | Focus |
| --- | --- |
| `privacy` | De-identification, privacy policy, audit, and re-identification risk |
| `interoperability` | Document intake, FHIR, and healthcare exchange standards |
| `coding` | Terminology, clinical coding, normalization, and mapping |
| `evaluation` | Synthetic evaluation, model evidence, and leakage gates |
| `research` | Cohorts, longitudinal analysis, biomedical research, and trials |

## Manifest contract

[`skills/packs/manifest.json`](../../skills/packs/manifest.json) is a local,
version-controlled contract. `manifest_version` is the schema version, each
pack has its own semantic `version`, and every entry in `skills` is a stable
skill identifier: the kebab-case directory name containing `SKILL.md`.

A skill may appear in only one topical pack. Skills that are not selected remain
available from the full catalog and are intentionally not copied into a pack.

Each pack declares two budgets:

- `max_skills` limits the number of selected skills.
- `max_bytes` limits the sum of regular files below those source skill
  directories. Symlink targets are not followed while measuring the budget.

The builder rejects missing skill directories, duplicate membership, invalid
identifiers, and budget overruns before it writes output.

## Validate and build locally

The builder uses only the Python standard library and reads the checked-in
manifest and skill folders. It never makes a network request:

```bash
python scripts/skills/build_packs.py --check
python scripts/skills/build_packs.py --output build/skill-packs
```

The default build creates one directory per pack. Each directory contains a
small `pack.json` selection record and a `skills/` directory of relative
symlinks back to the canonical source folders, so the `SKILL.md` content has a
single source of truth:

```text
build/skill-packs/privacy/
├── pack.json
└── skills/
    ├── auditing-deid-leakage -> the source skill directory
    └── deidentifying-clinical-text -> the source skill directory
```

Build one or more packs with repeated `--pack` options:

```bash
python scripts/skills/build_packs.py \
  --pack privacy \
  --pack evaluation \
  --output build/skill-packs
```

On a platform where directory symlinks are unavailable, generate the same
validated metadata and install selection without links:

```bash
python scripts/skills/build_packs.py \
  --pack privacy \
  --selection-only \
  --output build/skill-pack-selections
```

`pack.json` contains the pack identifier and version, selected stable skill
identifiers, declared budgets, and the measured byte size. It contains no
skill body, raw clinical text, or machine-specific absolute path.

The existing `install-skills.sh` command continues to install the complete
catalog. Packs are an opt-in context and installation selection; they do not
change OpenMed runtime behavior or provide a compliance certification.
