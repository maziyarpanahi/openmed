# Export and install Agent Skill bundles

OpenMed skills can be exported as a reproducible archive without cloning or
downloading anything during the export. The exporter reads the local
`skills/` directory, `skills/compatibility.json`, and the versioned topical-pack
manifest when present, then writes a bundle and an adjacent manifest. Duplicate
pack declarations must match the canonical manifest or export fails closed.

## Create a bundle

Export the complete local catalog as a ZIP archive:

```bash
python scripts/skills/export.py \
  --output openmed-skills.zip
```

Select individual skills, or use a pack declared in
[`skills/compatibility.json`](https://github.com/maziyarpanahi/openmed/blob/master/skills/compatibility.json):

```bash
python scripts/skills/export.py \
  --pack privacy \
  --host codex \
  --output openmed-privacy.zip

python scripts/skills/export.py \
  --skill deidentifying-clinical-text \
  --skill extracting-clinical-entities \
  --format tar.gz \
  --output openmed-clinical-skills.tar.gz
```

When no source revision is supplied, the exporter records the local Git
`HEAD`. This is a local lookup only. If the source was unpacked without Git
metadata, the manifest records `unknown`. Exporting never downloads models,
contacts a host, or makes a mandatory network call.

The archive contains this stable layout:

```text
manifest.json
compatibility.json
skill-packs.json  # present when the canonical pack manifest is available
skills/<skill-name>/SKILL.md
skills/<skill-name>/references/...
```

`manifest.json` records the selected skills and packs, the declared host
capabilities, the source revision, and a SHA-256 and byte size for every
source file. The same manifest is written beside the archive as
`<archive-name>.manifest.json`. Archive metadata, member order, and JSON
formatting are fixed so exporting the same source revision twice produces the
same bytes. Files below a skill's `scripts/` directory receive a deterministic
executable mode; all other archive members are non-executable.

## Install a bundle

The host paths are data in `compatibility.json`; the current declarations are:

| Host | Skills directory |
| --- | --- |
| Claude Code | `~/.claude/skills/` |
| OpenAI Codex | `~/.codex/skills/` |
| OpenCode | `~/.config/opencode/skills/` |
| Shared convention | `~/.agents/skills/` |

Inspect the manifest before installing. Use an archive tool's no-overwrite
mode to preserve existing skills:

```bash
bundle_dir="$(mktemp -d)"
unzip -n openmed-privacy.zip -d "$bundle_dir"
mkdir -p ~/.codex/skills
cp -R -n "$bundle_dir/skills/." ~/.codex/skills/
rm -rf "$bundle_dir"
```

The exporter refuses to replace an existing archive or manifest. Pass
`--force` only when replacing both outputs is intentional; it stages each file,
uses an atomic rename per output, and restores both prior files if finalization
fails. Output targets must be regular non-symlink files outside the source
`skills/` tree, and a requested format cannot conflict with the archive suffix.
Skill source symlinks, hidden paths, non-portable names, unsupported root files,
and case-folding path collisions are rejected rather than copied. In a Git
checkout, selected skill files must also be tracked, which prevents a local
scratch file from entering a bundle accidentally. Source archives without Git
metadata still use the same bounded skill-directory layout and size limits.
Compatibility and pack metadata use strict versioned schemas: duplicate JSON
keys, unknown fields, unsafe host paths, and unsupported capabilities fail
closed instead of being copied into the archive.

Keep real patient or customer data out of skill files, paths, logs, and
manifests. Committed examples and tests use synthetic data only.
