# Export and install Agent Skill bundles

OpenMed skills can be exported as a reproducible archive without cloning or
downloading anything during the export. The exporter reads the local
`skills/` directory and `skills/compatibility.json`, then writes a bundle and
an adjacent manifest.

## Create a bundle

Export the complete local catalog as a ZIP archive:

```bash
python scripts/skills/export.py \
  --output openmed-skills.zip \
  --source-revision "$(git rev-parse HEAD)"
```

Select individual skills, or use a pack declared in
[`skills/compatibility.json`](../../skills/compatibility.json):

```bash
python scripts/skills/export.py \
  --pack starter \
  --host codex \
  --output openmed-starter.zip

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
skills/<skill-name>/SKILL.md
skills/<skill-name>/references/...
```

`manifest.json` records the selected skills and packs, the declared host
capabilities, the source revision, and a SHA-256 and byte size for every
source file. The same manifest is written beside the archive as
`<archive-name>.manifest.json`. Archive metadata, member order, and JSON
formatting are fixed so exporting the same source revision twice produces the
same bytes.

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
mkdir -p /tmp/openmed-skill-bundle
unzip -n openmed-starter.zip -d /tmp/openmed-skill-bundle
mkdir -p ~/.codex/skills
cp -R -n /tmp/openmed-skill-bundle/skills/. ~/.codex/skills/
```

The exporter refuses to replace an existing archive or manifest. Pass
`--force` only when replacing both outputs is intentional; it stages each
output and finalizes it with an atomic rename. Skill files and symlinks outside
the selected skill folders are never followed into a bundle.

Keep real patient or customer data out of skill files, paths, logs, and
manifests. Committed examples and tests use synthetic data only.
