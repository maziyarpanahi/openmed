# Offline and air-gapped installation

OpenMed can package the Python wheels, a small PII model, documentation, and a
portable Hugging Face cache into one integrity-checked installation kit. Build
the kit on a connected machine with the same target architecture and Python
version as the clinic server, then transfer it by USB or another approved
offline medium.

The default bundle includes OpenMed's core package, the `cli` and required
model-runtime dependencies, and
`OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1`. The builder refuses to create
the default bundle if its exact payload exceeds 900,000,000 bytes.

## Build on a connected machine

Install OpenMed with its Hugging Face support, then create a directory bundle:

```console
python -m pip install "openmed[cli,hf]"
openmed airgap bundle ./openmed-airgap
```

Use `--archive` or a `.tar.gz` output path for a single transfer file:

```console
openmed airgap bundle ./openmed-airgap.tar.gz --archive
```

For a 64-bit Raspberry Pi-class Linux target running CPython 3.11, select the
manylinux platform and Python version explicitly:

```console
openmed airgap bundle ./openmed-pi-cp311 \
  --platform manylinux2014_aarch64 \
  --python-version cp311
```

`linux/aarch64` is accepted as an alias for `manylinux2014_aarch64`. Cross-target
downloads are binary-only: the command fails instead of silently including a
source distribution that would need build tools or internet access at the
clinic.

The `cli` extra is selected by default. Repeat `--extra` to choose another
supported profile:

```console
openmed airgap bundle ./openmed-onnx-kit \
  --extra cli \
  --extra onnx \
  --model OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1
```

Repeat `--model` to include multiple snapshots. The model runtime is always
included so the bundled PII model can run after installation.

## Verify before and after transfer

Every payload file is listed in `bundle-manifest.json` with its exact byte size
and SHA-256 digest. The manifest also records the target tags, package spec,
model snapshot paths, artifact count, and exact total payload size. The manifest
itself is excluded because a file cannot contain its own stable digest.

Verify either a directory or archive:

```console
openmed airgap verify ./openmed-airgap
openmed airgap verify ./openmed-airgap.tar.gz
```

The command exits non-zero for a changed, missing, unexpected, unreadable, or
duplicate file. Verify once on the connected machine and again after copying the
kit to local storage on the offline machine. Do not install directly from a USB
drive that has not passed verification.

## Install on the offline machine

Extract an archive if necessary, enter the bundle directory, and run:

```console
./install.sh
```

The installer runs pip with only the local wheelhouse:

```console
python3 -m pip install --no-index --find-links wheels/ "openmed[cli,hf]==<version>"
```

It then copies the bundled model repositories into `HF_HUB_CACHE`,
`$HF_HOME/hub`, or the standard `$HOME/.cache/huggingface/hub` location. Set
`PYTHON=/path/to/python` when a different interpreter should receive the
installation. Set `HF_HOME` or `HF_HUB_CACHE` before running the script when the
clinic uses a shared or non-default Hugging Face cache. The installer also links
the cached model repositories into OpenMed's default `$HOME/.cache/openmed`
location without duplicating model blobs. If the OpenMed configuration already
uses another `cache_dir`, set `OPENMED_AIRGAP_CACHE` to that directory while
running `install.sh`.

## Confirm local-only operation

Keep offline mode enabled whenever the installation must not open outbound
connections:

```console
export OPENMED_OFFLINE=1
openmed doctor
printf '%s\n' 'Patient Jane Example called 555-0100 on 2025-01-20.' | \
  openmed deid --model OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1
```

Use only synthetic text for this first check. `OPENMED_OFFLINE=1` enables the
Hugging Face and Transformers offline flags, forces cache-only model loading,
and activates OpenMed's socket guard around local-only inference.

!!! warning

    An air-gapped network does not make a model clinically validated. Review
    local policies, test direct-identifier recall with synthetic fixtures, and
    never place raw PHI in bundle manifests, transfer logs, or support tickets.

## Rehearse upgrade, rollback, and recovery offline

A kit can also carry everything an offline site needs to move between
releases. Supply the inputs as directories you already hold; nothing is
downloaded or substituted for them:

```console
openmed airgap bundle ./openmed-lifecycle \
  --upgrade-wheels ./next-release-wheels \
  --rollback-wheels ./previous-release-wheels \
  --migrations ./migrations \
  --recovery ./recovery-assets \
  --valid-until 2027-06-30T00:00:00Z
```

| Input | Bundle path | Purpose |
| --- | --- | --- |
| `--upgrade-wheels` | `lifecycle/upgrade/` | The next release's complete wheel/sdist set. |
| `--rollback-wheels` | `lifecycle/rollback/` | The previous release's artifacts, so a failed upgrade can be undone after an in-place install. |
| `--migrations` | `lifecycle/migrations/` | Ordered `NNNN-<name>` migration assets for the upgrade edge. |
| `--recovery` | `lifecycle/recovery/` | Recovery tools and synthetic backup assets for the restore phase. |
| `--upgrade-from` | manifest only | Release versions the upgrade supports (default: the bundled release). |
| `--valid-from`, `--valid-until` | manifest only | Validity window; an expired kit is refused. |

Every file is still listed with its size and SHA-256 digest. The manifest's
versioned `lifecycle` section adds, per set, the declared version, its files,
and a set digest, plus the compatibility window, the migration chain, and the
validity window. The build refuses a kit whose upgrade is not newer than the
release, whose rollback set is not the bundled release, whose upgrade window
excludes the release, whose migrations are out of order, or which is already
expired.

Rehearse the whole lifecycle before shipping the kit:

```console
openmed airgap rehearse ./openmed-lifecycle \
  --key-file ./rehearsal.key \
  --output rehearsal.json \
  --markdown rehearsal.md
openmed airgap verify-report rehearsal.json \
  --key-file ./rehearsal.key --require-signature
```

Every phase runs inside the OS-enforced no-egress boundary of
`scripts/security/no_egress_gate.py` (see
[Egress policy](serving/egress-policy.md)): a Linux user + network namespace, or
with `--boundary container` a local image (`--container-image` or
`OPENMED_NO_EGRESS_IMAGE`, never pulled) started with `--network none`. The
gate's IPv4, IPv6, TCP, UDP, DNS, child-process, and raw-libc negative controls
run before and after the phases and must all be blocked. The command needs a
source checkout, because the gate and its in-boundary entry point live under
`scripts/`.

| Phase | What must hold |
| --- | --- |
| `preflight` | Every digest, set membership, version order, upgrade window, rollback target, migration order, target interpreter, and the validity window check out before anything is installed. |
| `install` | A clean install of the release into a fresh virtual environment with `pip install --no-index --no-deps --require-hashes`. |
| `smoke` | The active release imports, its installed files match `RECORD`, and `pip check` passes. |
| `upgrade` | The first upgrade attempt is interrupted at `--inject-fault` (`before_activate`, `after_activate`, or `after_checkpoint`); recovery must land on exactly one declared state and the resumed upgrade must commit exactly once. |
| `injected_failure` | A corrupted installed file is detected and admission stays blocked. |
| `rollback` | The previous release reinstalls from the rollback assets and matches the declared rollback target. |
| `restore` | Backup restore and credential reload hooks. |
| `continuity` | The transaction ledger, then model activation, audit continuity, and durable work hooks; admission resumes only when all of them pass. |

Each transition writes a journal entry, builds and smoke-tests the new release
in its own environment, moves the active pointer, and writes the last-green
checkpoint last, so an interruption at any point recovers to the checkpoint.

Recovery features plug in through `--hook NAME=module:callable` (`migration`,
`backup_restore`, `credential_reload`, `model_activation`, `audit_continuity`,
`durable_work_resume`). A hook receives verified paths and returns only a
status, a controlled reason code, and an optional digest. A feature that is not
present reports `not_available`; that is never a pass.

The report is bounded and value-free: phase and check outcomes, controlled
reason codes, versions, counts, and digests only. It carries a `repro_hash`
over stable evidence and an `integrity_hash`, and an HMAC-SHA256 signature when
you supply `--key-file`. The key stays with you; it is never bundled or
written into the report.

| Exit code | Status | Meaning |
| ---: | --- | --- |
| 0 | `passed` | Every phase passed inside an isolated boundary. |
| 1 | `failed` | A phase, a network control, or the in-boundary runner failed. |
| 2 | usage | Invalid arguments, an unreadable bundle, or an unusable key. |
| 3 | `skipped` | No supported boundary on this host (for example macOS without a container image). |
| 4 | `incomplete` | Every executed phase passed, but at least one recovery feature is `not_available`. |

## Troubleshooting

- If `pip download` reports no compatible wheel, confirm the manylinux platform,
  Python version, implementation, and ABI supported by the clinic server.
- If installation reports a missing distribution, rebuild the complete bundle;
  do not download individual wheels on the offline machine.
- If a model is not found in offline mode, confirm that the cache variables used
  during installation match those present when OpenMed runs.
- If verification fails, discard or recopy the kit. Do not edit the manifest to
  match an unexplained file change.
- If `openmed airgap rehearse` exits 3, the host has no supported no-egress
  boundary. Run it on Linux with unprivileged user namespaces enabled, or pass
  a local `--container-image`. A skipped rehearsal is not evidence.
