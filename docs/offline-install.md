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

## Troubleshooting

- If `pip download` reports no compatible wheel, confirm the manylinux platform,
  Python version, implementation, and ABI supported by the clinic server.
- If installation reports a missing distribution, rebuild the complete bundle;
  do not download individual wheels on the offline machine.
- If a model is not found in offline mode, confirm that the cache variables used
  during installation match those present when OpenMed runs.
- If verification fails, discard or recopy the kit. Do not edit the manifest to
  match an unexplained file change.
