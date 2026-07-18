# OpenMed onboarding for users in China

[简体中文版](onboarding-china.zh.md)

This guide keeps the first OpenMed install and model download practical on
networks where the default PyPI and Hugging Face endpoints are slow or
unreachable. It covers package mirrors, a Hugging Face model mirror, a
transferable local cache, and the bundled PIPL-oriented policy profile.

The mirrors below are independent third-party services, not OpenMed-operated
or endorsed infrastructure. Check your organization's supply-chain policy,
pin package and model versions, and validate downloaded artifacts before
production use.

## Install OpenMed from a PyPI mirror

For a one-time install through the Tsinghua TUNA mirror:

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple openmed
```

Install the optional Hugging Face dependencies when this machine will download
or inspect model snapshots:

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "openmed[hf]"
```

The `simple` path segment is required. TUNA's
[PyPI mirror instructions](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)
also document the equivalent long service URL.

As an alternative, the Aliyun public mirror supports the same pip interface:

```bash
python -m pip install -i https://mirrors.aliyun.com/pypi/simple/ openmed
```

See the [Aliyun PyPI mirror page](https://developer.aliyun.com/mirror/pypi/)
for its public and ECS-specific endpoints.

### Pin the mirror in `pip.conf`

To make TUNA the default for the current user, let pip update the user-level
configuration rather than editing an unknown system file:

```bash
python -m pip config --user set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip config --user get global.index-url
```

Run `python -m pip config debug` to see the active `pip.conf` location. The
equivalent file content is:

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

Replace the URL with `https://mirrors.aliyun.com/pypi/simple/` to make Aliyun
the default instead. A mirror can lag the canonical index, so pin exact package
versions and investigate unexpected version differences before falling back to
another index.

## Download models through HF-Mirror

Install `openmed[hf]`, then set the mirror endpoint **before** importing
`huggingface_hub`, Transformers, or OpenMed in that process:

=== "Linux and macOS"

    ```bash
    export HF_ENDPOINT=https://hf-mirror.com
    ```

=== "Windows PowerShell"

    ```powershell
    $env:HF_ENDPOINT = "https://hf-mirror.com"
    ```

This is the environment-variable method documented by
[HF-Mirror](https://hf-mirror.com/). It changes where compatible Hugging Face
clients fetch model files; it does not route OpenMed inference to a hosted API.

## Pre-download once, then run from the local cache

The standard Hugging Face cache lives under `~/.cache/huggingface` unless
`HF_HOME` or `HF_HUB_CACHE` selects another location. Use a dedicated `HF_HOME`
when the cache must be archived or copied to an isolated machine.

### 1. Warm the cache on a connected machine

From an OpenMed source checkout, the example downloads only after an explicit
opt-in:

```bash
export HF_HOME="$PWD/openmed-hf-cache"
OPENMED_EXAMPLE_ALLOW_DOWNLOAD=1 \
  python examples/onboarding_china_mirrors.py
```

The example sets `HF_ENDPOINT=https://hf-mirror.com` before lazily importing
the Hub client and caches the default 44M PII model. To pre-download directly
from an installed environment instead:

```bash
export HF_HOME="$PWD/openmed-hf-cache"
export HF_ENDPOINT=https://hf-mirror.com
python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"))'
```

Archive or copy the entire `openmed-hf-cache` directory to the offline host,
preserving its directory structure and symlinks.

### 2. Resolve the cache with all Hub traffic disabled

On the offline host, point `HF_HOME` at the transferred directory and enable
offline mode before starting Python:

```bash
export HF_HOME="$PWD/openmed-hf-cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python examples/onboarding_china_mirrors.py
```

The example also passes `local_files_only=True`. If the snapshot is absent, it
reports a cache miss instead of attempting a network request. The same flags
can wrap an application after its models are cached:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python your_openmed_app.py
```

See the Hugging Face Hub documentation for the
[`HF_HOME`, `HF_HUB_CACHE`, and `HF_HUB_OFFLINE` behavior](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables).

## PIPL-oriented policy quickstart

PIPL compliance depends on purpose, consent or another lawful basis, data
classification, retention, cross-border transfer, access controls, and human
governance. A software policy profile is only one technical control; this
section is not legal advice or a claim of PIPL compliance.

OpenMed ships a `china_pipl` technical profile for sensitive personal
information. It uses the high-recall `strict_no_leak` threshold profile,
requires a safety sweep, replaces direct identifiers, and masks
quasi-identifiers and clinical concepts:

```python
from openmed.core.policy import list_policies, load_policy

policy_name = "china_pipl"
assert policy_name in list_policies()

policy = load_policy(policy_name)
assert policy.name == "china_pipl"
assert policy.threshold_profile == "strict_no_leak"
assert policy.safety_sweep_mandatory
assert policy.default_action == "replace"
assert policy.keep_mapping is True
assert policy.reversible_id is True

print(policy.name, policy.default_action)
```

This snippet runs entirely from the installed package and loads
`openmed/core/policies/china_pipl.json`; it does not need a model or network
connection. Pass the same `policy_name` to the normal runtime call as
`deidentify(synthetic_text, policy=policy_name)`. The profile intentionally
keeps a reversible mapping for replacement workflows, so its output remains
personal information under PIPL; only irreversible anonymization can remove
data from PIPL scope. Reassess the policy and your legal controls before
production processing of personal information or protected health information
(PHI).

## Privacy and network boundary

!!! important
    **Mirrors affect package and model downloads only. OpenMed inference stays
    fully local, and the OpenMed library sends no telemetry.** After required
    artifacts are cached, `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`
    provide an explicit cache-only boundary for compatible Hub and Transformers
    loaders.

Package installers and model-download clients are third-party tools with their
own configuration and policies. Keep real personally identifiable information
(PII) and PHI out of install commands, model repository names, logs, and mirror
requests. Use only synthetic data while verifying this setup.

## Troubleshooting

- `No matching distribution found`: confirm the mirror has synchronized the
  requested OpenMed version, then compare it with the canonical PyPI release.
- `LocalEntryNotFoundError` in offline mode: the selected model or revision is
  missing from the transferred cache. Warm that exact snapshot while connected.
- A request still leaves the machine: ensure all environment variables are set
  before Python starts, and audit any additional application dependencies that
  have their own network clients.
- A gated or private model fails through the mirror: follow the model owner's
  access terms and your organization's credential policy; do not place tokens
  in shell history, source files, or documentation.
