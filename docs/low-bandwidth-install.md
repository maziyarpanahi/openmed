# Low-bandwidth, mirror, and proxy installation

OpenMed can use an institutional Python package index, a Hugging Face mirror,
and an HTTP proxy without application changes. Keep one Hugging Face cache,
warm it while bandwidth is inexpensive, and then enable OpenMed's enforced
offline mode.

## Keep downloads in one cache

Choose the cache before downloading a model. The same `HF_HOME` must be visible
to the online prefetch step and the offline runtime.

```bash
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME/hub"
```

`HF_HUB_CACHE` can override the final Hub cache directory. Run `openmed doctor`
after setting the environment to confirm the active location.

## Install through a pip mirror

Set `PIP_INDEX_URL` to the `/simple` endpoint supplied by your institution or
regional mirror. The fallback below is the public PyPI index, so the block is
also runnable on a machine without a mirror configuration.

```bash
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"
python -m pip install \
  --index-url "$PIP_INDEX_URL" \
  --retries 20 \
  --timeout 120 \
  "openmed[hf]"
```

The command-line `--index-url` wins for that invocation. Setting
`PIP_INDEX_URL` applies the same mirror to later pip commands. Do not add
`--trusted-host` to bypass TLS verification; ask the network administrator for
the institution's CA certificate instead.

## Select a Hugging Face mirror

`prefetch_model()` delegates downloads to `huggingface_hub`, which reads
`HF_ENDPOINT`. OpenMed does not replace or auto-detect that value. Set the
mirror before starting Python or the `hf` CLI:

```bash
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
```

Use the exact HTTPS endpoint approved by the institution. The public Hub is the
fallback above. A mirror must implement the compatible Hub repository-info and
file-download routes; a generic caching web proxy is not automatically a Hub
mirror.

## Configure an HTTP proxy

Python, pip, and `huggingface_hub` honor the conventional proxy variables. Both
uppercase and lowercase spellings are common; keep the values consistent if a
host application sets both.

```bash
export HTTP_PROXY="http://proxy.example.org:8080"
export HTTPS_PROXY="http://proxy.example.org:8080"
export NO_PROXY="localhost,127.0.0.1,.example.org"
```

Replace the example host and port with the values supplied by the network
administrator. Keep mirror and internal hosts in `NO_PROXY` only when they are
directly reachable. Avoid embedding proxy passwords in scripts or support
logs; `openmed doctor` redacts URL credentials when it prints proxy settings.

## Make model downloads retryable

The Hugging Face cache records completed blobs and incomplete transfers. Rerun
the same command after an interruption: cached files are reused and partial
downloads resume when the server supports HTTP range requests. A single worker
is slower but more reliable on fragile links.

```bash
export MODEL_ID="OpenMed/OpenMed-NER-DiseaseDetect-TinyMed-135M"
hf download \
  "$MODEL_ID" \
  --cache-dir "$HF_HOME/hub" \
  --max-workers 1 \
  --no-force-download
```

OpenMed's Python helper warms the same cache and resolves registry aliases:

```bash
export OPENMED_MODEL_ALIAS="disease_detection_tiny"
python - <<'PY'
import os

from openmed import prefetch_model

print(prefetch_model(os.environ["OPENMED_MODEL_ALIAS"]))
PY
```

Do not set `OPENMED_OFFLINE=1` until the prefetch succeeds. Offline mode rejects
a cache miss instead of falling back to the network.

## Metered-connection checklist

1. Inspect the manifest estimate before downloading. This command reads local
   metadata unless `--remote` is explicitly requested:

    ```bash
    openmed models size disease_detection_tiny
    ```

2. On inexpensive Wi-Fi or a scheduled network window, disable stale offline
   flags and prefetch the model. Rerunning this block is safe:

    ```bash
    unset OPENMED_OFFLINE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
    export OPENMED_MODEL_ALIAS="disease_detection_tiny"
    python - <<'PY'
    import os

    from openmed import prefetch_model

    print(prefetch_model(os.environ["OPENMED_MODEL_ALIAS"]))
    PY
    ```

3. Disconnect from the network, enable enforced offline mode, and verify the
   cached snapshot plus diagnostics before starting the application:

    ```bash
    export OPENMED_OFFLINE=1
    export OPENMED_MODEL_ALIAS="disease_detection_tiny"
    python - <<'PY'
    import os

    from openmed import prefetch_model

    print(prefetch_model(os.environ["OPENMED_MODEL_ALIAS"]))
    PY
    openmed doctor
    ```

After the prefetch, the size check, cache verification, and doctor command are
local-only. Keep `OPENMED_OFFLINE=1` set when launching the real workload so a
missing file fails clearly instead of consuming metered data.

## Diagnose the active environment

```bash
openmed doctor
openmed doctor --json
```

The report includes the active Hub endpoint, `HTTP_PROXY`, `HTTPS_PROXY`,
`NO_PROXY`, Hugging Face cache, and `OPENMED_OFFLINE` state. It performs no
network request and never prints a Hugging Face token. Proxy URL credentials
are replaced with `***`.

If an online prefetch still fails, compare those fields with the institution's
network configuration. For an offline cache miss, temporarily disable offline
mode only during an approved network window, rerun the same prefetch command,
and restore `OPENMED_OFFLINE=1`.
