# Disk exhaustion runbook

Use this runbook when the volume behind the model cache, the async job store,
or the Journey artifact store is full or nearly full. Read the
[shared rules](service-runbooks.md#rules-for-tickets-logs-and-chat) first.

Where data is stored in the reference deployments:

| Data | Setting | Compose profile | Helm chart |
| --- | --- | --- | --- |
| Model cache | `HF_HOME`, `HF_HUB_CACHE`, `OPENMED_CACHE_DIR` | Compose volume `openmed-cache` at `/cache` | Helm `persistence.size` at Helm `persistence.mountPath` |
| Async job metadata | `OPENMED_SERVICE_JOBS_STORE_PATH` (default `~/.cache/openmed/service-jobs.json`) | Not set; uses the default | Not set; uses the default |
| Journey artifacts | `OPENMED_JOURNEY_ARTIFACT_ROOT` | Compose volume `openmed-cache` at `/cache/artifacts` | Helm `journey.artifacts.size` at Helm `journey.artifacts.mountPath` |
| Journey metadata | `OPENMED_JOURNEY_POSTGRES_DSN` | Compose volume `openmed-postgres` | External or operator-managed PostgreSQL |

In the Compose profile, models and Journey artifacts share the
Compose volume `openmed-cache`. If the model cache fills it, artifact writes
also fail. Under Helm they use separate claims.

Never free space by deleting Journey artifact blobs, the job store, PostgreSQL
data files, or SQLite `-wal` files. Artifacts are content-addressed evidence
that metadata rows reference. Deleting them turns a capacity incident into a
[store corruption incident](store-corruption.md).

## Model cache volume full

### Detection signals

- Model downloads fail, or the first request for a model that is not loaded
  yet returns `500` with `internal_error`.
- With `OPENMED_SERVICE_PRELOAD_MODELS` set, startup fails during preload, or
  `GET /readyz` stays at `503` with `not_ready`.
- In the Compose profile, the Journey probe also reports
  `artifact_write_failed`, because the volume is shared.
- Volume usage alerts from your platform.

### Containment

1. Stop new downloads. The Compose profile already sets `OPENMED_OFFLINE`,
   `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE`. Set them in other deployments.
2. Do not start more replicas on the same volume until space is free.

### Recovery

1. List cached models and their sizes. The output contains repository IDs,
   sizes and last-access times only:

    ```bash
    openmed models cache list --json
    ```

2. Remove models that are not in `OPENMED_SERVICE_PRELOAD_MODELS` and not
   served by any client:

    ```bash
    openmed models cache clear <repo-id> --yes
    ```

3. If the cache is still too small, expand the volume. Under Helm, raise
   Helm `persistence.size` if the storage class supports expansion, or move to
   a larger claim with Helm `persistence.existingClaim`.
4. A download interrupted by a full disk can leave a partial snapshot. Clear
   that model and fetch it again. Use `openmed models pull <repo-id>` on a
   connected host. For an air-gapped site, use `openmed airgap bundle` on a
   connected host and `openmed airgap verify` before you copy the kit in.
   Transactional download with rollback is pending in #3575.

### Verification

```bash
openmed models verify --all
```

Each model with integrity metadata must report `PASS`. Then confirm that
`GET /readyz` returns `200` and that a synthetic request to a preloaded model
succeeds.

### Keep out of tickets and logs

- Request text used to reproduce the failure. Use synthetic text only.
- Hugging Face tokens (`HF_TOKEN`) used to fetch models again. See the
  [Hugging Face token policy](../../security/hf-token-policy.md).
- Private host paths. Record the volume name, its size, and the used bytes.

## Async job store volume full

The job store writes a complete new file next to the old one and then
atomically replaces it. If the disk is full, the write fails. The temporary
file is removed, and the previous file stays intact.

### Detection signals

- `POST /jobs` returns `500` with `internal_error` because the new record
  cannot be saved.
- Jobs stop advancing. `GET /jobs/{job_id}` keeps returning an older `status`
  or progress because the worker cannot save updates.
- `openmed_service_request_total` shows `500` responses for `/jobs`.

### Containment

1. Stop or throttle `POST /jobs` at your proxy, or with the service rate limits
   `OPENMED_SERVICE_RATE_LIMIT_RPS` and `OPENMED_SERVICE_RATE_LIMIT_BURST`.
2. Lower `OPENMED_SERVICE_JOBS_TTL_SECONDS` before the next restart so that
   jobs finishing after the restart expire sooner. Records that already
   finished keep their stored expiry time. Expired records are removed when
   other records are created or read.

### Recovery

1. Free space on that volume by removing data that does not belong to OpenMed
   stores, or expand the volume.
2. To move the store, stop the service and copy the file to a larger volume
   with mode `0600`. Then set `OPENMED_SERVICE_JOBS_STORE_PATH` to the new
   path and restart.
3. Jobs that were queued or running when the service restarted cannot resume.
   Their text lived only in memory. Clients must submit them again. Durable
   job replay is pending in #3574.

### Verification

1. Submit a synthetic job with `POST /jobs` and poll `GET /jobs/{job_id}`
   until it reaches `done`.
2. Run the job store check in the
   [store corruption runbook](store-corruption.md#async-job-store). It must
   report `job store: loadable`.

### Keep out of tickets and logs

- The job store file, or excerpts from it. It contains document IDs, span
  offsets, labels and text hashes for every retained job.
- Job IDs together with document IDs. Record counts of affected jobs instead.

## Journey artifact volume full

### Detection signals

- The Journey worker readiness endpoint and the Journey probe command
  report the artifact component as unavailable with `artifact_write_failed`,
  or `artifact_verify_failed` for a truncated write. The worker writes and reads back a small synthetic probe artifact on
  each refresh.
- Ingestion calls return a `failure` store result with
  `artifact_write_failed`. A metadata write on a full PostgreSQL volume returns
  `transaction_failed` or `commit_status_unknown`.
- In the Compose profile, the `openmed` health check runs the probe. It turns
  the container unhealthy.

### Containment

1. Pause the jobs that write new artifacts. Leave the stored data in place.
2. Do not retry a write that returned `commit_status_unknown` blindly. The
   commit may have succeeded. Read the record back first.

### Recovery

1. Expand the volume. Under Helm, raise Helm `journey.artifacts.size` if the
   storage class supports expansion, or attach a larger claim with
   Helm `journey.artifacts.existingClaim` after copying the data with its
   permissions preserved.
2. In the Compose profile, free space on the shared volume by clearing unused
   models. See [Model cache volume full](#model-cache-volume-full).
3. For PostgreSQL, add capacity using your database tooling. OpenMed does not
   manage database storage. Backup and restore tooling for service state is
   pending in #3577.

### Verification

```bash
docker compose -f deploy/openmed-compose.yaml exec openmed \
  python3 -m openmed.deploy.journey probe \
  --worker-url http://journey-worker:8091/readyz \
  --model-url http://127.0.0.1:8080/readyz
```

The JSON output lists controlled states and codes only. The artifact component
must show `artifact_store_ready`, and the store component must show
`store_ready`. The probe reads the DSN from `OPENMED_JOURNEY_POSTGRES_DSN`
and never prints it. The Compose service `journey-golden` (profile `smoke`)
runs `python -m openmed.deploy.journey golden`. It writes one synthetic record
and expects `golden_journey_verified`. Run it only where a synthetic record in
the production store is acceptable.

### Keep out of tickets and logs

- Artifact blobs, file names under the artifact root, and content digests that
  you could match to a patient document. Record counts and used bytes.
- `OPENMED_JOURNEY_POSTGRES_DSN` and `OPENMED_POSTGRES_PASSWORD`. Do not pass
  the DSN on a command line.
