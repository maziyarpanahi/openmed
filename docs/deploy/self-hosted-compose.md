# Self-hosted Journey deployment

`deploy/openmed-compose.yaml` is the reference single-host Journey profile. It
starts the REST API, an operational worker, PostgreSQL, content-addressed
artifact storage, and a one-shot migration gate. The optional `mcp` profile
adds the streamable HTTP MCP surface; the optional `smoke` profile runs one
idempotent synthetic golden journey.

The runtime remains local-first. The application containers use an internal
Docker network, publish the API to `127.0.0.1:8080` by default, run as
`65532:65532`, drop Linux capabilities, and keep their root filesystems
read-only. Metrics, tracing, remote model access, the privacy gateway, and
OpenHIM are disabled unless explicitly enabled in a reviewed override.

## Prepare the image and secret

Build while the host can reach the required package and image registries:

```bash
docker compose -f deploy/openmed-compose.yaml build
docker pull postgres:17-alpine
```

Create a strong database password in the deployment secret manager and expose
it only to the Compose invocation as `OPENMED_POSTGRES_PASSWORD`. Do not write
it into this YAML, an image, a support bundle, or shell output. A clean
development machine can generate an ephemeral value for its current shell:

```bash
export OPENMED_POSTGRES_PASSWORD="$(openssl rand -hex 32)"
```

The value is required; Compose fails before startup when it is absent. For a
long-lived deployment, inject it from the host's secret service rather than a
checked-in `.env` file. `HF_TOKEN` is not configured by this profile.

## Start and run the golden journey

Start PostgreSQL, wait for it to become healthy, run the migration job, start
the worker, and finally admit the API:

```bash
docker compose -f deploy/openmed-compose.yaml up -d --no-build
docker compose -f deploy/openmed-compose.yaml ps
curl --fail http://127.0.0.1:8080/readyz
```

Run the synthetic deployment proof without an external model service:

```bash
docker compose -f deploy/openmed-compose.yaml \
  --profile smoke run --rm journey-golden
```

Success is a value-free JSON result with `state=passed`,
`code=golden_journey_verified`, and `artifact_count=1`. The command writes a
fixed synthetic artifact to the content-addressed volume, commits its metadata
to PostgreSQL, and verifies both read paths. Repeating it is idempotent. It
does not contain clinical input, identifiers, credentials, or an external API
call.

Enable MCP only when needed:

```bash
docker compose -f deploy/openmed-compose.yaml --profile mcp up -d mcp
```

The MCP container is reachable only on the internal Docker network. Publish it
through a reviewed TLS and authentication boundary if another host needs it.

## Startup and health semantics

The dependency order is part of the profile contract:

1. PostgreSQL passes `pg_isready`.
2. `journey-migrate` obtains the migration lock, applies the append-only schema
   chain, checks migration checksums, and exits successfully.
3. `journey-worker` verifies migration and metadata-store schema readiness plus
   artifact storage with a deterministic synthetic write/read marker, then exposes
   `/livez` and `/readyz` on the internal network.
4. The API starts after the migration and worker gates. Its container health
   command combines migration, store, artifact, worker, and local model
   readiness.

The combined response follows
`openmed.journey.deployment-health.v1` with compatibility major `1`. Every
component has one typed state: `ready`, `pending`, `unavailable`,
`incompatible`, or `disabled`. Reason codes and schema versions are safe for
operational logs; exception strings and configured URLs are not returned.

`/livez` means only that a process can answer. `/readyz` means its required
dependencies are usable. Keep both: restarting a live process because a
database is temporarily unavailable creates avoidable failure loops.

## Storage and permissions

- `openmed-postgres` owns PostgreSQL's durable database files.
- `openmed-cache:/cache` contains the application cache and the
  content-addressed artifact namespace at `/cache/artifacts`.
- `openmed-models:/models:ro` contains pre-staged model inputs and remains
  read-only.
- `/tmp` is a bounded `tmpfs` for the non-root application containers.

When replacing a named volume with a host path, use long syntax and
`create_host_path: false`; make the cache writable by `65532:65532` and keep
models read-only:

```yaml
services:
  openmed:
    volumes:
      - type: bind
        source: /srv/openmed/models
        target: /models
        read_only: true
        bind:
          create_host_path: false
```

The local log driver rotates three 10 MiB files. Operational output contains
states and counts only. It must never contain source text, artifact bytes,
database connection strings, vault values, or request bodies.

## Offline boundary

The application containers set `OPENMED_OFFLINE=1`, `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`, and `HF_DATASETS_OFFLINE=1`. They also set
`HF_HUB_DISABLE_TELEMETRY=1`, `DISABLE_TELEMETRY=1`, and `DO_NOT_TRACK=1`.
The `openmed-internal` network is internal, so containers have no default
egress-capable network after images and model artifacts are prepared.

The default port mapping is `127.0.0.1:8080:8080`. Changing
`OPENMED_BIND_ADDRESS` requires authentication, TLS, and an explicit trusted
host policy. A remote integration requires a separately declared network and
a documented egress allowlist. Telemetry is opt-in and must remain value-free.

## Backup and restore

Use application-consistent backups: stop admission and the worker, wait for
in-flight work to finish, then capture PostgreSQL and the artifact volume as
one named recovery point. `pg_dump --format=custom` is preferred for logical
database backups. Use the storage platform's snapshot or backup tool for
`openmed-cache`; never copy a live volume with a generic recursive command.

Record, outside the backup payload, the OpenMed image digest, PostgreSQL major
version, Journey schema version, artifact snapshot identity, and UTC recovery
point. Encrypt the backup and test restore into a separate namespace. After
restore, run `openmed-journey migrate`, then `journey-golden`, before admitting
traffic. Never run a restore over the active database or artifact volume.

## Upgrade and rollback

Pin immutable image digests in production. Before an upgrade:

1. take and verify one application-consistent backup;
2. read the target release's schema compatibility window;
3. pull/build images before the outage window;
4. let the one-shot migration gate finish before replacing API or worker
   processes;
5. run the golden journey and inspect all five component states.

Journey migrations are additive and checksum-verified. Roll back only to an
image whose documented compatibility major accepts the current persisted
schema. A container rollback does not reverse durable migrations. If the old
image is incompatible, keep traffic stopped and restore the paired database
and artifact recovery point into new volumes.

## Database secret rotation

Use overlapping credentials so a secret rotation does not strand old
containers:

1. create a second least-privilege PostgreSQL role with the same schema grants;
2. update the host secret source and recreate migration, worker, and API
   containers in that order;
3. verify the component health report and golden journey;
4. revoke the old role, then remove it after the rollback window.

Do not log either credential or place it on a command line visible to other
host users. Rotation changes authentication only; it must not change the
Journey schema or delete the old recovery point.

This deployment mechanism is operational packaging, not a compliance
certification or a clinical decision guarantee.
