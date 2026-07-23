# OpenHIM De-identification Mediator

OpenMed can run as an [OpenHIM mediator](https://openhim.org/docs/configuration/mediators/) that de-identifies FHIR JSON and plain-text transactions inside a health information exchange. The integration is feature-flagged and off by default. When enabled, the service registers with OpenHIM core at startup, sends an immediate heartbeat requesting current configuration, and continues heartbeats every 10 seconds.

!!! important "Keep PHI inside the trust boundary"

    Deploy the mediator on infrastructure controlled by the national, regional, or facility HIE. Route identifiable transactions to OpenMed before any system outside that trust boundary. OpenMed does not send mediator payloads to OpenHIM's management API; registration and heartbeat calls contain only mediator metadata and uptime.

## Registration payload

The example registration is in `examples/openhim-mediator/mediator-config.json`. It declares:

- URN `urn:openhim-mediator:openmed-deidentification`
- the default `/openmed/deidentify` OpenHIM channel
- the mediator route `/openhim/deidentify` on port `8080`
- the endpoint displayed in the OpenHIM Console

OpenHIM core accepts mediator registration at `POST /mediators` and heartbeats at `POST /mediators/:urn/heartbeat`. The first heartbeat includes `config: true`; later heartbeats include uptime only. Registration is safe to call repeatedly across service restarts, while a running service instance suppresses duplicate registration attempts.

## Configure the service

Set the following variables on the OpenMed service:

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `OPENMED_OPENHIM_MEDIATOR_ENABLED` | yes | `false` | Enables registration, heartbeat, and mediator routes. |
| `OPENMED_OPENHIM_CORE_URL` | when enabled | none | OpenHIM management API URL, including its API port, such as `https://openhim-core:8080`. |
| `OPENMED_OPENHIM_USERNAME` | when enabled | none | OpenHIM administrative API username. |
| `OPENMED_OPENHIM_PASSWORD` | when enabled | none | OpenHIM administrative API password. Supply it through a secret store, not the JSON config. |
| `OPENMED_OPENHIM_AUTH_MODE` | no | `basic` | Management API authentication: `basic`, or the deprecated OpenHIM `token` handshake for older deployments. |
| `OPENMED_OPENHIM_POLICY` | no | `hipaa_safe_harbor` | Bundled policy applied to mediator payloads. Validated once at startup. |
| `OPENMED_OPENHIM_METHOD` | no | `replace` | De-identification method applied to mediator payloads. Validated once at startup. |
| `OPENMED_OPENHIM_CONFIG_PATH` | no | built-in config | Mounted mediator registration JSON. |
| `OPENMED_OPENHIM_VERIFY_TLS` | no | `true` | Verifies the OpenHIM API certificate. Keep enabled outside isolated fixture tests. |
| `OPENMED_OPENHIM_ALLOW_INSECURE_HTTP` | no | `false` | Allows a plaintext management URL only for the bundled synthetic fixture. Never enable it for a real HIE. |
| `OPENMED_OPENHIM_HEARTBEAT_INTERVAL_SECONDS` | no | `10` | Heartbeat interval; accepted values are greater than zero and no more than 30 seconds. |
| `OPENMED_OPENHIM_REQUEST_TIMEOUT_SECONDS` | no | `10` | Registration and heartbeat request timeout. |

The management credentials authorize mediator registration and heartbeat, so keep them out of `mediator-config.json`, source control, images, and logs. OpenHIM's management API uses HTTPS; OpenMed rejects plaintext management URLs unless the fixture-only override is explicitly enabled. Basic authentication is the default. Set `OPENMED_OPENHIM_AUTH_MODE=token` only when connecting to an older deployment configured for OpenHIM's deprecated salt-and-timestamp token headers.

The policy and method are deployment settings, not request parameters. Incoming
query strings cannot weaken the configured de-identification posture. Change
these settings only through the mediator deployment configuration and restart
the service so startup validation runs again.

## Run with an existing OpenHIM core

From the repository root, set the real management API and credentials, then start the example:

```bash
export OPENMED_OPENHIM_CORE_URL=https://openhim-core.internal:8080
export OPENMED_OPENHIM_USERNAME=mediator-admin@example.org
export OPENMED_OPENHIM_PASSWORD='use-your-secret-store'

docker compose -f examples/openhim-mediator/docker-compose.yml \
  up --build openmed-mediator
```

The compose service mounts `mediator-config.json` read-only at `/etc/openmed/mediator-config.json`. If OpenHIM resolves the service under a different hostname, update `host` in both the channel route and endpoint before starting the mediator.

After registration, open the mediator in the OpenHIM Console and install the **OpenMed De-identification** default channel. The channel matches `POST /openmed/deidentify` on the OpenHIM router and forwards it to the mediator's `/openhim/deidentify` route.

## Send a FHIR transaction

Send a Bundle through the OpenHIM router, not its management API:

```bash
export OPENHIM_CLIENT_USERNAME=openhim-client

curl --request POST https://openhim-core.internal:5000/openmed/deidentify \
  --user "${OPENHIM_CLIENT_USERNAME}" \
  --header 'Content-Type: application/fhir+json' \
  --data @synthetic-bundle.json
```

With only the username supplied, `curl` prompts for the client password instead
of placing it in the command or shell history.

For FHIR resources and Bundles, OpenMed de-identifies free-text and direct identifier values with `openmed.interop.fhir_operations`. Coded elements, systems, resource IDs, `fullUrl` values, request blocks, and references remain unchanged. Plain `text/*` bodies use the same OpenMed privacy pipeline. Other media types pass through byte-for-byte instead of being forced into a JSON envelope. Only response-safe representation and correlation headers are reflected; credentials, cookies, and unbounded custom clinical metadata are not copied into mediator response metadata.

Transformed responses use the required `application/json+openhim` media type. The envelope contains `x-mediator-urn`, the client response, and an orchestration step for the OpenHIM Console. The orchestration request intentionally omits the raw body, so identifiable input is not duplicated into transaction metadata. Errors return PHI-free messages.

## Health and heartbeat

The existing `/health`, `/livez`, and `/readyz` routes retain their normal meanings. When mediator mode is enabled, `GET /openhim/heartbeat` reports local registration and heartbeat state without exposing credentials:

```bash
curl http://127.0.0.1:8080/openhim/heartbeat
```

A healthy response has `registered: true`, a non-null `last_heartbeat_at`, and `last_error: null`.

## Offline container smoke test

The compose file includes a small recorded handshake fixture, not OpenHIM core. It exists only to test registration and heartbeat without operating or bundling the real interoperability layer:

```bash
export OPENMED_OPENHIM_CORE_URL=http://openhim-fixture:8081
export OPENMED_OPENHIM_USERNAME=openhim@example.org
export OPENMED_OPENHIM_PASSWORD=synthetic-password
export OPENMED_OPENHIM_ALLOW_INSECURE_HTTP=true

docker compose -f examples/openhim-mediator/docker-compose.yml \
  --profile fixture up -d openhim-fixture

docker compose -f examples/openhim-mediator/docker-compose.yml \
  up -d --build openmed-mediator

curl http://127.0.0.1:8080/openhim/heartbeat
curl http://127.0.0.1:8081/health

docker compose -f examples/openhim-mediator/docker-compose.yml \
  --profile fixture down --volumes --remove-orphans
```

The fixture uses synthetic credentials and payloads. Never use it as an HIE
component. Unset those four variables before configuring a real OpenHIM
deployment.
