# OpenMed Go REST Client

Dependency-light Go client for the OpenMed REST service. It is built entirely on
the standard library (`net/http`) with **no third-party dependencies**, and it
tracks the operations and request schemas published in the committed OpenAPI
spec (`docs/api/openapi.json`).

- Typed request structs for every JSON request, typed structs for stable JSON
  responses, and an incremental typed decoder for the NDJSON endpoint.
- Enum-like typed string constants for the de-identification method, PII
  language, aggregation strategy, privacy policy, and job status.
- A configurable base URL and injectable `*http.Client`.
- A `context.Context` on every call for cancellation and deadlines.
- Non-2xx responses surfaced as a typed `*APIError` carrying the service error
  envelope (`code`, `message`, `details`, `request_id`).
- Bounded buffered responses, bounded stream events, redirects disabled by
  default, and HTTPS required for non-loopback hosts so clinical text and
  service credentials are not silently exposed in transit.

## Install

```bash
go get github.com/maziyarpanahi/openmed/clients/go
```

Import it as `openmed`:

```go
import openmed "github.com/maziyarpanahi/openmed/clients/go"
```

## Usage

```go
package main

import (
	"context"
	"fmt"
	"log"

	openmed "github.com/maziyarpanahi/openmed/clients/go"
)

func main() {
	client, err := openmed.New("http://localhost:8080")
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()

	// Liveness / readiness.
	health, err := client.Health(ctx)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("service:", health.Service, "version:", health.Version)

	// Analyze clinical text.
	strategy := openmed.AggregationSimple
	confidence := 0.25
	analysis, err := client.Analyze(ctx, openmed.AnalyzeRequest{
		Text:                "Patient started imatinib for CML.",
		ModelName:           "disease_detection_superclinical",
		ConfidenceThreshold: &confidence,
		AggregationStrategy: &strategy,
		KeepAlive:           "5m",
	})
	if err != nil {
		log.Fatal(err)
	}
	for _, e := range analysis.Entities {
		fmt.Printf("%s (%s) %.2f\n", e.Text, e.Label, e.Confidence)
	}

	// Extract PII.
	pii, err := client.ExtractPII(ctx, openmed.PIIExtractRequest{
		Text: "Paciente: Maria Garcia, DNI: 12345678Z",
		Lang: openmed.LangES,
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("pii spans:", len(pii.Entities))

	// De-identify, keeping the reversible mapping.
	deid, err := client.Deidentify(ctx, openmed.PIIDeidentifyRequest{
		Text:        "Paciente: Maria Garcia, DNI: 12345678Z",
		Method:      openmed.MethodMask,
		Lang:        openmed.LangES,
		KeepMapping: true,
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(deid.DeidentifiedText)

	// Inspect loaded models and unload one.
	loaded, err := client.LoadedModels(ctx)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("warm models:", loaded.WarmModels)

	if _, err := client.UnloadModels(ctx, openmed.ModelUnloadRequest{
		ModelName: "disease_detection_superclinical",
	}); err != nil {
		log.Fatal(err)
	}
	// Or unload everything inactive:
	// client.UnloadModels(ctx, openmed.ModelUnloadRequest{All: true})
}
```

### Custom `*http.Client`

Provide your own client for timeouts, transports, or test doubles:

```go
hc := &http.Client{Timeout: 30 * time.Second}
client, err := openmed.New("http://localhost:8080", openmed.WithHTTPClient(hc))
```

The SDK copies the supplied client's public configuration without copying its
internal synchronization state. If `CheckRedirect` is nil, redirects remain
disabled. A non-nil redirect policy is treated as an explicit override; take
particular care with 307/308 responses because Go may resend POST bodies that
contain clinical text or service credentials.

### Streaming and batch jobs

```go
// Incrementally consume the NDJSON stream of PII events.
stream, err := client.ExtractPIIStream(ctx, openmed.PIIExtractStreamRequest{
	Text:      longClinicalNote,
	ChunkSize: 2048,
})
if err != nil {
	log.Fatal(err)
}
defer stream.Close()
for stream.Next() {
	event := stream.Event()
	fmt.Println(event.Type)
}
if err := stream.Err(); err != nil {
	log.Fatal(err)
}

// Submit a batch de-identification job and poll it.
// Document IDs are echoed by the service, so keep them free of PHI.
job, err := client.CreateJob(ctx, openmed.DeidentifyJobRequest{
	Documents: []openmed.DeidentifyJobDocument{{ID: "note-1", Text: note}},
	Method:    openmed.MethodMask,
})
status, err := client.GetJob(ctx, job.ID)
```

## Error handling

Every non-2xx response is returned as a typed `*APIError`. It preserves the HTTP
status code and a valid JSON service error envelope with non-empty
`error.code` and `error.message`, including any `error.details`. If an upstream
proxy returns an empty, oversized, wrongly typed, or malformed body, the client
returns a synthetic `http_error` with nil details and does not retain that body;
the body may contain raw clinical text or credentials.

The default `error` string contains only the HTTP status. It deliberately omits
the service message and details so routine error logging does not copy possible
clinical text. Inspect the typed envelope explicitly only where that data can be
handled safely.

```go
_, err := client.Deidentify(ctx, openmed.PIIDeidentifyRequest{
	Text:   "   ",
	Method: openmed.MethodMask,
})
if apiErr, ok := openmed.AsAPIError(err); ok {
	fmt.Println("status:", apiErr.StatusCode)
	fmt.Println("code:", apiErr.Code())
	fmt.Println("request ID:", apiErr.Envelope.Error.RequestID)
	// Inspect Message and Details only inside a PHI-approved handling path.
}
```

## Transport safety and response limits

Buffered JSON success responses are limited to 64 MiB by default. Configure a
different positive limit with `WithMaxResponseBodyBytes`. The NDJSON endpoint is
not buffered as a whole; each decoded event is limited to 4 MiB by default and
can be configured with `WithMaxStreamEventBytes`.

Cleartext HTTP is accepted for loopback hosts such as `localhost`, `127.0.0.1`,
and `::1`. Non-loopback services must use HTTPS. If transport security is
provided outside the application on a trusted, isolated network, callers can
make the cleartext exception explicit with `WithInsecureHTTP()`; never use that
option across an untrusted network.

Redirects are disabled by default, including when a custom `*http.Client` has a
nil `CheckRedirect`. This prevents POST bodies containing clinical text or a
SMART private key from being forwarded to a redirect target. Only provide a
non-nil redirect policy when that forwarding behavior is deliberate.

## Endpoint coverage

| Method | Endpoint |
| --- | --- |
| `Analyze` | `POST /analyze` |
| `ExtractPII` | `POST /pii/extract` |
| `ExtractPIIStream` | `POST /pii/extract/stream` |
| `Deidentify` | `POST /pii/deidentify` |
| `PrivacyGateway` | `POST /privacy-gateway/complete` |
| `Health` | `GET /health` |
| `Livez` | `GET /livez` |
| `Readyz` | `GET /readyz` |
| `LoadedModels` | `GET /models/loaded` |
| `UnloadModels` | `POST /models/unload` |
| `CreateJob` | `POST /jobs` |
| `GetJob` | `GET /jobs/{job_id}` |
| `StartSMARTBackendIngestion` | `POST /fhir/smart-backend/ingestions` |
| `SMARTBackendIngestionStatus` | `GET /fhir/smart-backend/ingestions/{job_id}` |
| `SMARTBackendIngestionSummary` | `GET /fhir/smart-backend/ingestions/{job_id}/summary` |

A Python parity test (`tests/unit/service/test_go_client_parity.py`) guards
against drift: it asserts that every OpenAPI HTTP operation maps to an exported
Go method and that request fields, requiredness, and explicit-zero semantics
stay aligned with the committed schemas.

## Development

```bash
cd clients/go
go vet ./...
go test ./...
gofmt -l .   # must print nothing
```

The module is pure Go with no cgo. On toolchains where the host linker rejects
the default external link (for example a Homebrew Go build paired with a newer
Xcode linker), build and test with the internal linker: `CGO_ENABLED=0 go test ./...`.
