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
- Bounded buffered responses, bounded stream events, and redirects disabled by
  default so clinical text is not silently forwarded to another origin.

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

```go
_, err := client.Deidentify(ctx, openmed.PIIDeidentifyRequest{
	Text:   "   ",
	Method: openmed.MethodMask,
})
if apiErr, ok := openmed.AsAPIError(err); ok {
	fmt.Println("status:", apiErr.StatusCode)
	fmt.Println("code:", apiErr.Code())
	fmt.Println("message:", apiErr.Envelope.Error.Message)
	fmt.Println("details:", apiErr.Details())
}
```

## Response limits and redirects

Buffered JSON success responses are limited to 64 MiB by default. Configure a
different positive limit with `WithMaxResponseBodyBytes`. The NDJSON endpoint is
not buffered as a whole; each decoded event is limited to 4 MiB by default and
can be configured with `WithMaxStreamEventBytes`.

The default HTTP client does not follow redirects. This prevents POST bodies
containing clinical text from being forwarded to a redirect target. A custom
client supplied with `WithHTTPClient` uses that client's redirect policy, so set
`CheckRedirect` deliberately if you provide one.

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
