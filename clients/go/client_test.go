package openmed_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	openmed "github.com/maziyarpanahi/openmed/clients/go"
)

func ptrFloat(v float64) *float64 { return &v }

// newServer returns a test server whose handler asserts the request and writes
// the provided response, plus a client pointed at it.
func newServer(t *testing.T, handler http.HandlerFunc) (*openmed.Client, *httptest.Server) {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	client, err := openmed.New(srv.URL)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return client, srv
}

func decodeBody(t *testing.T, r *http.Request, out any) {
	t.Helper()
	body, err := io.ReadAll(r.Body)
	if err != nil {
		t.Fatalf("read request body: %v", err)
	}
	if err := json.Unmarshal(body, out); err != nil {
		t.Fatalf("decode request body: %v (%s)", err, body)
	}
}

func TestNewNormalizesBaseURL(t *testing.T) {
	client, err := openmed.New("http://example.com:8080///")
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if got := client.BaseURL(); got != "http://example.com:8080" {
		t.Fatalf("BaseURL = %q, want trailing slashes trimmed", got)
	}
}

func TestNewRejectsInvalidBaseURL(t *testing.T) {
	if _, err := openmed.New("://not-a-url"); err == nil {
		t.Fatal("expected an error for an invalid base URL")
	}
}

func TestAnalyze(t *testing.T) {
	client, _ := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/analyze" {
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		if ct := r.Header.Get("Content-Type"); ct != "application/json" {
			t.Fatalf("Content-Type = %q, want application/json", ct)
		}
		var req openmed.AnalyzeRequest
		decodeBody(t, r, &req)
		if req.Text != "Patient started imatinib." {
			t.Fatalf("Text = %q", req.Text)
		}
		if req.AggregationStrategy == nil || *req.AggregationStrategy != openmed.AggregationSimple {
			t.Fatalf("AggregationStrategy = %v, want simple", req.AggregationStrategy)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"text": "Patient started imatinib.",
			"entities": [{"text": "imatinib", "label": "CHEM", "confidence": 0.98, "start": 16, "end": 24, "metadata": {}}],
			"model_name": "disease_detection_superclinical",
			"timestamp": "2026-01-01T00:00:00Z",
			"processing_time": 0.01,
			"metadata": {}
		}`)
	})

	strategy := openmed.AggregationSimple
	resp, err := client.Analyze(context.Background(), openmed.AnalyzeRequest{
		Text:                "Patient started imatinib.",
		ModelName:           "disease_detection_superclinical",
		ConfidenceThreshold: ptrFloat(0.25),
		AggregationStrategy: &strategy,
		KeepAlive:           "5m",
	})
	if err != nil {
		t.Fatalf("Analyze: %v", err)
	}
	if len(resp.Entities) != 1 || resp.Entities[0].Label != "CHEM" {
		t.Fatalf("unexpected entities: %+v", resp.Entities)
	}
	if resp.Entities[0].Start == nil || *resp.Entities[0].Start != 16 {
		t.Fatalf("Start = %v, want 16", resp.Entities[0].Start)
	}
}

func TestExtractPII(t *testing.T) {
	client, _ := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/pii/extract" {
			t.Fatalf("path = %q", r.URL.Path)
		}
		var req openmed.PIIExtractRequest
		decodeBody(t, r, &req)
		if req.Lang != openmed.LangES {
			t.Fatalf("Lang = %q, want es", req.Lang)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"text": "Maria Garcia",
			"entities": [{"text": "Maria Garcia", "label": "PATIENT", "confidence": 0.99, "start": 0, "end": 12, "metadata": {}}],
			"model_name": "pii",
			"timestamp": "2026-01-01T00:00:00Z",
			"processing_time": null,
			"metadata": {}
		}`)
	})

	resp, err := client.ExtractPII(context.Background(), openmed.PIIExtractRequest{
		Text: "Maria Garcia",
		Lang: openmed.LangES,
	})
	if err != nil {
		t.Fatalf("ExtractPII: %v", err)
	}
	if resp.ProcessingTime != nil {
		t.Fatalf("ProcessingTime = %v, want nil", resp.ProcessingTime)
	}
	if len(resp.Entities) != 1 || resp.Entities[0].Label != "PATIENT" {
		t.Fatalf("unexpected entities: %+v", resp.Entities)
	}
}

func TestExtractPIIStream(t *testing.T) {
	const ndjson = "{\"event\":\"entity\"}\n{\"event\":\"done\"}\n"
	client, _ := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/pii/extract/stream" {
			t.Fatalf("path = %q", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = io.WriteString(w, ndjson)
	})

	body, err := client.ExtractPIIStream(context.Background(), openmed.PIIExtractStreamRequest{
		Text:      "hello",
		ChunkSize: 512,
	})
	if err != nil {
		t.Fatalf("ExtractPIIStream: %v", err)
	}
	if body != ndjson {
		t.Fatalf("stream body = %q", body)
	}
}

func TestDeidentify(t *testing.T) {
	client, _ := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/pii/deidentify" {
			t.Fatalf("path = %q", r.URL.Path)
		}
		var req openmed.PIIDeidentifyRequest
		decodeBody(t, r, &req)
		if req.Method != openmed.MethodMask {
			t.Fatalf("Method = %q, want mask", req.Method)
		}
		if !req.KeepMapping {
			t.Fatalf("KeepMapping = false, want true")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"original_text": "Maria Garcia",
			"deidentified_text": "[PATIENT]",
			"pii_entities": [],
			"method": "mask",
			"timestamp": "2026-01-01T00:00:00Z",
			"num_entities_redacted": 1,
			"metadata": {},
			"audit_report": null,
			"mapping": {"[PATIENT]": "Maria Garcia"}
		}`)
	})

	resp, err := client.Deidentify(context.Background(), openmed.PIIDeidentifyRequest{
		Text:        "Maria Garcia",
		Method:      openmed.MethodMask,
		Lang:        openmed.LangES,
		KeepMapping: true,
	})
	if err != nil {
		t.Fatalf("Deidentify: %v", err)
	}
	if resp.DeidentifiedText != "[PATIENT]" {
		t.Fatalf("DeidentifiedText = %q", resp.DeidentifiedText)
	}
	if resp.NumEntitiesRedacted != 1 {
		t.Fatalf("NumEntitiesRedacted = %d, want 1", resp.NumEntitiesRedacted)
	}
	if resp.Mapping["[PATIENT]"] != "Maria Garcia" {
		t.Fatalf("mapping missing entry: %+v", resp.Mapping)
	}
}

func TestHealth(t *testing.T) {
	client, _ := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/health" {
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"status":"ok","service":"openmed","version":"1.0.0","profile":"default"}`)
	})

	resp, err := client.Health(context.Background())
	if err != nil {
		t.Fatalf("Health: %v", err)
	}
	if resp.Status != "ok" || resp.Service != "openmed" {
		t.Fatalf("unexpected health: %+v", resp)
	}
}

func TestLoadedModels(t *testing.T) {
	client, _ := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models/loaded" {
			t.Fatalf("path = %q", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"default_keep_alive_seconds": 300,
			"max_resident_models": 2,
			"warm_models": ["pii"],
			"models": {"pii": {"pipelines": 1, "resident": true}},
			"resident_memory_bytes": 1024
		}`)
	})

	resp, err := client.LoadedModels(context.Background())
	if err != nil {
		t.Fatalf("LoadedModels: %v", err)
	}
	if resp.MaxResidentModels == nil || *resp.MaxResidentModels != 2 {
		t.Fatalf("MaxResidentModels = %v, want 2", resp.MaxResidentModels)
	}
	if _, ok := resp.Models["pii"]; !ok {
		t.Fatalf("expected pii in models: %+v", resp.Models)
	}
	if resp.Raw["resident_memory_bytes"] == nil {
		t.Fatalf("Raw should retain open-ended fields: %+v", resp.Raw)
	}
}

func TestUnloadModels(t *testing.T) {
	client, _ := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/models/unload" {
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		var req openmed.ModelUnloadRequest
		decodeBody(t, r, &req)
		if req.ModelName != "pii" {
			t.Fatalf("ModelName = %q, want pii", req.ModelName)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"unloaded": true, "model_name": "pii", "released": {"pipelines": 1}}`)
	})

	resp, err := client.UnloadModels(context.Background(), openmed.ModelUnloadRequest{ModelName: "pii"})
	if err != nil {
		t.Fatalf("UnloadModels: %v", err)
	}
	if resp.Unloaded == nil || !*resp.Unloaded {
		t.Fatalf("Unloaded = %v, want true", resp.Unloaded)
	}
	if resp.Released["pipelines"] != 1 {
		t.Fatalf("Released = %+v", resp.Released)
	}
}

func TestAPIErrorEnvelope(t *testing.T) {
	client, _ := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnprocessableEntity)
		_, _ = io.WriteString(w, `{"error":{"code":"validation_error","message":"Text must not be blank","details":[{"field":"text","message":"blank"}],"request_id":"req-1"}}`)
	})

	_, err := client.Deidentify(context.Background(), openmed.PIIDeidentifyRequest{Text: "   "})
	if err == nil {
		t.Fatal("expected an error")
	}
	apiErr, ok := openmed.AsAPIError(err)
	if !ok {
		t.Fatalf("expected *APIError, got %T", err)
	}
	if apiErr.StatusCode != http.StatusUnprocessableEntity {
		t.Fatalf("StatusCode = %d, want 422", apiErr.StatusCode)
	}
	if apiErr.Code() != "validation_error" {
		t.Fatalf("Code = %q, want validation_error", apiErr.Code())
	}
	if apiErr.Envelope.Error.Message != "Text must not be blank" {
		t.Fatalf("Message = %q", apiErr.Envelope.Error.Message)
	}
	if apiErr.Details() == nil {
		t.Fatal("Details should be populated")
	}
	if apiErr.Envelope.Error.RequestID != "req-1" {
		t.Fatalf("RequestID = %q, want req-1", apiErr.Envelope.Error.RequestID)
	}
}

func TestAPIErrorNonEnvelopeFallback(t *testing.T) {
	client, _ := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		_, _ = io.WriteString(w, "upstream boom")
	})

	_, err := client.Health(context.Background())
	apiErr, ok := openmed.AsAPIError(err)
	if !ok {
		t.Fatalf("expected *APIError, got %T", err)
	}
	if apiErr.StatusCode != http.StatusBadGateway {
		t.Fatalf("StatusCode = %d, want 502", apiErr.StatusCode)
	}
	if apiErr.Code() != "http_error" {
		t.Fatalf("Code = %q, want http_error", apiErr.Code())
	}
}

func TestContextCancellation(t *testing.T) {
	client, _ := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"status":"ok","service":"openmed","version":"1.0.0","profile":"default"}`)
	})

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := client.Health(ctx); err == nil {
		t.Fatal("expected a context cancellation error")
	}
}

func TestGetJobEscapesPath(t *testing.T) {
	client, _ := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.EscapedPath() != "/jobs/job%2F1" {
			t.Fatalf("escaped path = %q", r.URL.EscapedPath())
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"job/1","status":"done","progress_percent":100,"document_count":1,"processed_count":1,"failed_count":0,"label_histogram":{},"spans":[],"documents":[],"error":null,"webhook":null,"webhook_delivery":null,"created_at":"t","updated_at":"t","started_at":null,"completed_at":null,"expires_at":"t"}`)
	})

	resp, err := client.GetJob(context.Background(), "job/1")
	if err != nil {
		t.Fatalf("GetJob: %v", err)
	}
	if resp.Status != openmed.JobDone {
		t.Fatalf("Status = %q, want done", resp.Status)
	}
}
