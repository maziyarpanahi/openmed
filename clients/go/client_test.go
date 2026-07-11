package openmed_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	openmed "github.com/maziyarpanahi/openmed/clients/go"
)

func ptrFloat(v float64) *float64 { return &v }
func ptrInt(v int) *int           { return &v }

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

type roundTripFunc func(*http.Request) (*http.Response, error)

func (fn roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return fn(req)
}

type countingReadCloser struct {
	remaining int
	read      int
}

func (r *countingReadCloser) Read(p []byte) (int, error) {
	if r.remaining == 0 {
		return 0, io.EOF
	}
	n := len(p)
	if n > r.remaining {
		n = r.remaining
	}
	for i := 0; i < n; i++ {
		p[i] = 'x'
	}
	r.remaining -= n
	r.read += n
	return n, nil
}

func (r *countingReadCloser) Close() error { return nil }

func TestNewNormalizesBaseURL(t *testing.T) {
	client, err := openmed.New("https://example.com:8080///")
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if got := client.BaseURL(); got != "https://example.com:8080" {
		t.Fatalf("BaseURL = %q, want trailing slashes trimmed", got)
	}
}

func TestNewRejectsInvalidBaseURL(t *testing.T) {
	if _, err := openmed.New("://not-a-url"); err == nil {
		t.Fatal("expected an error for an invalid base URL")
	}
}

func TestNewRejectsNonHTTPBaseURLs(t *testing.T) {
	for _, baseURL := range []string{
		"localhost:8080",
		"/relative",
		"ftp://example.com",
		"http:///missing-host",
		"http://:8080",
		"http://user:secret@example.com",
		"http://example.com/api/../v1",
		"http://example.com/api/%2e%2e/v1",
		"http://example.com?token=secret",
		"http://example.com#fragment",
	} {
		t.Run(baseURL, func(t *testing.T) {
			if _, err := openmed.New(baseURL); err == nil {
				t.Fatalf("New(%q) succeeded, want an error", baseURL)
			}
		})
	}
}

func TestNewRequiresHTTPSForNonLoopbackHosts(t *testing.T) {
	for _, baseURL := range []string{
		"http://example.com:8080",
		"http://openmed.internal:8080",
		"http://0.0.0.0:8080",
		"http://[::]:8080",
	} {
		t.Run(baseURL, func(t *testing.T) {
			if _, err := openmed.New(baseURL); err == nil {
				t.Fatalf("New(%q) succeeded without an insecure-HTTP opt-in", baseURL)
			}
		})
	}

	for _, baseURL := range []string{
		"http://localhost:8080",
		"http://LOCALHOST.:8080",
		"http://127.0.0.1:8080",
		"http://127.42.0.1:8080",
		"http://[::1]:8080",
	} {
		t.Run(baseURL, func(t *testing.T) {
			if _, err := openmed.New(baseURL); err != nil {
				t.Fatalf("New(%q): %v", baseURL, err)
			}
		})
	}

	client, err := openmed.New(
		"http://openmed.internal:8080",
		openmed.WithInsecureHTTP(),
	)
	if err != nil {
		t.Fatalf("New with explicit insecure HTTP: %v", err)
	}
	if got := client.BaseURL(); got != "http://openmed.internal:8080" {
		t.Fatalf("BaseURL = %q", got)
	}
}

func TestNewValidationErrorsDoNotExposeBaseURLSecrets(t *testing.T) {
	tests := []struct {
		baseURL string
		secret  string
	}{
		{"http://user:synthetic-password@example.com", "synthetic-password"},
		{"http://example.com?token=synthetic-query-token", "synthetic-query-token"},
		{"://synthetic-malformed-secret", "synthetic-malformed-secret"},
		{"http://example.com/synthetic-path-secret", "synthetic-path-secret"},
	}
	for _, test := range tests {
		_, err := openmed.New(test.baseURL)
		if err == nil {
			t.Fatalf("New(%q) unexpectedly succeeded", test.baseURL)
		}
		if strings.Contains(err.Error(), test.secret) {
			t.Fatalf("validation error exposed secret %q: %v", test.secret, err)
		}
	}
}

func TestNewPreservesBasePath(t *testing.T) {
	client, _ := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/openmed/health" {
			t.Fatalf("path = %q, want /openmed/health", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"status":"ok","service":"openmed"}`)
	})

	client, err := openmed.New(client.BaseURL() + "/openmed/")
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := client.Health(context.Background()); err != nil {
		t.Fatalf("Health: %v", err)
	}
}

func TestDefaultClientDoesNotForwardClinicalTextAcrossRedirects(t *testing.T) {
	redirectTargetCalled := false
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		redirectTargetCalled = true
		w.WriteHeader(http.StatusOK)
	}))
	defer target.Close()

	redirector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL+r.URL.Path, http.StatusTemporaryRedirect)
	}))
	defer redirector.Close()
	client, err := openmed.New(redirector.URL)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = client.Deidentify(context.Background(), openmed.PIIDeidentifyRequest{
		Text: "synthetic patient Maria Garcia, MRN 12345",
	})
	apiErr, ok := openmed.AsAPIError(err)
	if !ok || apiErr.StatusCode != http.StatusTemporaryRedirect {
		t.Fatalf("Deidentify error = %v, want a 307 APIError", err)
	}
	if redirectTargetCalled {
		t.Fatal("default client forwarded clinical text to a redirect target")
	}
}

func TestCustomClientWithoutPolicyDoesNotForwardClinicalTextAcrossRedirects(t *testing.T) {
	redirectTargetCalled := false
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		redirectTargetCalled = true
		w.WriteHeader(http.StatusOK)
	}))
	defer target.Close()

	redirector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL+r.URL.Path, http.StatusTemporaryRedirect)
	}))
	defer redirector.Close()

	httpClient := &http.Client{Timeout: time.Second}
	client, err := openmed.New(
		redirector.URL,
		openmed.WithHTTPClient(httpClient),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = client.Deidentify(context.Background(), openmed.PIIDeidentifyRequest{
		Text: "synthetic patient Maria Garcia, MRN 12345",
	})
	apiErr, ok := openmed.AsAPIError(err)
	if !ok || apiErr.StatusCode != http.StatusTemporaryRedirect {
		t.Fatalf("Deidentify error = %v, want a 307 APIError", err)
	}
	if redirectTargetCalled {
		t.Fatal("custom client with a nil policy forwarded clinical text")
	}
	if httpClient.CheckRedirect != nil {
		t.Fatal("WithHTTPClient mutated the caller's client")
	}
}

func TestCustomClientPreservesExplicitRedirectPolicy(t *testing.T) {
	policyCalled := false
	redirector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "/elsewhere", http.StatusTemporaryRedirect)
	}))
	defer redirector.Close()

	httpClient := &http.Client{
		CheckRedirect: func(_ *http.Request, _ []*http.Request) error {
			policyCalled = true
			return http.ErrUseLastResponse
		},
	}
	client, err := openmed.New(redirector.URL, openmed.WithHTTPClient(httpClient))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = client.Health(context.Background())
	apiErr, ok := openmed.AsAPIError(err)
	if !ok || apiErr.StatusCode != http.StatusTemporaryRedirect {
		t.Fatalf("Health error = %v, want a 307 APIError", err)
	}
	if !policyCalled {
		t.Fatal("explicit redirect policy was not used")
	}
}

func TestWithHTTPClient(t *testing.T) {
	called := false
	httpClient := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		called = true
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     make(http.Header),
			Body:       io.NopCloser(strings.NewReader(`{"status":"ok","service":"openmed"}`)),
			Request:    req,
		}, nil
	})}
	client, err := openmed.New("https://openmed.invalid", openmed.WithHTTPClient(httpClient))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := client.Health(context.Background()); err != nil {
		t.Fatalf("Health: %v", err)
	}
	if !called {
		t.Fatal("custom HTTP client was not used")
	}
}

func TestTransportErrorsDoNotExposeBasePathsOrJobIDs(t *testing.T) {
	httpClient := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		return nil, errors.New("synthetic dial failure")
	})}
	client, err := openmed.New(
		"https://openmed.invalid/synthetic-base-secret",
		openmed.WithHTTPClient(httpClient),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = client.GetJob(context.Background(), "synthetic-patient-job-id")
	if err == nil {
		t.Fatal("GetJob unexpectedly succeeded")
	}
	for _, secret := range []string{"synthetic-base-secret", "synthetic-patient-job-id"} {
		if strings.Contains(err.Error(), secret) {
			t.Fatalf("transport error exposed %q: %v", secret, err)
		}
	}
}

func TestEndpointMethodsAndPaths(t *testing.T) {
	tests := []struct {
		name   string
		method string
		path   string
		call   func(*openmed.Client) error
	}{
		{"Analyze", http.MethodPost, "/analyze", func(c *openmed.Client) error {
			_, err := c.Analyze(context.Background(), openmed.AnalyzeRequest{Text: "note"})
			return err
		}},
		{"ExtractPII", http.MethodPost, "/pii/extract", func(c *openmed.Client) error {
			_, err := c.ExtractPII(context.Background(), openmed.PIIExtractRequest{Text: "note"})
			return err
		}},
		{"ExtractPIIStream", http.MethodPost, "/pii/extract/stream", func(c *openmed.Client) error {
			stream, err := c.ExtractPIIStream(context.Background(), openmed.PIIExtractStreamRequest{Text: "note"})
			if err != nil {
				return err
			}
			return stream.Close()
		}},
		{"Deidentify", http.MethodPost, "/pii/deidentify", func(c *openmed.Client) error {
			_, err := c.Deidentify(context.Background(), openmed.PIIDeidentifyRequest{Text: "note"})
			return err
		}},
		{"PrivacyGateway", http.MethodPost, "/privacy-gateway/complete", func(c *openmed.Client) error {
			_, err := c.PrivacyGateway(context.Background(), openmed.PrivacyGatewayRequest{Text: "note"})
			return err
		}},
		{"Health", http.MethodGet, "/health", func(c *openmed.Client) error {
			_, err := c.Health(context.Background())
			return err
		}},
		{"Livez", http.MethodGet, "/livez", func(c *openmed.Client) error {
			_, err := c.Livez(context.Background())
			return err
		}},
		{"Readyz", http.MethodGet, "/readyz", func(c *openmed.Client) error {
			_, err := c.Readyz(context.Background())
			return err
		}},
		{"LoadedModels", http.MethodGet, "/models/loaded", func(c *openmed.Client) error {
			_, err := c.LoadedModels(context.Background())
			return err
		}},
		{"UnloadModels", http.MethodPost, "/models/unload", func(c *openmed.Client) error {
			_, err := c.UnloadModels(context.Background(), openmed.ModelUnloadRequest{All: true})
			return err
		}},
		{"CreateJob", http.MethodPost, "/jobs", func(c *openmed.Client) error {
			_, err := c.CreateJob(context.Background(), openmed.DeidentifyJobRequest{
				Documents: []openmed.DeidentifyJobDocument{{Text: "note"}},
			})
			return err
		}},
		{"GetJob", http.MethodGet, "/jobs/job%2F1", func(c *openmed.Client) error {
			_, err := c.GetJob(context.Background(), "job/1")
			return err
		}},
		{"StartSMARTBackendIngestion", http.MethodPost, "/fhir/smart-backend/ingestions", func(c *openmed.Client) error {
			_, err := c.StartSMARTBackendIngestion(context.Background(), openmed.SMARTBackendIngestionRequest{})
			return err
		}},
		{"SMARTBackendIngestionStatus", http.MethodGet, "/fhir/smart-backend/ingestions/job%2F1", func(c *openmed.Client) error {
			_, err := c.SMARTBackendIngestionStatus(context.Background(), "job/1")
			return err
		}},
		{"SMARTBackendIngestionSummary", http.MethodGet, "/fhir/smart-backend/ingestions/job%2F1/summary", func(c *openmed.Client) error {
			_, err := c.SMARTBackendIngestionSummary(context.Background(), "job/1")
			return err
		}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client, _ := newServer(t, func(w http.ResponseWriter, r *http.Request) {
				if r.Method != test.method || r.URL.EscapedPath() != test.path {
					t.Fatalf("request = %s %s, want %s %s", r.Method, r.URL.EscapedPath(), test.method, test.path)
				}
				contentType := "application/json"
				if test.path == "/pii/extract/stream" {
					contentType = "application/x-ndjson"
				}
				w.Header().Set("Content-Type", contentType)
				_, _ = io.WriteString(w, "{}")
			})
			if err := test.call(client); err != nil {
				t.Fatalf("call failed: %v", err)
			}
		})
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
	const ndjson = "{\"type\":\"emit\",\"entity_id\":\"ent-1\",\"span\":{\"id\":\"ent-1\",\"label\":\"NAME\",\"start\":0,\"end\":5,\"byte_start\":0,\"byte_end\":5,\"score\":0.99,\"text\":\"Maria\"},\"audit\":{\"type\":\"emit\"}}\n{\"type\":\"final\",\"audit\":{\"type\":\"final\"}}\n"
	client, _ := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/pii/extract/stream" {
			t.Fatalf("path = %q", r.URL.Path)
		}
		if accept := r.Header.Get("Accept"); accept != "application/x-ndjson" {
			t.Fatalf("Accept = %q, want application/x-ndjson", accept)
		}
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = io.WriteString(w, ndjson)
	})

	stream, err := client.ExtractPIIStream(context.Background(), openmed.PIIExtractStreamRequest{
		Text:      "hello",
		ChunkSize: 512,
	})
	if err != nil {
		t.Fatalf("ExtractPIIStream: %v", err)
	}
	defer stream.Close()

	if !stream.Next() {
		t.Fatalf("first Next = false: %v", stream.Err())
	}
	event := stream.Event()
	if event.Type != "emit" || event.Span == nil || event.Span.Label != "NAME" {
		t.Fatalf("unexpected first event: %+v", event)
	}
	if event.Span.Text == nil || *event.Span.Text != "Maria" {
		t.Fatalf("span text = %v, want Maria", event.Span.Text)
	}
	if !stream.Next() || stream.Event().Type != "final" {
		t.Fatalf("unexpected final event: %+v, err=%v", stream.Event(), stream.Err())
	}
	if stream.Next() {
		t.Fatal("unexpected third stream event")
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream error: %v", err)
	}
}

func TestExtractPIIStreamReturnsBeforeResponseCompletes(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = io.WriteString(w, "{\"type\":\"emit\",\"audit\":{\"type\":\"emit\"}}\n")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Error("test response writer does not support flushing")
			return
		}
		flusher.Flush()
		<-release
		_, _ = io.WriteString(w, "{\"type\":\"final\",\"audit\":{\"type\":\"final\"}}\n")
	}))
	defer func() {
		select {
		case <-release:
		default:
			close(release)
		}
		srv.Close()
	}()

	client, err := openmed.New(srv.URL)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	type result struct {
		stream *openmed.PIIExtractStream
		err    error
	}
	resultCh := make(chan result, 1)
	go func() {
		stream, streamErr := client.ExtractPIIStream(
			context.Background(),
			openmed.PIIExtractStreamRequest{Text: "synthetic note"},
		)
		resultCh <- result{stream: stream, err: streamErr}
	}()

	var got result
	select {
	case got = <-resultCh:
	case <-time.After(time.Second):
		t.Fatal("ExtractPIIStream waited for the response to finish")
	}
	if got.err != nil {
		t.Fatalf("ExtractPIIStream: %v", got.err)
	}
	defer got.stream.Close()
	if !got.stream.Next() || got.stream.Event().Type != "emit" {
		t.Fatalf("first event = %+v, err=%v", got.stream.Event(), got.stream.Err())
	}
	close(release)
}

func TestExtractPIIStreamBoundsEachEvent(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = io.WriteString(w, "{\"type\":\"emit\",\"audit\":{\"padding\":\"")
		_, _ = io.WriteString(w, strings.Repeat("x", 1024))
		_, _ = io.WriteString(w, "\"}}\n")
	}))
	defer srv.Close()
	client, err := openmed.New(srv.URL, openmed.WithMaxStreamEventBytes(128))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	stream, err := client.ExtractPIIStream(
		context.Background(),
		openmed.PIIExtractStreamRequest{Text: "synthetic note"},
	)
	if err != nil {
		t.Fatalf("ExtractPIIStream: %v", err)
	}
	defer stream.Close()
	if stream.Next() {
		t.Fatal("oversized event unexpectedly decoded")
	}
	if stream.Err() == nil {
		t.Fatal("oversized event should produce a stream error")
	}
}

func TestExtractPIIStreamRejectsNonNDJSONSuccess(t *testing.T) {
	client, _ := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"error":{"code":"proxy_error"}}`)
	})

	stream, err := client.ExtractPIIStream(
		context.Background(),
		openmed.PIIExtractStreamRequest{Text: "synthetic note"},
	)
	if err == nil || !strings.Contains(err.Error(), "non-NDJSON") {
		t.Fatalf("ExtractPIIStream error = %v, want a non-NDJSON error", err)
	}
	if stream != nil {
		t.Fatal("non-NDJSON response returned a stream")
	}
}

func TestExtractPIIStreamRejectsNonObjectEvent(t *testing.T) {
	client, _ := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson; charset=utf-8")
		_, _ = io.WriteString(w, "null\n")
	})

	stream, err := client.ExtractPIIStream(
		context.Background(),
		openmed.PIIExtractStreamRequest{Text: "synthetic note"},
	)
	if err != nil {
		t.Fatalf("ExtractPIIStream: %v", err)
	}
	defer stream.Close()
	if stream.Next() {
		t.Fatal("null event unexpectedly decoded")
	}
	if err := stream.Err(); err == nil || !strings.Contains(err.Error(), "not a JSON object") {
		t.Fatalf("stream error = %v, want a JSON-object error", err)
	}
}

func TestExtractPIIStreamRejectsObjectWithoutEventType(t *testing.T) {
	client, _ := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = io.WriteString(w, `{"error":{"message":"synthetic upstream body"}}`+"\n")
	})

	stream, err := client.ExtractPIIStream(
		context.Background(),
		openmed.PIIExtractStreamRequest{Text: "synthetic note"},
	)
	if err != nil {
		t.Fatalf("ExtractPIIStream: %v", err)
	}
	defer stream.Close()
	if stream.Next() {
		t.Fatal("object without an event type unexpectedly decoded")
	}
	if err := stream.Err(); err == nil || !strings.Contains(err.Error(), "has no type") {
		t.Fatalf("stream error = %v, want a missing-type error", err)
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
	if got := apiErr.Error(); !strings.Contains(got, "422") ||
		strings.Contains(got, "validation_error") ||
		strings.Contains(got, "Text must not be blank") {
		t.Fatalf("Error() = %q, want a PHI-safe status-only summary", got)
	}
}

func TestAPIErrorDefaultStringOmitsEnvelopeMessageAndDetails(t *testing.T) {
	const sensitive = "synthetic patient Maria Garcia, MRN 12345"
	client, _ := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnprocessableEntity)
		_, _ = io.WriteString(w, `{"error":{"code":"validation_error","message":"`+sensitive+`","details":"`+sensitive+`"}}`)
	})

	_, err := client.Deidentify(context.Background(), openmed.PIIDeidentifyRequest{
		Text: "synthetic note",
	})
	apiErr, ok := openmed.AsAPIError(err)
	if !ok {
		t.Fatalf("expected *APIError, got %T", err)
	}
	if apiErr.Envelope.Error.Message != sensitive || apiErr.Details() != sensitive {
		t.Fatal("typed APIError did not preserve the explicit service envelope")
	}
	if strings.Contains(apiErr.Error(), sensitive) {
		t.Fatal("default APIError string exposed service message or details")
	}
}

func TestAPIErrorNonEnvelopeFallback(t *testing.T) {
	const sensitiveBody = "upstream rendered patient: Maria Garcia, MRN 12345"
	client, _ := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		_, _ = io.WriteString(w, sensitiveBody)
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
	if apiErr.Details() != nil {
		t.Fatalf("Details = %#v, want fail-closed nil", apiErr.Details())
	}
	encoded, marshalErr := json.Marshal(apiErr.Envelope)
	if marshalErr != nil {
		t.Fatalf("marshal fallback envelope: %v", marshalErr)
	}
	if strings.Contains(apiErr.Error(), sensitiveBody) || strings.Contains(string(encoded), sensitiveBody) {
		t.Fatal("malformed upstream body was retained in APIError")
	}
}

func TestAPIErrorRejectsUntrustedOrMalformedEnvelopes(t *testing.T) {
	const sensitive = "Maria Garcia, MRN 12345"
	tests := []struct {
		name        string
		contentType string
		body        string
	}{
		{
			name:        "wrong content type",
			contentType: "text/plain",
			body:        `{"error":{"code":"proxy_error","message":"` + sensitive + `","details":"` + sensitive + `"}}`,
		},
		{
			name:        "missing message",
			contentType: "application/json",
			body:        `{"error":{"code":"proxy_error","details":"` + sensitive + `"}}`,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client, _ := newServer(t, func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", test.contentType)
				w.WriteHeader(http.StatusBadGateway)
				_, _ = io.WriteString(w, test.body)
			})
			_, err := client.Health(context.Background())
			apiErr, ok := openmed.AsAPIError(err)
			if !ok {
				t.Fatalf("expected *APIError, got %T", err)
			}
			encoded, marshalErr := json.Marshal(apiErr.Envelope)
			if marshalErr != nil {
				t.Fatalf("marshal fallback envelope: %v", marshalErr)
			}
			if apiErr.Details() != nil || strings.Contains(string(encoded), sensitive) {
				t.Fatalf("untrusted error body was retained: %s", encoded)
			}
		})
	}
}

func TestAPIErrorFallbackReadsMalformedBodyWithBound(t *testing.T) {
	body := &countingReadCloser{remaining: 4 << 20}
	httpClient := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusBadGateway,
			Header:     make(http.Header),
			Body:       body,
			Request:    req,
		}, nil
	})}
	client, err := openmed.New("https://openmed.invalid", openmed.WithHTTPClient(httpClient))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = client.Health(context.Background())
	apiErr, ok := openmed.AsAPIError(err)
	if !ok {
		t.Fatalf("expected *APIError, got %T", err)
	}
	if apiErr.Details() != nil {
		t.Fatalf("Details = %#v, want fail-closed nil", apiErr.Details())
	}
	if body.read >= 4<<20 {
		t.Fatalf("client read an unbounded error body: %d bytes", body.read)
	}
}

func TestBufferedResponseBodyIsBounded(t *testing.T) {
	client, _ := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, strings.Repeat("x", 1024))
	})
	limited, err := openmed.New(client.BaseURL(), openmed.WithMaxResponseBodyBytes(128))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := limited.Health(context.Background()); !errors.Is(err, openmed.ErrResponseTooLarge) {
		t.Fatalf("Health error = %v, want ErrResponseTooLarge", err)
	}
}

func TestMaximumResponseLimitDoesNotOverflow(t *testing.T) {
	client, _ := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"status":"ok","service":"openmed"}`)
	})
	unbounded, err := openmed.New(
		client.BaseURL(),
		openmed.WithMaxResponseBodyBytes(math.MaxInt64),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	response, err := unbounded.Health(context.Background())
	if err != nil {
		t.Fatalf("Health: %v", err)
	}
	if response.Status != "ok" {
		t.Fatalf("Health status = %q, want ok", response.Status)
	}
}

func TestTypedResponsesRequireJSONObject(t *testing.T) {
	for _, body := range []string{"", "null", "[]", `"ok"`} {
		t.Run(body, func(t *testing.T) {
			client, _ := newServer(t, func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, body)
			})
			response, err := client.Health(context.Background())
			if err == nil || !strings.Contains(err.Error(), "not a JSON object") {
				t.Fatalf("Health error = %v, want a JSON-object error", err)
			}
			if response != nil {
				t.Fatalf("Health response = %+v, want nil", response)
			}
		})
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

func TestJobMethodsRejectAmbiguousPathSegments(t *testing.T) {
	called := false
	httpClient := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		called = true
		return nil, errors.New("unexpected request")
	})}
	client, err := openmed.New("https://openmed.invalid", openmed.WithHTTPClient(httpClient))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	calls := []struct {
		name string
		call func(string) error
	}{
		{"GetJob", func(id string) error { _, err := client.GetJob(context.Background(), id); return err }},
		{"SMARTBackendIngestionStatus", func(id string) error {
			_, err := client.SMARTBackendIngestionStatus(context.Background(), id)
			return err
		}},
		{"SMARTBackendIngestionSummary", func(id string) error {
			_, err := client.SMARTBackendIngestionSummary(context.Background(), id)
			return err
		}},
	}
	for _, call := range calls {
		for _, jobID := range []string{"", "   ", ".", ".."} {
			t.Run(call.name+"/"+jobID, func(t *testing.T) {
				if err := call.call(jobID); err == nil {
					t.Fatalf("job ID %q unexpectedly succeeded", jobID)
				}
			})
		}
	}
	if called {
		t.Fatal("invalid job ID reached the HTTP transport")
	}
}

func TestCreateJobSupportsFractionalWebhookBackoff(t *testing.T) {
	client, _ := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/jobs" {
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		var req openmed.DeidentifyJobRequest
		decodeBody(t, r, &req)
		if req.Webhook == nil || req.Webhook.BackoffSeconds == nil || *req.Webhook.BackoffSeconds != 0.25 {
			t.Fatalf("Webhook = %+v, want a 0.25-second backoff", req.Webhook)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"job-1",
			"status":"queued",
			"webhook":{
				"configured":true,
				"url_hash":"sha256",
				"max_attempts":3,
				"backoff_seconds":0.25
			}
		}`)
	})

	resp, err := client.CreateJob(context.Background(), openmed.DeidentifyJobRequest{
		Documents: []openmed.DeidentifyJobDocument{{Text: "synthetic note"}},
		Webhook: &openmed.JobWebhookRequest{
			URL:            "https://example.com/callback",
			Secret:         "synthetic-secret",
			MaxAttempts:    3,
			BackoffSeconds: ptrFloat(0.25),
		},
	})
	if err != nil {
		t.Fatalf("CreateJob: %v", err)
	}
	if resp.Webhook == nil || resp.Webhook.BackoffSeconds != 0.25 {
		t.Fatalf("Webhook = %+v, want a 0.25-second backoff", resp.Webhook)
	}
}

func TestOptionalZeroValuesAreSent(t *testing.T) {
	client, _ := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/pii/extract/stream":
			var req openmed.PIIExtractStreamRequest
			decodeBody(t, r, &req)
			if req.TokenizerContextChars == nil || *req.TokenizerContextChars != 0 {
				t.Fatalf("TokenizerContextChars = %v, want explicit zero", req.TokenizerContextChars)
			}
			w.Header().Set("Content-Type", "application/x-ndjson")
			_, _ = io.WriteString(w, "{\"event\":\"done\"}\n")
		case "/fhir/smart-backend/ingestions":
			var req openmed.SMARTBackendIngestionRequest
			decodeBody(t, r, &req)
			if req.PollIntervalSeconds == nil || *req.PollIntervalSeconds != 0 {
				t.Fatalf("PollIntervalSeconds = %v, want explicit zero", req.PollIntervalSeconds)
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"job_id":"job-1","status":"queued"}`)
		case "/jobs":
			var req openmed.DeidentifyJobRequest
			decodeBody(t, r, &req)
			if req.Webhook == nil || req.Webhook.BackoffSeconds == nil || *req.Webhook.BackoffSeconds != 0 {
				t.Fatalf("Webhook = %+v, want an explicit zero-second backoff", req.Webhook)
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"id":"job-1","status":"queued"}`)
		default:
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
	})

	stream, err := client.ExtractPIIStream(context.Background(), openmed.PIIExtractStreamRequest{
		Text:                  "synthetic note",
		TokenizerContextChars: ptrInt(0),
	})
	if err != nil {
		t.Fatalf("ExtractPIIStream: %v", err)
	}
	if err := stream.Close(); err != nil {
		t.Fatalf("close ExtractPIIStream: %v", err)
	}
	if _, err := client.StartSMARTBackendIngestion(context.Background(), openmed.SMARTBackendIngestionRequest{
		FHIRBaseURL:         "https://fhir.example.com",
		TokenURL:            "https://fhir.example.com/token",
		ClientID:            "client-id",
		PrivateKeyPEM:       "synthetic-key",
		OutputDir:           "/tmp/openmed-output",
		PollIntervalSeconds: ptrFloat(0),
	}); err != nil {
		t.Fatalf("StartSMARTBackendIngestion: %v", err)
	}
	if _, err := client.CreateJob(context.Background(), openmed.DeidentifyJobRequest{
		Documents: []openmed.DeidentifyJobDocument{{Text: "synthetic note"}},
		Webhook: &openmed.JobWebhookRequest{
			URL:            "https://example.com/callback",
			Secret:         "synthetic-secret",
			BackoffSeconds: ptrFloat(0),
		},
	}); err != nil {
		t.Fatalf("CreateJob: %v", err)
	}
}
