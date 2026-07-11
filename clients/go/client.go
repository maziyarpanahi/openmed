// Package openmed provides a dependency-light Go client for the OpenMed REST
// service. It is built on the standard library net/http package and mirrors the
// service request/response schemas published in the committed OpenAPI spec
// (docs/api/openapi.json).
//
// The client keeps enums (deidentification method, PII language, aggregation
// strategy) as typed string constants so callers get compile-time help, and it
// surfaces non-2xx responses as a typed *APIError carrying the service error
// envelope. Every request method takes a context.Context for cancellation and
// deadlines.
package openmed

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
)

// DefaultBaseURL is the base URL assumed when none is provided.
const DefaultBaseURL = "http://localhost:8080"

// Path templates for the endpoints that take a path parameter. The {job_id}
// placeholder mirrors the OpenAPI path exactly and is substituted with the
// URL-escaped job identifier at call time.
const (
	pathGetJob                       = "/jobs/{job_id}"
	pathSMARTBackendIngestionStatus  = "/fhir/smart-backend/ingestions/{job_id}"
	pathSMARTBackendIngestionSummary = "/fhir/smart-backend/ingestions/{job_id}/summary"
)

// AggregationStrategy selects how token-level predictions are grouped by the
// /analyze endpoint.
type AggregationStrategy string

// Aggregation strategies accepted by /analyze.
const (
	AggregationSimple  AggregationStrategy = "simple"
	AggregationFirst   AggregationStrategy = "first"
	AggregationAverage AggregationStrategy = "average"
	AggregationMax     AggregationStrategy = "max"
)

// PIILanguage is one of the languages supported by the PII endpoints.
type PIILanguage string

// Languages supported by the PII endpoints.
const (
	LangEN PIILanguage = "en"
	LangFR PIILanguage = "fr"
	LangDE PIILanguage = "de"
	LangIT PIILanguage = "it"
	LangES PIILanguage = "es"
	LangNL PIILanguage = "nl"
	LangHI PIILanguage = "hi"
	LangTE PIILanguage = "te"
	LangPT PIILanguage = "pt"
	LangAR PIILanguage = "ar"
	LangHE PIILanguage = "he"
	LangJA PIILanguage = "ja"
	LangTR PIILanguage = "tr"
	LangID PIILanguage = "id"
	LangTH PIILanguage = "th"
	LangKO PIILanguage = "ko"
	LangRO PIILanguage = "ro"
)

// DeidentificationMethod selects how detected PII spans are transformed by the
// /pii/deidentify endpoint and de-identification jobs.
type DeidentificationMethod string

// De-identification methods accepted by /pii/deidentify.
const (
	MethodMask       DeidentificationMethod = "mask"
	MethodRemove     DeidentificationMethod = "remove"
	MethodReplace    DeidentificationMethod = "replace"
	MethodHash       DeidentificationMethod = "hash"
	MethodShiftDates DeidentificationMethod = "shift_dates"
)

// PrivacyPolicy names a redaction policy for /pii/deidentify and
// /privacy-gateway/complete.
type PrivacyPolicy string

// Built-in privacy policies. Callers may also pass any custom policy name the
// service recognizes.
const (
	PolicyStrict   PrivacyPolicy = "strict"
	PolicyBalanced PrivacyPolicy = "balanced"
	PolicyMinimal  PrivacyPolicy = "minimal"
)

// JobStatus enumerates the lifecycle states of a de-identification job.
type JobStatus string

// De-identification job statuses.
const (
	JobQueued  JobStatus = "queued"
	JobRunning JobStatus = "running"
	JobDone    JobStatus = "done"
	JobFailed  JobStatus = "failed"
)

// JSONObject is an untyped JSON object, used for open-ended metadata fields.
type JSONObject = map[string]any

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

// AnalyzeRequest is the request body for POST /analyze.
type AnalyzeRequest struct {
	Text                string               `json:"text"`
	ModelName           string               `json:"model_name,omitempty"`
	ConfidenceThreshold *float64             `json:"confidence_threshold,omitempty"`
	GroupEntities       bool                 `json:"group_entities,omitempty"`
	AggregationStrategy *AggregationStrategy `json:"aggregation_strategy,omitempty"`
	SentenceDetection   *bool                `json:"sentence_detection,omitempty"`
	SentenceLanguage    string               `json:"sentence_language,omitempty"`
	SentenceClean       bool                 `json:"sentence_clean,omitempty"`
	UseFastTokenizer    *bool                `json:"use_fast_tokenizer,omitempty"`
	KeepAlive           any                  `json:"keep_alive,omitempty"`
}

// PIIExtractRequest is the request body for POST /pii/extract.
type PIIExtractRequest struct {
	Text                string      `json:"text"`
	ModelName           string      `json:"model_name,omitempty"`
	ConfidenceThreshold *float64    `json:"confidence_threshold,omitempty"`
	UseSmartMerging     *bool       `json:"use_smart_merging,omitempty"`
	Lang                PIILanguage `json:"lang,omitempty"`
	NormalizeAccents    *bool       `json:"normalize_accents,omitempty"`
	KeepAlive           any         `json:"keep_alive,omitempty"`
}

// PIIExtractStreamRequest is the request body for POST /pii/extract/stream.
type PIIExtractStreamRequest struct {
	Text                  string      `json:"text"`
	ModelName             string      `json:"model_name,omitempty"`
	ConfidenceThreshold   *float64    `json:"confidence_threshold,omitempty"`
	UseSmartMerging       *bool       `json:"use_smart_merging,omitempty"`
	Lang                  PIILanguage `json:"lang,omitempty"`
	NormalizeAccents      *bool       `json:"normalize_accents,omitempty"`
	KeepAlive             any         `json:"keep_alive,omitempty"`
	ChunkSize             int         `json:"chunk_size,omitempty"`
	WindowChars           int         `json:"window_chars,omitempty"`
	TokenizerContextChars *int        `json:"tokenizer_context_chars,omitempty"`
	MaxEntityChars        int         `json:"max_entity_chars,omitempty"`
	IncludeText           *bool       `json:"include_text,omitempty"`
}

// PIIDeidentifyRequest is the request body for POST /pii/deidentify.
type PIIDeidentifyRequest struct {
	Text                string                 `json:"text"`
	Method              DeidentificationMethod `json:"method,omitempty"`
	ModelName           string                 `json:"model_name,omitempty"`
	ConfidenceThreshold *float64               `json:"confidence_threshold,omitempty"`
	KeepYear            bool                   `json:"keep_year,omitempty"`
	ShiftDates          *bool                  `json:"shift_dates,omitempty"`
	DateShiftDays       *int                   `json:"date_shift_days,omitempty"`
	KeepMapping         bool                   `json:"keep_mapping,omitempty"`
	Policy              PrivacyPolicy          `json:"policy,omitempty"`
	UseSmartMerging     *bool                  `json:"use_smart_merging,omitempty"`
	UseSafetySweep      *bool                  `json:"use_safety_sweep,omitempty"`
	Lang                PIILanguage            `json:"lang,omitempty"`
	NormalizeAccents    *bool                  `json:"normalize_accents,omitempty"`
	KeepAlive           any                    `json:"keep_alive,omitempty"`
}

// PrivacyGatewayRequest is the request body for POST /privacy-gateway/complete.
type PrivacyGatewayRequest struct {
	Text                       string        `json:"text"`
	ModelName                  string        `json:"model_name,omitempty"`
	ConfidenceThreshold        *float64      `json:"confidence_threshold,omitempty"`
	DetectorConfidenceFloor    *float64      `json:"detector_confidence_floor,omitempty"`
	Policy                     PrivacyPolicy `json:"policy,omitempty"`
	DisallowedEntityCategories []string      `json:"disallowed_entity_categories,omitempty"`
	UseSmartMerging            *bool         `json:"use_smart_merging,omitempty"`
	Lang                       PIILanguage   `json:"lang,omitempty"`
	NormalizeAccents           *bool         `json:"normalize_accents,omitempty"`
	KeepAlive                  any           `json:"keep_alive,omitempty"`
}

// ModelUnloadRequest is the request body for POST /models/unload. Provide a
// ModelName to unload a single model, or set All to unload every inactive
// model.
type ModelUnloadRequest struct {
	ModelName string `json:"model_name,omitempty"`
	All       bool   `json:"all,omitempty"`
}

// DeidentifyJobDocument is a single document submitted to a batch job.
type DeidentifyJobDocument struct {
	Text string `json:"text"`
	ID   string `json:"id,omitempty"`
}

// JobWebhookRequest configures an optional completion webhook for a job.
type JobWebhookRequest struct {
	URL            string   `json:"url"`
	Secret         string   `json:"secret"`
	MaxAttempts    int      `json:"max_attempts,omitempty"`
	BackoffSeconds *float64 `json:"backoff_seconds,omitempty"`
}

// DeidentifyJobRequest is the request body for POST /jobs.
type DeidentifyJobRequest struct {
	Documents           []DeidentifyJobDocument `json:"documents"`
	Webhook             *JobWebhookRequest      `json:"webhook,omitempty"`
	Method              DeidentificationMethod  `json:"method,omitempty"`
	ModelName           string                  `json:"model_name,omitempty"`
	ConfidenceThreshold *float64                `json:"confidence_threshold,omitempty"`
	KeepYear            bool                    `json:"keep_year,omitempty"`
	ShiftDates          *bool                   `json:"shift_dates,omitempty"`
	DateShiftDays       *int                    `json:"date_shift_days,omitempty"`
	KeepMapping         bool                    `json:"keep_mapping,omitempty"`
	Policy              PrivacyPolicy           `json:"policy,omitempty"`
	UseSmartMerging     *bool                   `json:"use_smart_merging,omitempty"`
	UseSafetySweep      *bool                   `json:"use_safety_sweep,omitempty"`
	Lang                PIILanguage             `json:"lang,omitempty"`
	NormalizeAccents    *bool                   `json:"normalize_accents,omitempty"`
	KeepAlive           any                     `json:"keep_alive,omitempty"`
}

// SMARTBackendIngestionRequest is the request body for
// POST /fhir/smart-backend/ingestions.
type SMARTBackendIngestionRequest struct {
	FHIRBaseURL           string                 `json:"fhir_base_url"`
	TokenURL              string                 `json:"token_url"`
	ClientID              string                 `json:"client_id"`
	PrivateKeyPEM         string                 `json:"private_key_pem"`
	OutputDir             string                 `json:"output_dir"`
	CheckpointPath        string                 `json:"checkpoint_path,omitempty"`
	KeyID                 string                 `json:"key_id,omitempty"`
	Scope                 string                 `json:"scope,omitempty"`
	ExportPath            string                 `json:"export_path,omitempty"`
	MaxInflightDownloads  int                    `json:"max_inflight_downloads,omitempty"`
	PollIntervalSeconds   *float64               `json:"poll_interval_seconds,omitempty"`
	RequestTimeoutSeconds float64                `json:"request_timeout_seconds,omitempty"`
	Policy                PrivacyPolicy          `json:"policy,omitempty"`
	Method                DeidentificationMethod `json:"method,omitempty"`
	ModelName             string                 `json:"model_name,omitempty"`
	ConfidenceThreshold   *float64               `json:"confidence_threshold,omitempty"`
	UseSmartMerging       *bool                  `json:"use_smart_merging,omitempty"`
	UseSafetySweep        *bool                  `json:"use_safety_sweep,omitempty"`
	Lang                  PIILanguage            `json:"lang,omitempty"`
	NormalizeAccents      *bool                  `json:"normalize_accents,omitempty"`
	KeepAlive             any                    `json:"keep_alive,omitempty"`
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

// EntityPrediction is a single detected span in an /analyze or /pii/extract
// response.
type EntityPrediction struct {
	Text       string     `json:"text"`
	Label      string     `json:"label"`
	Confidence float64    `json:"confidence"`
	Start      *int       `json:"start"`
	End        *int       `json:"end"`
	Metadata   JSONObject `json:"metadata"`
}

// PredictionResult is the response for both /analyze and /pii/extract.
type PredictionResult struct {
	Text           string             `json:"text"`
	Entities       []EntityPrediction `json:"entities"`
	ModelName      string             `json:"model_name"`
	Timestamp      string             `json:"timestamp"`
	ProcessingTime *float64           `json:"processing_time"`
	Metadata       JSONObject         `json:"metadata"`
}

// AnalyzeResponse is returned by /analyze.
type AnalyzeResponse = PredictionResult

// PIIExtractResponse is returned by /pii/extract.
type PIIExtractResponse = PredictionResult

// PIIDeidentifiedEntity describes one redacted span in a /pii/deidentify
// response.
type PIIDeidentifiedEntity struct {
	Text           string     `json:"text"`
	Label          string     `json:"label"`
	Confidence     float64    `json:"confidence"`
	Start          *int       `json:"start"`
	End            *int       `json:"end"`
	Metadata       JSONObject `json:"metadata"`
	EntityType     string     `json:"entity_type"`
	RedactedText   *string    `json:"redacted_text"`
	CanonicalLabel *string    `json:"canonical_label"`
	Sources        []string   `json:"sources"`
	Evidence       JSONObject `json:"evidence"`
	Threshold      *float64   `json:"threshold"`
	Action         *string    `json:"action"`
	Surrogate      *string    `json:"surrogate"`
	ReversibleID   string     `json:"reversible_id,omitempty"`
}

// PIIDeidentifyResponse is returned by /pii/deidentify.
type PIIDeidentifyResponse struct {
	OriginalText        string                  `json:"original_text"`
	DeidentifiedText    string                  `json:"deidentified_text"`
	PIIEntities         []PIIDeidentifiedEntity `json:"pii_entities"`
	Method              string                  `json:"method"`
	Timestamp           string                  `json:"timestamp"`
	NumEntitiesRedacted int                     `json:"num_entities_redacted"`
	Metadata            JSONObject              `json:"metadata"`
	AuditReport         JSONObject              `json:"audit_report"`
	Mapping             map[string]string       `json:"mapping,omitempty"`
}

// PrivacyGatewayResponse is returned by /privacy-gateway/complete.
type PrivacyGatewayResponse struct {
	RequestID         string         `json:"request_id"`
	RedactedPrompt    string         `json:"redacted_prompt"`
	ExternalResponse  string         `json:"external_response"`
	ReidentifiedText  string         `json:"reidentified_text"`
	EntityCounts      map[string]int `json:"entity_counts"`
	PlaceholderHashes []string       `json:"placeholder_hashes"`
	Audit             struct {
		RecordHash string `json:"record_hash"`
		Verified   bool   `json:"verified"`
	} `json:"audit"`
}

// HealthResponse is returned by /health.
type HealthResponse struct {
	Status  string `json:"status"`
	Service string `json:"service"`
	Version string `json:"version"`
	Profile string `json:"profile"`
}

// ProbeResponse is returned by the /livez and /readyz probes.
type ProbeResponse struct {
	Status  string `json:"status"`
	Service string `json:"service"`
}

// LoadedModelsResponse is returned by /models/loaded. The service returns an
// open-ended map of runtime statistics, so the exposed struct captures the
// stable top-level fields and keeps the rest in Raw.
type LoadedModelsResponse struct {
	DefaultKeepAliveSeconds *float64              `json:"default_keep_alive_seconds"`
	MaxResidentModels       *int                  `json:"max_resident_models"`
	WarmModels              []string              `json:"warm_models"`
	Models                  map[string]JSONObject `json:"models"`
	Raw                     JSONObject            `json:"-"`
}

// ModelUnloadResponse is returned by /models/unload.
type ModelUnloadResponse struct {
	Unloaded       *bool              `json:"unloaded"`
	ModelName      string             `json:"model_name"`
	Released       map[string]float64 `json:"released"`
	ActiveRequests *int               `json:"active_requests"`
	Raw            JSONObject         `json:"-"`
}

// SMARTBackendFileResult describes one exported FHIR file in a job summary.
type SMARTBackendFileResult struct {
	Index                 int    `json:"index"`
	ResourceType          string `json:"resource_type"`
	OutputFile            string `json:"output_file"`
	ExpectedCount         *int   `json:"expected_count"`
	LinesProcessed        int    `json:"lines_processed"`
	ResourcesDeidentified int    `json:"resources_deidentified"`
	BlankLines            int    `json:"blank_lines"`
	ErrorCount            int    `json:"error_count"`
	OutputSHA256          string `json:"output_sha256"`
	Resumed               bool   `json:"resumed"`
}

// SMARTBackendIngestionSummary is returned by
// /fhir/smart-backend/ingestions/{job_id}/summary.
type SMARTBackendIngestionSummary struct {
	JobID                        string                   `json:"job_id"`
	Status                       string                   `json:"status"`
	FilesTotal                   int                      `json:"files_total"`
	FilesCompleted               int                      `json:"files_completed"`
	ResourcesDeidentified        int                      `json:"resources_deidentified"`
	LinesProcessed               int                      `json:"lines_processed"`
	ErrorCount                   int                      `json:"error_count"`
	OutputSHA256                 string                   `json:"output_sha256"`
	MaxInflightDownloadsObserved int                      `json:"max_inflight_downloads_observed"`
	StartedAt                    float64                  `json:"started_at"`
	FinishedAt                   float64                  `json:"finished_at"`
	Files                        []SMARTBackendFileResult `json:"files"`
}

// SMARTBackendJobStatus is returned by POST /fhir/smart-backend/ingestions and
// GET /fhir/smart-backend/ingestions/{job_id}.
type SMARTBackendJobStatus struct {
	JobID     string                        `json:"job_id"`
	Status    string                        `json:"status"`
	CreatedAt float64                       `json:"created_at"`
	UpdatedAt float64                       `json:"updated_at"`
	Summary   *SMARTBackendIngestionSummary `json:"summary"`
	Error     *string                       `json:"error"`
}

// JobDocumentMetadata describes one document in a job response. Only offsets and
// hashes are surfaced; no plaintext PHI is returned.
type JobDocumentMetadata struct {
	ID       string `json:"id"`
	Length   int    `json:"length"`
	TextHash string `json:"text_hash"`
}

// JobSpan is a detected span in a job response, carrying offsets and a hash
// instead of plaintext.
type JobSpan struct {
	DocumentID string   `json:"document_id"`
	Start      int      `json:"start"`
	End        int      `json:"end"`
	Label      string   `json:"label"`
	TextHash   string   `json:"text_hash"`
	Confidence *float64 `json:"confidence"`
}

// JobWebhookMetadata reports webhook configuration on a job.
type JobWebhookMetadata struct {
	Configured     bool    `json:"configured"`
	URLHash        string  `json:"url_hash"`
	MaxAttempts    int     `json:"max_attempts"`
	BackoffSeconds float64 `json:"backoff_seconds"`
}

// JobWebhookDelivery reports the outcome of a webhook delivery attempt.
type JobWebhookDelivery struct {
	Success    bool    `json:"success"`
	Attempts   int     `json:"attempts"`
	StatusCode *int    `json:"status_code"`
	Error      *string `json:"error"`
}

// JobErrorSummary describes a job-level error.
type JobErrorSummary struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// JobResponse is returned by POST /jobs and GET /jobs/{job_id}.
type JobResponse struct {
	ID              string                `json:"id"`
	Status          JobStatus             `json:"status"`
	ProgressPercent float64               `json:"progress_percent"`
	DocumentCount   int                   `json:"document_count"`
	ProcessedCount  int                   `json:"processed_count"`
	FailedCount     int                   `json:"failed_count"`
	LabelHistogram  map[string]int        `json:"label_histogram"`
	Spans           []JobSpan             `json:"spans"`
	Documents       []JobDocumentMetadata `json:"documents"`
	Error           *JobErrorSummary      `json:"error"`
	Webhook         *JobWebhookMetadata   `json:"webhook"`
	WebhookDelivery *JobWebhookDelivery   `json:"webhook_delivery"`
	CreatedAt       string                `json:"created_at"`
	UpdatedAt       string                `json:"updated_at"`
	StartedAt       *string               `json:"started_at"`
	CompletedAt     *string               `json:"completed_at"`
	ExpiresAt       string                `json:"expires_at"`
	StatusURL       string                `json:"status_url,omitempty"`
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

// ErrorEnvelope mirrors the standardized service error body: {"error": {...}}.
type ErrorEnvelope struct {
	Error ErrorBody `json:"error"`
}

// ErrorBody is the inner error payload returned by the service.
type ErrorBody struct {
	Code      string `json:"code"`
	Message   string `json:"message"`
	Details   any    `json:"details"`
	RequestID string `json:"request_id,omitempty"`
}

// APIError is returned for any non-2xx response. It preserves the HTTP status
// code and the parsed service error envelope.
type APIError struct {
	StatusCode int
	Envelope   ErrorEnvelope
}

// Error implements the error interface.
func (e *APIError) Error() string {
	return fmt.Sprintf(
		"openmed: request failed with status %d (%s): %s",
		e.StatusCode, e.Envelope.Error.Code, e.Envelope.Error.Message,
	)
}

// Code returns the service error code from the envelope.
func (e *APIError) Code() string { return e.Envelope.Error.Code }

// Details returns the service error details from the envelope.
func (e *APIError) Details() any { return e.Envelope.Error.Details }

// AsAPIError reports whether err is (or wraps) an *APIError and returns it.
func AsAPIError(err error) (*APIError, bool) {
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		return apiErr, true
	}
	return nil, false
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

// Client is a REST client for the OpenMed de-identification service. The zero
// value is not ready for use; construct one with New.
type Client struct {
	baseURL    string
	httpClient *http.Client
}

// Option configures a Client.
type Option func(*Client)

// WithHTTPClient sets a custom *http.Client (for timeouts, transports, or
// test doubles). A nil client is ignored.
func WithHTTPClient(hc *http.Client) Option {
	return func(c *Client) {
		if hc != nil {
			c.httpClient = hc
		}
	}
}

// New constructs a Client for the given base URL (for example
// "http://localhost:8080"). If baseURL is empty, DefaultBaseURL is used.
func New(baseURL string, opts ...Option) (*Client, error) {
	trimmed := strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if trimmed == "" {
		trimmed = DefaultBaseURL
	}
	parsed, err := url.ParseRequestURI(trimmed)
	if err != nil {
		return nil, fmt.Errorf("openmed: invalid base URL %q: %w", baseURL, err)
	}
	if (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" {
		return nil, fmt.Errorf(
			"openmed: invalid base URL %q: expected an absolute HTTP(S) URL",
			baseURL,
		)
	}
	if parsed.RawQuery != "" || parsed.ForceQuery || parsed.Fragment != "" {
		return nil, fmt.Errorf(
			"openmed: invalid base URL %q: query parameters and fragments are not supported",
			baseURL,
		)
	}
	c := &Client{
		baseURL:    trimmed,
		httpClient: http.DefaultClient,
	}
	for _, opt := range opts {
		opt(c)
	}
	return c, nil
}

// BaseURL returns the normalized base URL the client targets.
func (c *Client) BaseURL() string { return c.baseURL }

// Analyze calls POST /analyze.
func (c *Client) Analyze(ctx context.Context, req AnalyzeRequest) (*AnalyzeResponse, error) {
	var out AnalyzeResponse
	if err := c.post(ctx, "/analyze", req, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// ExtractPII calls POST /pii/extract.
func (c *Client) ExtractPII(ctx context.Context, req PIIExtractRequest) (*PIIExtractResponse, error) {
	var out PIIExtractResponse
	if err := c.post(ctx, "/pii/extract", req, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// ExtractPIIStream calls POST /pii/extract/stream and returns the raw
// newline-delimited JSON (NDJSON) stream body. Each line is a JSON object.
func (c *Client) ExtractPIIStream(ctx context.Context, req PIIExtractStreamRequest) (string, error) {
	body, err := c.rawWithAccept(
		ctx,
		http.MethodPost,
		"/pii/extract/stream",
		req,
		"application/x-ndjson",
	)
	if err != nil {
		return "", err
	}
	return string(body), nil
}

// Deidentify calls POST /pii/deidentify.
func (c *Client) Deidentify(ctx context.Context, req PIIDeidentifyRequest) (*PIIDeidentifyResponse, error) {
	var out PIIDeidentifyResponse
	if err := c.post(ctx, "/pii/deidentify", req, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// PrivacyGateway calls POST /privacy-gateway/complete.
func (c *Client) PrivacyGateway(ctx context.Context, req PrivacyGatewayRequest) (*PrivacyGatewayResponse, error) {
	var out PrivacyGatewayResponse
	if err := c.post(ctx, "/privacy-gateway/complete", req, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// Health calls GET /health.
func (c *Client) Health(ctx context.Context) (*HealthResponse, error) {
	var out HealthResponse
	if err := c.get(ctx, "/health", &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// Livez calls GET /livez.
func (c *Client) Livez(ctx context.Context) (*ProbeResponse, error) {
	var out ProbeResponse
	if err := c.get(ctx, "/livez", &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// Readyz calls GET /readyz.
func (c *Client) Readyz(ctx context.Context) (*ProbeResponse, error) {
	var out ProbeResponse
	if err := c.get(ctx, "/readyz", &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// LoadedModels calls GET /models/loaded.
func (c *Client) LoadedModels(ctx context.Context) (*LoadedModelsResponse, error) {
	body, err := c.raw(ctx, http.MethodGet, "/models/loaded", nil)
	if err != nil {
		return nil, err
	}
	var out LoadedModelsResponse
	if err := json.Unmarshal(body, &out); err != nil {
		return nil, fmt.Errorf("openmed: decode /models/loaded response: %w", err)
	}
	if err := json.Unmarshal(body, &out.Raw); err != nil {
		return nil, fmt.Errorf("openmed: decode /models/loaded response: %w", err)
	}
	return &out, nil
}

// UnloadModels calls POST /models/unload.
func (c *Client) UnloadModels(ctx context.Context, req ModelUnloadRequest) (*ModelUnloadResponse, error) {
	body, err := c.raw(ctx, http.MethodPost, "/models/unload", req)
	if err != nil {
		return nil, err
	}
	var out ModelUnloadResponse
	if err := json.Unmarshal(body, &out); err != nil {
		return nil, fmt.Errorf("openmed: decode /models/unload response: %w", err)
	}
	if err := json.Unmarshal(body, &out.Raw); err != nil {
		return nil, fmt.Errorf("openmed: decode /models/unload response: %w", err)
	}
	return &out, nil
}

// CreateJob calls POST /jobs.
func (c *Client) CreateJob(ctx context.Context, req DeidentifyJobRequest) (*JobResponse, error) {
	var out JobResponse
	if err := c.post(ctx, "/jobs", req, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// GetJob calls GET /jobs/{job_id}.
func (c *Client) GetJob(ctx context.Context, jobID string) (*JobResponse, error) {
	var out JobResponse
	if err := c.get(ctx, fillJobID(pathGetJob, jobID), &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// StartSMARTBackendIngestion calls POST /fhir/smart-backend/ingestions.
func (c *Client) StartSMARTBackendIngestion(ctx context.Context, req SMARTBackendIngestionRequest) (*SMARTBackendJobStatus, error) {
	var out SMARTBackendJobStatus
	if err := c.post(ctx, "/fhir/smart-backend/ingestions", req, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// SMARTBackendIngestionStatus calls GET /fhir/smart-backend/ingestions/{job_id}.
func (c *Client) SMARTBackendIngestionStatus(ctx context.Context, jobID string) (*SMARTBackendJobStatus, error) {
	var out SMARTBackendJobStatus
	if err := c.get(ctx, fillJobID(pathSMARTBackendIngestionStatus, jobID), &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// SMARTBackendIngestionSummary calls
// GET /fhir/smart-backend/ingestions/{job_id}/summary.
func (c *Client) SMARTBackendIngestionSummary(ctx context.Context, jobID string) (*SMARTBackendIngestionSummary, error) {
	var out SMARTBackendIngestionSummary
	if err := c.get(ctx, fillJobID(pathSMARTBackendIngestionSummary, jobID), &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// ---------------------------------------------------------------------------
// Transport helpers
// ---------------------------------------------------------------------------

func (c *Client) get(ctx context.Context, path string, out any) error {
	body, err := c.raw(ctx, http.MethodGet, path, nil)
	if err != nil {
		return err
	}
	return decodeInto(path, body, out)
}

func (c *Client) post(ctx context.Context, path string, in, out any) error {
	body, err := c.raw(ctx, http.MethodPost, path, in)
	if err != nil {
		return err
	}
	return decodeInto(path, body, out)
}

// fillJobID substitutes the {job_id} placeholder in a path template with the
// URL-escaped job identifier.
func fillJobID(template, jobID string) string {
	return strings.Replace(template, "{job_id}", url.PathEscape(jobID), 1)
}

func decodeInto(path string, body []byte, out any) error {
	if out == nil || len(body) == 0 {
		return nil
	}
	if err := json.Unmarshal(body, out); err != nil {
		return fmt.Errorf("openmed: decode %s response: %w", path, err)
	}
	return nil
}

// raw issues a request and returns the response body for 2xx responses, or an
// *APIError for any non-2xx response.
func (c *Client) raw(ctx context.Context, method, path string, in any) ([]byte, error) {
	return c.rawWithAccept(ctx, method, path, in, "application/json")
}

func (c *Client) rawWithAccept(
	ctx context.Context,
	method, path string,
	in any,
	accept string,
) ([]byte, error) {
	var reader io.Reader
	if in != nil {
		encoded, err := json.Marshal(in)
		if err != nil {
			return nil, fmt.Errorf("openmed: encode %s request: %w", path, err)
		}
		reader = bytes.NewReader(encoded)
	}

	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, reader)
	if err != nil {
		return nil, fmt.Errorf("openmed: build %s request: %w", path, err)
	}
	req.Header.Set("Accept", accept)
	if in != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("openmed: %s %s: %w", method, path, err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("openmed: read %s response: %w", path, err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, newAPIError(resp.StatusCode, body)
	}
	return body, nil
}

func newAPIError(status int, body []byte) *APIError {
	apiErr := &APIError{StatusCode: status}
	var envelope ErrorEnvelope
	if len(body) > 0 && json.Unmarshal(body, &envelope) == nil && envelope.Error.Code != "" {
		apiErr.Envelope = envelope
		return apiErr
	}
	// Fall back to a synthetic envelope when the body is not a recognized
	// service error (e.g. an upstream proxy error page).
	apiErr.Envelope = ErrorEnvelope{
		Error: ErrorBody{
			Code:    "http_error",
			Message: fmt.Sprintf("request failed with status %d", status),
			Details: string(body),
		},
	}
	return apiErr
}
