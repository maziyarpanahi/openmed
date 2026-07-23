// Package openmed provides a dependency-light Go client for the OpenMed REST
// service. It is built on the standard library net/http package and tracks the
// operations and request schemas published in the committed OpenAPI spec
// (docs/api/openapi.json), with typed representations for stable responses.
//
// The client keeps enums (deidentification method, PII language, aggregation
// strategy) as typed string constants so callers get compile-time help, and it
// surfaces non-2xx responses as a typed *APIError carrying the service error
// envelope. Every request method takes a context.Context for cancellation and
// deadlines.
package openmed

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"mime"
	"net"
	"net/http"
	"net/url"
	"strings"
)

// DefaultBaseURL is the base URL assumed when none is provided.
const DefaultBaseURL = "http://localhost:8080"

// DefaultMaxResponseBodyBytes bounds buffered JSON responses. Streaming
// responses are read incrementally and are not subject to this total limit.
const DefaultMaxResponseBodyBytes int64 = 64 << 20

// DefaultMaxStreamEventBytes bounds one NDJSON event returned by
// /pii/extract/stream. The total stream remains unbounded and is consumed
// incrementally.
const DefaultMaxStreamEventBytes = 4 << 20

const maxErrorBodyBytes int64 = 1 << 20

// ErrResponseTooLarge is returned when a buffered success response exceeds the
// configured response-body limit.
var ErrResponseTooLarge = errors.New("openmed: response body exceeds configured limit")

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
	LangAM PIILanguage = "am"
	LangAS PIILanguage = "as"
	LangBN PIILanguage = "bn"
	LangEN PIILanguage = "en"
	LangFR PIILanguage = "fr"
	LangDE PIILanguage = "de"
	LangIT PIILanguage = "it"
	LangES PIILanguage = "es"
	LangNL PIILanguage = "nl"
	LangHI PIILanguage = "hi"
	LangGU PIILanguage = "gu"
	LangKN PIILanguage = "kn"
	LangML PIILanguage = "ml"
	LangMR PIILanguage = "mr"
	LangOR PIILanguage = "or"
	LangPA PIILanguage = "pa"
	LangTA PIILanguage = "ta"
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
	LangRU PIILanguage = "ru"
	LangSV PIILanguage = "sv"
	LangDA PIILanguage = "da"
	LangNO PIILanguage = "no"
	LangSW PIILanguage = "sw"
	LangZU PIILanguage = "zu"
	LangXH PIILanguage = "xh"
	LangZH PIILanguage = "zh"
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

// PIIExtractStreamSpan is one entity span in a streaming PII event.
type PIIExtractStreamSpan struct {
	ID        string  `json:"id"`
	Label     string  `json:"label"`
	Start     int     `json:"start"`
	End       int     `json:"end"`
	ByteStart *int    `json:"byte_start"`
	ByteEnd   *int    `json:"byte_end"`
	Score     float64 `json:"score"`
	Text      *string `json:"text,omitempty"`
}

// PIIExtractStreamEvent is one NDJSON object returned by
// POST /pii/extract/stream.
type PIIExtractStreamEvent struct {
	Type        string                 `json:"type"`
	EntityID    *string                `json:"entity_id,omitempty"`
	Span        *PIIExtractStreamSpan  `json:"span,omitempty"`
	Reason      *string                `json:"reason,omitempty"`
	FinalSpans  []PIIExtractStreamSpan `json:"final_spans,omitempty"`
	LatencyMS   *float64               `json:"latency_ms,omitempty"`
	WindowChars *int                   `json:"window_chars,omitempty"`
	Audit       JSONObject             `json:"audit"`
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

// JobDocumentMetadata describes one document in a job response. Document text
// is represented by its length and hash. ID echoes the caller-supplied
// identifier, so callers should keep document identifiers free of PHI.
type JobDocumentMetadata struct {
	ID       string `json:"id"`
	Length   int    `json:"length"`
	TextHash string `json:"text_hash"`
}

// JobSpan is a detected span in a job response, carrying offsets and a hash
// instead of plaintext. DocumentID echoes the caller-supplied document ID.
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

// Error implements the error interface with a status-only summary. Service
// messages and details remain available on Envelope but are omitted here so
// routine error logging does not copy possible clinical text.
func (e *APIError) Error() string {
	return fmt.Sprintf("openmed: request failed with status %d", e.StatusCode)
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
	baseURL              string
	httpClient           *http.Client
	maxResponseBodyBytes int64
	maxStreamEventBytes  int
	allowInsecureHTTP    bool
}

// PIIExtractStream incrementally decodes the NDJSON response from
// /pii/extract/stream. Callers must close the stream, normally with defer.
type PIIExtractStream struct {
	body    io.ReadCloser
	scanner *bufio.Scanner
	event   PIIExtractStreamEvent
	err     error
	closed  bool
}

// Next advances to the next event. It returns false at EOF or after an error.
func (s *PIIExtractStream) Next() bool {
	if s == nil || s.closed || s.err != nil {
		return false
	}
	if !s.scanner.Scan() {
		if err := s.scanner.Err(); err != nil {
			s.err = fmt.Errorf("openmed: read /pii/extract/stream event: %w", err)
		}
		_ = s.closeBody()
		return false
	}

	var event PIIExtractStreamEvent
	if err := decodeJSONObject(
		"/pii/extract/stream event",
		s.scanner.Bytes(),
		&event,
	); err != nil {
		s.err = err
		_ = s.closeBody()
		return false
	}
	if strings.TrimSpace(event.Type) == "" {
		s.err = errors.New("openmed: /pii/extract/stream event has no type")
		_ = s.closeBody()
		return false
	}
	s.event = event
	return true
}

// Event returns the event decoded by the most recent successful Next call.
func (s *PIIExtractStream) Event() PIIExtractStreamEvent {
	if s == nil {
		return PIIExtractStreamEvent{}
	}
	return s.event
}

// Err returns the first stream scanning or JSON decoding error, if any.
func (s *PIIExtractStream) Err() error {
	if s == nil {
		return nil
	}
	return s.err
}

// Close closes the response body. It is safe to call more than once.
func (s *PIIExtractStream) Close() error {
	if s == nil {
		return nil
	}
	return s.closeBody()
}

func (s *PIIExtractStream) closeBody() error {
	if s.closed {
		return nil
	}
	s.closed = true
	return s.body.Close()
}

// Option configures a Client.
type Option func(*Client)

// WithHTTPClient sets a custom *http.Client (for timeouts, transports, redirect
// policy, or test doubles). A nil client is ignored. The configuration is
// copied without copying the client's internal synchronization state. When the
// supplied client has no CheckRedirect policy, redirects remain disabled. A
// non-nil policy is treated as an explicit override.
func WithHTTPClient(hc *http.Client) Option {
	return func(c *Client) {
		if hc != nil {
			cloned := &http.Client{
				Transport:     hc.Transport,
				CheckRedirect: hc.CheckRedirect,
				Jar:           hc.Jar,
				Timeout:       hc.Timeout,
			}
			if cloned.CheckRedirect == nil {
				cloned.CheckRedirect = refuseRedirect
			}
			c.httpClient = cloned
		}
	}
}

// WithInsecureHTTP explicitly permits cleartext HTTP for a non-loopback base
// URL. Prefer HTTPS. This option is intended only for trusted, isolated
// networks where transport security is provided outside the application.
func WithInsecureHTTP() Option {
	return func(c *Client) {
		c.allowInsecureHTTP = true
	}
}

// WithMaxResponseBodyBytes sets the maximum size of a buffered JSON success
// response. Non-positive values are ignored. Streaming endpoint bodies are not
// buffered and are governed by WithMaxStreamEventBytes instead.
func WithMaxResponseBodyBytes(limit int64) Option {
	return func(c *Client) {
		if limit > 0 {
			c.maxResponseBodyBytes = limit
		}
	}
}

// WithMaxStreamEventBytes sets the maximum size of one NDJSON event. The total
// stream is consumed incrementally and is not capped. Non-positive values are
// ignored.
func WithMaxStreamEventBytes(limit int) Option {
	return func(c *Client) {
		if limit > 0 {
			c.maxStreamEventBytes = limit
		}
	}
}

// New constructs a Client for the given base URL (for example
// "http://localhost:8080"). If baseURL is empty, DefaultBaseURL is used.
// Cleartext HTTP is accepted for loopback hosts only unless WithInsecureHTTP is
// passed. Redirects are disabled unless a custom client supplies an explicit
// CheckRedirect policy.
func New(baseURL string, opts ...Option) (*Client, error) {
	trimmed := strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if trimmed == "" {
		trimmed = DefaultBaseURL
	}
	parsed, err := url.ParseRequestURI(trimmed)
	if err != nil {
		return nil, errors.New("openmed: invalid base URL: malformed URL")
	}
	if (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Hostname() == "" {
		return nil, errors.New(
			"openmed: invalid base URL: expected an absolute HTTP(S) URL",
		)
	}
	if parsed.User != nil {
		return nil, errors.New(
			"openmed: invalid base URL: user information is not supported",
		)
	}
	if parsed.RawQuery != "" || parsed.ForceQuery || parsed.Fragment != "" {
		return nil, errors.New(
			"openmed: invalid base URL: query parameters and fragments are not supported",
		)
	}
	for _, segment := range strings.Split(parsed.Path, "/") {
		if segment == "." || segment == ".." {
			return nil, errors.New(
				"openmed: invalid base URL: dot path segments are not supported",
			)
		}
	}
	c := &Client{
		baseURL: trimmed,
		httpClient: &http.Client{
			CheckRedirect: refuseRedirect,
		},
		maxResponseBodyBytes: DefaultMaxResponseBodyBytes,
		maxStreamEventBytes:  DefaultMaxStreamEventBytes,
	}
	for _, opt := range opts {
		if opt != nil {
			opt(c)
		}
	}
	if parsed.Scheme == "http" &&
		!isLoopbackHost(parsed.Hostname()) &&
		!c.allowInsecureHTTP {
		return nil, errors.New(
			"openmed: invalid base URL: use HTTPS for non-loopback hosts or explicitly pass WithInsecureHTTP",
		)
	}
	return c, nil
}

func refuseRedirect(_ *http.Request, _ []*http.Request) error {
	return http.ErrUseLastResponse
}

func isLoopbackHost(host string) bool {
	normalized := strings.TrimSuffix(strings.ToLower(host), ".")
	if normalized == "localhost" {
		return true
	}
	if zone := strings.LastIndexByte(normalized, '%'); zone >= 0 {
		normalized = normalized[:zone]
	}
	ip := net.ParseIP(normalized)
	return ip != nil && ip.IsLoopback()
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

// ExtractPIIStream calls POST /pii/extract/stream and returns an incremental,
// bounded-per-event NDJSON decoder. The caller must close the returned stream.
func (c *Client) ExtractPIIStream(ctx context.Context, req PIIExtractStreamRequest) (*PIIExtractStream, error) {
	resp, err := c.do(
		ctx,
		http.MethodPost,
		"/pii/extract/stream",
		req,
		"application/x-ndjson",
	)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		defer resp.Body.Close()
		return nil, apiErrorFromResponse(resp)
	}
	if !isNDJSONContentType(resp.Header.Get("Content-Type")) {
		_ = resp.Body.Close()
		return nil, errors.New(
			"openmed: /pii/extract/stream returned a non-NDJSON response",
		)
	}

	scanner := bufio.NewScanner(resp.Body)
	initialBufferSize := 64 << 10
	if c.maxStreamEventBytes < initialBufferSize {
		initialBufferSize = c.maxStreamEventBytes
	}
	scanner.Buffer(make([]byte, initialBufferSize), c.maxStreamEventBytes)
	return &PIIExtractStream{body: resp.Body, scanner: scanner}, nil
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
	if err := decodeInto("/models/loaded", body, &out); err != nil {
		return nil, err
	}
	if err := decodeInto("/models/loaded", body, &out.Raw); err != nil {
		return nil, err
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
	if err := decodeInto("/models/unload", body, &out); err != nil {
		return nil, err
	}
	if err := decodeInto("/models/unload", body, &out.Raw); err != nil {
		return nil, err
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
	path, err := fillJobID(pathGetJob, jobID)
	if err != nil {
		return nil, err
	}
	var out JobResponse
	if err := c.get(ctx, path, &out); err != nil {
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
	path, err := fillJobID(pathSMARTBackendIngestionStatus, jobID)
	if err != nil {
		return nil, err
	}
	var out SMARTBackendJobStatus
	if err := c.get(ctx, path, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// SMARTBackendIngestionSummary calls
// GET /fhir/smart-backend/ingestions/{job_id}/summary.
func (c *Client) SMARTBackendIngestionSummary(ctx context.Context, jobID string) (*SMARTBackendIngestionSummary, error) {
	path, err := fillJobID(pathSMARTBackendIngestionSummary, jobID)
	if err != nil {
		return nil, err
	}
	var out SMARTBackendIngestionSummary
	if err := c.get(ctx, path, &out); err != nil {
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
func fillJobID(template, jobID string) (string, error) {
	if strings.TrimSpace(jobID) == "" || jobID == "." || jobID == ".." {
		return "", errors.New("openmed: job ID must not be empty or a dot path segment")
	}
	return strings.Replace(template, "{job_id}", url.PathEscape(jobID), 1), nil
}

func decodeInto(path string, body []byte, out any) error {
	if out == nil {
		return nil
	}
	return decodeJSONObject(endpointLabel(path)+" response", body, out)
}

func decodeJSONObject(label string, body []byte, out any) error {
	trimmed := bytes.TrimSpace(body)
	if len(trimmed) == 0 || trimmed[0] != '{' {
		return fmt.Errorf("openmed: %s is not a JSON object", label)
	}
	if err := json.Unmarshal(trimmed, out); err != nil {
		return fmt.Errorf("openmed: decode %s: %w", label, err)
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
	resp, err := c.do(ctx, method, path, in, accept)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, apiErrorFromResponse(resp)
	}
	body, err := readBounded(resp.Body, c.maxResponseBodyBytes)
	if err != nil {
		return nil, fmt.Errorf(
			"openmed: read %s response: %w",
			endpointLabel(path),
			err,
		)
	}
	return body, nil
}

func (c *Client) do(
	ctx context.Context,
	method, path string,
	in any,
	accept string,
) (*http.Response, error) {
	label := endpointLabel(path)
	var reader io.Reader
	if in != nil {
		encoded, err := json.Marshal(in)
		if err != nil {
			return nil, fmt.Errorf("openmed: encode %s request: %w", label, err)
		}
		reader = bytes.NewReader(encoded)
	}

	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, reader)
	if err != nil {
		return nil, fmt.Errorf("openmed: build %s request", label)
	}
	req.Header.Set("Accept", accept)
	if in != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf(
			"openmed: %s %s: %w",
			method,
			label,
			transportErrorCause(err),
		)
	}
	return resp, nil
}

func endpointLabel(path string) string {
	if strings.HasPrefix(path, "/jobs/") {
		return pathGetJob
	}
	if strings.HasPrefix(path, "/fhir/smart-backend/ingestions/") {
		if strings.HasSuffix(path, "/summary") {
			return pathSMARTBackendIngestionSummary
		}
		return pathSMARTBackendIngestionStatus
	}
	return path
}

func transportErrorCause(err error) error {
	var urlErr *url.Error
	if errors.As(err, &urlErr) && urlErr.Err != nil {
		return urlErr.Err
	}
	return err
}

func readBounded(reader io.Reader, limit int64) ([]byte, error) {
	if limit == math.MaxInt64 {
		return io.ReadAll(reader)
	}
	body, err := io.ReadAll(io.LimitReader(reader, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(body)) > limit {
		return nil, ErrResponseTooLarge
	}
	return body, nil
}

func apiErrorFromResponse(resp *http.Response) *APIError {
	mediaType, _, err := mime.ParseMediaType(resp.Header.Get("Content-Type"))
	if err != nil || mediaType != "application/json" {
		return newAPIError(resp.StatusCode, nil)
	}
	body, err := readBounded(resp.Body, maxErrorBodyBytes)
	if err != nil {
		body = nil
	}
	return newAPIError(resp.StatusCode, body)
}

func isNDJSONContentType(value string) bool {
	mediaType, _, err := mime.ParseMediaType(value)
	if err != nil {
		return false
	}
	return mediaType == "application/x-ndjson" || mediaType == "application/ndjson"
}

func newAPIError(status int, body []byte) *APIError {
	apiErr := &APIError{StatusCode: status}
	var envelope ErrorEnvelope
	if len(body) > 0 &&
		json.Unmarshal(body, &envelope) == nil &&
		envelope.Error.Code != "" &&
		envelope.Error.Message != "" {
		apiErr.Envelope = envelope
		return apiErr
	}
	// Fall back to a synthetic envelope when the body is not a recognized
	// service error (for example, an upstream proxy error page). Never retain
	// an unrecognized body: it may contain raw clinical text or credentials.
	apiErr.Envelope = ErrorEnvelope{
		Error: ErrorBody{
			Code:    "http_error",
			Message: fmt.Sprintf("request failed with status %d", status),
			Details: nil,
		},
	}
	return apiErr
}
