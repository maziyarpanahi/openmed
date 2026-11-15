package openmed_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	openmed "github.com/maziyarpanahi/openmed/clients/go"
)

// decisionFixtureCase is one exchange of the shared POST /v1/decisions
// fixture that the Python, TypeScript and Go clients replay.
type decisionFixtureCase struct {
	Name     string          `json:"name"`
	Shape    *string         `json:"shape"`
	Typed    bool            `json:"typed"`
	Request  json.RawMessage `json:"request"`
	Status   int             `json:"status"`
	Response json.RawMessage `json:"response"`
	Error    *struct {
		Code    string `json:"code"`
		Details any    `json:"details"`
	} `json:"error"`
}

func loadDecisionFixture(t *testing.T) []decisionFixtureCase {
	t.Helper()
	path := filepath.Join("..", "..", "tests", "fixtures", "service", "decision-negotiation-v1.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read shared decision fixture: %v", err)
	}
	var document struct {
		Cases []decisionFixtureCase `json:"cases"`
	}
	if err := json.Unmarshal(data, &document); err != nil {
		t.Fatalf("decode shared decision fixture: %v", err)
	}
	return document.Cases
}

func decodeJSON(t *testing.T, data []byte) any {
	t.Helper()
	var value any
	if err := json.Unmarshal(data, &value); err != nil {
		t.Fatalf("decode JSON: %v", err)
	}
	return value
}

func TestDecisionNegotiationFixturesRoundTrip(t *testing.T) {
	ran := 0
	for _, tc := range loadDecisionFixture(t) {
		if !tc.Typed || tc.Shape == nil {
			continue
		}
		ran++
		tc := tc
		t.Run(tc.Name, func(t *testing.T) {
			var sent []byte
			client, _ := newServer(t, func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodPost || r.URL.Path != "/v1/decisions" {
					t.Errorf("unexpected %s %s", r.Method, r.URL.Path)
				}
				sent, _ = io.ReadAll(r.Body)
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(tc.Status)
				if tc.Status == http.StatusOK {
					_, _ = w.Write(tc.Response)
					return
				}
				_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{
					"code":    tc.Error.Code,
					"message": "Synthetic refusal",
					"details": tc.Error.Details,
				}})
			})
			ctx := context.Background()
			var err error
			switch *tc.Shape {
			case "openmed":
				var req openmed.FixedOptionDecisionRequest
				if decodeErr := json.Unmarshal(tc.Request, &req); decodeErr != nil {
					t.Fatalf("typed request: %v", decodeErr)
				}
				var got *openmed.FixedOptionDecisionResult
				got, err = client.Decision(ctx, req)
				if err == nil {
					want := decodeJSON(t, tc.Response).(map[string]any)
					if string(got.State) != want["state"] || got.SchemaVersion != want["schema_version"] {
						t.Fatalf("result state/version = %s/%s", got.State, got.SchemaVersion)
					}
					if got.AutonomousAction || got.Advisory != want["advisory"] {
						t.Fatal("review-only provenance was not preserved")
					}
					questions, _ := want["questions"].([]any)
					if len(got.Questions) != len(questions) {
						t.Fatalf("questions = %d, want %d", len(got.Questions), len(questions))
					}
				}
			case "openai_decisions":
				var req openmed.OpenAIDecisionsRequest
				if decodeErr := json.Unmarshal(tc.Request, &req); decodeErr != nil {
					t.Fatalf("typed request: %v", decodeErr)
				}
				var got *openmed.OpenAIDecisionsResponse
				got, err = client.OpenAIDecisions(ctx, req)
				if err == nil {
					want := decodeJSON(t, tc.Response).(map[string]any)
					answers := want["answers"].([]any)
					if got.Model != want["model"] || len(got.Answers) != len(answers) {
						t.Fatalf("model/answers = %s/%d", got.Model, len(got.Answers))
					}
					for i, raw := range answers {
						answer := raw.(map[string]any)
						if got.Answers[i].Name != answer["name"] || got.Answers[i].Type != answer["type"] {
							t.Fatalf("answer %d = %s/%s", i, got.Answers[i].Name, got.Answers[i].Type)
						}
						if state, ok := answer["state"]; ok && (got.Answers[i].State == nil || string(*got.Answers[i].State) != state) {
							t.Fatalf("answer %d lost its non-success state", i)
						}
					}
					review := want["openmed"].(map[string]any)
					if !reflect.DeepEqual(got.OpenMed, openmed.JSONObject(review)) {
						t.Fatal("openmed review envelope was not preserved")
					}
				}
			default:
				t.Fatalf("unknown shape %q", *tc.Shape)
			}
			if !reflect.DeepEqual(decodeJSON(t, sent), decodeJSON(t, tc.Request)) {
				t.Fatalf("wire body differs from the shared fixture: %s", sent)
			}
			if tc.Status == http.StatusOK {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return
			}
			apiErr, ok := openmed.AsAPIError(err)
			if !ok || apiErr.StatusCode != tc.Status || apiErr.Code() != tc.Error.Code {
				t.Fatalf("error = %v, want %d %s", err, tc.Status, tc.Error.Code)
			}
		})
	}
	if ran < 4 {
		t.Fatalf("only %d typed fixture cases ran", ran)
	}
}
