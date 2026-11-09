package openmed_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httptest"

	openmed "github.com/maziyarpanahi/openmed/clients/go"
)

// The examples below run against in-process httptest servers that return
// synthetic JSON, so they need no running OpenMed service and send no data off
// the machine. Real callers pass the base URL of their own service instead.

func ExampleNew() {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"status":"healthy","service":"openmed-rest","version":"0.0.0-example","profile":"dev"}`)
	}))
	defer server.Close()

	// httptest listens on loopback, where cleartext HTTP is already accepted.
	// WithInsecureHTTP is passed here only because this is a local test server;
	// never use it for a service reached across an untrusted network.
	client, err := openmed.New(
		server.URL,
		openmed.WithHTTPClient(server.Client()),
		openmed.WithInsecureHTTP(),
	)
	if err != nil {
		log.Fatal(err)
	}
	health, err := client.Health(context.Background())
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(health.Status, health.Service)

	// Without the option, a cleartext URL for a non-loopback host is refused.
	_, err = openmed.New("http://openmed.example:8080")
	fmt.Println(err)
	// Output:
	// healthy openmed-rest
	// openmed: invalid base URL: use HTTPS for non-loopback hosts or explicitly pass WithInsecureHTTP
}

func ExampleClient_Deidentify() {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req openmed.PIIDeidentifyRequest
		if r.Method != http.MethodPost || r.URL.Path != "/pii/deidentify" ||
			json.NewDecoder(r.Body).Decode(&req) != nil || req.Method != openmed.MethodMask {
			http.Error(w, "unexpected request", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{
			"original_text": "Synthetic note for example-person.",
			"deidentified_text": "Synthetic note for [NAME].",
			"pii_entities": [{"text": "example-person", "label": "NAME",
				"entity_type": "NAME", "confidence": 0.99, "start": 19, "end": 33}],
			"method": "mask",
			"timestamp": "2000-01-01T00:00:00Z",
			"num_entities_redacted": 1
		}`)
	}))
	defer server.Close()

	client, err := openmed.New(server.URL, openmed.WithHTTPClient(server.Client()))
	if err != nil {
		log.Fatal(err)
	}
	result, err := client.Deidentify(context.Background(), openmed.PIIDeidentifyRequest{
		Text:   "Synthetic note for example-person.",
		Method: openmed.MethodMask,
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.DeidentifiedText)
	for _, entity := range result.PIIEntities {
		fmt.Printf("%s at [%d, %d)\n", entity.Label, *entity.Start, *entity.End)
	}
	// Output:
	// Synthetic note for [NAME].
	// NAME at [19, 33)
}

func ExampleClient_DeidentifyStream() {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		io.WriteString(w, `{"type":"chunk","index":0,"redacted_text":"Synthetic note for [NAME]."}`+"\n")
		io.WriteString(w, `{"type":"chunk","index":1,"redacted_text":" Follow-up in [DATE]."}`+"\n")
		io.WriteString(w, `{"type":"done"}`+"\n")
	}))
	defer server.Close()

	client, err := openmed.New(server.URL, openmed.WithHTTPClient(server.Client()))
	if err != nil {
		log.Fatal(err)
	}
	stream, err := client.DeidentifyStream(context.Background(), openmed.PIIDeidentifyStreamRequest{
		Text:      "Synthetic note for example-person. Follow-up in example-month.",
		Method:    openmed.MethodMask,
		ChunkSize: 32,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer stream.Close()
	for stream.Next() {
		event := stream.Event()
		fmt.Printf("%s %q\n", event.Type, event.RedactedText)
	}
	if err := stream.Err(); err != nil {
		log.Fatal(err)
	}
	// Output:
	// chunk "Synthetic note for [NAME]."
	// chunk " Follow-up in [DATE]."
	// done ""
}

func ExampleAsAPIError() {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnprocessableEntity)
		fmt.Fprint(w, `{"error":{"code":"validation_error","message":"text must not be blank",`+
			`"details":null,"request_id":"example-request"}}`)
	}))
	defer server.Close()

	client, err := openmed.New(server.URL, openmed.WithHTTPClient(server.Client()))
	if err != nil {
		log.Fatal(err)
	}
	_, err = client.Deidentify(context.Background(), openmed.PIIDeidentifyRequest{
		Text:   "   ",
		Method: openmed.MethodMask,
	})
	// The error string carries only the status, so it is safe to log.
	fmt.Println(err)
	if apiErr, ok := openmed.AsAPIError(err); ok {
		fmt.Println("status:", apiErr.StatusCode)
		fmt.Println("code:", apiErr.Code())
		fmt.Println("request ID:", apiErr.Envelope.Error.RequestID)
	}

	// Errors that did not come from a service response are not API errors.
	_, ok := openmed.AsAPIError(errors.New("connection refused"))
	fmt.Println("API error:", ok)
	// Output:
	// openmed: request failed with status 422
	// status: 422
	// code: validation_error
	// request ID: example-request
	// API error: false
}
